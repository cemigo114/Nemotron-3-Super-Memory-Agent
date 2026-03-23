"""
Agentic conversation loop for Nemotron-3-Super with persistent memory.

Uses the OpenAI-compatible API exposed by vLLM to send chat completions
with tool definitions. When the model emits a tool_call for the memory
tool, the call is routed to MemoryBackend and the result is fed back
until the model produces a final text response (finish_reason == "stop").

Requires a running vLLM server:
    vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
      --kv-cache-dtype fp8 --tensor-parallel-size 4 --trust-remote-code \
      --served-model-name nemotron \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser nemotron_v3
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from memory_backend import MemoryBackend, MEMORY_TOOL_SCHEMA

load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "nemotron")

MAX_TOOL_ROUNDS = 20
CONTEXT_TOKEN_BUDGET = 30_000
KEEP_TOOL_RESULTS = 3

SYSTEM_PROMPT = """\
You are a helpful assistant with persistent memory.

IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Use the `view` command of your `memory` tool to check /memories for earlier progress.
2. Work on the user's task.
3. As you make progress, record status / progress / thoughts in your memory.
ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk \
losing any progress that is not recorded in your memory directory.

Memory guidelines:
- Do NOT store raw conversation history.
- Store facts about the user and their preferences.
- Before responding, check memory to adjust technical depth and response style.
- Keep memories up-to-date — remove outdated info, add new details as you learn them.
- Use an xml format like <fact type="preference">John prefers dark mode</fact>.
- When editing your memory folder, keep content coherent and organized.
"""


def build_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


def trim_context(messages: list[dict], keep_system: bool = True) -> list[dict]:
    """Drop oldest assistant+tool-result turns when the conversation is too long.

    Drops entire turns (assistant message + all its tool results) to avoid
    orphaning tool_call IDs. Uses 4 chars ~ 1 token as a rough heuristic.
    """
    total_chars = sum(
        len(json.dumps(m, default=str)) for m in messages
    )
    approx_tokens = total_chars // 4

    if approx_tokens <= CONTEXT_TOKEN_BUDGET:
        return messages

    trimmed = list(messages)
    start = 1 if keep_system and trimmed and trimmed[0]["role"] == "system" else 0

    # Identify complete tool turns: (assistant_with_tool_calls, [tool_result, ...])
    turns_to_drop: list[set[int]] = []
    i = start
    while i < len(trimmed):
        m = trimmed[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            turn_indices = {i}
            tc_ids = {tc["id"] for tc in m["tool_calls"]}
            j = i + 1
            while j < len(trimmed) and trimmed[j].get("role") == "tool":
                if trimmed[j].get("tool_call_id") in tc_ids:
                    turn_indices.add(j)
                j += 1
            turns_to_drop.append(turn_indices)
            i = j
        else:
            i += 1

    # Drop oldest turns first, keeping the most recent KEEP_TOOL_RESULTS
    drop_count = max(0, len(turns_to_drop) - KEEP_TOOL_RESULTS)
    indices_to_drop: set[int] = set()
    for turn in turns_to_drop[:drop_count]:
        indices_to_drop.update(turn)

    return [m for i, m in enumerate(trimmed) if i not in indices_to_drop]


def agent_turn(
    client: OpenAI,
    messages: list[dict],
    memory: MemoryBackend,
) -> str:
    """Run one full agent turn: call the model, handle tool calls, repeat."""
    tools = [MEMORY_TOOL_SCHEMA]

    for round_num in range(MAX_TOOL_ROUNDS):
        messages = trim_context(messages)

        try:
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                tools=tools,
                temperature=1.0,
                top_p=0.95,
                max_tokens=2048,
            )
        except Exception as exc:
            return f"[API error: {exc}]"

        if not response.choices:
            return "[API error: Empty response — no choices returned]"

        choice = response.choices[0]
        assistant_msg: dict = {"role": "assistant"}

        if choice.message.content:
            assistant_msg["content"] = choice.message.content

        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        messages.append(assistant_msg)

        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            return choice.message.content or ""

        for tc in choice.message.tool_calls:
            try:
                raw = tc.function.arguments
                if raw is None:
                    raise TypeError("arguments is None")
                args = json.loads(raw)
                if not isinstance(args, dict):
                    raise TypeError(f"Expected dict, got {type(args).__name__}")
            except (json.JSONDecodeError, TypeError) as parse_err:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: Failed to parse tool arguments: {parse_err}",
                    }
                )
                continue

            result = memory.execute(args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return "[Max tool rounds reached. Forcing response.]"


def conversation_loop() -> None:
    client = build_client()
    memory = MemoryBackend()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Nemotron Memory Agent")
    print("Commands: /quit  /clear  /memory_view  /memory_clear  /debug")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared.")
            continue

        if user_input.lower() == "/memory_view":
            result = memory.execute({"command": "view", "path": "/memories"})
            print(f"\n{result}")
            continue

        if user_input.lower() == "/memory_clear":
            print(memory.clear_all())
            continue

        if user_input.lower() == "/debug":
            total_chars = sum(len(json.dumps(m, default=str)) for m in messages)
            print(f"\nMessages: {len(messages)}")
            print(f"Approx tokens: {total_chars // 4:,}")
            for i, m in enumerate(messages):
                role = m["role"].upper()
                if m.get("tool_calls"):
                    names = [tc["function"]["name"] for tc in m["tool_calls"]]
                    print(f"  [{i}] {role}: tool_calls={names}")
                elif m["role"] == "tool":
                    preview = m.get("content", "")[:80]
                    print(f"  [{i}] {role}: {preview}...")
                else:
                    preview = str(m.get("content", ""))[:80]
                    print(f"  [{i}] {role}: {preview}...")
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        reply = agent_turn(client, messages, memory)
        print(reply)

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    conversation_loop()
