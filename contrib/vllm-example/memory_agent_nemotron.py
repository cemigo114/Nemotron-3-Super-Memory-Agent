"""
Memory Agent with Nemotron-3-Super via vLLM
============================================

This example demonstrates how to build a persistent memory agent using
NVIDIA Nemotron-3-Super-120B served via vLLM with tool calling enabled.

The agent can read, write, and organize files in a local ./memories/
directory, allowing it to retain facts and preferences across conversations.
This pattern is inspired by the Anthropic memory tool specification but
works with any OpenAI-compatible server.

Prerequisites:
    pip install openai

Start the vLLM server first:
    vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
        --kv-cache-dtype fp8 \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --served-model-name nemotron \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser nemotron_v3

Then run:
    python memory_agent_nemotron.py
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("VLLM_MODEL", "nemotron")
MEMORY_DIR = Path("./memories")

MAX_TOOL_ROUNDS = 20

SYSTEM_PROMPT = """\
You are a helpful assistant with persistent memory.

IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Use the `view` command of your `memory` tool to check /memories for earlier progress.
2. Work on the user's task.
3. As you make progress, record status / progress / thoughts in your memory.

Memory guidelines:
- Do NOT store raw conversation history.
- Store facts about the user and their preferences.
- Use an xml format like <fact type="preference">John prefers dark mode</fact>.
"""

# ---------------------------------------------------------------------------
# Memory tool schema (OpenAI function calling format)
# ---------------------------------------------------------------------------

MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "Persistent memory store. Commands: view (list dir or read file), "
            "create (new file), str_replace (edit), insert (insert at line), "
            "delete (remove), rename (move). All paths under /memories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace",
                             "insert", "delete", "rename"],
                },
                "path": {"type": "string"},
                "file_text": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "insert_line": {"type": "integer"},
                "insert_text": {"type": "string"},
                "view_range": {"type": "array", "items": {"type": "integer"}},
                "old_path": {"type": "string"},
                "new_path": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}

# ---------------------------------------------------------------------------
# Memory backend (file-based, sandboxed under MEMORY_DIR)
# ---------------------------------------------------------------------------


def resolve_path(virtual: str) -> Path:
    """Map /memories/... to a real path, blocking traversal."""
    clean = virtual.replace("\\", "/")
    if clean.startswith("/memories"):
        clean = clean[len("/memories"):]
    clean = clean.lstrip("/")
    real = (MEMORY_DIR / clean).resolve()
    if not str(real).startswith(str(MEMORY_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: {virtual!r}")
    return real


def execute_memory(args: dict[str, Any]) -> str:
    """Execute a single memory tool call and return the result string."""
    cmd = args.get("command", "")
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if cmd == "view":
        path = resolve_path(args.get("path", "/memories"))
        if not path.exists():
            return f"The path {args.get('path')} does not exist."
        if path.is_dir():
            lines = [f"Contents of {args.get('path', '/memories')}:"]
            for item in sorted(path.rglob("*")):
                rel = item.relative_to(path)
                if len(rel.parts) > 2 or any(p.startswith(".") for p in rel.parts):
                    continue
                size = f"{item.stat().st_size}B" if item.is_file() else ""
                lines.append(f"  {size}\t{rel}")
            return "\n".join(lines)
        text = path.read_text(encoding="utf-8", errors="replace")
        numbered = "\n".join(
            f"{i+1:6d}\t{line}" for i, line in enumerate(text.splitlines())
        )
        return f"Content of {args.get('path')}:\n{numbered}"

    if cmd == "create":
        path = resolve_path(args.get("path", ""))
        if path.exists():
            return f"Error: File {args.get('path')} already exists"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.get("file_text", ""), encoding="utf-8")
        return f"File created at {args.get('path')}"

    if cmd == "str_replace":
        path = resolve_path(args.get("path", ""))
        if not path.exists() or path.is_dir():
            return f"Error: {args.get('path')} does not exist"
        text = path.read_text(encoding="utf-8")
        old = args.get("old_str", "")
        if text.count(old) != 1:
            return f"Error: old_str not found or not unique in {args.get('path')}"
        path.write_text(text.replace(old, args.get("new_str", ""), 1), encoding="utf-8")
        return "File edited."

    if cmd == "insert":
        path = resolve_path(args.get("path", ""))
        if not path.exists() or path.is_dir():
            return f"Error: {args.get('path')} does not exist"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        idx = args.get("insert_line", 0)
        new_text = args.get("insert_text", "") + "\n"
        lines.insert(idx, new_text)
        path.write_text("".join(lines), encoding="utf-8")
        return f"Inserted at line {idx}."

    if cmd == "delete":
        path = resolve_path(args.get("path", ""))
        if not path.exists():
            return f"Error: {args.get('path')} does not exist"
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return f"Deleted {args.get('path')}"

    if cmd == "rename":
        old = resolve_path(args.get("old_path", args.get("path", "")))
        new = resolve_path(args.get("new_path", ""))
        if not old.exists():
            return f"Error: {args.get('old_path')} does not exist"
        if new.exists():
            return f"Error: {args.get('new_path')} already exists"
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        return f"Renamed to {args.get('new_path')}"

    return f"Unknown command: {cmd}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(user_message: str, messages: list[dict]) -> tuple[str, list[dict]]:
    """Send a user message and handle the tool-call loop until a text reply."""
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[MEMORY_TOOL],
            temperature=1.0,
            top_p=0.95,
            max_tokens=2048,
        )
        choice = resp.choices[0]

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
            return choice.message.content or "", messages

        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"command": "view", "path": "/memories"}
            result = execute_memory(args)
            print(f"  [memory:{args.get('command', '?')}] {result[:80]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "[Max tool rounds reached]", messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Nemotron-3-Super Memory Agent (vLLM)")
    print("Type a message, or /quit to exit.")
    print("=" * 60)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("/quit", "/exit"):
            break

        reply, messages = run_agent(user_input, messages)
        print(f"\nAssistant: {reply}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
