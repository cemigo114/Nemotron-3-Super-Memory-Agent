# Persistent Memory for Open LLMs: Nemotron-3-Super + vLLM + Memory-as-Tool

**TL;DR** — We show how to give NVIDIA Nemotron-3-Super-120B persistent memory using a file-based tool calling pattern. The model runs locally via vLLM, stores facts in XML files on disk, and can optionally run inside a NemoClaw sandbox for OS-level security. No vendor API keys needed for inference. All code is open source.

---

## The Problem: LLM Agents Forget Everything

Every LLM conversation starts from zero. The model has no memory of who you are, what you've told it before, or what decisions were made in past sessions. For one-off chat, that's fine. For agents that manage your calendar, triage your inbox, or maintain a codebase — it's a dealbreaker.

Cloud providers have started addressing this. Anthropic ships a [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) that lets Claude read and write files to persist facts across conversations. It's a clean pattern: the model decides what to remember, writes it to a directory, and reads it back at the start of the next session.

But it requires calling Anthropic's API. What if you want the same capability with an open model, running locally, with no data leaving your machine?

## The Insight: Memory-as-Tool Is Model-Agnostic

The memory tool pattern doesn't depend on any vendor-specific API. It's just tool calling:

1. Define a `memory` tool with commands: `view`, `create`, `str_replace`, `insert`, `delete`, `rename`
2. The model emits `tool_call` messages to read/write files under a `/memories` directory
3. Your application executes those calls against the local filesystem
4. Results are fed back to the model as `tool` messages

Any model that supports function calling can do this. NVIDIA Nemotron-3-Super-120B is optimized for exactly this kind of agentic workflow — and vLLM 0.17.1+ supports it natively with the `--enable-auto-tool-choice` flag.

## The Stack

```
User
  │
  ▼
agent.py ──► vLLM (Nemotron-3-Super-120B-A12B-NVFP4)
  │                │
  │           tool_call: memory(view, /memories)
  │                │
  ▼                ▼
memory_backend.py ◄── execute ──► ./memories/
  │
  ▼
tool_result fed back to vLLM
  │
  ▼
... (repeat until finish_reason == "stop")
  │
  ▼
Final text response
```

### Serving the Model

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --served-model-name nemotron \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3
```

Key flags:
- `--enable-auto-tool-choice` — lets the model decide when to call tools
- `--tool-call-parser qwen3_coder` — the parser format Nemotron uses for structured tool calls
- `--reasoning-parser nemotron_v3` — surfaces the model's chain-of-thought reasoning

The NVFP4 quantization delivers 4x higher throughput compared to FP8 on H100, per NVIDIA's benchmarks, while maintaining accuracy.

### The Memory Tool Schema

We define a single function tool in OpenAI format:

```python
MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": "Persistent memory store. Commands: view, create, ...",
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
                # ... other command-specific parameters
            },
            "required": ["command"],
        },
    },
}
```

### The Agent Loop

The core loop is straightforward:

```python
for _ in range(MAX_TOOL_ROUNDS):
    resp = client.chat.completions.create(
        model="nemotron",
        messages=messages,
        tools=[MEMORY_TOOL],
        temperature=1.0,
        top_p=0.95,
    )
    choice = resp.choices[0]
    messages.append(assistant_message_from(choice))

    if choice.finish_reason == "stop":
        return choice.message.content

    for tc in choice.message.tool_calls:
        args = json.loads(tc.function.arguments)
        result = memory.execute(args)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

The model typically:
1. Calls `memory(view, /memories)` to check for prior context
2. Reads relevant files
3. Responds to the user
4. Writes updated facts back to memory

### Path Traversal Protection

All memory operations are sandboxed:

```python
def resolve_path(virtual: str) -> Path:
    clean = virtual.replace("\\", "/")
    if clean.startswith("/memories"):
        clean = clean[len("/memories"):]
    real = (MEMORY_DIR / clean.lstrip("/")).resolve()
    if not str(real).startswith(str(MEMORY_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: {virtual!r}")
    return real
```

This prevents the model from accessing anything outside the `./memories/` directory, even if it tries paths like `../../etc/passwd`.

### Context Management

Long conversations will exceed the context window. We implement a sliding window that drops the oldest tool results while keeping the most recent ones:

```python
def trim_context(messages, budget=30_000, keep=3):
    # Approximate token count from character length
    approx_tokens = sum(len(json.dumps(m)) for m in messages) // 4
    if approx_tokens <= budget:
        return messages
    # Drop oldest tool results, keeping the most recent `keep`
    ...
```

This mirrors Anthropic's `clear_tool_uses` context management strategy.

## Running Inside NemoClaw (Optional but Recommended)

For production use, the agent should run inside a [NemoClaw](https://github.com/NVIDIA/NemoClaw) sandbox. NemoClaw adds OS-level enforcement:

- **Network policy**: blocks all outbound connections except explicitly listed endpoints
- **Filesystem policy**: restricts reads/writes to `/sandbox` and `/tmp`
- **Process policy**: blocks privilege escalation and dangerous syscalls

We've contributed a Google Workspace policy preset (`google.yaml`) to NemoClaw that allowlists Gmail, Calendar, Drive, and OAuth endpoints — enabling the memory agent to interact with Google services while staying sandboxed.

## What This Enables

The persistent memory pattern unlocks use cases that stateless chat cannot:

| Use Case | How Memory Helps |
|---|---|
| **Personal assistant** | Remembers your preferences, schedule patterns, communication style |
| **Code review agent** | Accumulates coding conventions and past PR decisions per-repo |
| **Multi-agent coordination** | Multiple agents share a `./memories/` directory for state |
| **RAG-free knowledge base** | Agent writes XML summaries of documents; retrieves via `view` |
| **Incident response** | Remembers past incidents and resolutions; improves over time |

The key advantage over cloud-based memory: **everything stays local**. No data leaves your machine. The model, the memory files, and the sandbox are all under your control.

## Try It

The full implementation is open source:

- **Memory agent**: [github.com/yfama/NemoClaw](https://github.com/yfama/NemoClaw)
- **vLLM example**: Contributed as `memory_agent_nemotron.py` to the vLLM examples
- **NemoClaw Google preset**: Contributed as `google.yaml` to NemoClaw presets

```bash
pip install openai python-dotenv
# Start vLLM server (see above), then:
python agent.py
```

## Acknowledgments

- NVIDIA for Nemotron-3-Super and NemoClaw
- The vLLM team for native tool calling support
- Anthropic for the memory tool specification that inspired this pattern
