# Nemotron Memory Agent

Persistent memory for open LLMs. An agentic conversation loop that gives [NVIDIA Nemotron-3-Super-120B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) the ability to read, write, and organize files in a local `./memories/` directory — inspired by the [Anthropic memory tool pattern](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).

The model runs locally via [vLLM](https://vllm.ai) with tool calling enabled. No vendor API keys required for inference.

## Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| vLLM | 0.17.1+ |
| GPU | 4x H100 (BF16) or equivalent for NVFP4 |

## Quickstart

### 1. Start the vLLM server

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

The `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` flags are **required** for tool calling to work. Without them, the model will not emit structured `tool_calls`.

### 2. Install dependencies and run

```bash
pip install -r requirements.txt
cp .env.example .env   # edit if your vLLM server is not on localhost:8000
python agent.py
```

### 3. Chat

```
You: What's the capital of France?
Assistant: The capital of France is Paris. Let me save that in my memory in case it's useful later.
[Memory tool called: create]

You: /memory_view
Here're the files and directories up to 2 levels deep in /memories, ...
```

## CLI Commands

| Command | Description |
|---|---|
| `/quit` or `/exit` | Exit the session |
| `/clear` | Reset conversation history |
| `/memory_view` | Show all memory files |
| `/memory_clear` | Delete all memory files |
| `/debug` | Show message count, approximate token count, and conversation structure |

## How It Works

```
User message
    │
    ▼
agent.py ──► vLLM (Nemotron-3-Super)
    │              │
    │         tool_call: memory(view, /memories)
    │              │
    ▼              ▼
memory_backend.py ◄── execute ──► ./memories/
    │
    ▼
tool_result fed back to vLLM
    │
    ▼
... (repeat until finish_reason == "stop")
    │
    ▼
Final text response shown to user
```

The agent loop runs up to 20 consecutive tool calls before forcing a text response. Context is automatically trimmed when the conversation exceeds ~30k tokens by dropping the oldest tool results.

## Memory Tool Commands

The memory tool exposes 6 commands via OpenAI function calling:

| Command | Description |
|---|---|
| `view` | List a directory or read a file (with optional line range) |
| `create` | Create a new file |
| `str_replace` | Find-and-replace text in a file (must be unique match) |
| `insert` | Insert text at a specific line number |
| `delete` | Delete a file or directory |
| `rename` | Rename or move a file/directory |

All paths are sandboxed under `./memories/` with path traversal protection.

## Security

- **Path traversal guard**: All paths are resolved to their canonical form and verified to stay within the `./memories/` directory. Attempts to escape (e.g., `../../etc/passwd`) raise an error.
- **No secrets in memory**: The system prompt instructs the model to store facts, not credentials. For production use, add additional validation to strip sensitive data.
- **For stronger isolation**: Run inside a [NemoClaw](https://github.com/NVIDIA/NemoClaw) sandbox with filesystem and network policies enforced at the OS level.

## Configuration

Edit `.env` or set environment variables:

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `VLLM_API_KEY` | `EMPTY` | API key (vLLM default is `EMPTY`) |
| `VLLM_MODEL` | `nemotron` | Model name matching `--served-model-name` |

## License

Apache 2.0
