"""
Memory backend for an LLM agent using the Anthropic memory tool pattern.

Implements the client-side file-based memory store that an LLM agent
invokes via tool calls. The tool schema follows the OpenAI function
calling format so it works with any OpenAI-compatible server (vLLM, etc.).

Commands: view, create, str_replace, insert, delete, rename.
All paths are sandboxed under a configurable root directory.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


MEMORY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "A persistent memory store. Use this to save facts, preferences, "
            "and progress across conversations. Commands: view (list dir or "
            "read file), create (new file), str_replace (edit text in file), "
            "insert (insert text at line), delete (remove file/dir), "
            "rename (move file/dir). All paths are under /memories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
                    "description": "The memory operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Target path (e.g. /memories or /memories/notes.txt).",
                },
                "file_text": {
                    "type": "string",
                    "description": "File contents for the 'create' command.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Text to find for 'str_replace'.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text for 'str_replace'.",
                },
                "insert_line": {
                    "type": "integer",
                    "description": "0-indexed line number for 'insert'.",
                },
                "insert_text": {
                    "type": "string",
                    "description": "Text to insert for 'insert'.",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [start, end] line range for 'view'.",
                },
                "old_path": {
                    "type": "string",
                    "description": "Source path for 'rename'.",
                },
                "new_path": {
                    "type": "string",
                    "description": "Destination path for 'rename'.",
                },
            },
            "required": ["command"],
        },
    },
}


MAX_LINES = 999_999


class MemoryBackend:
    """File-based memory store sandboxed under a root directory."""

    def __init__(self, root: str | Path = "./memories") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path security
    # ------------------------------------------------------------------

    def _resolve(self, virtual_path: str) -> Path:
        """Map a virtual path like /memories/foo.txt to a real filesystem path.

        Raises ValueError on any path traversal attempt.
        """
        clean = virtual_path.replace("\\", "/")
        # Strip the leading /memories prefix if present
        if clean.startswith("/memories"):
            clean = clean[len("/memories"):]
        clean = clean.lstrip("/")

        real = (self.root / clean).resolve()
        try:
            real.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"Path traversal blocked: {virtual_path!r} resolves outside the memory directory."
            )
        return real

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def execute(self, args: dict[str, Any]) -> str:
        cmd = args.get("command", "")
        dispatch = {
            "view": self._view,
            "create": self._create,
            "str_replace": self._str_replace,
            "insert": self._insert,
            "delete": self._delete,
            "rename": self._rename,
        }
        handler = dispatch.get(cmd)
        if handler is None:
            return f"Error: Unknown command {cmd!r}. Valid: {', '.join(dispatch)}"
        try:
            return handler(args)
        except ValueError as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _view(self, args: dict[str, Any]) -> str:
        path = self._resolve(args.get("path", "/memories"))
        view_range = args.get("view_range")

        if not path.exists():
            return f"The path {args.get('path')} does not exist. Please provide a valid path."

        if path.is_dir():
            return self._list_dir(path, args.get("path", "/memories"))

        return self._read_file(path, args.get("path", ""), view_range)

    def _list_dir(self, real: Path, virtual: str) -> str:
        lines = [
            f"Here're the files and directories up to 2 levels deep in {virtual}, "
            "excluding hidden items and node_modules:"
        ]
        for item in sorted(real.rglob("*")):
            rel = item.relative_to(real)
            if any(p.startswith(".") or p == "node_modules" for p in rel.parts):
                continue
            if len(rel.parts) > 2:
                continue
            size = self._human_size(item.stat().st_size) if item.is_file() else ""
            lines.append(f"{size}\t{virtual.rstrip('/')}/{rel}")
        return "\n".join(lines)

    def _read_file(self, real: Path, virtual: str, view_range: list[int] | None) -> str:
        text = real.read_text(encoding="utf-8", errors="replace")
        file_lines = text.splitlines(keepends=True)

        if len(file_lines) > MAX_LINES:
            return f"File {virtual} exceeds maximum line limit of {MAX_LINES:,} lines."

        start = 1
        end = len(file_lines)
        if view_range and len(view_range) == 2:
            start, end = max(1, view_range[0]), min(len(file_lines), view_range[1])

        header = f"Here's the content of {virtual} with line numbers:"
        numbered = []
        for i in range(start, end + 1):
            line_content = file_lines[i - 1].rstrip("\n")
            numbered.append(f"{i:6d}\t{line_content}")

        return header + "\n" + "\n".join(numbered)

    def _create(self, args: dict[str, Any]) -> str:
        path = self._resolve(args.get("path", ""))
        if path.exists():
            return f"Error: File {args.get('path')} already exists"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.get("file_text", ""), encoding="utf-8")
        return f"File created successfully at: {args.get('path')}"

    def _str_replace(self, args: dict[str, Any]) -> str:
        path = self._resolve(args.get("path", ""))
        virtual = args.get("path", "")

        if not path.exists() or path.is_dir():
            return f"Error: The path {virtual} does not exist. Please provide a valid path."

        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        text = path.read_text(encoding="utf-8", errors="replace")

        occurrences = text.count(old_str)
        if occurrences == 0:
            return (
                f"No replacement was performed, old_str `{old_str}` "
                f"did not appear verbatim in {virtual}."
            )
        if occurrences > 1:
            lines_with = [
                str(i + 1) for i, line in enumerate(text.splitlines()) if old_str in line
            ]
            return (
                f"No replacement was performed. Multiple occurrences of "
                f"old_str `{old_str}` in lines: {', '.join(lines_with)}. "
                "Please ensure it is unique"
            )

        new_text = text.replace(old_str, new_str, 1)
        path.write_text(new_text, encoding="utf-8")

        # Show a snippet around the replacement
        new_lines = new_text.splitlines()
        replace_line = next(
            (i for i, l in enumerate(new_lines) if new_str in l), 0
        )
        start = max(0, replace_line - 2)
        end = min(len(new_lines), replace_line + 3)
        snippet = "\n".join(f"{i + 1:6d}\t{new_lines[i]}" for i in range(start, end))
        return f"The memory file has been edited.\n{snippet}"

    def _insert(self, args: dict[str, Any]) -> str:
        path = self._resolve(args.get("path", ""))
        virtual = args.get("path", "")
        insert_line = args.get("insert_line")

        if not path.exists() or path.is_dir():
            return f"Error: The path {virtual} does not exist"

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        n = len(lines)

        if insert_line is None or not (0 <= insert_line <= n):
            return (
                f"Error: Invalid `insert_line` parameter: {insert_line}. "
                f"It should be within the range of lines of the file: [0, {n}]"
            )

        insert_text = args.get("insert_text", "")
        if not insert_text.endswith("\n"):
            insert_text += "\n"
        lines.insert(insert_line, insert_text)
        path.write_text("".join(lines), encoding="utf-8")
        return f"The file {virtual} has been edited."

    def _delete(self, args: dict[str, Any]) -> str:
        path = self._resolve(args.get("path", ""))
        virtual = args.get("path", "")

        if not path.exists():
            return f"Error: The path {virtual} does not exist"

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return f"Successfully deleted {virtual}"

    def _rename(self, args: dict[str, Any]) -> str:
        old = self._resolve(args.get("old_path", args.get("path", "")))
        new = self._resolve(args.get("new_path", ""))
        old_v = args.get("old_path", args.get("path", ""))
        new_v = args.get("new_path", "")

        if not old.exists():
            return f"Error: The path {old_v} does not exist"
        if new.exists():
            return f"Error: The destination {new_v} already exists"

        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        return f"Successfully renamed {old_v} to {new_v}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _human_size(nbytes: int) -> str:
        for unit in ("", "K", "M", "G", "T"):
            if abs(nbytes) < 1024:
                if unit == "":
                    return f"{nbytes}B"
                return f"{nbytes:.1f}{unit}"
            nbytes /= 1024  # type: ignore[assignment]
        return f"{nbytes:.1f}P"

    def clear_all(self) -> str:
        """Remove everything under the memory root."""
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        return "All memory files deleted."
