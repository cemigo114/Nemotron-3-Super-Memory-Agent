"""Tests for memory_backend.py — covers path security, all 6 commands, and edge cases."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from memory_backend import MemoryBackend


@pytest.fixture
def mem(tmp_path: Path) -> MemoryBackend:
    return MemoryBackend(root=tmp_path / "memories")


# ------------------------------------------------------------------
# Path traversal security
# ------------------------------------------------------------------

class TestPathTraversal:
    def test_dotdot_blocked(self, mem: MemoryBackend):
        result = mem.execute({"command": "view", "path": "/memories/../../etc/passwd"})
        assert "Error" in result

    def test_backslash_dotdot_blocked(self, mem: MemoryBackend):
        result = mem.execute({"command": "view", "path": "/memories/..\\..\\etc\\passwd"})
        assert "Error" in result

    def test_memories_prefix_confusion(self, mem: MemoryBackend):
        """Ensures /memories2/../../etc/passwd is not accepted."""
        result = mem.execute({"command": "view", "path": "/memories2/../../etc/passwd"})
        assert "Error" in result or "does not exist" in result

    def test_null_byte_rejected(self, mem: MemoryBackend):
        result = mem.execute({"command": "view", "path": "/memories/foo\x00bar"})
        assert "Error" in result

    def test_non_string_path_rejected(self, mem: MemoryBackend):
        result = mem.execute({"command": "view", "path": 123})  # type: ignore
        assert "Error" in result

    def test_normal_path_allowed(self, mem: MemoryBackend):
        mem.execute({"command": "create", "path": "/memories/test.txt", "file_text": "hello"})
        result = mem.execute({"command": "view", "path": "/memories/test.txt"})
        assert "hello" in result

    def test_root_view_allowed(self, mem: MemoryBackend):
        result = mem.execute({"command": "view", "path": "/memories"})
        assert "files and directories" in result.lower() or "contents" in result.lower() or "Here" in result


# ------------------------------------------------------------------
# Command: view
# ------------------------------------------------------------------

class TestView:
    def test_view_nonexistent(self, mem: MemoryBackend):
        result = mem.execute({"command": "view", "path": "/memories/nope.txt"})
        assert "does not exist" in result

    def test_view_empty_file(self, mem: MemoryBackend):
        (mem.root / "empty.txt").write_text("", encoding="utf-8")
        result = mem.execute({"command": "view", "path": "/memories/empty.txt"})
        assert "content of" in result.lower()

    def test_view_with_line_numbers(self, mem: MemoryBackend):
        (mem.root / "lines.txt").write_text("a\nb\nc\n", encoding="utf-8")
        result = mem.execute({"command": "view", "path": "/memories/lines.txt"})
        assert "1" in result and "a" in result

    def test_view_range(self, mem: MemoryBackend):
        (mem.root / "lines.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
        result = mem.execute({"command": "view", "path": "/memories/lines.txt", "view_range": [2, 3]})
        assert "b" in result
        assert "d" not in result

    def test_view_range_invalid_type_ignored(self, mem: MemoryBackend):
        (mem.root / "lines.txt").write_text("a\nb\n", encoding="utf-8")
        result = mem.execute({"command": "view", "path": "/memories/lines.txt", "view_range": "bad"})
        assert "a" in result  # returns full file


# ------------------------------------------------------------------
# Command: create
# ------------------------------------------------------------------

class TestCreate:
    def test_create_success(self, mem: MemoryBackend):
        result = mem.execute({"command": "create", "path": "/memories/new.txt", "file_text": "hello"})
        assert "created" in result.lower()
        assert (mem.root / "new.txt").read_text() == "hello"

    def test_create_already_exists(self, mem: MemoryBackend):
        (mem.root / "exists.txt").write_text("x")
        result = mem.execute({"command": "create", "path": "/memories/exists.txt", "file_text": "y"})
        assert "already exists" in result

    def test_create_nested_dir(self, mem: MemoryBackend):
        result = mem.execute({"command": "create", "path": "/memories/sub/deep.txt", "file_text": "nested"})
        assert "created" in result.lower()
        assert (mem.root / "sub" / "deep.txt").exists()


# ------------------------------------------------------------------
# Command: str_replace
# ------------------------------------------------------------------

class TestStrReplace:
    def test_replace_success(self, mem: MemoryBackend):
        (mem.root / "edit.txt").write_text("color: blue\n")
        result = mem.execute({
            "command": "str_replace",
            "path": "/memories/edit.txt",
            "old_str": "blue",
            "new_str": "green",
        })
        assert "edited" in result.lower()
        assert "green" in (mem.root / "edit.txt").read_text()

    def test_replace_not_found(self, mem: MemoryBackend):
        (mem.root / "edit.txt").write_text("color: blue\n")
        result = mem.execute({
            "command": "str_replace",
            "path": "/memories/edit.txt",
            "old_str": "red",
            "new_str": "green",
        })
        assert "did not appear" in result

    def test_replace_multiple_occurrences(self, mem: MemoryBackend):
        (mem.root / "edit.txt").write_text("a a a\n")
        result = mem.execute({
            "command": "str_replace",
            "path": "/memories/edit.txt",
            "old_str": "a",
            "new_str": "b",
        })
        assert "Multiple occurrences" in result

    def test_replace_empty_old_str(self, mem: MemoryBackend):
        (mem.root / "edit.txt").write_text("text\n")
        result = mem.execute({
            "command": "str_replace",
            "path": "/memories/edit.txt",
            "old_str": "",
            "new_str": "x",
        })
        assert "Error" in result

    def test_replace_nonexistent_file(self, mem: MemoryBackend):
        result = mem.execute({
            "command": "str_replace",
            "path": "/memories/nope.txt",
            "old_str": "a",
            "new_str": "b",
        })
        assert "does not exist" in result


# ------------------------------------------------------------------
# Command: insert
# ------------------------------------------------------------------

class TestInsert:
    def test_insert_success(self, mem: MemoryBackend):
        (mem.root / "ins.txt").write_text("line1\nline2\n")
        result = mem.execute({
            "command": "insert",
            "path": "/memories/ins.txt",
            "insert_line": 1,
            "insert_text": "new line",
        })
        assert "edited" in result.lower()
        lines = (mem.root / "ins.txt").read_text().splitlines()
        assert lines[1] == "new line"

    def test_insert_out_of_range(self, mem: MemoryBackend):
        (mem.root / "ins.txt").write_text("line1\n")
        result = mem.execute({
            "command": "insert",
            "path": "/memories/ins.txt",
            "insert_line": 99,
            "insert_text": "x",
        })
        assert "Invalid" in result or "Error" in result

    def test_insert_invalid_type(self, mem: MemoryBackend):
        (mem.root / "ins.txt").write_text("line1\n")
        result = mem.execute({
            "command": "insert",
            "path": "/memories/ins.txt",
            "insert_line": "two",
            "insert_text": "x",
        })
        assert "Error" in result

    def test_insert_bool_rejected(self, mem: MemoryBackend):
        (mem.root / "ins.txt").write_text("line1\n")
        result = mem.execute({
            "command": "insert",
            "path": "/memories/ins.txt",
            "insert_line": True,
            "insert_text": "x",
        })
        assert "Error" in result


# ------------------------------------------------------------------
# Command: delete
# ------------------------------------------------------------------

class TestDelete:
    def test_delete_file(self, mem: MemoryBackend):
        (mem.root / "del.txt").write_text("x")
        result = mem.execute({"command": "delete", "path": "/memories/del.txt"})
        assert "deleted" in result.lower()
        assert not (mem.root / "del.txt").exists()

    def test_delete_directory(self, mem: MemoryBackend):
        d = mem.root / "subdir"
        d.mkdir()
        (d / "file.txt").write_text("x")
        result = mem.execute({"command": "delete", "path": "/memories/subdir"})
        assert "deleted" in result.lower()
        assert not d.exists()

    def test_delete_nonexistent(self, mem: MemoryBackend):
        result = mem.execute({"command": "delete", "path": "/memories/nope"})
        assert "does not exist" in result


# ------------------------------------------------------------------
# Command: rename
# ------------------------------------------------------------------

class TestRename:
    def test_rename_success(self, mem: MemoryBackend):
        (mem.root / "old.txt").write_text("content")
        result = mem.execute({
            "command": "rename",
            "old_path": "/memories/old.txt",
            "new_path": "/memories/new.txt",
        })
        assert "renamed" in result.lower()
        assert not (mem.root / "old.txt").exists()
        assert (mem.root / "new.txt").read_text() == "content"

    def test_rename_destination_exists(self, mem: MemoryBackend):
        (mem.root / "a.txt").write_text("a")
        (mem.root / "b.txt").write_text("b")
        result = mem.execute({
            "command": "rename",
            "old_path": "/memories/a.txt",
            "new_path": "/memories/b.txt",
        })
        assert "already exists" in result

    def test_rename_source_missing(self, mem: MemoryBackend):
        result = mem.execute({
            "command": "rename",
            "old_path": "/memories/nope.txt",
            "new_path": "/memories/new.txt",
        })
        assert "does not exist" in result


# ------------------------------------------------------------------
# Dispatch / edge cases
# ------------------------------------------------------------------

class TestDispatch:
    def test_unknown_command(self, mem: MemoryBackend):
        result = mem.execute({"command": "drop_table"})
        assert "Unknown command" in result

    def test_non_dict_args(self, mem: MemoryBackend):
        result = mem.execute("not a dict")  # type: ignore
        assert "Error" in result

    def test_clear_all(self, mem: MemoryBackend):
        (mem.root / "a.txt").write_text("a")
        (mem.root / "b.txt").write_text("b")
        result = mem.clear_all()
        assert "deleted" in result.lower()
        assert len(list(mem.root.iterdir())) == 0
