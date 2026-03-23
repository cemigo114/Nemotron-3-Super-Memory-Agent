"""Tests for agent.py — covers trim_context and argument parsing edge cases."""

from __future__ import annotations

import json
import pytest

from agent import trim_context, CONTEXT_TOKEN_BUDGET, KEEP_TOOL_RESULTS


def _make_system():
    return {"role": "system", "content": "You are a test assistant."}


def _make_user(text: str = "hello"):
    return {"role": "user", "content": text}


def _make_assistant_with_tools(tool_call_ids: list[str], content: str = ""):
    msg: dict = {
        "role": "assistant",
        "tool_calls": [
            {"id": tid, "type": "function", "function": {"name": "memory", "arguments": "{}"}}
            for tid in tool_call_ids
        ],
    }
    if content:
        msg["content"] = content
    return msg


def _make_tool_result(tool_call_id: str, content: str = "ok"):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _make_assistant_text(text: str = "sure"):
    return {"role": "assistant", "content": text}


class TestTrimContextNoTrimNeeded:
    def test_short_conversation_untouched(self):
        msgs = [_make_system(), _make_user(), _make_assistant_text()]
        result = trim_context(msgs)
        assert result == msgs

    def test_empty_list(self):
        assert trim_context([]) == []

    def test_system_only(self):
        msgs = [_make_system()]
        assert trim_context(msgs) == msgs


class TestTrimContextDropsTurns:
    def _bloated_messages(self, n_turns: int = 20):
        """Create a conversation big enough to exceed CONTEXT_TOKEN_BUDGET."""
        msgs = [_make_system()]
        big_content = "x" * 10000
        for i in range(n_turns):
            tid = f"call_{i}"
            msgs.append(_make_user(f"Question {i}"))
            msgs.append(_make_assistant_with_tools([tid]))
            msgs.append(_make_tool_result(tid, big_content))
            msgs.append(_make_assistant_text(f"Answer {i}"))
        return msgs

    def test_over_budget_trims(self):
        msgs = self._bloated_messages(20)
        trimmed = trim_context(msgs)
        assert len(trimmed) < len(msgs)

    def test_system_message_preserved(self):
        msgs = self._bloated_messages(20)
        trimmed = trim_context(msgs)
        assert trimmed[0]["role"] == "system"

    def test_no_orphaned_tool_calls(self):
        """Every assistant with tool_calls must have matching tool results."""
        msgs = self._bloated_messages(20)
        trimmed = trim_context(msgs)

        expected_tool_ids: set[str] = set()
        for m in trimmed:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    expected_tool_ids.add(tc["id"])

        actual_tool_ids: set[str] = set()
        for m in trimmed:
            if m.get("role") == "tool":
                actual_tool_ids.add(m.get("tool_call_id", ""))

        assert expected_tool_ids == actual_tool_ids, (
            f"Orphaned tool_call IDs: {expected_tool_ids - actual_tool_ids}, "
            f"Orphaned tool results: {actual_tool_ids - expected_tool_ids}"
        )

    def test_multi_tool_call_turn_dropped_atomically(self):
        """When an assistant calls 2 tools, both results must be dropped together."""
        msgs = [_make_system()]
        big = "y" * 20000
        for i in range(10):
            tids = [f"call_{i}_a", f"call_{i}_b"]
            msgs.append(_make_user(f"Q {i}"))
            msgs.append(_make_assistant_with_tools(tids))
            msgs.append(_make_tool_result(tids[0], big))
            msgs.append(_make_tool_result(tids[1], big))
            msgs.append(_make_assistant_text(f"A {i}"))

        trimmed = trim_context(msgs)

        expected_tool_ids: set[str] = set()
        for m in trimmed:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    expected_tool_ids.add(tc["id"])

        actual_tool_ids: set[str] = set()
        for m in trimmed:
            if m.get("role") == "tool":
                actual_tool_ids.add(m.get("tool_call_id", ""))

        assert expected_tool_ids == actual_tool_ids

    def test_assistant_with_both_content_and_tool_calls_dropped_cleanly(self):
        """If an assistant has both content and tool_calls, the entire turn is dropped."""
        msgs = [_make_system()]
        big = "z" * 20000
        for i in range(10):
            tid = f"call_{i}"
            msgs.append(_make_user(f"Q {i}"))
            msgs.append(_make_assistant_with_tools([tid], content=f"Thinking about {i}..."))
            msgs.append(_make_tool_result(tid, big))
            msgs.append(_make_assistant_text(f"A {i}"))

        trimmed = trim_context(msgs)

        expected_tool_ids: set[str] = set()
        for m in trimmed:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    expected_tool_ids.add(tc["id"])

        actual_tool_ids: set[str] = set()
        for m in trimmed:
            if m.get("role") == "tool":
                actual_tool_ids.add(m.get("tool_call_id", ""))

        assert expected_tool_ids == actual_tool_ids
