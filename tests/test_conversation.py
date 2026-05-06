"""Tests for ConversationHistory — trim-oldest legacy + summarizer path."""

import pytest

from conversation import ConversationHistory


def _fill_with_long_messages(history: ConversationHistory, n: int = 6, chars: int = 600):
    """Fill history past its budget with predictable long messages."""
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.add(role, f"msg-{i} " + ("x" * chars))


def test_trim_oldest_when_no_summarizer():
    """Default trim-oldest behaviour preserved when summarizer is None."""
    history = ConversationHistory(max_tokens=100)  # ~400 chars
    _fill_with_long_messages(history, n=6, chars=600)
    # Always keeps at least 2 messages
    assert len(history.messages) >= 2
    # No residue inserted (summarizer is None)
    assert all(not m["content"].startswith("[Earlier in this conversation:") for m in history.messages)


def test_summarizer_invoked_when_budget_exceeded():
    """When the summarizer is wired, it gets called with the dropped messages."""
    calls = []

    def fake_summarizer(dropped):
        calls.append(list(dropped))
        return f"Compressed {len(dropped)} earlier turn(s)."

    history = ConversationHistory(max_tokens=100, summarizer=fake_summarizer)
    _fill_with_long_messages(history, n=6, chars=600)

    assert len(calls) >= 1
    # Each summarizer call received non-empty dropped messages
    assert all(len(d) > 0 for d in calls)


def test_summary_inserted_as_residue_turn():
    """Compressed history starts with the synthetic residue turn."""
    def fake_summarizer(dropped):
        return f"User asked about {len(dropped)} earlier topics including revenue and region."

    history = ConversationHistory(max_tokens=100, summarizer=fake_summarizer)
    _fill_with_long_messages(history, n=6, chars=600)

    assert history.messages[0]["content"].startswith("[Earlier in this conversation:")
    assert history.messages[0]["content"].endswith("]")
    assert "revenue and region" in history.messages[0]["content"]


def test_turn_count_increments_through_summarization():
    """turn_count is monotonic regardless of compression."""
    def fake_summarizer(dropped):
        return "summary"

    history = ConversationHistory(max_tokens=100, summarizer=fake_summarizer)
    _fill_with_long_messages(history, n=6, chars=600)

    # Six adds, six turns
    assert history.turn_count == 6


def test_to_json_from_json_round_trip_preserves_history():
    """Serializing and deserializing keeps the message list intact."""
    def fake_summarizer(dropped):
        return "compressed summary"

    history = ConversationHistory(max_tokens=100, summarizer=fake_summarizer, session_id="round-trip")
    _fill_with_long_messages(history, n=6, chars=600)

    raw = history.to_json()
    restored = ConversationHistory.from_json(raw)

    assert restored.messages == history.messages
    assert restored.session_id == "round-trip"
    assert restored.turn_count == history.turn_count
