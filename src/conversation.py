"""
src/conversation.py — Stateful conversation history for multi-turn LLM sessions.

Maintains an ordered list of user/assistant message pairs and trims from
the oldest end when the estimated token count exceeds the budget.

Token counting is approximate (4 chars ≈ 1 token) — precise enough to
stay safely within context limits without a tokenizer dependency.

Usage:
    history = ConversationHistory(max_tokens=3000)

    history.add("user",      "Which region had the highest revenue?")
    history.add("assistant", "The North region led with $1.24M in revenue...")
    history.add("user",      "What about by profit margin instead?")

    # Pass the full list to the API each turn
    messages = history.get_messages()
    reply    = backend.chat(messages, context=ctx)
    history.add("assistant", reply)

    # Persist between sessions
    history.save("sessions/2025-11-05.json")
    restored = ConversationHistory.load("sessions/2025-11-05.json")
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
import json
import logging

log = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4   # rough approximation
_RESIDUE_PREFIX  = "[Earlier in this conversation: "


class ConversationHistory:
    """
    Ordered message history with automatic token-budget enforcement.

    When the total estimated token count exceeds max_tokens, the oldest
    messages are dropped first. The two most recent messages (one user,
    one assistant) are always preserved so the LLM has at least the
    immediate prior context.

    Args:
        max_tokens: Estimated token budget for the stored messages.
                    Does not include the system prompt or DataFrame context —
                    those are accounted for separately. Default 3000.
        session_id: Optional string identifier for logging and persistence.
                    Auto-generated from current UTC time if not provided.
        summarizer: Optional callable that compresses a list of dropped
                    messages into a single summary string. When provided,
                    over-budget trims replace the dropped messages with a
                    synthetic ``[Earlier in this conversation: ...]`` turn
                    so entity references (column names, numbers, regions)
                    survive the compression. Default None preserves the
                    older trim-oldest behaviour.

    Attributes:
        messages:    List of {"role": str, "content": str} message dicts.
        turn_count:  Cumulative number of add() calls (never decremented).
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        session_id: Optional[str] = None,
        summarizer: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ):
        self.max_tokens  = max_tokens
        self.session_id  = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.summarizer  = summarizer
        self.messages:   List[Dict[str, str]] = []
        self.turn_count: int = 0

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """
        Append a message to history, then trim if over token budget.

        Args:
            role:    "user" or "assistant".
            content: Message text.

        Raises:
            ValueError: If role is not "user" or "assistant".
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")

        self.messages.append({"role": role, "content": content})
        self.turn_count += 1

        if self._token_estimate() <= self.max_tokens:
            return

        if self.summarizer is None:
            # Trim oldest messages to stay within budget, always keeping ≥2 messages
            while self._token_estimate() > self.max_tokens and len(self.messages) > 2:
                dropped = self.messages.pop(0)
                log.debug(
                    f"[{self.session_id}] Trimmed oldest message "
                    f"({len(dropped['content'])} chars) to stay within {self.max_tokens}-token budget."
                )
            return

        self._compress_oldest()

    def _compress_oldest(self) -> None:
        """
        Pull dropped messages off the front, compress them via summarizer,
        and replace them with a single synthetic residue turn. If the
        summarizer returns empty (e.g. API failed), fall back to trim-oldest.
        """
        dropped: List[Dict[str, str]] = []
        existing_residue_index: Optional[int] = None
        existing_residue: Optional[str] = None

        # If the front of history is already a residue from a prior compression,
        # pull it forward so we re-summarize cumulatively rather than stack
        # nested "[Earlier in this conversation: ...]" turns.
        if self.messages and self.messages[0]["content"].startswith(_RESIDUE_PREFIX):
            existing_residue_index = 0
            existing_residue = self.messages[0]["content"][len(_RESIDUE_PREFIX):].rstrip("]").strip()

        while self._token_estimate() > self.max_tokens and len(self.messages) > 2:
            idx = 0
            if existing_residue_index is not None and len(self.messages) > 1:
                idx = 1  # skip the residue we already have queued for re-summarization
            dropped.append(self.messages.pop(idx))

        if existing_residue is not None:
            # Re-summarize residue + new dropped messages together.
            self.messages.pop(existing_residue_index)
            dropped.insert(0, {"role": "user", "content": existing_residue})

        try:
            summary = self.summarizer(dropped) if dropped else ""
        except Exception as exc:
            log.warning(f"[{self.session_id}] Summarizer raised ({exc}); residue suppressed.")
            summary = ""

        if summary:
            residue = {
                "role": "user",
                "content": f"{_RESIDUE_PREFIX}{summary}]",
            }
            self.messages.insert(0, residue)
            log.debug(
                f"[{self.session_id}] Compressed {len(dropped)} dropped message(s) "
                f"into a residue of {len(summary)} chars."
            )
        else:
            log.debug(
                f"[{self.session_id}] Summarizer returned empty; "
                f"dropped {len(dropped)} message(s) without residue."
            )

    def clear(self) -> None:
        """Remove all messages. Does not reset turn_count."""
        self.messages.clear()
        log.debug(f"[{self.session_id}] History cleared.")

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_messages(self) -> List[Dict[str, str]]:
        """Return a shallow copy of the message list safe to pass to the API."""
        return deepcopy(self.messages)

    def last_user_message(self) -> Optional[str]:
        """Content of the most recent user message, or None."""
        return next(
            (m["content"] for m in reversed(self.messages) if m["role"] == "user"),
            None,
        )

    def last_assistant_message(self) -> Optional[str]:
        """Content of the most recent assistant message, or None."""
        return next(
            (m["content"] for m in reversed(self.messages) if m["role"] == "assistant"),
            None,
        )

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"ConversationHistory("
            f"messages={len(self.messages)}, "
            f"~{self._token_estimate()} tokens, "
            f"session={self.session_id})"
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialise history to a JSON string."""
        return json.dumps(
            {
                "session_id":  self.session_id,
                "turn_count":  self.turn_count,
                "max_tokens":  self.max_tokens,
                "messages":    self.messages,
                "saved_at":    datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str, max_tokens: Optional[int] = None) -> "ConversationHistory":
        """
        Restore a ConversationHistory from a JSON string.

        Args:
            json_str:   JSON string produced by to_json().
            max_tokens: Override the stored token budget if provided.
        """
        data     = json.loads(json_str)
        instance = cls(
            max_tokens = max_tokens or data.get("max_tokens", 3000),
            session_id = data.get("session_id"),
        )
        instance.messages    = data.get("messages", [])
        instance.turn_count  = data.get("turn_count", len(instance.messages))
        return instance

    def save(self, path: str) -> None:
        """Write conversation to a JSON file, creating parent dirs as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        log.info(f"[{self.session_id}] Saved to {path} ({len(self.messages)} messages).")

    @classmethod
    def load(cls, path: str, max_tokens: Optional[int] = None) -> "ConversationHistory":
        """Load and return a ConversationHistory from a JSON file."""
        return cls.from_json(Path(path).read_text(), max_tokens=max_tokens)

    # ── Private ────────────────────────────────────────────────────────────────

    def _token_estimate(self) -> int:
        return sum(len(m["content"]) for m in self.messages) // _CHARS_PER_TOKEN
