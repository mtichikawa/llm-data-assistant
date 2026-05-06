"""
src/model_router.py — Pick a Claude model per query based on complexity.

Defaults to Haiku for short, factual lookups. Escalates to Sonnet when:
  - the query contains an escalation keyword ("explain", "why", "compare", ...)
  - the conversation history has 3+ turns (likely deeper analysis)
  - the query is over 200 chars (likely complex)

The escalation rules are intentionally cheap and explicit. Embedding-based
routing (KNN over training queries) is the natural next step but lives a
rev away — for the data-assistant scope, keyword + length + turn-count
gets ~90% of the right routing decisions at zero per-query cost.

Usage:
    router = ModelRouter()
    model  = router.select(query="why is revenue down?", history_turns=4)
    # → "claude-sonnet-4-6"
"""

from typing import Optional

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

ESCALATION_KEYWORDS = {
    "explain", "why", "how does", "interpret", "compare",
    "tradeoff", "trade-off", "recommend", "predict", "forecast",
    "trend", "outlier", "anomaly", "root cause", "drilldown",
    "analyze", "analysis", "what story", "what drove",
}

_LONG_QUERY_CHARS = 200
_HISTORY_ESCALATION_TURNS = 3


class ModelRouter:
    """
    Pick a model per query.

    Args:
        default:   Model used for short, factual queries. Default Haiku 4.5.
        escalated: Model used when an escalation rule fires. Default Sonnet 4.6.
    """

    def __init__(self, default: str = HAIKU, escalated: str = SONNET):
        self.default   = default
        self.escalated = escalated

    def select(self, query: str, history_turns: int = 0) -> str:
        """
        Return the model name that should handle ``query``.

        Args:
            query:         The user's question text.
            history_turns: Number of prior turns in the conversation.
                           Triggers escalation at 3+.

        Returns:
            A model identifier string ready to pass to ``AnthropicBackend``.
        """
        if not query or not query.strip():
            return self.default

        q = query.lower().strip()
        if any(kw in q for kw in ESCALATION_KEYWORDS):
            return self.escalated
        if history_turns >= _HISTORY_ESCALATION_TURNS:
            return self.escalated
        if len(query) > _LONG_QUERY_CHARS:
            return self.escalated
        return self.default

    def explain(self, query: str, history_turns: int = 0) -> Optional[str]:
        """
        Return a short reason string for the routing decision, or None
        when the default model was selected.

        Useful for debugging and for the multi-turn example in the README.
        """
        if not query or not query.strip():
            return None

        q = query.lower().strip()
        for kw in ESCALATION_KEYWORDS:
            if kw in q:
                return f"keyword '{kw}' triggered escalation"
        if history_turns >= _HISTORY_ESCALATION_TURNS:
            return f"history depth ({history_turns} turns) triggered escalation"
        if len(query) > _LONG_QUERY_CHARS:
            return f"long query ({len(query)} chars) triggered escalation"
        return None
