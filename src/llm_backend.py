"""
src/llm_backend.py — Anthropic API client for the data analysis assistant.

Wraps the Anthropic Messages API with:
    - Exponential backoff retry for rate limits and 5xx errors
    - Context injection: DataFrame summaries go into the system prompt, keeping
      user-facing messages clean and conversation history portable
    - JSON response mode for structured output queries

This module is intentionally stateless — it does not track conversation
history. That belongs to ConversationHistory (see conversation.py).

Usage:
    backend = AnthropicBackend(api_key=os.getenv("ANTHROPIC_API_KEY"))
    reply = backend.chat(
        messages=[{"role": "user", "content": "Which region had the highest revenue?"}],
        system_prompt="You are a concise data analyst.",
        context="Dataset: 100 rows. Column 'region': North, South, East, West...",
    )
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_MODEL       = "claude-haiku-4-5-20251001"
_MAX_RETRIES = 3
_BASE_DELAY  = 1.0   # seconds
_MAX_DELAY   = 30.0  # seconds cap


class AnthropicBackend:
    """
    Thin, stateless wrapper around the Anthropic Messages API.

    Args:
        api_key:    Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model:      Model to use. Defaults to claude-haiku-4-5.
        max_tokens: Max tokens in the response. Default 1024.

    Raises:
        ImportError: If the 'anthropic' package is not installed.
        ValueError:  If no API key is found.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model:      str           = _MODEL,
        max_tokens: int           = 1024,
    ):
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY env var "
                "or pass api_key= to AnthropicBackend()."
            )

        self.client     = self._anthropic.Anthropic(api_key=key)
        self.model      = model
        self.max_tokens = max_tokens
        log.info(f"AnthropicBackend ready — model={model}")

    # ── Public interface ───────────────────────────────────────────────────────

    def chat(
        self,
        messages:      List[Dict[str, str]],
        system_prompt: str = "",
        context:       str = "",
    ) -> str:
        """
        Send a conversation to the API and return the assistant's text reply.

        The context string (DataFrame schema + stats + sample rows) is
        prepended to the system prompt so the LLM has data to reason about
        without polluting the user-visible message history.

        Args:
            messages:      Full conversation history as list of
                           {"role": "user"/"assistant", "content": str}.
                           Pass the complete history for multi-turn support.
            system_prompt: Behavioural instructions for the assistant.
            context:       DataFrame summary injected into the system prompt.

        Returns:
            Assistant reply as a plain string.

        Raises:
            RuntimeError: After all retries are exhausted.
        """
        system = self._build_system(system_prompt, context)

        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=messages,
                )
                text = response.content[0].text
                log.debug(
                    f"LLM reply: {len(text)} chars | "
                    f"in={response.usage.input_tokens} "
                    f"out={response.usage.output_tokens} tokens"
                )
                return text

            except self._anthropic.RateLimitError:
                wait = self._backoff(attempt)
                log.warning(f"Rate limit (attempt {attempt+1}/{_MAX_RETRIES}). Waiting {wait:.1f}s.")
                time.sleep(wait)

            except self._anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = self._backoff(attempt)
                    log.warning(f"API {exc.status_code} (attempt {attempt+1}). Waiting {wait:.1f}s.")
                    time.sleep(wait)
                else:
                    log.error(f"Non-retryable API error {exc.status_code}: {exc.message}")
                    raise

            except Exception as exc:
                log.error(f"Unexpected error calling Anthropic API: {exc}")
                raise

        raise RuntimeError(f"Anthropic API failed after {_MAX_RETRIES} retries.")

    def chat_json(
        self,
        messages:      List[Dict[str, str]],
        system_prompt: str = "",
        context:       str = "",
    ) -> Any:
        """
        Like chat(), but instructs the model to respond with JSON and
        parses the result automatically.

        Strips markdown code fences before parsing so the model doesn't
        need to be perfectly obedient about formatting.

        Returns:
            Parsed Python object (dict or list).

        Raises:
            json.JSONDecodeError: If the response is not valid JSON.
        """
        json_system = (
            system_prompt
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
              "No explanation, no markdown fences, no text outside the JSON."
        )
        raw = self.chat(messages, system_prompt=json_system, context=context)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        return json.loads(cleaned)

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_system(self, base: str, context: str) -> str:
        if not context:
            return base
        return (
            "=== DATA CONTEXT ===\n"
            f"{context}\n"
            "=== END DATA CONTEXT ===\n\n"
            + base
        )

    @staticmethod
    def _backoff(attempt: int) -> float:
        import random
        return min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), _MAX_DELAY)
