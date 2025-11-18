"""
src/context_builder.py — Builds concise DataFrame context strings for LLM prompts.

The LLM can't see the DataFrame directly. This module converts it into a
structured text summary: dataset shape, per-column statistics, and a few
sample rows. When a query is provided, columns whose names overlap with
the query are promoted to the top of the list so they land within the
model's attention window.

Output stays under ~6000 characters (~1500 tokens), leaving room for
conversation history and the model's response.

Usage:
    builder  = ContextBuilder(df, dataset_name="sales_q3")
    context  = builder.build_context("Which region had the highest revenue?")
    reply    = backend.chat(messages, context=context)
"""

import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_CHAR_BUDGET = 6_000
_SAMPLE_ROWS = 5


class ContextBuilder:
    """
    Converts a DataFrame into a concise text summary for LLM prompts.

    Summaries include:
        - Dataset name and shape
        - Per-column type, missing rate, and statistics
        - A small sample of rows
        - Query-aware column ordering (relevant columns listed first)

    Args:
        df:           DataFrame to summarise.
        dataset_name: Human-readable name shown in the context header.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        self.df   = df
        self.name = dataset_name

    def build_context(self, query: Optional[str] = None) -> str:
        """
        Build a context string, optionally ranked by relevance to a query.

        When a query is given, columns whose names appear in the query
        (exact token match or substring) are listed first. All columns
        are still included — relevance only affects ordering.

        Args:
            query: The natural-language question the LLM will answer.
                   Pass None for a general-purpose summary.

        Returns:
            Plain text string under _CHAR_BUDGET characters.
        """
        lines: List[str] = []

        lines.append(f"Dataset: '{self.name}'")
        lines.append(f"Shape: {len(self.df):,} rows × {len(self.df.columns)} columns")
        lines.append(f"Columns: {', '.join(self.df.columns)}")
        lines.append("")

        lines.append("=== Column Statistics ===")
        for col in self._ranked_columns(query):
            lines.append(self._column_summary(col))
        lines.append("")

        sample_header = f"=== Sample Rows (first {_SAMPLE_ROWS}) ==="
        lines.append(sample_header)
        lines.append(self.df.head(_SAMPLE_ROWS).to_string(max_cols=10, max_colwidth=40))
        lines.append("")

        full = "\n".join(lines)

        # If over budget, drop sample rows (least critical for reasoning)
        if len(full) > _CHAR_BUDGET:
            idx   = lines.index(sample_header)
            full  = "\n".join(lines[:idx])
            log.debug(f"Context trimmed to {len(full)} chars — sample rows dropped.")

        return full

    # ── Private ────────────────────────────────────────────────────────────────

    def _ranked_columns(self, query: Optional[str]) -> List[str]:
        """Return columns sorted by relevance to the query (most relevant first)."""
        if not query:
            return list(self.df.columns)

        q_tokens = set(re.split(r"[\s_\-,]+", query.lower()))

        def score(col: str) -> int:
            col_parts = set(re.split(r"[\s_\-]+", col.lower()))
            if col_parts & q_tokens:
                return 2
            if any(part in query.lower() for part in col_parts if len(part) > 3):
                return 1
            return 0

        return sorted(self.df.columns, key=score, reverse=True)

    def _column_summary(self, col: str) -> str:
        """Return a one-line statistics summary for a single column."""
        s        = self.df[col]
        dtype    = str(s.dtype)
        n_miss   = int(s.isnull().sum())
        pct_miss = f"{n_miss / len(self.df) * 100:.1f}%" if n_miss else "0%"
        n_unique = s.nunique()

        base = f"  {col} [{dtype}]  missing={n_miss}({pct_miss})  unique={n_unique}"

        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if len(clean):
                return (
                    f"{base}  |  "
                    f"mean={clean.mean():,.2f}  "
                    f"std={clean.std():,.2f}  "
                    f"min={clean.min():,.2f}  "
                    f"max={clean.max():,.2f}  "
                    f"p50={clean.median():,.2f}"
                )

        elif pd.api.types.is_datetime64_any_dtype(s):
            clean = s.dropna()
            if len(clean):
                return f"{base}  |  range=[{clean.min().date()} → {clean.max().date()}]"

        else:
            top = s.value_counts().head(5)
            top_str = ", ".join(f"'{v}'({c})" for v, c in zip(top.index, top.values))
            return f"{base}  |  top=[{top_str}]"

        return base
