"""
src/data_assistant.py — LLM-powered data analysis assistant.

Architecture
============
LLMDataAssistant
├── DataFrameEmbedder    converts the DataFrame into searchable text chunks
├── QueryProcessor       rule-based fast path for common query types
├── ContextBuilder       builds concise LLM-readable DataFrame summaries
├── AnthropicBackend     Anthropic API client with retry logic
└── ConversationHistory  tracks multi-turn dialogue with token-aware trimming

Query routing
=============
1. High-confidence rule match → answered instantly, free, deterministic
   (correlations, missing data, averages, groupbys, row/column counts, etc.)
2. Low-confidence or open-ended query → routed to Anthropic API with full
   DataFrame context and conversation history attached
3. LLM unavailable (no key, network error) → graceful fallback to help text

The two-path design makes the assistant fast and free for ~80% of typical
analytical questions, while the LLM handles ambiguous phrasing, multi-step
reasoning, and follow-up questions that reference earlier answers.

Multi-turn example
==================
    assistant = LLMDataAssistant(api_key=os.getenv("ANTHROPIC_API_KEY"))
    assistant.load_data("q3_sales.csv")

    r1 = assistant.ask("Which region had the highest revenue?")
    r2 = assistant.ask("How does that compare to last quarter?")   # references r1
    r3 = assistant.ask("Break it down month by month")             # references r2

    assistant.save_conversation("sessions/q3_analysis.json")
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.context_builder import ContextBuilder
from src.conversation import ConversationHistory

log = logging.getLogger(__name__)

# Confidence threshold: rule-based results below this are escalated to the LLM
_RULE_CONFIDENCE_THRESHOLD = 0.7

_SYSTEM_PROMPT = """\
You are a precise, concise data analyst assistant.

The DATA CONTEXT block in the system prompt contains a summary of the dataset.
Answer the user's question based on that data. Guidelines:
  - Give direct, specific answers with numbers where possible
  - If the question references a previous answer, use the conversation history
  - If you cannot answer from the provided context, say so clearly
  - Keep answers under 150 words unless a detailed breakdown is requested
  - Format numbers with commas (e.g. 1,234.56)
"""


# ── DataFrameEmbedder ──────────────────────────────────────────────────────────

class DataFrameEmbedder:
    """
    Converts a DataFrame into a list of searchable text chunks.

    Chunks are one-unit-of-information text snippets: a dataset summary,
    a column description, or a sample row. Used by QueryProcessor for
    keyword routing and by ContextBuilder when assembling LLM prompts.
    """

    def __init__(self):
        self.chunks:   List[Dict] = []
        self.metadata: Dict[str, Any] = {}

    def process_dataframe(self, df: pd.DataFrame, name: str = "dataset") -> List[Dict]:
        """Build and store the chunk list. Returns the list for inspection."""
        chunks: List[Dict] = []

        # Dataset-level summary
        chunks.append({
            "content":  (
                f"Dataset '{name}' has {len(df)} rows and {len(df.columns)} columns. "
                f"Columns: {', '.join(df.columns)}"
            ),
            "type":     "summary",
            "metadata": {"name": name, "shape": df.shape},
        })

        # Per-column chunks
        for col in df.columns:
            info = self._analyse_column(df, col)
            chunks.append({
                "content":  f"Column '{col}': {info['description']}",
                "type":     "column",
                "metadata": {"column": col, **info},
            })

        # Sample row chunks (up to 5)
        for i in range(min(5, len(df))):
            row_text = f"Row {i}: " + ", ".join(
                f"{col}={df.iloc[i][col]}" for col in df.columns[:6]
            )
            chunks.append({"content": row_text, "type": "sample", "metadata": {"row": i}})

        self.chunks   = chunks
        self.metadata = {"name": name, "df": df}
        return chunks

    def _analyse_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "dtype":   str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "unique":  int(df[col].nunique()),
        }
        if df[col].dtype in ("int64", "float64"):
            info["description"] = (
                f"Numeric with mean={df[col].mean():.2f}, "
                f"range=[{df[col].min():.2f}, {df[col].max():.2f}]"
            )
            info["stats"] = {
                "mean": float(df[col].mean()), "std":  float(df[col].std()),
                "min":  float(df[col].min()),  "max":  float(df[col].max()),
            }
        else:
            top = df[col].value_counts().head(1)
            info["description"] = (
                f"Categorical with {info['unique']} unique values, "
                f"most common: {top.index[0]}" if len(top) else
                f"Categorical with {info['unique']} unique values"
            )
            info["top_values"] = df[col].value_counts().head(5).to_dict()
        return info


# ── QueryProcessor ─────────────────────────────────────────────────────────────

class QueryProcessor:
    """
    Deterministic query router for common analytical question types.

    Returns a confidence score (0.0–1.0) alongside each answer so the
    caller can decide whether to accept the result or escalate to the LLM.
    confidence=1.0 means the match was unambiguous; <0.7 means the LLM
    should be tried.
    """

    def __init__(self, embedder: DataFrameEmbedder):
        self.df: pd.DataFrame = embedder.metadata["df"]

    def process_query(self, query: str) -> Dict[str, Any]:
        """Attempt to answer the query via rule-based matching."""
        q = query.lower()

        if "correlat" in q:
            return {**self._correlation(), "confidence": 1.0}
        if "missing" in q or ("null" in q and "value" in q):
            return {**self._missing(), "confidence": 1.0}
        if "average" in q or "mean" in q:
            return {**self._average(q), "confidence": 1.0}
        if any(k in q for k in ("highest", "lowest", "top", "bottom", "which", "by region", "by category", "group")):
            return {**self._group(q), "confidence": 0.85}
        if "column" in q or "field" in q or "what column" in q:
            return {**self._columns(), "confidence": 1.0}
        if ("row" in q and "how many" in q) or "how many record" in q or "dataset size" in q:
            return {**self._rows(), "confidence": 1.0}
        if "summary" in q or "describe" in q or "overview" in q:
            return {**self._summary(), "confidence": 1.0}
        if any(k in q for k in ("visuali", "plot", "chart", "graph")):
            return {**self._visualizations(), "confidence": 1.0}

        return {"answer": None, "data": None, "type": "unmatched", "confidence": 0.0}

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _columns(self) -> Dict:
        return {
            "answer": f"The dataset has {len(self.df.columns)} columns: {', '.join(self.df.columns)}",
            "data":   list(self.df.columns),
            "type":   "column_list",
        }

    def _rows(self) -> Dict:
        return {"answer": f"The dataset contains {len(self.df):,} rows.", "data": len(self.df), "type": "row_count"}

    def _correlation(self) -> Dict:
        numeric = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric) < 2:
            return {"answer": "Need at least 2 numeric columns for correlation analysis.", "data": None, "type": "error"}
        corr = self.df[numeric].corr()
        pairs = sorted([
            {"col1": corr.columns[i], "col2": corr.columns[j], "r": corr.iloc[i, j]}
            for i in range(len(corr.columns))
            for j in range(i + 1, len(corr.columns))
        ], key=lambda x: abs(x["r"]), reverse=True)
        lines = [f"  {p['col1']} ↔ {p['col2']}: r={p['r']:.3f}" for p in pairs[:5]]
        return {"answer": "Top correlations:\n" + "\n".join(lines), "data": pairs[:5], "type": "correlation", "visualization": "heatmap"}

    def _average(self, q: str) -> Dict:
        numeric  = self.df.select_dtypes(include=[np.number]).columns.tolist()
        targeted = [c for c in numeric if c.lower() in q or c.replace("_", " ") in q]
        cols     = targeted or numeric
        lines    = [f"  {c}: {self.df[c].mean():,.2f}" for c in cols]
        return {"answer": "Average values:\n" + "\n".join(lines), "data": {c: float(self.df[c].mean()) for c in cols}, "type": "average"}

    def _group(self, q: str) -> Dict:
        cat_cols = self.df.select_dtypes(include="object").columns.tolist()
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not cat_cols or not num_cols:
            return {"answer": "No categorical columns available for grouping.", "data": None, "type": "group"}
        group_col = next((c for c in cat_cols if c.lower() in q), cat_cols[0])
        value_col = next((c for c in num_cols if c.lower() in q or c.replace("_", " ") in q), num_cols[0])
        grouped   = self.df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
        lines     = [f"  {idx}: {val:,.0f}" for idx, val in grouped.items()]
        return {"answer": f"By {group_col}, highest {value_col} is '{grouped.index[0]}':\n" + "\n".join(lines), "data": grouped.to_dict(), "type": "group"}

    def _missing(self) -> Dict:
        m    = self.df.isnull().sum()
        cols = m[m > 0]
        if len(cols) == 0:
            return {"answer": "No missing values found.", "data": {}, "type": "missing"}
        lines = [f"  {c}: {n} ({n / len(self.df) * 100:.1f}%)" for c, n in cols.items()]
        return {"answer": f"Missing values in {len(cols)} column(s):\n" + "\n".join(lines), "data": cols.to_dict(), "type": "missing"}

    def _summary(self) -> Dict:
        stats = self.df.describe()
        return {"answer": f"Dataset summary:\n{stats}", "data": stats.to_dict(), "type": "summary"}

    def _visualizations(self) -> Dict:
        numeric     = self.df.select_dtypes(include=[np.number]).columns
        suggestions = []
        if len(numeric) >= 2:
            suggestions += [f"Scatter: {numeric[0]} vs {numeric[1]}", "Correlation heatmap"]
        for c in numeric[:3]:
            suggestions.append(f"Histogram of {c}")
        return {"answer": "Suggested visualizations:\n" + "\n".join(f"  - {s}" for s in suggestions), "data": suggestions, "type": "visualization_suggestions"}


# ── LLMDataAssistant ──────────────────────────────────────────────────────────

class LLMDataAssistant:
    """
    LLM-powered data analysis assistant with deterministic fast path.

    Answers natural-language questions about a loaded CSV dataset using a
    two-path strategy:

        Fast path  — common, structured queries answered instantly via
                     keyword routing (averages, correlations, groupbys, etc.)
        LLM path   — ambiguous, open-ended, or follow-up queries routed to
                     the Anthropic API with full DataFrame context and
                     conversation history

    The LLM path is optional. Without an API key the assistant still works
    for all common query types and returns a descriptive help message for
    anything it can't handle deterministically.

    Args:
        api_key:            Anthropic API key. Falls back to ANTHROPIC_API_KEY
                            env var. Omit or set to None for rule-based mode.
        max_history_tokens: Token budget for stored conversation history.
                            Default 3000 — leaves room for context + response.

    Example:
        >>> assistant = LLMDataAssistant()          # rule-based only
        >>> assistant = LLMDataAssistant(api_key=os.getenv("ANTHROPIC_API_KEY"))
        >>> assistant.load_data("data/sales_q3.csv")
        >>> print(assistant.ask("Which region had the highest revenue?")["answer"])
        >>> print(assistant.ask("What drove that? Anything unusual?")["answer"])
    """

    def __init__(
        self,
        api_key:            Optional[str] = None,
        max_history_tokens: int           = 3000,
    ):
        self.embedder:        Optional[DataFrameEmbedder] = None
        self.processor:       Optional[QueryProcessor]    = None
        self.context_builder: Optional[ContextBuilder]    = None
        self.df:              Optional[pd.DataFrame]      = None
        self.history = ConversationHistory(max_tokens=max_history_tokens)
        self._llm:   Any = None

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if key:
            try:
                from src.llm_backend import AnthropicBackend
                self._llm = AnthropicBackend(api_key=key)
                log.info("LLM backend ready — hybrid mode active.")
            except Exception as exc:
                log.warning(f"LLM backend failed to init ({exc}). Rule-based mode only.")
        else:
            log.info("No API key — rule-based mode only.")

    # ── Data loading ───────────────────────────────────────────────────────────

    def load_data(self, filepath: str) -> None:
        """
        Load a CSV and initialise all analysis components.

        Resets conversation history — follow-up questions from a previous
        dataset won't bleed into a new session.

        Args:
            filepath: Path to the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.df           = pd.read_csv(filepath)
        name              = path.stem
        self.embedder     = DataFrameEmbedder()
        self.embedder.process_dataframe(self.df, name)
        self.processor       = QueryProcessor(self.embedder)
        self.context_builder = ContextBuilder(self.df, dataset_name=name)
        self.history.clear()

        print(f"✅ Loaded '{name}': {len(self.df):,} rows × {len(self.df.columns)} columns")

    # ── Core query interface ───────────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Answer a natural-language question about the loaded dataset.

        Routes to the rule-based handler if confidence ≥ threshold.
        Otherwise calls the LLM with DataFrame context and conversation
        history. Falls back gracefully if the LLM is unavailable.

        Args:
            question: Any natural-language question about the data.

        Returns:
            Dict with keys:
                answer     — human-readable response string
                data       — structured data (dict/list) when available
                type       — answer category identifier
                source     — "rules" or "llm"
                confidence — rule-based confidence score (0.0–1.0)
        """
        if self.processor is None:
            return {
                "answer": "Please call load_data() first.",
                "data": None, "type": "error", "source": "rules", "confidence": 1.0,
            }

        # ── Fast path: rule-based ──────────────────────────────────────────────
        rule_result = self.processor.process_query(question)
        if rule_result.get("confidence", 0.0) >= _RULE_CONFIDENCE_THRESHOLD:
            self.history.add("user",      question)
            self.history.add("assistant", rule_result["answer"])
            return {**rule_result, "source": "rules"}

        # ── LLM path ──────────────────────────────────────────────────────────
        if self._llm is not None:
            try:
                context = self.context_builder.build_context(query=question)
                self.history.add("user", question)
                reply = self._llm.chat(
                    messages=self.history.get_messages(),
                    system_prompt=_SYSTEM_PROMPT,
                    context=context,
                )
                self.history.add("assistant", reply)
                return {
                    "answer": reply, "data": None,
                    "type": "llm_response", "source": "llm",
                    "confidence": rule_result.get("confidence", 0.0),
                }
            except Exception as exc:
                log.warning(f"LLM call failed: {exc}. Falling back to help message.")
                # Roll back the user message we already added
                if self.history.messages and self.history.messages[-1]["role"] == "user":
                    self.history.messages.pop()

        # ── Fallback: help ────────────────────────────────────────────────────
        fallback = (
            "I can answer questions about: column names, row counts, missing values, "
            "correlations, averages, group aggregations (highest/lowest by category), "
            "dataset summaries, and visualization suggestions.\n\n"
            "For open-ended, multi-step, or follow-up questions, "
            "set ANTHROPIC_API_KEY to enable the LLM path."
        )
        self.history.add("user",      question)
        self.history.add("assistant", fallback)
        return {"answer": fallback, "data": None, "type": "help", "source": "rules", "confidence": 0.0}

    # ── Insights ───────────────────────────────────────────────────────────────

    def generate_insights(self) -> List[str]:
        """
        Automatically surface notable patterns in the loaded dataset.

        Checks for significant missing data, strong correlations,
        outlier-heavy columns, and dominant categorical values.

        Returns:
            List of insight strings prefixed with emoji indicators.
        """
        if self.df is None:
            return []

        insights: List[str] = []

        total_missing = int(self.df.isnull().sum().sum())
        if total_missing:
            insights.append(f"⚠️  {total_missing:,} missing values across {int(self.df.isnull().any().sum())} column(s)")

        numeric = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric) >= 2:
            corr   = self.df[numeric].corr()
            strong = [
                (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                for i in range(len(corr.columns))
                for j in range(i + 1, len(corr.columns))
                if abs(corr.iloc[i, j]) > 0.7
            ]
            if strong:
                best = max(strong, key=lambda x: abs(x[2]))
                insights.append(
                    f"🔍 {len(strong)} strong correlation(s) — "
                    f"strongest: {best[0]} ↔ {best[1]} (r={best[2]:.3f})"
                )

        for col in numeric:
            if "date" in col.lower():
                continue
            q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            iqr    = q3 - q1
            n_out  = int(((self.df[col] < q1 - 1.5 * iqr) | (self.df[col] > q3 + 1.5 * iqr)).sum())
            if n_out > len(self.df) * 0.05:
                insights.append(f"📊 {col}: {n_out:,} potential outliers ({n_out / len(self.df):.1%} of rows)")

        for col in self.df.select_dtypes(include="object").columns:
            top_count = self.df[col].value_counts().iloc[0]
            pct       = top_count / len(self.df) * 100
            if pct > 40:
                insights.append(f"📌 {col}: '{self.df[col].value_counts().index[0]}' dominates ({pct:.1f}% of rows)")

        return insights

    # ── Conversation management ────────────────────────────────────────────────

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return a copy of the full conversation message list."""
        return self.history.get_messages()

    def reset_conversation(self) -> None:
        """Clear conversation history without reloading data."""
        self.history.clear()

    def save_conversation(self, path: str) -> None:
        """Persist conversation history to a JSON file."""
        self.history.save(path)

    def load_conversation(self, path: str) -> None:
        """Restore conversation from a JSON file saved by save_conversation()."""
        self.history = ConversationHistory.load(path)


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    """
    End-to-end demo of LLMDataAssistant.

    Set ANTHROPIC_API_KEY to see the LLM path handle the open-ended
    questions at the end. Rule-based questions work without a key.
    """
    import tempfile

    np.random.seed(42)
    df = pd.DataFrame({
        "date":     pd.date_range("2024-01-01", periods=100),
        "sales":    np.random.poisson(100, 100) + np.arange(100) * 0.5,
        "revenue":  np.random.normal(5000, 1000, 100) + np.arange(100) * 10,
        "region":   np.random.choice(["North", "South", "East", "West"], 100),
        "category": np.random.choice(["A", "B", "C"], 100),
    })
    df.loc[np.random.choice(100, 10), "revenue"] = np.nan

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        filepath = f.name

    print("LLM Data Analysis Assistant Demo")
    print("=" * 55)

    assistant = LLMDataAssistant()
    assistant.load_data(filepath)

    questions = [
        "What columns are in this dataset?",
        "Are there any missing values?",
        "What are the strongest correlations?",
        "Which region has the highest sales?",
        "What is the average revenue?",
        # Open-ended — route to LLM if key is set
        "What story does the revenue trend tell over the 100 days?",
        "Based on these patterns, where would you invest next quarter and why?",
    ]

    print("\n📝 Questions:\n")
    for q in questions:
        result = assistant.ask(q)
        source = "🤖 LLM" if result.get("source") == "llm" else "⚡ Rules"
        print(f"Q: {q}")
        print(f"A [{source}]: {result['answer']}\n")

    print("\n💡 Automatic Insights:")
    for insight in assistant.generate_insights():
        print(f"  {insight}")

    print(f"\n📜 {len(assistant.history)} messages in conversation history")


if __name__ == "__main__":
    demo()
