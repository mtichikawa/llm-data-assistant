"""Smoke tests for LLMDataAssistant — no API key needed (rule-based path only)."""

import os
import tempfile

import pandas as pd
import pytest

from data_assistant import LLMDataAssistant


@pytest.fixture
def assistant_no_key(monkeypatch):
    """Construct an assistant explicitly without an API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return LLMDataAssistant(api_key=None)


@pytest.fixture
def sales_csv():
    """Write a small CSV to a temporary file and return its path."""
    df = pd.DataFrame({
        "region":  ["North", "South", "East", "West"] * 5,
        "revenue": [100, 200, 150, 175] * 5,
        "units":   [10, 20, 15, 17] * 5,
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    yield path
    os.unlink(path)


def test_assistant_constructs_without_api_key(assistant_no_key):
    """No key → rule-based mode only, both backends are None."""
    assert assistant_no_key._llm_default is None
    assert assistant_no_key._llm_escalated is None


def test_load_data_accepts_csv(assistant_no_key, sales_csv):
    """A valid CSV path initialises the embedder and processor."""
    assistant_no_key.load_data(sales_csv)
    assert assistant_no_key.df is not None
    assert len(assistant_no_key.df) == 20
    assert list(assistant_no_key.df.columns) == ["region", "revenue", "units"]


def test_rule_based_path_returns_deterministic_answer(assistant_no_key, sales_csv):
    """Column-list query routes to the rule path with high confidence."""
    assistant_no_key.load_data(sales_csv)
    result = assistant_no_key.ask("What columns are in this dataset?")
    assert result["source"] == "rules"
    assert result["confidence"] >= 0.7
    assert "region" in result["answer"]
    assert "revenue" in result["answer"]
