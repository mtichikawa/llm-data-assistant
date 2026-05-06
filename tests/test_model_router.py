"""Tests for ModelRouter — purely deterministic routing, no API calls."""

from model_router import HAIKU, SONNET, ModelRouter


def test_default_model_for_short_factual_query():
    router = ModelRouter()
    assert router.select("What columns are in this dataset?") == HAIKU


def test_escalation_keyword_routes_to_sonnet():
    router = ModelRouter()
    # 'why' and 'compare' should both trigger escalation
    assert router.select("Why is revenue down this quarter?") == SONNET
    assert router.select("Compare Q3 to Q2 spending") == SONNET


def test_history_depth_triggers_escalation():
    router = ModelRouter()
    short_query = "Show top categories"
    # 0, 1, 2 turns: stay on default. 3+ turns: escalate.
    assert router.select(short_query, history_turns=0) == HAIKU
    assert router.select(short_query, history_turns=2) == HAIKU
    assert router.select(short_query, history_turns=3) == SONNET
    assert router.select(short_query, history_turns=10) == SONNET


def test_long_query_routes_to_sonnet():
    router = ModelRouter()
    long_query = (
        "Take the dataset I just loaded and tell me about the relationship "
        "between every numeric column and revenue, broken down by region, "
        "with seasonal effects considered carefully and excluded from the headline number."
    )
    assert len(long_query) > 200
    assert router.select(long_query) == SONNET


def test_empty_query_falls_through_to_default():
    router = ModelRouter()
    assert router.select("") == HAIKU
    assert router.select("   ") == HAIKU


def test_configurable_models_are_honoured():
    router = ModelRouter(default="custom-default", escalated="custom-escalated")
    assert router.select("simple lookup") == "custom-default"
    assert router.select("explain the trend") == "custom-escalated"
    assert router.select("simple", history_turns=5) == "custom-escalated"
