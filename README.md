# LLM-Powered Data Analysis Assistant

A Python library: load a CSV into `LLMDataAssistant` and ask questions about it in natural language. It also generates basic insights and charts. There is no upload UI.

## Features

- **Natural Language Queries** - "What columns exist?", "Show correlations"
- **Automatic Insights** - Detects missing data, outliers, correlations
- **Smart Model Routing** - Haiku 4.5 for short lookups, Sonnet 4.6 escalation on complexity
- **Conversation Summarization** - Older turns compressed to a residue, not dropped
- **Smart Visualizations** - Distributions, heatmaps, time series
- **Statistical Analysis** - Summary stats, correlation analysis
- **Export Reports** - Generate comprehensive visual reports

## Architecture

### Two-path query routing

1. **Fast path (rule-based)** - `QueryProcessor` handles common query types
   (correlations, groupbys, averages, missing values) with a confidence
   score. Queries scoring >=0.7 are answered deterministically, free, with
   no API call.
2. **LLM path** - queries below threshold are sent to Anthropic with a
   `ContextBuilder`-assembled DataFrame summary and `ConversationHistory`
   for multi-turn support.

### Smart model routing

`ModelRouter` picks between Haiku 4.5 and Sonnet 4.6 for each query that
reaches the LLM path. The routing decision is rule-based and free; the
model call it routes to is a billed API call. Default is Haiku. Escalates
to Sonnet when:

- the query contains a complexity keyword (`why`, `compare`, `explain`,
  `interpret`, `recommend`, `forecast`, `outlier`, `root cause`, ...)
- the conversation already holds 3 or more messages (each question and
  each answer counts as one), so from the third question on
- the query is over 200 chars (likely complex)

The router is a constructor parameter on `LLMDataAssistant`, so callers
can swap in custom rules or different model tiers without touching the
assistant code.

### Conversation summarization

When the conversation history exceeds its token budget,
`ConversationHistory` no longer drops the oldest messages outright.
With a `summarizer` callable wired in (the assistant points it at the
cheap Haiku backend), dropped messages are compressed into a single
synthetic `[Earlier in this conversation: ...]` turn that preserves
entity references, numbers, and column names. Subsequent compressions
re-summarize the existing residue plus newly dropped messages so
nesting doesn't accumulate.

Without a summarizer, the older trim-oldest behaviour is preserved
unchanged - it's purely additive.

## Quick Start

```bash
pip install -r requirements.txt
mkdir -p data
cd examples && python quick_start.py
```

The script adds `../src` to the import path and writes its sample CSV to
`../data/`, so run it from `examples/`. The LLM path needs the `anthropic`
package (in `requirements.txt`) and
`ANTHROPIC_API_KEY`. Without them, every question uses the rule-based path.

## Usage

The modules in `src/` import each other by bare name, so put `src/` on the path (for example `PYTHONPATH=src`).

```python
from data_assistant import LLMDataAssistant

assistant = LLMDataAssistant()
assistant.load_data('your_data.csv')

# Ask questions
result = assistant.ask("What columns are in the dataset?")
print(result['answer'])

# Get insights
insights = assistant.generate_insights()

# Create visualizations
from visualizations import VisualizationGenerator
viz = VisualizationGenerator(assistant.df)
plots = viz.generate_all()
```

### Multi-turn with smart routing

```python
assistant = LLMDataAssistant(api_key=os.getenv("ANTHROPIC_API_KEY"))
assistant.load_data("q3_sales.csv")

r1 = assistant.ask("What columns are here?")          # → rules path (free)
r2 = assistant.ask("Average revenue?")                # → rules path (free)
r3 = assistant.ask("Why did revenue dip in week 6?")  # → Sonnet ('why' keyword)
r4 = assistant.ask("Compare to Q2")                   # → Sonnet ('compare' keyword)
r5 = assistant.ask("Anything else?")                  # → Sonnet (8 prior messages)

# r3..r5 each return result["model"] so you can see which tier ran
```

## Jupyter Notebook

See `notebooks/llm_data_assistant_analysis.ipynb` for a full demonstration.

## What's Inside

- `src/data_assistant.py` - LLMDataAssistant orchestrator (rule-based + LLM)
- `src/model_router.py` - Per-query model selection (Haiku <-> Sonnet)
- `src/conversation.py` - Multi-turn history with optional summarizer compression
- `src/llm_backend.py` - Anthropic API wrapper with retry + JSON mode
- `src/context_builder.py` - DataFrame summary assembly for the system prompt
- `src/visualizations.py` - Auto-visualization
- `notebooks/llm_data_assistant_analysis.ipynb` - Full workflow
- `examples/quick_start.py` - 5-minute demo
- `tests/` - 14 tests covering ModelRouter, ConversationHistory, smoke path

## What I Learned

- Building LLM context from a DataFrame summary, plus rule-based routing
- Natural language to data queries
- Automatic insight generation

Contact: Mike Ichikawa - projects.ichikawa@gmail.com

> _Note: the fast-path / LLM-path routing threshold defaults to 0.7. Lower it to send more queries to the deterministic fast path (saves API cost, risks lower-quality answers on edge cases); raise it to fall back to the LLM more aggressively._
