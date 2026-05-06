# LLM-Powered Data Analysis Assistant

Upload CSV files and ask questions in natural language. Get automatic insights and visualizations.

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

`ModelRouter` picks between Haiku 4.5 and Sonnet 4.6 per query at zero
per-call cost. Default is Haiku. Escalates to Sonnet when:

- the query contains a complexity keyword (`why`, `compare`, `explain`,
  `interpret`, `recommend`, `forecast`, `outlier`, `root cause`, ...)
- the conversation history has 3 or more turns (likely deep analysis)
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
python examples/quick_start.py
```

## Usage

```python
from src.data_assistant import LLMDataAssistant

assistant = LLMDataAssistant()
assistant.load_data('your_data.csv')

# Ask questions
result = assistant.ask("What columns are in the dataset?")
print(result['answer'])

# Get insights
insights = assistant.generate_insights()

# Create visualizations
from src.visualizations import VisualizationGenerator
viz = VisualizationGenerator(assistant.df)
plots = viz.generate_all()
```

### Multi-turn with smart routing

```python
assistant = LLMDataAssistant(api_key=os.getenv("ANTHROPIC_API_KEY"))
assistant.load_data("q3_sales.csv")

r1 = assistant.ask("What columns are here?")          # → Haiku (factual lookup)
r2 = assistant.ask("Average revenue?")                # → rules path (free)
r3 = assistant.ask("Why did revenue dip in week 6?")  # → Sonnet ('why' keyword)
r4 = assistant.ask("Compare to Q2")                   # → Sonnet ('compare' keyword)
r5 = assistant.ask("Anything else?")                  # → Sonnet (4 prior turns)

# r3..r5 each return result["model"] so you can see which tier ran
```

## Jupyter Notebook

See `notebooks/complete_demo.ipynb` for full demonstration.

## What's Inside

- `src/data_assistant.py` - LLMDataAssistant orchestrator (rule-based + LLM)
- `src/model_router.py` - Per-query model selection (Haiku <-> Sonnet)
- `src/conversation.py` - Multi-turn history with optional summarizer compression
- `src/llm_backend.py` - Anthropic API wrapper with retry + JSON mode
- `src/context_builder.py` - DataFrame summary assembly for the system prompt
- `src/visualizations.py` - Auto-visualization
- `notebooks/complete_demo.ipynb` - Full workflow
- `examples/quick_start.py` - 5-minute demo
- `tests/` - 14 tests covering ModelRouter, ConversationHistory, smoke path

## What I Learned

- Building RAG systems for structured data
- Natural language to data queries
- Automatic insight generation
- Production-ready data analysis tools

Contact: Mike Ichikawa - projects.ichikawa@gmail.com

# 2025-10-05
# 2025-10-05
# 2025-10-09
# 2025-10-14
# 2025-10-18
# 2025-10-23
# 2025-10-28
# 2025-11-02
# 2025-11-07
# 2025-11-11
# 2025-11-16
# 2025-11-21
# 2025-11-26
# 2025-12-01
# 2025-12-05
# 2025-12-10
# 2025-12-15
# 2025-12-20
# 2025-12-24
# 2025-12-28
# 2026-01-03