# LLM-Powered Data Analysis Assistant

Upload CSV files and ask questions in natural language. Get automatic insights and visualizations.

## Features

- **Natural Language Queries** - "What columns exist?", "Show correlations"
- **Automatic Insights** - Detects missing data, outliers, correlations
- **Smart Visualizations** - Distributions, heatmaps, time series
- **Statistical Analysis** - Summary stats, correlation analysis
- **Export Reports** - Generate comprehensive visual reports

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

## Jupyter Notebook

See `notebooks/complete_demo.ipynb` for full demonstration.

## What's Inside

- `src/data_assistant.py` - Main RAG system (360 lines)
- `src/visualizations.py` - Auto-visualization (240 lines)
- `notebooks/complete_demo.ipynb` - Full workflow
- `examples/quick_start.py` - 5-minute demo

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