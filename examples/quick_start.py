'''Quick Start - LLM Data Assistant'''
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from data_assistant import LLMDataAssistant
import pandas as pd, numpy as np

np.random.seed(42)
data = pd.DataFrame({
    'sales': np.random.poisson(100, 100),
    'revenue': np.random.normal(5000, 1000, 100)
})
sample_csv = ROOT / 'data' / 'sample.csv'
sample_csv.parent.mkdir(exist_ok=True)
data.to_csv(sample_csv, index=False)

assistant = LLMDataAssistant()
assistant.load_data(str(sample_csv))

print('Q: What columns?')
print('A:', assistant.ask("What columns?")['answer'])
print('\nInsights:', assistant.generate_insights())
