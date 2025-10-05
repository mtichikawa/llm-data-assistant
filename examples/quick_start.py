'''Quick Start - LLM Data Assistant'''
import sys
sys.path.append('../src')
from data_assistant import LLMDataAssistant
import pandas as pd, numpy as np

np.random.seed(42)
data = pd.DataFrame({
    'sales': np.random.poisson(100, 100),
    'revenue': np.random.normal(5000, 1000, 100)
})
data.to_csv('../data/sample.csv', index=False)

assistant = LLMDataAssistant()
assistant.load_data('../data/sample.csv')

print('Q: What columns?')
print('A:', assistant.ask("What columns?")['answer'])
print('\nInsights:', assistant.generate_insights())
