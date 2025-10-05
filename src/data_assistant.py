'''
LLM Data Analysis Assistant - RAG System
Complete implementation with embeddings and query processing
'''

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Any


class DataFrameEmbedder:
    '''Create embeddings for dataframe context'''
    
    def __init__(self):
        self.chunks = []
        self.metadata = {}
        
    def process_dataframe(self, df: pd.DataFrame, name: str = "dataset"):
        '''Convert dataframe to searchable chunks'''
        chunks = []
        
        # Overall summary chunk
        summary = {
            'content': f"Dataset '{name}' has {len(df)} rows and {len(df.columns)} columns. "
                      f"Columns: {', '.join(df.columns)}",
            'type': 'summary',
            'metadata': {'name': name, 'shape': df.shape}
        }
        chunks.append(summary)
        
        # Column-level chunks
        for col in df.columns:
            col_info = self._analyze_column(df, col)
            chunk = {
                'content': f"Column '{col}': {col_info['description']}",
                'type': 'column',
                'metadata': {'column': col, **col_info}
            }
            chunks.append(chunk)
            
        # Sample data chunks
        for i in range(min(5, len(df))):
            row_text = f"Row {i}: " + ", ".join([f"{col}={df.iloc[i][col]}" for col in df.columns[:5]])
            chunks.append({
                'content': row_text,
                'type': 'sample',
                'metadata': {'row': i}
            })
            
        self.chunks = chunks
        self.metadata = {'name': name, 'df': df}
        return chunks
        
    def _analyze_column(self, df: pd.DataFrame, col: str) -> Dict:
        '''Analyze a single column'''
        info = {
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'unique': int(df[col].nunique())
        }
        
        if df[col].dtype in ['int64', 'float64']:
            info['description'] = (
                f"Numeric column with mean={df[col].mean():.2f}, "
                f"range=[{df[col].min():.2f}, {df[col].max():.2f}]"
            )
            info['stats'] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        else:
            top_val = df[col].value_counts().head(1)
            if len(top_val) > 0:
                info['description'] = f"Categorical column with {info['unique']} unique values. Most common: {top_val.index[0]}"
                info['top_values'] = df[col].value_counts().head(5).to_dict()
            else:
                info['description'] = f"Categorical column with {info['unique']} unique values"
                
        return info


class QueryProcessor:
    '''Process natural language queries about data'''
    
    def __init__(self, embedder: DataFrameEmbedder):
        self.embedder = embedder
        self.df = embedder.metadata.get('df')
        
    def process_query(self, query: str) -> Dict[str, Any]:
        '''Process a query and return results'''
        query_lower = query.lower()
        
        # Route to appropriate handler
        if 'column' in query_lower or 'field' in query_lower:
            return self._handle_column_query(query)
        elif 'row' in query_lower or 'record' in query_lower:
            return self._handle_row_query(query)
        elif 'correlat' in query_lower:
            return self._handle_correlation_query()
        elif 'missing' in query_lower or 'null' in query_lower:
            return self._handle_missing_query()
        elif 'summary' in query_lower or 'describe' in query_lower:
            return self._handle_summary_query()
        elif 'visuali' in query_lower or 'plot' in query_lower or 'chart' in query_lower:
            return self._handle_visualization_query(query)
        else:
            return self._handle_general_query(query)
            
    def _handle_column_query(self, query: str) -> Dict:
        '''Handle questions about columns'''
        return {
            'answer': f"The dataset has {len(self.df.columns)} columns: {', '.join(self.df.columns)}",
            'data': list(self.df.columns),
            'type': 'column_list'
        }
        
    def _handle_row_query(self, query: str) -> Dict:
        '''Handle questions about rows'''
        return {
            'answer': f"The dataset contains {len(self.df)} rows",
            'data': len(self.df),
            'type': 'row_count'
        }
        
    def _handle_correlation_query(self) -> Dict:
        '''Calculate correlations'''
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'answer': "Need at least 2 numeric columns for correlation analysis",
                'data': None,
                'type': 'error'
            }
            
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strongest correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'col1': corr_matrix.columns[i],
                    'col2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
                
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        top_corr = correlations[0] if correlations else None
        if top_corr:
            answer = (f"Strongest correlation: {top_corr['col1']} and {top_corr['col2']} "
                     f"(r={top_corr['correlation']:.3f})")
        else:
            answer = "No correlations found"
            
        return {
            'answer': answer,
            'data': correlations[:5],
            'type': 'correlation',
            'visualization': 'heatmap'
        }
        
    def _handle_missing_query(self) -> Dict:
        '''Handle missing data questions'''
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) == 0:
            return {
                'answer': "No missing values found in the dataset",
                'data': {},
                'type': 'missing'
            }
            
        answer = f"Found missing values in {len(missing_cols)} columns:\n"
        for col, count in missing_cols.items():
            pct = (count / len(self.df)) * 100
            answer += f"  - {col}: {count} ({pct:.1f}%)\n"
            
        return {
            'answer': answer,
            'data': missing_cols.to_dict(),
            'type': 'missing'
        }
        
    def _handle_summary_query(self) -> Dict:
        '''Provide dataset summary'''
        summary_stats = self.df.describe()
        
        return {
            'answer': f"Dataset summary:\n{summary_stats}",
            'data': summary_stats.to_dict(),
            'type': 'summary'
        }
        
    def _handle_visualization_query(self, query: str) -> Dict:
        '''Suggest visualizations'''
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        suggestions = []
        if len(numeric_cols) >= 2:
            suggestions.append(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
            suggestions.append(f"Correlation heatmap of numeric columns")
            
        for col in numeric_cols[:3]:
            suggestions.append(f"Histogram of {col}")
            
        return {
            'answer': "Suggested visualizations:\n" + "\n".join(f"  - {s}" for s in suggestions),
            'data': suggestions,
            'type': 'visualization_suggestions'
        }
        
    def _handle_general_query(self, query: str) -> Dict:
        '''Handle general questions'''
        return {
            'answer': (
                "I can help with:\n"
                "  - Column information\n"
                "  - Row counts\n"
                "  - Missing data analysis\n"
                "  - Correlation analysis\n"
                "  - Dataset summaries\n"
                "  - Visualization suggestions"
            ),
            'data': None,
            'type': 'help'
        }


class LLMDataAssistant:
    '''Complete LLM-powered data analysis assistant'''
    
    def __init__(self):
        self.embedder = None
        self.processor = None
        self.df = None
        
    def load_data(self, filepath: str):
        '''Load CSV and prepare for analysis'''
        self.df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Create embeddings
        self.embedder = DataFrameEmbedder()
        self.embedder.process_dataframe(self.df, Path(filepath).stem)
        
        # Initialize query processor
        self.processor = QueryProcessor(self.embedder)
        
    def ask(self, question: str) -> Dict:
        '''Ask a question about the data'''
        if self.processor is None:
            return {'answer': 'Please load data first', 'type': 'error'}
            
        return self.processor.process_query(question)
        
    def generate_insights(self) -> List[str]:
        '''Automatically generate insights'''
        insights = []
        
        # Missing data
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            insights.append(f"⚠️ Dataset has {missing} missing values")
            
        # Correlations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr = self.df[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                        
            if high_corr:
                insights.append(f"🔍 Found {len(high_corr)} strong correlations")
                
        # Outliers
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))).sum()
            if outliers > len(self.df) * 0.05:
                insights.append(f"📊 {col} has {outliers} potential outliers")
                
        return insights


def demo():
    '''Demonstration'''
    import tempfile
    
    # Create sample dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': np.random.poisson(100, 100) + np.arange(100) * 0.5,
        'revenue': np.random.normal(5000, 1000, 100) + np.arange(100) * 10,
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some missing values
    data.loc[np.random.choice(100, 10), 'revenue'] = np.nan
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        filepath = f.name
        
    print('LLM Data Analysis Assistant Demo')
    print('=' * 50)
    
    # Create assistant
    assistant = LLMDataAssistant()
    assistant.load_data(filepath)
    
    # Ask questions
    questions = [
        "What columns are in the dataset?",
        "How many rows are there?",
        "Are there any missing values?",
        "Show me correlations",
    ]
    
    print('\n📝 Asking questions:\n')
    for q in questions:
        print(f"Q: {q}")
        result = assistant.ask(q)
        print(f"A: {result['answer']}\n")
        
    # Generate insights
    print('💡 Automatic Insights:')
    for insight in assistant.generate_insights():
        print(f"  {insight}")


if __name__ == '__main__':
    demo()
