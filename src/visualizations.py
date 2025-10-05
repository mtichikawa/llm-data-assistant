'''
Visualization Generator
Automatically create charts and plots for data analysis
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class VisualizationGenerator:
    '''Generate visualizations for data analysis'''
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.output_dir = Path('assets')
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_all(self) -> Dict[str, str]:
        '''Generate all relevant visualizations'''
        plots = {}
        
        # Distribution plots for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limit to 3
            filepath = self.plot_distribution(col)
            if filepath:
                plots[f'distribution_{col}'] = filepath
                
        # Correlation heatmap
        if len(numeric_cols) >= 2:
            filepath = self.plot_correlation_heatmap()
            if filepath:
                plots['correlation_heatmap'] = filepath
                
        # Time series if date column exists
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            filepath = self.plot_time_series(date_cols[0], numeric_cols[0])
            if filepath:
                plots['time_series'] = filepath
                
        # Missing data visualization
        if self.df.isnull().sum().sum() > 0:
            filepath = self.plot_missing_data()
            if filepath:
                plots['missing_data'] = filepath
                
        return plots
        
    def plot_distribution(self, column: str, save: bool = True) -> str:
        '''Plot distribution of a numeric column'''
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.df[column].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        
        # Add statistics
        mean = self.df[column].mean()
        median = self.df[column].median()
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
        ax.legend()
        
        if save:
            filepath = self.output_dir / f'distribution_{column}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def plot_correlation_heatmap(self, save: bool = True) -> str:
        '''Plot correlation heatmap'''
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        if save:
            filepath = self.output_dir / 'correlation_heatmap.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def plot_time_series(self, date_col: str, value_col: str, save: bool = True) -> str:
        '''Plot time series'''
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.df[date_col], self.df[value_col], linewidth=2, marker='o', markersize=3)
        ax.set_title(f'{value_col} over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.grid(alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        if save:
            filepath = self.output_dir / 'time_series.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def plot_missing_data(self, save: bool = True) -> str:
        '''Visualize missing data'''
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        missing.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Missing Values')
        ax.grid(axis='x', alpha=0.3)
        
        if save:
            filepath = self.output_dir / 'missing_data.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
            
    def create_summary_report(self) -> str:
        '''Create comprehensive visual report'''
        fig = plt.figure(figsize=(16, 12))
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Layout: 2x2 grid
        # Top left: First numeric distribution
        if len(numeric_cols) >= 1:
            ax1 = plt.subplot(2, 2, 1)
            self.df[numeric_cols[0]].hist(bins=30, ax=ax1, edgecolor='black', alpha=0.7)
            ax1.set_title(f'Distribution: {numeric_cols[0]}')
            ax1.grid(alpha=0.3)
            
        # Top right: Second numeric distribution
        if len(numeric_cols) >= 2:
            ax2 = plt.subplot(2, 2, 2)
            self.df[numeric_cols[1]].hist(bins=30, ax=ax2, edgecolor='black', alpha=0.7, color='green')
            ax2.set_title(f'Distribution: {numeric_cols[1]}')
            ax2.grid(alpha=0.3)
            
        # Bottom left: Correlation heatmap
        if len(numeric_cols) >= 2:
            ax3 = plt.subplot(2, 2, 3)
            corr = self.df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax3)
            ax3.set_title('Correlations')
            
        # Bottom right: Missing data
        ax4 = plt.subplot(2, 2, 4)
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing = missing[missing > 0]
            missing.plot(kind='barh', ax=ax4, color='coral')
            ax4.set_title('Missing Values')
            ax4.grid(axis='x', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Missing Values ✓', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            ax4.axis('off')
            
        plt.suptitle('Data Analysis Summary Report', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = self.output_dir / 'summary_report.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)


def demo():
    '''Demo visualization generator'''
    import tempfile
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'sales': np.random.poisson(100, 200) + np.arange(200) * 0.5,
        'revenue': np.random.normal(5000, 1000, 200) + np.arange(200) * 10,
        'profit': np.random.normal(1000, 300, 200),
        'customers': np.random.randint(50, 150, 200)
    })
    
    # Add some missing data
    data.loc[np.random.choice(200, 20), 'revenue'] = np.nan
    
    print('Visualization Generator Demo')
    print('=' * 50)
    
    viz = VisualizationGenerator(data)
    
    print('\nGenerating all visualizations...')
    plots = viz.generate_all()
    
    print(f'\n✅ Generated {len(plots)} visualizations:')
    for name, path in plots.items():
        print(f'  - {name}: {path}')
        
    print('\nCreating summary report...')
    report_path = viz.create_summary_report()
    print(f'  ✅ Summary report: {report_path}')


if __name__ == '__main__':
    demo()
