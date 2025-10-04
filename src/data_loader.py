"""
Data Loading and Initial Exploration Module
Implements Phase 1 of the exoplanet detection project plan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import warnings
warnings.filterwarnings('ignore')

from config import *

class ExoplanetDataLoader:
    """
    Handles loading and initial exploration of exoplanet datasets
    Following methodologies from research papers for data selection
    """
    
    def __init__(self):
        self.raw_data = None
        self.data_info = {}
        
    def load_nasa_exoplanet_archive(self, save_local=True):
        """
        Load data from NASA Exoplanet Archive
        Primary dataset selection based on research findings
        """
        print("Loading NASA Exoplanet Archive data...")
        
        # Download the dataset
        try:
            # Using a simplified query for confirmed exoplanets
            query_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars+where+default_flag=1&format=csv"
            
            self.raw_data = pd.read_csv(query_url)
            
            if save_local:
                save_path = RAW_DATA_DIR / "nasa_exoplanet_archive.csv"
                self.raw_data.to_csv(save_path, index=False)
                print(f"Data saved to: {save_path}")
                
            print(f"Successfully loaded {len(self.raw_data)} records")
            return self.raw_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Attempting to load from local backup...")
            return self.load_local_backup()
    
    def load_local_backup(self):
        """Load from local file if download fails"""
        try:
            backup_path = RAW_DATA_DIR / "nasa_exoplanet_archive.csv"
            if backup_path.exists():
                self.raw_data = pd.read_csv(backup_path)
                print(f"Loaded backup data: {len(self.raw_data)} records")
                return self.raw_data
            else:
                print("No local backup found. Please check internet connection.")
                return None
        except Exception as e:
            print(f"Error loading backup: {e}")
            return None
    
    def initial_data_inspection(self):
        """
        Perform initial data inspection as outlined in the research methodology
        Documents dataset characteristics for Phase 1 requirements
        """
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        print("=== INITIAL DATA INSPECTION ===")
        
        # Basic dataset information
        print(f"Dataset shape: {self.raw_data.shape}")
        print(f"Number of columns: {self.raw_data.shape[1]}")
        print(f"Number of samples: {self.raw_data.shape[0]}")
        
        # Data types analysis
        print("\n=== DATA TYPES ===")
        dtype_summary = self.raw_data.dtypes.value_counts()
        print(dtype_summary)
        
        # Missing values analysis
        print("\n=== MISSING VALUES ANALYSIS ===")
        missing_values = self.raw_data.isnull().sum()
        missing_percentage = (missing_values / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("Top 20 columns with missing values:")
        print(missing_df.head(20))
        
        # Store information for later use
        self.data_info = {
            'shape': self.raw_data.shape,
            'dtypes': dtype_summary.to_dict(),
            'missing_values': missing_df,
            'column_names': list(self.raw_data.columns)
        }
        
        return self.data_info
    
    def explore_target_variable(self, target_col=None):
        """
        Explore potential target variables for classification
        Based on research methodology for problem framing
        """
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        print("=== TARGET VARIABLE EXPLORATION ===")
        
        # Look for discovery method or confirmation status columns
        potential_targets = [col for col in self.raw_data.columns 
                           if any(keyword in col.lower() for keyword in 
                                ['method', 'status', 'flag', 'confirmed', 'candidate'])]
        
        print("Potential target variables found:")
        for col in potential_targets:
            print(f"- {col}")
            if self.raw_data[col].dtype == 'object':
                print(f"  Unique values: {self.raw_data[col].unique()}")
            print(f"  Value counts:\n{self.raw_data[col].value_counts()}")
            print()
        
        return potential_targets
    
    def generate_data_quality_report(self):
        """
        Generate comprehensive data quality report
        Following research methodology for data assessment
        """
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        print("=== DATA QUALITY REPORT ===")
        
        # Duplicate analysis
        duplicates = self.raw_data.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Numerical columns statistics
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        print(f"\nNumerical columns: {len(numerical_cols)}")
        
        # Categorical columns analysis
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {len(categorical_cols)}")
        
        # High cardinality categorical variables
        print("\n=== HIGH CARDINALITY CATEGORICALS ===")
        for col in categorical_cols:
            unique_count = self.raw_data[col].nunique()
            if unique_count > 50:
                print(f"{col}: {unique_count} unique values")
        
        # Columns with single unique value (potential for removal)
        print("\n=== SINGLE VALUE COLUMNS (CANDIDATES FOR REMOVAL) ===")
        single_value_cols = []
        for col in self.raw_data.columns:
            if self.raw_data[col].nunique() <= 1:
                single_value_cols.append(col)
                print(f"- {col}")
        
        self.data_info['single_value_columns'] = single_value_cols
        self.data_info['numerical_columns'] = list(numerical_cols)
        self.data_info['categorical_columns'] = list(categorical_cols)
        
        return self.data_info
    
    def visualize_data_overview(self):
        """
        Create visualizations for data overview
        Supporting Phase 1 exploration requirements
        """
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Missing values heatmap
        missing_data = self.raw_data.isnull().sum().sort_values(ascending=False).head(20)
        sns.barplot(x=missing_data.values, y=missing_data.index, ax=axes[0,0])
        axes[0,0].set_title('Top 20 Columns with Missing Values')
        axes[0,0].set_xlabel('Number of Missing Values')
        
        # Data types distribution
        dtype_counts = self.raw_data.dtypes.value_counts()
        axes[0,1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Distribution of Data Types')
        
        # Numerical columns distribution sample
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns[:5]
        if len(numerical_cols) > 0:
            self.raw_data[numerical_cols].hist(ax=axes[1,0], bins=20)
            axes[1,0].set_title('Sample Numerical Distributions')
        
        # Missing values percentage
        missing_pct = (self.raw_data.isnull().sum() / len(self.raw_data) * 100).sort_values(ascending=False).head(15)
        sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=axes[1,1])
        axes[1,1].set_title('Missing Values Percentage (Top 15)')
        axes[1,1].set_xlabel('Percentage Missing')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'data_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Data overview visualization saved to: {RESULTS_DIR / 'data_overview.png'}")

# Example usage function
def run_phase1_data_exploration():
    """
    Execute Phase 1: Data Selection, Exploration, and Preprocessing
    """
    print("Starting Phase 1: Data Selection, Exploration, and Preprocessing")
    print("=" * 60)
    
    # Initialize data loader
    loader = ExoplanetDataLoader()
    
    # Load data
    data = loader.load_nasa_exoplanet_archive()
    
    if data is not None:
        # Perform initial inspection
        data_info = loader.initial_data_inspection()
        
        # Explore target variables
        targets = loader.explore_target_variable()
        
        # Generate quality report
        quality_report = loader.generate_data_quality_report()
        
        # Create visualizations
        loader.visualize_data_overview()
        
        print("\nPhase 1 exploration completed successfully!")
        return loader, data_info
    else:
        print("Failed to load data. Please check your internet connection or data sources.")
        return None, None

if __name__ == "__main__":
    run_phase1_data_exploration()
