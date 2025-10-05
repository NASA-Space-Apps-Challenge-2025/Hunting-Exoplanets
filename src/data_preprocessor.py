"""
Data Preprocessing Module
Implements Phase 1 preprocessing strategies based on research methodology
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

class ExoplanetDataPreprocessor:
    """
    Handles data cleaning, preprocessing, and feature engineering
    Following research-based methodologies for exoplanet classification
    """
    
    def __init__(self):
        self.cleaned_data = None
        self.target_column = None
        self.features_to_remove = []
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        
    def load_raw_data(self, data_path=None):
        """Load raw data for preprocessing"""
        if data_path is None:
            data_path = RAW_DATA_DIR / "nasa_exoplanet_archive.csv"
        
        try:
            self.raw_data = pd.read_csv(data_path)
            print(f"Loaded raw data: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def identify_columns_for_removal(self, missing_threshold=0.8, single_value_threshold=0.99):
        """
        Identify columns to remove based on research methodology
        - High missing value percentage
        - Single/near-single value columns
        - Non-informative columns
        """
        print("=== IDENTIFYING COLUMNS FOR REMOVAL ===")
        
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return []
        
        columns_to_remove = []
        
        # 1. High missing value columns
        missing_pct = self.raw_data.isnull().sum() / len(self.raw_data)
        high_missing_cols = missing_pct[missing_pct > missing_threshold].index.tolist()
        columns_to_remove.extend(high_missing_cols)
        print(f"Columns with >{missing_threshold*100}% missing values: {len(high_missing_cols)}")
        
        # 2. Single value or near-single value columns
        for col in self.raw_data.columns:
            if col not in columns_to_remove:
                value_counts = self.raw_data[col].value_counts(normalize=True)
                if len(value_counts) == 1 or value_counts.iloc[0] > single_value_threshold:
                    columns_to_remove.append(col)
        
        # 3. Non-informative columns (based on common patterns)
        non_informative_patterns = ['url', 'reference', 'bibcode', 'facility', 'telescope']
        for col in self.raw_data.columns:
            if any(pattern in col.lower() for pattern in non_informative_patterns):
                if col not in columns_to_remove:
                    columns_to_remove.append(col)
        
        self.features_to_remove = columns_to_remove
        print(f"Total columns identified for removal: {len(columns_to_remove)}")
        
        return columns_to_remove
    
    def create_target_variable(self, method='discovery_method'):
        """
        Create and encode target variable for classification
        Based on research methodology for problem framing
        """
        print("=== CREATING TARGET VARIABLE ===")
        
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Option 1: Based on discovery method (most common approach)
        if method == 'discovery_method' and 'discoverymethod' in self.raw_data.columns:
            self.target_column = 'discoverymethod'
            target_data = self.raw_data[self.target_column].copy()
            
            # Group similar methods for better class balance
            method_mapping = {
                'Transit': 'Transit',
                'Radial Velocity': 'Radial_Velocity', 
                'Microlensing': 'Microlensing',
                'Direct Imaging': 'Direct_Imaging',
                'Astrometry': 'Other',
                'Eclipse Timing Variations': 'Other',
                'Orbital Brightness Modulation': 'Other',
                'Pulsar Timing': 'Other',
                'Pulsation Timing Variations': 'Other'
            }
            
            # Apply mapping
            target_encoded = target_data.map(method_mapping)
            target_encoded = target_encoded.fillna('Other')
            
        # Option 2: Create binary classification (confirmed vs candidate)
        elif method == 'binary_confirmed':
            # Create binary target: 1 for confirmed exoplanets, 0 for candidates
            if 'pl_controv_flag' in self.raw_data.columns:
                target_encoded = (self.raw_data['pl_controv_flag'] == 0).astype(int)
                self.target_column = 'confirmed_planet'
            else:
                # Fallback: use any available confirmation indicator
                target_encoded = pd.Series([1] * len(self.raw_data), name='confirmed_planet')
                self.target_column = 'confirmed_planet'
        
        # Option 3: Size-based classification
        elif method == 'size_classification' and 'pl_rade' in self.raw_data.columns:
            radius = self.raw_data['pl_rade']
            
            # Classification based on planetary radius (Earth radii)
            def classify_by_size(radius):
                if pd.isna(radius):
                    return 'Unknown'
                elif radius < 1.25:
                    return 'Earth_like'
                elif radius < 2.0:
                    return 'Super_Earth'
                elif radius < 6.0:
                    return 'Mini_Neptune'
                elif radius < 14.3:
                    return 'Neptune_like'
                else:
                    return 'Jupiter_like'
            
            target_encoded = radius.apply(classify_by_size)
            self.target_column = 'size_class'
        
        else:
            print(f"Method '{method}' not available or column not found.")
            print("Available columns:", list(self.raw_data.columns))
            return None
        
        # Encode the target variable
        if target_encoded.dtype == 'object':
            le = LabelEncoder()
            target_encoded = le.fit_transform(target_encoded.fillna('Unknown'))
            self.encoders['target'] = le
            print(f"Target classes: {le.classes_}")
        
        print(f"Target variable created: {self.target_column}")
        print(f"Class distribution:\n{pd.Series(target_encoded).value_counts()}")
        
        return target_encoded
    
    def clean_and_preprocess_features(self, target):
        """
        Clean and preprocess feature variables
        Following research methodology for data cleaning
        """
        print("=== CLEANING AND PREPROCESSING FEATURES ===")
        
        # Remove identified columns
        cleaned_data = self.raw_data.drop(columns=self.features_to_remove, errors='ignore')
        print(f"Removed {len(self.features_to_remove)} columns")
        
        # Add target variable
        cleaned_data['target'] = target
        
        # Separate numerical and categorical features
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        categorical_cols = cleaned_data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        # Handle missing values in numerical columns
        if numerical_cols:
            # Use median imputation for numerical features (robust to outliers)
            imputer_num = SimpleImputer(strategy='median')
            cleaned_data[numerical_cols] = imputer_num.fit_transform(cleaned_data[numerical_cols])
            self.scalers['imputer_numerical'] = imputer_num
        
        # Handle missing values in categorical columns
        if categorical_cols:
            # Use mode imputation for categorical features
            imputer_cat = SimpleImputer(strategy='most_frequent')
            cleaned_data[categorical_cols] = imputer_cat.fit_transform(cleaned_data[categorical_cols])
            self.scalers['imputer_categorical'] = imputer_cat
        
        # Encode categorical variables
        for col in categorical_cols:
            if cleaned_data[col].nunique() < 50:  # Avoid high cardinality
                le = LabelEncoder()
                cleaned_data[col] = le.fit_transform(cleaned_data[col].astype(str))
                self.encoders[col] = le
            else:
                # Drop high cardinality categorical columns
                cleaned_data = cleaned_data.drop(columns=[col])
                print(f"Dropped high cardinality column: {col}")
        
        self.cleaned_data = cleaned_data
        print(f"Final cleaned dataset shape: {cleaned_data.shape}")
        
        return cleaned_data
    
    def apply_feature_scaling(self, method='standard'):
        """
        Apply feature scaling based on research recommendations
        StandardScaler is commonly recommended for ensemble methods
        """
        print(f"=== APPLYING FEATURE SCALING ({method.upper()}) ===")
        
        if self.cleaned_data is None:
            print("No cleaned data available. Please run preprocessing first.")
            return None
        
        # Separate features and target
        X = self.cleaned_data.drop(columns=['target'])
        y = self.cleaned_data['target']
        
        # Select scaler based on method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            print(f"Unknown scaling method: {method}")
            return None
        
        # Apply scaling
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Store scaler for later use
        self.scalers['feature_scaler'] = scaler
        
        # Combine scaled features with target
        self.processed_data = pd.concat([X_scaled_df, y], axis=1)
        
        print(f"Feature scaling completed using {method} method")
        print(f"Scaled dataset shape: {self.processed_data.shape}")
        
        return self.processed_data
    
    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets
        Following research methodology for model validation
        """
        print("=== SPLITTING DATA ===")
        
        if self.processed_data is None:
            print("No processed data available. Please run preprocessing and scaling first.")
            return None
        
        X = self.processed_data.drop(columns=['target'])
        y = self.processed_data['target']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        self.data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples") 
        print(f"Test set: {X_test.shape[0]} samples")
        
        return self.data_splits
    
    def save_processed_data(self):
        """Save processed data and preprocessing objects"""
        if self.processed_data is not None:
            # Save processed dataset
            processed_path = PROCESSED_DATA_DIR / "processed_exoplanet_data.csv"
            self.processed_data.to_csv(processed_path, index=False)
            print(f"Processed data saved to: {processed_path}")
            
            # Save preprocessing objects
            objects_path = PROCESSED_DATA_DIR / "preprocessing_objects.joblib"
            preprocessing_objects = {
                'scalers': self.scalers,
                'encoders': self.encoders,
                'features_to_remove': self.features_to_remove,
                'target_column': self.target_column
            }
            joblib.dump(preprocessing_objects, objects_path)
            print(f"Preprocessing objects saved to: {objects_path}")
            
            # Save data splits if available
            if hasattr(self, 'data_splits'):
                splits_path = PROCESSED_DATA_DIR / "data_splits.joblib"
                joblib.dump(self.data_splits, splits_path)
                print(f"Data splits saved to: {splits_path}")

def run_phase1_preprocessing(target_method='discovery_method'):
    """
    Execute Phase 1 preprocessing pipeline
    """
    print("Starting Phase 1: Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = ExoplanetDataPreprocessor()
    
    # Load raw data
    data = preprocessor.load_raw_data()
    
    if data is not None:
        # Identify columns for removal
        cols_to_remove = preprocessor.identify_columns_for_removal()
        
        # Create target variable
        target = preprocessor.create_target_variable(method=target_method)
        
        if target is not None:
            # Clean and preprocess features
            cleaned_data = preprocessor.clean_and_preprocess_features(target)
            
            # Apply feature scaling
            scaled_data = preprocessor.apply_feature_scaling(method=FEATURE_CONFIG['scaling_method'].lower())
            
            # Split data
            data_splits = preprocessor.split_data()
            
            # Save processed data
            preprocessor.save_processed_data()
            
            print("\nPhase 1 preprocessing completed successfully!")
            return preprocessor
        else:
            print("Failed to create target variable.")
            return None
    else:
        print("Failed to load data.")
        return None

if __name__ == "__main__":
    run_phase1_preprocessing()
