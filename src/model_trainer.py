"""
Model Training and Evaluation Module
Implements Phase 2 of the exoplanet detection project plan
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import joblib
import optuna
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost, LightGBM, CatBoost (may not be installed initially)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from config import *

class ExoplanetModelTrainer:
    """
    Handles model selection, training, and evaluation
    Following research-based methodologies for exoplanet classification
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.data_splits = None
        
    def load_processed_data(self):
        """Load preprocessed data and splits"""
        try:
            splits_path = PROCESSED_DATA_DIR / "data_splits.joblib"
            self.data_splits = joblib.load(splits_path)
            print("Loaded preprocessed data splits successfully")
            
            # Print data split information
            print(f"Training samples: {self.data_splits['X_train'].shape[0]}")
            print(f"Validation samples: {self.data_splits['X_val'].shape[0]}")
            print(f"Test samples: {self.data_splits['X_test'].shape[0]}")
            print(f"Features: {self.data_splits['X_train'].shape[1]}")
            
            return True
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return False
    
    def initialize_models(self):
        """
        Initialize models based on research findings
        Priority given to ensemble methods as per literature
        """
        print("=== INITIALIZING MODELS ===")
        
        # Base models with initial parameters
        base_models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG['random_state'],
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG['random_state'],
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG['random_state']
            ),
            'LogisticRegression': LogisticRegression(
                random_state=MODEL_CONFIG['random_state'],
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=MODEL_CONFIG['random_state'],
                probability=True
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=MODEL_CONFIG['random_state']
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'MLP': MLPClassifier(
                random_state=MODEL_CONFIG['random_state'],
                max_iter=500
            )
        }
        
        # Add advanced ensemble methods if available
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = XGBClassifier(
                random_state=MODEL_CONFIG['random_state'],
                eval_metric='logloss'
            )
            print("XGBoost available and added")
        
        if LIGHTGBM_AVAILABLE:
            base_models['LightGBM'] = LGBMClassifier(
                random_state=MODEL_CONFIG['random_state'],
                verbose=-1
            )
            print("LightGBM available and added")
        
        if CATBOOST_AVAILABLE:
            base_models['CatBoost'] = CatBoostClassifier(
                random_state=MODEL_CONFIG['random_state'],
                verbose=False
            )
            print("CatBoost available and added")
        
        self.models = base_models
        print(f"Initialized {len(self.models)} models")
        
        return self.models
    
    def cross_validate_models(self, cv_folds=5):
        """
        Perform cross-validation on all models
        Following research methodology for model validation
        """
        print("=== CROSS-VALIDATION EVALUATION ===")
        
        if self.data_splits is None:
            print("No data loaded. Please load processed data first.")
            return None
        
        # Combine train and validation for cross-validation
        X_cv = pd.concat([self.data_splits['X_train'], self.data_splits['X_val']])
        y_cv = pd.concat([self.data_splits['y_train'], self.data_splits['y_val']])
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=MODEL_CONFIG['random_state'])
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Perform cross-validation for multiple metrics
            scores = {}
            for metric in MODEL_CONFIG['scoring_metrics']:
                try:
                    score = cross_val_score(model, X_cv, y_cv, cv=cv, 
                                          scoring=metric, n_jobs=-1)
                    scores[metric] = {
                        'mean': score.mean(),
                        'std': score.std(),
                        'scores': score
                    }
                except Exception as e:
                    print(f"Warning: Could not compute {metric} for {name}: {e}")
                    scores[metric] = {'mean': 0, 'std': 0, 'scores': [0]*cv_folds}
            
            cv_results[name] = scores
        
        self.results['cross_validation'] = cv_results
        
        # Display results
        self.display_cv_results()
        
        return cv_results
    
    def display_cv_results(self):
        """Display cross-validation results in a formatted table"""
        if 'cross_validation' not in self.results:
            return
        
        print("\n=== CROSS-VALIDATION RESULTS ===")
        
        # Create results DataFrame
        results_data = []
        for model_name, scores in self.results['cross_validation'].items():
            row = {'Model': model_name}
            for metric, values in scores.items():
                row[f'{metric}_mean'] = f"{values['mean']:.4f}"
                row[f'{metric}_std'] = f"{values['std']:.4f}"
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        # Find best model for each metric
        print("\n=== BEST MODELS PER METRIC ===")
        for metric in MODEL_CONFIG['scoring_metrics']:
            metric_col = f'{metric}_mean'
            if metric_col in results_df.columns:
                best_idx = results_df[metric_col].astype(float).idxmax()
                best_model = results_df.loc[best_idx, 'Model']
                best_score = results_df.loc[best_idx, metric_col]
                print(f"{metric}: {best_model} ({best_score})")
    
    def train_final_models(self):
        """
        Train final models on training data and evaluate on validation set
        """
        print("=== TRAINING FINAL MODELS ===")
        
        if self.data_splits is None:
            print("No data loaded. Please load processed data first.")
            return None
        
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        X_val = self.data_splits['X_val']
        y_val = self.data_splits['y_val']
        
        trained_models = {}
        validation_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                trained_model = model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = trained_model.predict(X_val)
                y_prob = trained_model.predict_proba(X_val)[:, 1] if hasattr(trained_model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1': f1_score(y_val, y_pred, average='weighted')
                }
                
                if y_prob is not None and len(np.unique(y_val)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_val, y_prob)
                
                trained_models[name] = trained_model
                validation_results[name] = metrics
                
                print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        self.trained_models = trained_models
        self.results['validation'] = validation_results
        
        return trained_models, validation_results
    
    def select_best_model(self, metric='f1'):
        """
        Select the best model based on specified metric
        """
        if 'validation' not in self.results:
            print("No validation results available. Please train models first.")
            return None
        
        best_score = -1
        best_model_name = None
        
        for model_name, metrics in self.results['validation'].items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.trained_models[best_model_name]
            self.best_model_name = best_model_name
            print(f"Best model selected: {best_model_name} ({metric}: {best_score:.4f})")
            
            return self.best_model
        else:
            print(f"Could not find best model for metric: {metric}")
            return None
    
    def hyperparameter_tuning(self, model_name, param_grid, cv_folds=3):
        """
        Perform hyperparameter tuning for a specific model
        Using GridSearchCV as mentioned in research methodology
        """
        print(f"=== HYPERPARAMETER TUNING FOR {model_name.upper()} ===")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found in initialized models")
            return None
        
        if self.data_splits is None:
            print("No data loaded. Please load processed data first.")
            return None
        
        # Combine train and validation for hyperparameter tuning
        X_tune = pd.concat([self.data_splits['X_train'], self.data_splits['X_val']])
        y_tune = pd.concat([self.data_splits['y_train'], self.data_splits['y_val']])
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.models[model_name],
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                             random_state=MODEL_CONFIG['random_state']),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting grid search...")
        grid_search.fit(X_tune, y_tune)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_test_performance(self):
        """
        Final evaluation on test set
        Critical for scientific validation as mentioned in research
        """
        print("=== FINAL TEST SET EVALUATION ===")
        
        if self.best_model is None:
            print("No best model selected. Please select a model first.")
            return None
        
        X_test = self.data_splits['X_test']
        y_test = self.data_splits['y_test']
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # Calculate comprehensive metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        if y_prob is not None and len(np.unique(y_test)) == 2:
            test_metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        # Print detailed results
        print(f"Model: {self.best_model_name}")
        print("Test Set Performance:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.results['test'] = test_metrics
        
        return test_metrics
    
    def save_models_and_results(self):
        """Save trained models and results"""
        # Save best model
        if self.best_model is not None:
            model_path = MODELS_DIR / f"best_model_{self.best_model_name}.joblib"
            joblib.dump(self.best_model, model_path)
            print(f"Best model saved to: {model_path}")
        
        # Save all trained models
        if hasattr(self, 'trained_models'):
            all_models_path = MODELS_DIR / "all_trained_models.joblib"
            joblib.dump(self.trained_models, all_models_path)
            print(f"All trained models saved to: {all_models_path}")
        
        # Save results
        results_path = RESULTS_DIR / "model_evaluation_results.joblib"
        joblib.dump(self.results, results_path)
        print(f"Results saved to: {results_path}")

def run_phase2_model_training():
    """
    Execute Phase 2: Model Selection and Training Strategy
    """
    print("Starting Phase 2: Model Selection and Training Strategy")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ExoplanetModelTrainer()
    
    # Load processed data
    if not trainer.load_processed_data():
        print("Failed to load processed data. Please complete Phase 1 first.")
        return None
    
    # Initialize models
    models = trainer.initialize_models()
    
    # Cross-validate models
    cv_results = trainer.cross_validate_models(cv_folds=MODEL_CONFIG['cv_folds'])
    
    # Train final models
    trained_models, val_results = trainer.train_final_models()
    
    # Select best model
    best_model = trainer.select_best_model(metric='f1')
    
    # Evaluate on test set
    test_results = trainer.evaluate_test_performance()
    
    # Save models and results
    trainer.save_models_and_results()
    
    print("\nPhase 2 model training completed successfully!")
    return trainer

if __name__ == "__main__":
    run_phase2_model_training()
