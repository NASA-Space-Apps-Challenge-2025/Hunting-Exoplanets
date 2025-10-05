import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import numpy as np
import pickle
import json
import os
import sys
from pathlib import Path

# Add models directory to path to import ydf_ensemble
MODELS_DIR = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(MODELS_DIR))

from ydf_ensemble import HonestYDFEnsemble


class ModelManager:
    """Manages model training, inference, and session storage"""

    # Required features from ydf_ensemble
    REQUIRED_FEATURES = [
        'koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_impact',
        'koi_model_snr', 'koi_max_mult_ev', 'koi_num_transits',
        'koi_steff', 'koi_srad', 'koi_kepmag', 'koi_insol', 'koi_teq'
    ]

    # Model storage paths
    BASE_DIR = Path(__file__).parent / "models"
    BASE_MODEL_DIR = BASE_DIR / "base_model"
    CONFIG_PATH = BASE_DIR / "config.json"

    def __init__(self):
        """Initialize model manager with base model and session storage"""
        self.base_model: Optional[HonestYDFEnsemble] = None
        self.base_model_metrics: Optional[Dict[str, Any]] = None
        self.base_model_metadata: Optional[Dict[str, Any]] = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.trained_models: Dict[str, HonestYDFEnsemble] = {}
        self.data_api_url: Optional[str] = None  # Will be set when needed

        # Load config
        self.optimal_hyperparameters = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load optimal hyperparameters from config file"""
        try:
            if self.CONFIG_PATH.exists():
                with open(self.CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    return config.get('optimal_hyperparameters', {})
        except Exception as e:
            print(f"Error loading config: {e}")

        # Return default hyperparameters for optimized single GBT model
        # These are the optimized parameters from Optuna tuning (Trial #133, Oct 2025)
        return {
            "gbt_num_trees": 1100,
            "gbt_max_depth": 6,
            "gbt_shrinkage": 0.007094192442381623,
            "gbt_subsample": 0.7314794732228673,
            "gbt_min_examples": 4,
            "gbt_early_stopping": 60,
            "random_seed": 42,
            "verbose": True
        }

    def set_data_api_url(self, url: str):
        """Set the URL for fetching training data"""
        self.data_api_url = url

    def save_base_model(self):
        """Save base model, metrics, and metadata to disk"""
        if not self.base_model:
            raise ValueError("No base model to save")

        # Create directory if it doesn't exist
        self.BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = self.BASE_MODEL_DIR / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.base_model, f)

        # Save metrics
        metrics_path = self.BASE_MODEL_DIR / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self._format_metrics(self.base_model_metrics or {}), f, indent=2)

        # Save metadata
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": self.REQUIRED_FEATURES,
            "hyperparameters": self.base_model.get_params() if self.base_model else {},
            "training_samples": len(self.base_model.feature_names) if self.base_model and self.base_model.feature_names else 0
        }
        metadata_path = self.BASE_MODEL_DIR / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Base model saved to {self.BASE_MODEL_DIR}")

    def load_base_model(self, data: pd.DataFrame = None, target: pd.Series = None):
        """
        Load base model from disk or train a new one

        Args:
            data: Training data for base model (if training new)
            target: Target values for base model (if training new)

        If data and target are provided, trains a new base model
        Otherwise, attempts to load from disk
        """
        model_path = self.BASE_MODEL_DIR / "model.pkl"
        metrics_path = self.BASE_MODEL_DIR / "metrics.json"
        metadata_path = self.BASE_MODEL_DIR / "metadata.json"

        # Try to load from disk first
        if model_path.exists() and data is None and target is None:
            try:
                with open(model_path, 'rb') as f:
                    self.base_model = pickle.load(f)

                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        self.base_model_metrics = json.load(f)

                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.base_model_metadata = json.load(f)

                print(f"Base model loaded from {model_path}")
                return
            except Exception as e:
                print(f"Error loading base model: {e}")
                raise

        # Train new base model if data provided
        if data is not None and target is not None:
            # Train new base model with optimal hyperparameters
            self.base_model = HonestYDFEnsemble(**self.optimal_hyperparameters, verbose=False)

            # Split for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                data, target, test_size=0.2, random_state=42, stratify=target
            )

            self.base_model.fit(X_train, y_train, X_val, y_val)

            # Store base model metrics
            self.base_model_metrics = self.base_model.metrics.get('validation', {})

            # Store metadata
            self.base_model_metadata = {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "features": self.REQUIRED_FEATURES,
                "training_samples": len(X_train),
                "validation_samples": len(X_val)
            }

            print("New base model trained successfully")
        else:
            raise ValueError("No saved model found and no training data provided")

    def validate_features(self, data: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Validate that DataFrame contains all required features

        Args:
            data: DataFrame to validate

        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing_features = [f for f in self.REQUIRED_FEATURES if f not in data.columns]
        return len(missing_features) == 0, missing_features

    def fetch_existing_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch existing training data from NASA Exoplanet Archive TAP API

        Returns:
            DataFrame with existing training data, or None if fetch fails
        """
        import requests
        from io import StringIO

        try:
            # TAP API query for Kepler Objects of Interest (KOI) cumulative table
            tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            query = """
            SELECT *
            FROM cumulative
            """

            params = {
                'query': query.strip(),
                'format': 'csv'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ExoplanetAnalysis/1.0)',
                'Accept': 'text/csv,application/csv'
            }

            response = requests.get(tap_url, params=params, headers=headers, timeout=90)

            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))

                # Print data info
                print(f"   ✓ Fetched {len(df)} rows and {len(df.columns)} columns", flush=True)
                print(f"   ✓ Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"   ✓ Columns: {list(df.columns)}", flush=True)

                return df
            else:
                print(f"Failed to fetch data from API. Status code: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error fetching data from API: {str(e)}")
            return None

    def train_model(self, session_id: str, data: pd.DataFrame, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a new model with provided data

        Args:
            session_id: Unique session identifier
            data: Training data (must include 'target' column or be passed separately)
            hyperparams: Hyperparameters for model training

        Returns:
            Dictionary containing metrics and comparison data
        """
        # Check minimum data requirements
        MIN_TRAINING_SAMPLES = 20
        if len(data) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient training data: {len(data)} rows provided, "
                f"but at least {MIN_TRAINING_SAMPLES} rows are required for reliable training."
            )
        
        # Validate features
        is_valid, missing = self.validate_features(data)
        if not is_valid:
            raise ValueError(f"Missing required features: {missing}")

        # Extract features and target
        # Assume target is in a column named 'koi_disposition' or 'target'
        if 'target' in data.columns:
            y = data['target']
            # Select features in the correct order
            X = data[self.REQUIRED_FEATURES].copy()
        elif 'koi_disposition' in data.columns:
            # Convert to binary: CONFIRMED + CANDIDATE -> 1, FALSE POSITIVE -> 0
            # This follows the binary classification strategy documented in models/README.md
            y = data['koi_disposition'].map({
                'CONFIRMED': 1,
                'CANDIDATE': 1,
                'FALSE POSITIVE': 0
            })
            # Select features in the correct order
            X = data[self.REQUIRED_FEATURES].copy()
        else:
            raise ValueError("Data must contain 'target' or 'koi_disposition' column")

        # Ensure columns are in the correct order (pandas will reorder automatically)
        # This is important for model consistency
        X = X[self.REQUIRED_FEATURES]

        # Map user-provided hyperparameters to YDF ensemble parameters
        # Only expose: learning_rate, max_depth, num_trees
        # Use optimal hyperparameters from config for all others
        ensemble_params = self.optimal_hyperparameters.copy()

        # Override only the user-provided hyperparameters
        if 'learning_rate' in hyperparams:
            ensemble_params['gbt_shrinkage'] = hyperparams['learning_rate']
        if 'max_depth' in hyperparams:
            ensemble_params['gbt_max_depth'] = hyperparams['max_depth']
        if 'num_trees' in hyperparams:
            ensemble_params['gbt_num_trees'] = hyperparams['num_trees']

        # Always suppress verbose output
        ensemble_params['verbose'] = False

        # Create and train new model
        new_model = HonestYDFEnsemble(**ensemble_params)

        # Split for validation
        from sklearn.model_selection import train_test_split
        
        # Check if we have enough samples for stratification
        # We need at least 2 samples per class for stratification
        min_class_count = y.value_counts().min()
        use_stratify = min_class_count >= 2 and len(y) >= 10
        
        if use_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            print(f"   ⚠️  Warning: Dataset too small for stratification (min class count: {min_class_count}), using random split", flush=True)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        new_model.fit(X_train, y_train, X_val, y_val)

        # Get new model metrics
        new_metrics = self._format_metrics(new_model.metrics.get('validation', {}))

        # Get base model metrics for comparison
        base_metrics = self._get_base_model_metrics()

        # Create comparison JSON
        comparison = {
            "base_model": base_metrics,
            "new_model": new_metrics,
            "improvements": {
                "accuracy_delta": new_metrics["accuracy"] - base_metrics["accuracy"],
                "precision_delta": new_metrics["precision"] - base_metrics["precision"],
                "recall_delta": new_metrics["recall"] - base_metrics["recall"],
                "f1_score_delta": new_metrics["f1_score"] - base_metrics["f1_score"]
            },
            "training_info": {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "features": self.REQUIRED_FEATURES,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

        # Store the trained model
        self.trained_models[session_id] = new_model

        return {
            "metrics": new_metrics,
            "comparison": comparison
        }

    def predict(self, data: pd.DataFrame, session_id: Optional[str] = None) -> List[Any]:
        """
        Perform prediction on input data

        Args:
            data: Input data for prediction
            session_id: Optional session ID to use retrained model

        Returns:
            List of predictions
        """
        # Validate features
        is_valid, missing = self.validate_features(data)
        if not is_valid:
            raise ValueError(f"Missing required features: {missing}")

        # Select model
        if session_id and session_id in self.trained_models:
            model = self.trained_models[session_id]
        elif self.base_model:
            model = self.base_model
        else:
            raise ValueError("No model available. Train a base model first.")

        # Make predictions - ensure columns are in correct order
        X = data[self.REQUIRED_FEATURES].copy()
        X = X[self.REQUIRED_FEATURES]  # Reorder columns to match training order

        inference_result = model.infer(X, return_proba=True, return_details=False)

        # Return predictions as list
        return inference_result['predictions'].tolist()

    def store_session(
        self,
        session_id: str,
        metrics: Dict[str, Any],
        comparison: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        timestamp: str
    ):
        """
        Store session data for later retrieval

        Args:
            session_id: Unique session identifier
            metrics: Model performance metrics
            comparison: Comparison between base and new model
            hyperparameters: Hyperparameters used for training
            timestamp: Session creation timestamp
        """
        self.sessions[session_id] = {
            "metrics": metrics,
            "comparison": comparison,
            "hyperparameters": hyperparameters,
            "timestamp": timestamp,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data by ID

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        return self.sessions.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists

        Args:
            session_id: Session identifier

        Returns:
            True if session exists, False otherwise
        """
        return session_id in self.sessions

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions

        Returns:
            List of session summaries
        """
        return [
            {
                "session_id": session_id,
                "timestamp": data["timestamp"],
                "accuracy": data["metrics"].get("accuracy", 0.0),
                "created_at": data["created_at"]
            }
            for session_id, data in self.sessions.items()
        ]

    def get_base_model_info(self) -> Dict[str, Any]:
        """
        Get base model information including metrics and training details

        Returns:
            Dictionary containing model type, metrics, and training info
        """
        if not self.base_model:
            # Return placeholder if no base model
            return {
                "model_type": "YDF Stacking Ensemble (Not Trained)",
                "metrics": self._get_base_model_metrics(),
                "hyperparameters": {
                    "gbt_shrinkage": 0.03,
                    "gbt_max_depth": 8,
                    "gbt_num_trees": 700,
                    "rf_num_trees": 800,
                    "et_num_trees": 300
                },
                "training_info": {
                    "training_samples": 0,
                    "features": self.REQUIRED_FEATURES,
                    "trained_at": "Not trained yet"
                }
            }

        # Get actual base model info
        params = self.base_model.get_params()
        metrics = self._format_metrics(self.base_model_metrics or {})

        return {
            "model_type": "YDF Stacking Ensemble",
            "metrics": metrics,
            "hyperparameters": {
                "gbt_shrinkage": params.get('gbt_shrinkage', 0.03),
                "gbt_max_depth": params.get('gbt_max_depth', 8),
                "gbt_num_trees": params.get('gbt_num_trees', 700),
                "rf_num_trees": params.get('rf_num_trees', 800),
                "et_num_trees": params.get('et_num_trees', 300)
            },
            "training_info": {
                "training_samples": len(self.base_model.feature_names) if self.base_model.feature_names else 0,
                "features": self.REQUIRED_FEATURES,
                "trained_at": datetime.now(timezone.utc).isoformat()
            }
        }

    def _get_base_model_metrics(self) -> Dict[str, float]:
        """
        Get base model metrics for comparison

        Returns:
            Dictionary of base model metrics
        """
        if self.base_model_metrics:
            return self._format_metrics(self.base_model_metrics)

        # Return placeholder metrics if no base model
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "roc_auc": 0.0
        }

    def _format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics from YDF ensemble to API format

        Args:
            metrics: Metrics from YDF ensemble

        Returns:
            Formatted metrics dictionary
        """
        # Handle confusion matrix
        cm = metrics.get('confusion_matrix', np.array([[0, 0], [0, 0]]))
        if isinstance(cm, np.ndarray):
            cm_list = cm.tolist()
        else:
            cm_list = cm

        # Handle F1 score - YDF uses 'f1' key, but saved metrics use 'f1_score'
        f1_value = metrics.get('f1_score', metrics.get('f1', 0.0))

        return {
            "accuracy": float(metrics.get('accuracy', 0.0)),
            "precision": float(metrics.get('precision', 0.0)),
            "recall": float(metrics.get('recall', 0.0)),
            "f1_score": float(f1_value),
            "confusion_matrix": cm_list,
            "roc_auc": float(metrics.get('roc_auc', 0.0))
        }
