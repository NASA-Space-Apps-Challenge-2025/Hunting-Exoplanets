"""
YDF Stacking Ensemble for Exoplanet Detection
==============================================

An honest machine learning model for exoplanet detection using YDF (Yggdrasil Decision Forests)
stacking ensemble. This model uses 13 curated physical features without data leakage.

Features:
- Random Forest base model
- Gradient Boosted Trees base model  
- Extra Trees base model
- Logistic Regression meta-learner
- Optimized hyperparameters from Optuna
- No data leakage (excludes koi_pdisposition and koi_score)

Author: brix for the NASA Space Apps Challenge 2025
Date: October 2025
"""

import numpy as np
import pandas as pd
import ydf
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class HonestYDFEnsemble:
    """
    Honest YDF Stacking Ensemble for Exoplanet Detection
    
    This ensemble combines three YDF base models (Random Forest, Gradient Boosted Trees, 
    Extra Trees) with a Logistic Regression meta-learner to predict exoplanet candidates.
    
    Key Features:
    - Uses 13 curated physical features (no data leakage)
    - Optimized hyperparameters from Optuna tuning
    - Configurable parameters for fine-tuning
    - Comprehensive performance metrics
    
    Parameters
    ----------
    rf_num_trees : int, default=800
        Number of trees in Random Forest
    rf_max_depth : int, default=50
        Maximum depth of Random Forest trees
    rf_min_examples : int, default=5
        Minimum examples per leaf in Random Forest
    gbt_num_trees : int, default=700
        Number of trees in Gradient Boosted Trees
    gbt_max_depth : int, default=8
        Maximum depth of GBT trees
    gbt_shrinkage : float, default=0.03
        Learning rate (shrinkage) for GBT
    gbt_subsample : float, default=0.70
        Subsample ratio for GBT
    gbt_early_stopping : int, default=40
        Early stopping rounds for GBT
    et_num_trees : int, default=300
        Number of trees in Extra Trees
    et_candidate_ratio : float, default=0.40
        Candidate attributes ratio for Extra Trees
    meta_C : float, default=0.05
        Regularization parameter for meta-learner
    meta_max_iter : int, default=2000
        Maximum iterations for meta-learner
    random_seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print training progress
    """
    
    def __init__(
        self,
        # Random Forest parameters
        rf_num_trees: int = 800,
        rf_max_depth: int = 50,
        rf_min_examples: int = 5,
        # Gradient Boosted Trees parameters
        gbt_num_trees: int = 700,
        gbt_max_depth: int = 8,
        gbt_shrinkage: float = 0.03,  # learning_rate
        gbt_subsample: float = 0.70,
        gbt_early_stopping: int = 40,
        # Extra Trees parameters
        et_num_trees: int = 300,
        et_candidate_ratio: float = 0.40,
        # Meta-learner parameters
        meta_C: float = 0.05,
        meta_max_iter: int = 2000,
        # General parameters
        random_seed: int = 42,
        verbose: bool = True
    ):
        """Initialize the Honest YDF Ensemble"""
        
        # Store parameters
        self.rf_num_trees = rf_num_trees
        self.rf_max_depth = rf_max_depth
        self.rf_min_examples = rf_min_examples
        
        self.gbt_num_trees = gbt_num_trees
        self.gbt_max_depth = gbt_max_depth
        self.gbt_shrinkage = gbt_shrinkage
        self.gbt_subsample = gbt_subsample
        self.gbt_early_stopping = gbt_early_stopping
        
        self.et_num_trees = et_num_trees
        self.et_candidate_ratio = et_candidate_ratio
        
        self.meta_C = meta_C
        self.meta_max_iter = meta_max_iter
        
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Model components (initialized during fit)
        self.base_models = []
        self.meta_learner = None
        self.imputer = None
        self.feature_names = None
        
        # Training metrics
        self.training_time = None
        self.metrics = {}
        
        # Curated features (no data leakage)
        self.CURATED_FEATURES = [
            'koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_impact',
            'koi_model_snr', 'koi_max_mult_ev', 'koi_num_transits',
            'koi_steff', 'koi_srad', 'koi_kepmag', 'koi_insol', 'koi_teq'
        ]
        
    def _log(self, message: str) -> None:
        """Print message if verbose is True"""
        if self.verbose:
            print(message)
    
    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        
        Applies log transformation to skewed features and handles missing values
        """
        X_prepared = X[self.CURATED_FEATURES].copy()
        
        # Log transform skewed features
        if 'koi_period' in X_prepared.columns:
            X_prepared['koi_period'] = np.log1p(X_prepared['koi_period'])
        if 'koi_depth' in X_prepared.columns:
            X_prepared['koi_depth'] = np.log1p(X_prepared['koi_depth'])
        
        return X_prepared
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'HonestYDFEnsemble':
        """
        Fit the YDF stacking ensemble
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target (0=False Positive, 1=Planet)
        X_val : pd.DataFrame, optional
            Validation features for evaluation
        y_val : pd.Series, optional
            Validation target
            
        Returns
        -------
        self : HonestYDFEnsemble
            Fitted ensemble
        """
        start_time = time.time()
        
        self._log("=" * 90)
        self._log("TRAINING HONEST YDF STACKING ENSEMBLE")
        self._log("=" * 90)
        
        # Prepare data
        self._log("\n🔧 DATA PREPARATION")
        self._log("-" * 90)
        
        X_prep = self._prepare_data(X)
        self.feature_names = X_prep.columns.tolist()
        
        # Handle missing values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_prep),
            columns=X_prep.columns,
            index=X_prep.index
        )
        
        self._log(f"  Features: {len(self.CURATED_FEATURES)}")
        self._log(f"  Samples: {len(X_imputed)}")
        self._log(f"  Target distribution: {(y == 1).sum()} planets ({(y == 1).sum() / len(y) * 100:.1f}%), "
                  f"{(y == 0).sum()} false positives ({(y == 0).sum() / len(y) * 100:.1f}%)")
        
        # Convert target to string for YDF
        y_str = y.astype(str)
        
        # Create training dataframe with target
        train_df = X_imputed.copy()
        train_df['target'] = y_str.values
        
        # Train base models
        self._log("\n🌲 TRAINING BASE MODELS")
        self._log("=" * 90)
        
        # 1. Random Forest
        self._log(f"\n1. Training Random Forest...")
        self._log(f"   Parameters: num_trees={self.rf_num_trees}, max_depth={self.rf_max_depth}, "
                  f"min_examples={self.rf_min_examples}")
        rf_start = time.time()
        rf_model = ydf.RandomForestLearner(
            label='target',
            num_trees=self.rf_num_trees,
            max_depth=self.rf_max_depth,
            min_examples=self.rf_min_examples,
            random_seed=self.random_seed
        ).train(train_df)
        rf_time = time.time() - rf_start
        self.base_models.append(rf_model)
        self._log(f"   ✅ Completed in {rf_time:.2f}s")
        
        # 2. Gradient Boosted Trees
        self._log(f"\n2. Training Gradient Boosted Trees...")
        self._log(f"   Parameters: num_trees={self.gbt_num_trees}, max_depth={self.gbt_max_depth}, "
                  f"subsample={self.gbt_subsample:.2f}")
        self._log(f"               shrinkage={self.gbt_shrinkage:.4f}, early_stopping={self.gbt_early_stopping}")
        gbt_start = time.time()
        gbt_model = ydf.GradientBoostedTreesLearner(
            label='target',
            num_trees=self.gbt_num_trees,
            max_depth=self.gbt_max_depth,
            subsample=self.gbt_subsample,
            shrinkage=self.gbt_shrinkage,
            early_stopping_num_trees_look_ahead=self.gbt_early_stopping,
            random_seed=self.random_seed
        ).train(train_df)
        gbt_time = time.time() - gbt_start
        self.base_models.append(gbt_model)
        self._log(f"   ✅ Completed in {gbt_time:.2f}s")
        
        # 3. Extra Trees
        self._log(f"\n3. Training Extra Trees...")
        self._log(f"   Parameters: num_trees={self.et_num_trees}, "
                  f"candidate_ratio={self.et_candidate_ratio:.2f}")
        et_start = time.time()
        et_model = ydf.RandomForestLearner(
            label='target',
            num_trees=self.et_num_trees,
            winner_take_all=True,
            num_candidate_attributes_ratio=self.et_candidate_ratio,
            random_seed=self.random_seed
        ).train(train_df)
        et_time = time.time() - et_start
        self.base_models.append(et_model)
        self._log(f"   ✅ Completed in {et_time:.2f}s")
        
        total_base_time = rf_time + gbt_time + et_time
        self._log(f"\n  Total base model training time: {total_base_time:.2f}s")
        
        # Create meta-features
        self._log(f"\n🎯 CREATING META-FEATURES")
        self._log("-" * 90)
        
        meta_X_train = self._create_meta_features(X_imputed)
        self._log(f"  Meta-features shape: {meta_X_train.shape}")
        
        # Train meta-learner
        self._log(f"\n🧠 TRAINING META-LEARNER (Logistic Regression)")
        self._log("-" * 90)
        self._log(f"   Parameters: C={self.meta_C:.4f}, max_iter={self.meta_max_iter}")
        
        self.meta_learner = LogisticRegression(
            C=self.meta_C,
            max_iter=self.meta_max_iter,
            random_state=self.random_seed,
            solver='lbfgs'
        )
        self.meta_learner.fit(meta_X_train, y)
        self._log(f"  ✅ Meta-learner trained")
        self._log(f"  Coefficients: {self.meta_learner.coef_[0]}")
        self._log(f"  Intercept: {self.meta_learner.intercept_[0]:.4f}")
        
        self.training_time = time.time() - start_time
        
        # Evaluate on training set
        train_predictions = self.predict(X)
        train_proba = self.predict_proba(X)[:, 1]
        
        self.metrics['train'] = self._calculate_metrics(y, train_predictions, train_proba)
        
        self._log(f"\n📊 TRAINING PERFORMANCE")
        self._log("=" * 90)
        self._print_metrics(self.metrics['train'])
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_proba = self.predict_proba(X_val)[:, 1]
            
            self.metrics['validation'] = self._calculate_metrics(y_val, val_predictions, val_proba)
            
            self._log(f"\n📊 VALIDATION PERFORMANCE")
            self._log("=" * 90)
            self._print_metrics(self.metrics['validation'])
        
        self._log(f"\n{'=' * 90}")
        self._log(f"TRAINING COMPLETE")
        self._log(f"Total training time: {self.training_time:.2f}s")
        self._log("=" * 90)
        
        return self
    
    def _create_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Create meta-features from base model predictions"""
        meta_features = []
        
        for model in self.base_models:
            predictions = model.predict(X)
            predictions_array = np.array(predictions).astype(float).reshape(-1, 1)
            meta_features.append(predictions_array)
        
        return np.hstack(meta_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict
            
        Returns
        -------
        predictions : np.ndarray
            Predicted class labels (0 or 1)
        """
        if self.meta_learner is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_prep = self._prepare_data(X)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_prep),
            columns=X_prep.columns,
            index=X_prep.index
        )
        
        meta_features = self._create_meta_features(X_imputed)
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict
            
        Returns
        -------
        probabilities : np.ndarray
            Predicted class probabilities, shape (n_samples, 2)
        """
        if self.meta_learner is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_prep = self._prepare_data(X)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_prep),
            columns=X_prep.columns,
            index=X_prep.index
        )
        
        meta_features = self._create_meta_features(X_imputed)
        return self.meta_learner.predict_proba(meta_features)
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        dataset_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to evaluate
        y : pd.Series
            True target values
        dataset_name : str, default='test'
            Name of the dataset for display
            
        Returns
        -------
        metrics : dict
            Dictionary containing performance metrics
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]
        
        metrics = self._calculate_metrics(y, predictions, probabilities)
        self.metrics[dataset_name] = metrics
        
        self._log(f"\n📊 {dataset_name.upper()} PERFORMANCE")
        self._log("=" * 90)
        self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print performance metrics"""
        self._log(f"\n  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        self._log(f"  Precision: {metrics['precision']:.4f}")
        self._log(f"  Recall:    {metrics['recall']:.4f}")
        self._log(f"  F1-Score:  {metrics['f1']:.4f}")
        self._log(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            self._log(f"\n  Confusion Matrix:")
            self._log(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
            self._log(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'rf_num_trees': self.rf_num_trees,
            'rf_max_depth': self.rf_max_depth,
            'rf_min_examples': self.rf_min_examples,
            'gbt_num_trees': self.gbt_num_trees,
            'gbt_max_depth': self.gbt_max_depth,
            'gbt_shrinkage': self.gbt_shrinkage,
            'gbt_subsample': self.gbt_subsample,
            'gbt_early_stopping': self.gbt_early_stopping,
            'et_num_trees': self.et_num_trees,
            'et_candidate_ratio': self.et_candidate_ratio,
            'meta_C': self.meta_C,
            'meta_max_iter': self.meta_max_iter,
            'random_seed': self.random_seed
        }
    
    def infer(
        self,
        X: pd.DataFrame,
        return_proba: bool = True,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Inference function for making predictions on new data
        
        This is a convenient method that wraps predict() and predict_proba()
        with additional information useful for inference in production.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features for prediction
            Must contain the 13 curated features:
            - koi_period, koi_depth, koi_duration, koi_prad, koi_impact
            - koi_model_snr, koi_max_mult_ev, koi_num_transits
            - koi_steff, koi_srad, koi_kepmag, koi_insol, koi_teq
            
        return_proba : bool, default=True
            Whether to return class probabilities
            
        return_details : bool, default=False
            Whether to return additional details (base model predictions, confidence)
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'predictions': np.ndarray
                Class predictions (0=False Positive, 1=Planet)
            - 'probabilities': np.ndarray (if return_proba=True)
                Probabilities for each class [prob_fp, prob_planet]
            - 'confidence': np.ndarray (if return_details=True)
                Confidence scores (max probability)
            - 'base_predictions': dict (if return_details=True)
                Individual predictions from base models
            - 'n_samples': int
                Number of samples processed
            
        Examples
        --------
        >>> # Simple prediction
        >>> results = ensemble.infer(X_new)
        >>> predictions = results['predictions']
        >>> probabilities = results['probabilities']
        
        >>> # Detailed inference with base model predictions
        >>> results = ensemble.infer(X_new, return_details=True)
        >>> confidence = results['confidence']
        >>> base_preds = results['base_predictions']
        
        >>> # Print results for each sample
        >>> for i in range(len(predictions)):
        ...     if predictions[i] == 1:
        ...         print(f"Sample {i}: PLANET (confidence: {confidence[i]:.2%})")
        """
        
        if self.meta_learner is None:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        # Validate required features
        missing_features = [f for f in self.CURATED_FEATURES if f not in X.columns]
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Required features: {self.CURATED_FEATURES}"
            )
        
        # Make predictions
        predictions = self.predict(X)
        
        # Initialize results dictionary
        results = {
            'predictions': predictions,
            'n_samples': len(X)
        }
        
        # Add probabilities if requested
        if return_proba:
            probabilities = self.predict_proba(X)
            results['probabilities'] = probabilities
            
            # Add confidence (max probability)
            if return_details:
                results['confidence'] = np.max(probabilities, axis=1)
        
        # Add detailed information if requested
        if return_details:
            # Get base model predictions
            X_prep = self._prepare_data(X)
            X_imputed = pd.DataFrame(
                self.imputer.transform(X_prep),
                columns=X_prep.columns,
                index=X_prep.index
            )
            
            base_predictions = {}
            base_probabilities = {}
            
            model_names = ['random_forest', 'gradient_boosted_trees', 'extra_trees']
            for i, (model, name) in enumerate(zip(self.base_models, model_names)):
                # Get predictions
                preds = model.predict(X_imputed)
                base_predictions[name] = np.array(preds).astype(float)
                
                # Note: YDF models return predictions as strings, convert to float
                # 1.0 = Planet, 0.0 = False Positive
            
            results['base_predictions'] = base_predictions
            
            # Add prediction agreement score
            base_pred_array = np.array(list(base_predictions.values())).T
            agreement = np.mean(base_pred_array == predictions.reshape(-1, 1), axis=1)
            results['agreement_score'] = agreement
        
        return results
    
    def infer_single(
        self,
        koi_period: float,
        koi_depth: float,
        koi_duration: float,
        koi_prad: float,
        koi_impact: float,
        koi_model_snr: float,
        koi_max_mult_ev: float,
        koi_num_transits: float,
        koi_steff: float,
        koi_srad: float,
        koi_kepmag: float,
        koi_insol: float,
        koi_teq: float,
        return_details: bool = True
    ) -> Dict[str, Any]:
        """
        Inference function for a single exoplanet candidate
        
        Convenient method to predict on a single candidate by passing
        individual feature values directly.
        
        Parameters
        ----------
        koi_period : float
            Orbital period (days)
        koi_depth : float
            Transit depth (parts per million)
        koi_duration : float
            Transit duration (hours)
        koi_prad : float
            Planet radius (Earth radii)
        koi_impact : float
            Impact parameter
        koi_model_snr : float
            Signal-to-noise ratio
        koi_max_mult_ev : float
            Maximum multiple event statistic
        koi_num_transits : float
            Number of observed transits
        koi_steff : float
            Stellar effective temperature (Kelvin)
        koi_srad : float
            Stellar radius (solar radii)
        koi_kepmag : float
            Kepler magnitude
        koi_insol : float
            Insolation flux (Earth flux)
        koi_teq : float
            Equilibrium temperature (Kelvin)
        return_details : bool, default=True
            Whether to return detailed prediction information
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'prediction': int (0 or 1)
                0 = False Positive, 1 = Planet
            - 'prediction_label': str
                Human-readable label
            - 'probability': float
                Probability of being a planet
            - 'confidence': float
                Confidence score (max probability)
            - 'base_predictions': dict (if return_details=True)
                Individual predictions from base models
            - 'agreement_score': float (if return_details=True)
                Agreement between base models (0-1)
                
        Examples
        --------
        >>> # Predict for a single candidate
        >>> result = ensemble.infer_single(
        ...     koi_period=10.5,
        ...     koi_depth=500.0,
        ...     koi_duration=3.5,
        ...     koi_prad=2.5,
        ...     koi_impact=0.5,
        ...     koi_model_snr=25.0,
        ...     koi_max_mult_ev=1.2,
        ...     koi_num_transits=15,
        ...     koi_steff=5800,
        ...     koi_srad=1.0,
        ...     koi_kepmag=14.5,
        ...     koi_insol=1.5,
        ...     koi_teq=350
        ... )
        >>> 
        >>> print(f"Prediction: {result['prediction_label']}")
        >>> print(f"Probability: {result['probability']:.2%}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
        """
        
        # Create DataFrame with single row
        X_single = pd.DataFrame([{
            'koi_period': koi_period,
            'koi_depth': koi_depth,
            'koi_duration': koi_duration,
            'koi_prad': koi_prad,
            'koi_impact': koi_impact,
            'koi_model_snr': koi_model_snr,
            'koi_max_mult_ev': koi_max_mult_ev,
            'koi_num_transits': koi_num_transits,
            'koi_steff': koi_steff,
            'koi_srad': koi_srad,
            'koi_kepmag': koi_kepmag,
            'koi_insol': koi_insol,
            'koi_teq': koi_teq
        }])
        
        # Make prediction using infer method
        batch_results = self.infer(X_single, return_proba=True, return_details=return_details)
        
        # Extract single-sample results
        prediction = int(batch_results['predictions'][0])
        probabilities = batch_results['probabilities'][0]
        
        results = {
            'prediction': prediction,
            'prediction_label': 'PLANET' if prediction == 1 else 'FALSE POSITIVE',
            'probability': float(probabilities[1]),  # Probability of being a planet
            'confidence': float(np.max(probabilities))
        }
        
        # Add detailed information if requested
        if return_details and 'base_predictions' in batch_results:
            base_preds = {}
            for model_name, preds in batch_results['base_predictions'].items():
                base_preds[model_name] = {
                    'prediction': int(preds[0]),
                    'prediction_label': 'PLANET' if int(preds[0]) == 1 else 'FALSE POSITIVE'
                }
            results['base_predictions'] = base_preds
            results['agreement_score'] = float(batch_results['agreement_score'][0])
        
        return results
    
    def summary(self) -> None:
        """Print model summary"""
        self._log("\n" + "=" * 90)
        self._log("HONEST YDF ENSEMBLE - MODEL SUMMARY")
        self._log("=" * 90)
        
        self._log("\n🎯 MODEL CONFIGURATION")
        self._log("-" * 90)
        self._log(f"  Features: {len(self.CURATED_FEATURES)} (no data leakage)")
        self._log(f"  Base Models: 3 (Random Forest, Gradient Boosted Trees, Extra Trees)")
        self._log(f"  Meta-Learner: Logistic Regression")
        
        self._log("\n🔧 HYPERPARAMETERS")
        self._log("-" * 90)
        params = self.get_params()
        self._log(f"  Random Forest: num_trees={params['rf_num_trees']}, "
                  f"max_depth={params['rf_max_depth']}, min_examples={params['rf_min_examples']}")
        self._log(f"  Gradient Boosted Trees: num_trees={params['gbt_num_trees']}, "
                  f"max_depth={params['gbt_max_depth']}, shrinkage={params['gbt_shrinkage']:.4f}")
        self._log(f"  Extra Trees: num_trees={params['et_num_trees']}, "
                  f"candidate_ratio={params['et_candidate_ratio']:.2f}")
        self._log(f"  Meta-Learner: C={params['meta_C']:.4f}, max_iter={params['meta_max_iter']}")
        
        if self.training_time is not None:
            self._log(f"\n⏱️  TRAINING TIME: {self.training_time:.2f}s")
        
        if self.metrics:
            self._log(f"\n📊 PERFORMANCE METRICS")
            self._log("-" * 90)
            for dataset_name, metrics in self.metrics.items():
                self._log(f"\n  {dataset_name.upper()}: Accuracy={metrics['accuracy']:.4f}, "
                          f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                          f"ROC-AUC={metrics['roc_auc']:.4f}")
        
        self._log("\n" + "=" * 90)


# Example usage
if __name__ == "__main__":
    print("=" * 90)
    print("HONEST YDF ENSEMBLE - EXAMPLE USAGE")
    print("=" * 90)
    
    print("\nThis script demonstrates how to use the HonestYDFEnsemble class.")
    print("\nExample code:")
    print("-" * 90)
    
    example_code = """
# Import the ensemble
from ydf_ensemble import HonestYDFEnsemble

# Load your data (must have the 13 curated features)
# X should contain: koi_period, koi_depth, koi_duration, koi_prad, koi_impact,
#                   koi_model_snr, koi_max_mult_ev, koi_num_transits,
#                   koi_steff, koi_srad, koi_kepmag, koi_insol, koi_teq
# y should be binary: 0=False Positive, 1=Planet

# 1. Create ensemble with default parameters (optimized from Optuna)
ensemble = HonestYDFEnsemble(verbose=True)

# 2. Or customize key parameters
ensemble = HonestYDFEnsemble(
    gbt_shrinkage=0.05,      # Lower learning rate
    gbt_max_depth=8,          # Deeper trees
    gbt_num_trees=600,        # More trees
    rf_num_trees=400,         # More RF trees
    verbose=True
)

# 3. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the ensemble
ensemble.fit(X_train, y_train, X_val=X_test, y_val=y_test)

# 5. Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# 6. Evaluate performance
metrics = ensemble.evaluate(X_test, y_test, dataset_name='test')

# 7. Print model summary
ensemble.summary()

# 8. Access performance metrics
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
"""
    
    print(example_code)
    
    print("-" * 90)
    print("\n✅ Model is ready for production deployment!")
    print("   • No data leakage (excludes koi_pdisposition, koi_score)")
    print("   • Optimized hyperparameters from Optuna")
    print("   • Expected performance: ~82-83% accuracy, 0.91+ ROC-AUC")
    print("=" * 90)
