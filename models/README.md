# YDF Stacking Ensemble for Exoplanet Detection

This directory contains the production-ready implementation of the Honest YDF Stacking Ensemble for exoplanet detection.

## 📁 Files

### Core Implementation
- **`ydf_ensemble.py`** - Main ensemble class with configurable hyperparameters
- **`train_ensemble.py`** - Complete training pipeline script

## 🚀 Quick Start

### Installation

First, ensure you have the required dependencies:

```bash
pip install numpy pandas ydf scikit-learn requests
```

### Basic Usage

```python
from ydf_ensemble import HonestYDFEnsemble
from sklearn.model_selection import train_test_split

# Load your data (must contain the 13 curated features)
# X: DataFrame with features
# y: Series with binary target (0=False Positive, 1=Planet)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create ensemble with default optimized parameters
ensemble = HonestYDFEnsemble(verbose=True)

# Train
ensemble.fit(X_train, y_train, X_val=X_test, y_val=y_test)

# Predict
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# Evaluate
metrics = ensemble.evaluate(X_test, y_test)

# Print summary
ensemble.summary()
```

### Run Complete Training Pipeline

```bash
python train_ensemble.py
```

This script will:
1. Load exoplanet data from NASA API
2. Prepare features and target
3. Split data into train/validation/test sets
4. Train the YDF ensemble
5. Display comprehensive metrics

## 🎛️ Configurable Parameters

The `HonestYDFEnsemble` class allows you to override default parameters:

### Random Forest Parameters
- `rf_num_trees` (default: 300) - Number of trees
- `rf_max_depth` (default: 20) - Maximum tree depth
- `rf_min_examples` (default: 5) - Minimum examples per leaf

### Gradient Boosted Trees Parameters
- `gbt_num_trees` (default: 500) - Number of trees
- `gbt_max_depth` (default: 6) - Maximum tree depth
- `gbt_shrinkage` (default: 0.1) - **Learning rate** (lower = more conservative)
- `gbt_subsample` (default: 0.8) - Subsample ratio
- `gbt_early_stopping` (default: 30) - Early stopping rounds

### Extra Trees Parameters
- `et_num_trees` (default: 200) - Number of trees
- `et_candidate_ratio` (default: 0.5) - Candidate attributes ratio

### Meta-Learner Parameters
- `meta_C` (default: 1.0) - Regularization parameter
- `meta_max_iter` (default: 1000) - Maximum iterations

### General Parameters
- `random_seed` (default: 42) - Random seed for reproducibility
- `verbose` (default: True) - Print training progress

## 🔧 Example: Custom Parameters

```python
# Create ensemble with custom hyperparameters
ensemble = HonestYDFEnsemble(
    # Adjust learning rate (shrinkage)
    gbt_shrinkage=0.05,       # Lower learning rate for more careful learning
    
    # Adjust tree depth
    gbt_max_depth=8,           # Deeper GBT trees
    rf_max_depth=25,           # Deeper RF trees
    
    # Adjust number of trees
    gbt_num_trees=600,         # More GBT trees
    rf_num_trees=400,          # More RF trees
    et_num_trees=300,          # More ET trees
    
    # Other parameters
    gbt_subsample=0.7,         # Lower subsample ratio
    verbose=True
)

# Train with custom parameters
ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

## 📊 Performance Metrics

The model automatically calculates and displays:

- **Accuracy** - Overall classification accuracy
- **Precision** - Positive predictive value
- **Recall** - True positive rate (sensitivity)
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve
- **Confusion Matrix** - True/False Positives/Negatives

### Expected Performance

Based on the NASA Exoplanet Archive dataset:

| Metric | Expected Value |
|--------|---------------|
| Test Accuracy | ~82-83% |
| Precision | ~82-83% |
| Recall | ~82-83% |
| ROC-AUC | ~0.91-0.92 |

## 🎯 Key Features

✅ **No Data Leakage**
- Excludes `koi_pdisposition` and `koi_score`
- Uses only 13 curated physical features

✅ **Optimized Hyperparameters**
- Default parameters tuned with Optuna
- 50 trials of Bayesian optimization

✅ **Fast Training**
- Trains in under 2 seconds on most systems
- Efficient YDF implementation

✅ **Production Ready**
- Clean API with fit/predict/evaluate methods
- Comprehensive error handling
- Detailed logging and metrics

## 🎯 Target Variable & Classification Approach

### Original Dataset Categories

The NASA Kepler dataset contains **3 categories** in the `koi_disposition` field:

1. **CONFIRMED** - Confirmed exoplanets (verified through multiple methods)
2. **CANDIDATE** - Candidate exoplanets (detected but not yet confirmed)
3. **FALSE POSITIVE** - Not actual planets (stellar eclipsing binaries, instrumental artifacts, etc.)

### Binary Classification Strategy

This model uses a **binary classification approach** where:

- **Class 0 (FALSE POSITIVE)** = Not a planet
- **Class 1 (PLANET)** = CONFIRMED + CANDIDATE combined

**Rationale for Binary Classification:**

✅ **Scientifically Sound**: Both CONFIRMED and CANDIDATE represent real planet detections. The difference is in verification status, not in the physical characteristics we predict from.

✅ **Practical Utility**: For most applications, the key question is "Is this a planet or not?" rather than "Is this confirmed vs. candidate?"

✅ **Better Performance**: Binary classification achieves higher accuracy (~82-83%) compared to 3-class classification.

✅ **Model Confidence = Verification Level**: The probability scores naturally reflect confidence:
- High probability (>0.85) ≈ "Confirmed-like" detection
- Moderate probability (0.60-0.85) ≈ "Candidate-like" detection
- Low probability (<0.60) ≈ Uncertain

### Interpreting Predictions

```python
# Example: Interpret probability as confidence level
result = ensemble.infer_single(...)
probability = result['probability']

if result['prediction'] == 0:
    status = "FALSE POSITIVE"
elif probability > 0.85:
    status = "CONFIRMED PLANET (high confidence)"
elif probability > 0.60:
    status = "CANDIDATE PLANET (moderate confidence)"
else:
    status = "CANDIDATE PLANET (low confidence)"
```

**Note**: If you need true 3-class classification (to distinguish CONFIRMED from CANDIDATE), the model would need to be retrained with a different target encoding and meta-learner (e.g., multi-class logistic regression or softmax).

## 📋 Required Features

The model expects the following 13 features:

### Core Transit Properties (5 features)
- `koi_period` - Orbital period (days)
- `koi_depth` - Transit depth (ppm)
- `koi_duration` - Transit duration (hours)
- `koi_prad` - Planetary radius (Earth radii)
- `koi_impact` - Impact parameter

### Signal Quality Metrics (3 features)
- `koi_model_snr` - Transit signal-to-noise ratio
- `koi_max_mult_ev` - Maximum multiple event statistic
- `koi_num_transits` - Number of transits observed

### Stellar Characteristics (3 features)
- `koi_steff` - Stellar effective temperature (K)
- `koi_srad` - Stellar radius (solar radii)
- `koi_kepmag` - Kepler-band magnitude

### Derived Physical Properties (2 features)
- `koi_insol` - Insolation flux (Earth flux)
- `koi_teq` - Equilibrium temperature (K)

## 🔬 Data Preprocessing

The model automatically handles:
- **Missing values** - Median imputation
- **Log transformation** - Applied to `koi_period` and `koi_depth` (skewed distributions)
- **Feature validation** - Ensures all required features are present

## 🎓 Model Architecture

```
Input Features (13 curated)
         ↓
    Data Preprocessing
    (log transform, imputation)
         ↓
    ┌─────────────────────┐
    │   Base Models (3)   │
    ├─────────────────────┤
    │ 1. Random Forest    │
    │ 2. Gradient Boosted │
    │ 3. Extra Trees      │
    └─────────────────────┘
         ↓
    Meta-Features (3)
    (base model predictions)
         ↓
    ┌─────────────────────┐
    │   Meta-Learner      │
    │ Logistic Regression │
    └─────────────────────┘
         ↓
    Final Prediction
    (0=False Positive, 1=Planet)
```

## 📖 API Reference

### `HonestYDFEnsemble`

Main ensemble class for exoplanet detection.

#### Methods

**`fit(X, y, X_val=None, y_val=None)`**
- Train the ensemble on training data
- Optionally evaluate on validation set
- Returns: self

**`predict(X)`**
- Predict class labels
- Returns: np.ndarray of predictions (0 or 1)

**`predict_proba(X)`**
- Predict class probabilities
- Returns: np.ndarray of shape (n_samples, 2)

**`evaluate(X, y, dataset_name='test')`**
- Evaluate model performance
- Returns: dict of metrics

**`get_params()`**
- Get model parameters
- Returns: dict of hyperparameters

**`summary()`**
- Print comprehensive model summary
- Shows configuration, hyperparameters, training time, and metrics

## 🐛 Troubleshooting

### Import Errors

```python
# If you get "ModuleNotFoundError: No module named 'ydf'"
pip install ydf

# If you get "ModuleNotFoundError: No module named 'ydf_ensemble'"
# Make sure you're in the correct directory or adjust your Python path
import sys
sys.path.append('path/to/models/directory')
```

### Feature Errors

```python
# If you get "KeyError: 'koi_period'" or similar
# Ensure your DataFrame has all 13 required features:
required_features = [
    'koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_impact',
    'koi_model_snr', 'koi_max_mult_ev', 'koi_num_transits',
    'koi_steff', 'koi_srad', 'koi_kepmag', 'koi_insol', 'koi_teq'
]

missing = [f for f in required_features if f not in X.columns]
print(f"Missing features: {missing}")
```

### Memory Errors

If you encounter memory issues with large datasets:

```python
# Reduce number of trees
ensemble = HonestYDFEnsemble(
    rf_num_trees=200,
    gbt_num_trees=300,
    et_num_trees=150
)
```

## 📚 References

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- **YDF Documentation**: https://ydf.readthedocs.io/
- **Optuna**: https://optuna.readthedocs.io/

## 📄 License

This project was developed as part of the NASA Space Apps Challenge 2025.

## 🤝 Contributing

For questions or contributions, please refer to the main repository README.

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Status**: Production Ready ✅
