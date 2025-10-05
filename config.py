# Exoplanet Detection Project Configuration

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data sources
EXOPLANET_DATASETS = {
    'nasa_exoplanet_archive': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars',
    'kepler_koi': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative',
    'tess_toi': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi'
}

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'cv_folds': 5,
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
}

# Feature engineering settings
FEATURE_CONFIG = {
    'scaling_method': 'StandardScaler',  # or 'MinMaxScaler', 'RobustScaler'
    'handle_outliers': True,
    'outlier_method': 'IQR',  # or 'isolation_forest', 'local_outlier_factor'
    'feature_selection': True,
    'selection_method': 'mutual_info',  # or 'chi2', 'f_classif', 'rfe'
}

# Algorithms to test (based on research findings)
ALGORITHMS = {
    'ensemble_methods': [
        'RandomForestClassifier',
        'ExtraTreesClassifier',
        'GradientBoostingClassifier',
        'XGBClassifier',
        'LGBMClassifier',
        'CatBoostClassifier'
    ],
    'base_classifiers': [
        'LogisticRegression',
        'SVC',
        'DecisionTreeClassifier',
        'KNeighborsClassifier'
    ],
    'neural_networks': [
        'MLPClassifier'
    ]
}

# Hyperparameter tuning configuration
HYPEROPT_CONFIG = {
    'n_trials': 100,
    'timeout': 3600,  # 1 hour
    'n_jobs': -1
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file']
}
