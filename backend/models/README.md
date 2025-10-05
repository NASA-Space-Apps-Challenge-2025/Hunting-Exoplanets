# Model Storage Directory

This directory stores trained model weights and configurations.

## Structure

```
models/
├── config.json              # Optimal hyperparameters and model configuration
├── base_model/             # Base model storage
│   ├── model.pkl          # Serialized base model
│   ├── metrics.json       # Base model performance metrics
│   └── metadata.json      # Training metadata (timestamp, samples, etc.)
└── README.md              # This file
```

## config.json

Contains optimal hyperparameters from Optuna tuning:
- Random Forest parameters (num_trees, max_depth, min_examples)
- Gradient Boosted Trees parameters (num_trees, max_depth, shrinkage, subsample, early_stopping)
- Extra Trees parameters (num_trees, candidate_ratio)
- Meta-learner parameters (C, max_iter)
- Random seed for reproducibility

## Base Model

The base model is the pre-trained YDF Stacking Ensemble that serves as the reference for:
- Initial inference requests
- Comparison with retrained models
- Performance benchmarking

### Files:
- `model.pkl`: Pickled HonestYDFEnsemble object
- `metrics.json`: Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- `metadata.json`: Training date, sample count, feature list, etc.

## Usage

### Save Base Model
```python
from model_manager import ModelManager

manager = ModelManager()
# ... train base model ...
manager.save_base_model()
```

### Load Base Model
```python
manager = ModelManager()
manager.load_base_model()
```

## Notes

- Session-specific retrained models are stored in memory (not persisted to disk)
- For production, consider using cloud storage (S3, GCS) for model versioning
- Model files can be large (50-500MB depending on data size)
