# Exoplanet ML Model API

FastAPI backend for hosting an ML model with training and inference capabilities.

## Features

- **Inference Endpoint**: Run predictions on CSV data using base or retrained models
- **Retraining Endpoint**: Train new models with additional data and track performance
- **Model Comparison**: Compare retrained models against the base model
- **Session Management**: Track and retrieve different model versions
- **Visualization Support**: Export comparison data for frontend visualization

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

### Local Development

```bash
# Development mode
python app.py

# Or with uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production Deployment

See [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) for detailed instructions on deploying to Render.com's free tier.

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-04T12:00:00.000000"
}
```

---

### `POST /inference`
Perform inference on uploaded CSV data.

**Query Parameters:**
- `session` (optional): Session ID to use a retrained model instead of base model

**Request:**
- Body: CSV file (multipart/form-data)

**Response:**
```json
{
  "session": "base_model",
  "predictions": [
    {
      "prediction": 1,
      "label": "Exoplanet",
      "confidence": null
    },
    {
      "prediction": 0,
      "label": "Not Exoplanet",
      "confidence": null
    },
    {
      "prediction": 1,
      "label": "Exoplanet",
      "confidence": null
    }
  ],
  "num_predictions": 3,
  "timestamp": "2025-10-04T12:00:00.000000"
}
```

**Example:**
```bash
# Using base model
curl -X POST "http://localhost:8000/inference" \
  -F "file=@data.csv"

# Using retrained model
curl -X POST "http://localhost:8000/inference?session=abc-123-def" \
  -F "file=@data.csv"
```

**Common Error - 405 Method Not Allowed:**
- ❌ Using GET: `curl http://localhost:8000/inference` (WRONG)
- ✅ Use POST: `curl -X POST http://localhost:8000/inference -F "file=@data.csv"` (CORRECT)

---

### `POST /retrain`
Retrain the model with new data. Fetches existing data from API, combines it with new data, trains a new model, and returns performance metrics.

**Request:**
- Body: CSV file with new training data (multipart/form-data)
- Form parameters (optional):
  - `learning_rate` (float, default: 0.03): Learning rate for gradient boosting
  - `max_depth` (int, default: 8): Maximum depth of gradient boosted trees
  - `num_trees` (int, default: 700): Number of gradient boosted trees

**Note:** All other hyperparameters (Random Forest, Extra Trees, Meta-learner) use optimal values from config.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.94,
    "f1_score": 0.935,
    "confusion_matrix": [...],
    "roc_auc": 0.96
  },
  "comparison": {
    "base_model": {...},
    "new_model": {...},
    "improvements": {
      "accuracy_delta": 0.05,
      "precision_delta": 0.03,
      "recall_delta": 0.04,
      "f1_score_delta": 0.035
    },
    "training_info": {
      "training_samples": 1000,
      "features": ["feature1", "feature2", ...],
      "timestamp": "2025-10-04T12:00:00.000000"
    }
  },
  "training_data_size": 1000,
  "new_data_size": 100,
  "timestamp": "2025-10-04T12:00:00.000000"
}
```

**Example:**
```bash
# With default hyperparameters
curl -X POST "http://localhost:8000/retrain" \
  -F "file=@new_training_data.csv"

# With custom hyperparameters
curl -X POST "http://localhost:8000/retrain" \
  -F "file=@new_training_data.csv" \
  -F "learning_rate=0.05" \
  -F "max_depth=10" \
  -F "num_trees=500"
```

---

### `GET /graph`
Get model comparison data for visualization.

**Query Parameters:**
- `identifier` (required): Session ID to fetch comparison data

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "comparison": {
    "base_model": {...},
    "new_model": {...},
    "improvements": {...},
    "training_info": {...}
  },
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.94,
    "f1_score": 0.935,
    "roc_auc": 0.96
  },
  "timestamp": "2025-10-04T12:00:00.000000"
}
```

**Example:**
```bash
curl "http://localhost:8000/graph?identifier=550e8400-e29b-41d4-a716-446655440000"
```

---

### `GET /sessions`
List all available training sessions.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2025-10-04T12:00:00.000000",
      "accuracy": 0.95,
      "created_at": "2025-10-04T12:00:00.000000"
    }
  ],
  "count": 1
}
```

**Example:**
```bash
curl "http://localhost:8000/sessions"
```

---

## Project Structure

```
backend/
├── app.py              # Main FastAPI application with endpoints
├── model_manager.py    # Model training, inference, and session management
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Implementation TODO

The following functions in `model_manager.py` need to be implemented:

### `fetch_existing_data()` (Line 19)
Fetch existing training data from external API.
```python
def fetch_existing_data(self) -> Optional[pd.DataFrame]:
    # TODO: Implement API call to fetch existing training data
    pass
```

### `train_model()` (Line 29)
Train a new model with provided data and calculate metrics.
```python
def train_model(self, session_id: str, data: pd.DataFrame) -> Dict[str, Any]:
    # TODO: Implement actual model training logic
    # - Train your ML model
    # - Calculate accuracy, precision, recall, F1, ROC-AUC
    # - Generate confusion matrix
    # - Store trained model in self.trained_models[session_id]
    pass
```

### `predict()` (Line 74)
Perform prediction on input data.
```python
def predict(self, data: pd.DataFrame, session_id: Optional[str] = None) -> List[Any]:
    # TODO: Implement actual prediction logic
    # - Use self.trained_models[session_id] if session_id provided
    # - Otherwise use self.base_model
    # - Return predictions as list
    pass
```

### `_get_base_model_metrics()` (Line 153)
Get base model metrics for comparison.
```python
def _get_base_model_metrics(self) -> Dict[str, float]:
    # TODO: Implement actual base model metrics retrieval
    # - Return metrics from your base model
    pass
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid CSV format, missing data)
- `404`: Resource not found (invalid session ID)
- `500`: Internal server error

Error response format:
```json
{
  "detail": "Error message here"
}
```

## CORS

CORS is enabled for all origins. Modify the `CORSMiddleware` configuration in `app.py` to restrict origins in production.

## Development Notes

- Sessions are stored in memory. For production, consider using a database or Redis.
- Models are stored in memory. For production, consider serializing models to disk or blob storage.
- The API currently has no authentication. Add authentication middleware for production use.
