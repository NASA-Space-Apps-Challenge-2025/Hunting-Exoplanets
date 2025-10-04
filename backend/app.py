from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from model_manager import ModelManager


class HyperParameters(BaseModel):
    learning_rate: float = Field(0.01, description="Learning rate for the model")
    max_depth: int = Field(5, description="Maximum depth of trees")
    num_trees: int = Field(100, description="Number of trees")
    use_hessian_gain: bool = Field(False, description="Whether to use Hessian gain")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")
    timestamp: str = Field(..., description="Current UTC timestamp")


class InferenceResponse(BaseModel):
    session: str = Field(..., description="Session ID used for inference or 'base_model'")
    predictions: List[Any] = Field(..., description="List of predictions")
    num_predictions: int = Field(..., description="Number of predictions made")
    timestamp: str = Field(..., description="UTC timestamp of inference")


class ModelMetrics(BaseModel):
    accuracy: float = Field(..., description="Model accuracy score")
    precision: float = Field(..., description="Model precision score")
    recall: float = Field(..., description="Model recall score")
    f1_score: float = Field(..., description="Model F1 score")
    confusion_matrix: List[Any] = Field(..., description="Confusion matrix")
    roc_auc: float = Field(..., description="ROC AUC score")


class ComparisonData(BaseModel):
    base_model: ModelMetrics = Field(..., description="Base model metrics")
    new_model: ModelMetrics = Field(..., description="New model metrics")
    improvements: Dict[str, float] = Field(..., description="Metric improvements (deltas)")
    training_info: Dict[str, Any] = Field(..., description="Training information")


class RetrainResponse(BaseModel):
    session_id: str = Field(..., description="Unique session ID for this trained model")
    metrics: ModelMetrics = Field(..., description="Performance metrics of the newly trained model")
    comparison: ComparisonData = Field(..., description="Comparison between base and new model")
    hyperparameters: HyperParameters = Field(..., description="Hyperparameters used for training")
    training_data_size: int = Field(..., description="Total size of training data")
    new_data_size: int = Field(..., description="Size of newly added data")
    timestamp: str = Field(..., description="UTC timestamp of training completion")


class TrainingInfo(BaseModel):
    training_samples: int = Field(..., description="Number of training samples")
    features: List[str] = Field(..., description="List of feature names")
    trained_at: str = Field(..., description="UTC timestamp when model was trained")


class ModelInfoResponse(BaseModel):
    model_type: str = Field(..., description="Type of ML model")
    metrics: ModelMetrics = Field(..., description="Model performance metrics")
    hyperparameters: HyperParameters = Field(..., description="Model hyperparameters")
    training_info: TrainingInfo = Field(..., description="Training metadata")
    timestamp: str = Field(..., description="Current UTC timestamp")


class GraphResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    comparison: ComparisonData = Field(..., description="Comparison data for visualization")
    metrics: ModelMetrics = Field(..., description="Model metrics")
    timestamp: str = Field(..., description="UTC timestamp")


app = FastAPI(
    title="Exoplanet ML Model API",
    version="1.0.0",
    description="API for exoplanet detection model training, inference, and performance tracking"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and healthy"
)
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post(
    "/inference",
    response_model=InferenceResponse,
    summary="Run Model Inference",
    description="Upload a CSV file to get predictions from the base model or a retrained model"
)
async def inference(
    file: UploadFile = File(..., description="CSV file containing features for prediction"),
    session: Optional[str] = Query(None, description="Session ID to use retrained model instead of base model")
):
    """
    Perform inference on uploaded CSV data

    Args:
        file: CSV file with data for prediction
        session: Optional session ID to use a retrained model

    Returns:
        Predictions and metadata
    """
    try:
        # Read and parse CSV
        contents = await file.read()
        try:
            csv_data = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        # Validate session if provided
        if session and not model_manager.session_exists(session):
            raise HTTPException(status_code=404, detail=f"Session {session} not found")

        # Perform inference
        predictions = model_manager.predict(csv_data, session)

        return {
            "session": session or "base_model",
            "predictions": predictions,
            "num_predictions": len(predictions) if predictions else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Retrain Model",
    description="Upload new training data to retrain the model with custom hyperparameters. Returns session ID and performance metrics."
)
async def retrain(
    file: UploadFile = File(..., description="CSV file with new training data"),
    learning_rate: Optional[float] = Form(0.01, description="Learning rate for gradient boosting (default: 0.01)"),
    max_depth: Optional[int] = Form(5, description="Maximum depth of decision trees (default: 5)"),
    num_trees: Optional[int] = Form(100, description="Number of trees in the ensemble (default: 100)"),
    use_hessian_gain: Optional[bool] = Form(False, description="Use Hessian-based gain calculation (default: False)")
):
    """
    Retrain the model with new data

    Args:
        file: CSV file with new training data
        learning_rate: Learning rate for the model (default: 0.01)
        max_depth: Maximum depth of trees (default: 5)
        num_trees: Number of trees (default: 100)
        use_hessian_gain: Whether to use Hessian gain (default: False)

    Returns:
        Session ID, model metrics, and comparison with base model
    """
    try:
        # Read and parse CSV
        contents = await file.read()
        try:
            new_data = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        # Prepare hyperparameters
        hyperparams = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_trees": num_trees,
            "use_hessian_gain": use_hessian_gain
        }

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Fetch existing training data from API
        existing_data = model_manager.fetch_existing_data()

        # Combine existing and new data
        if existing_data is not None:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data

        # Train new model and get metrics
        training_result = model_manager.train_model(session_id, combined_data, hyperparams)

        # Store session data
        model_manager.store_session(
            session_id=session_id,
            metrics=training_result['metrics'],
            comparison=training_result['comparison'],
            hyperparameters=hyperparams,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        return {
            "session_id": session_id,
            "metrics": training_result['metrics'],
            "comparison": training_result['comparison'],
            "hyperparameters": hyperparams,
            "training_data_size": len(combined_data),
            "new_data_size": len(new_data),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get Base Model Info",
    description="Retrieve information about the pre-trained base model including metrics, hyperparameters, and training details"
)
async def info():
    """
    Get pre-trained base model information and metrics

    Returns:
        Base model metrics and information
    """
    try:
        model_info = model_manager.get_base_model_info()
        return {
            "model_type": model_info.get("model_type", "base_model"),
            "metrics": model_info.get("metrics", {}),
            "hyperparameters": model_info.get("hyperparameters", {}),
            "training_info": model_info.get("training_info", {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {str(e)}")


@app.get(
    "/graph",
    response_model=GraphResponse,
    summary="Get Model Comparison Data",
    description="Retrieve comparison data between base model and retrained model for a specific session. Used for visualization."
)
async def graph(identifier: str = Query(..., description="Session ID from the retrain endpoint")):
    """
    Get model comparison data for visualization

    Args:
        identifier: Session ID to fetch comparison data

    Returns:
        Comparison metrics and visualization data
    """
    try:
        # Fetch session data
        session_data = model_manager.get_session_data(identifier)

        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {identifier} not found")

        return {
            "session_id": identifier,
            "comparison": session_data['comparison'],
            "metrics": session_data['metrics'],
            "timestamp": session_data['timestamp']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
