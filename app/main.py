from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
# from fastapi.testclient import TestClient
import asyncio
from .config import LOGISTIC_MODEL, RF_MODEL, MODELS_DIR
from .auth import verify_api_key
import joblib
import os
import numpy as np

classes = ['Setosa', 'Versicolor', 'Virginica']
models = {}

def load_model(path: str):
    if not path:
        return None
    model = None
    with open(path, "rb") as f:
        model = joblib.load(f)
    return model

class IrisData(BaseModel):
    sepal_length: float = Field(default=1.1, ge=0, le=8, description="Sepal length in cm (must be positive)")
    sepal_width: float = Field(default=1.0, ge=0, le=5, description="Sepal width in cm (must be positive)")
    petal_length: float = Field(default=1.0, ge=0, le=7, description="Petal length in cm (must be positive)")
    petal_width: float = Field(default=1.0, ge=0, le=3, description="Petal width in cm (must be positive)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    models["logistic_model"] = load_model(LOGISTIC_MODEL)
    models["rf_model"] = load_model(RF_MODEL)

    print("Models loaded successfully.")
    yield
    # Cleanup on shutdown (if needed)
    models.clear()
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
# Note: Remove the incorrect dependency override line

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/{model_name}")
async def predict(input_data: IrisData, model_name: str, background_tasks: BackgroundTasks, api_key_valid: bool = Depends(verify_api_key)):
    """
    General prediction endpoint that selects model based on query parameter.
    Args:
        input_data (IrisInput): `IrisInput` object containing list of observations.
        Eg.: `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}`
        model_name (str): Model to use for prediction (`lr` or `rd`).
    Returns:
        dict: Prediction result with label and name.
    """
    # Use security dependency
    if not api_key_valid:
        raise HTTPException(status_code=403, detail="Forbidden")

    await asyncio.sleep(1)
    if model_name == "lr":
        model = models["logistic_model"]
    elif model_name == "rd":
        model = models["rf_model"]
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")
    features = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
    prediction = model.predict(np.array(features))
    name = classes[prediction[0]]

    background_tasks.add_task(log_prediction, int(prediction[0]), model_name)
    return {"prediction label": int(prediction[0]), "name": name}


# Route: Prediction using logistic_regression
@app.post("/prediction/logistic_reg")
async def predict_lr(input_data: IrisData, background_tasks: BackgroundTasks, api_key_valid: bool = Depends(verify_api_key)):
    """
    Prediction endpoint using logistic regression model.
    Args:
        input_data (IrisInput): `IrisInput` object containing list of observations.
        Eg.: `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}`
    Returns:
        dict: Prediction result with label and name.
    """
    if not api_key_valid:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    await asyncio.sleep(1)
    model = models["logistic_model"]
    features = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
    prediction = model.predict(np.array(features))
    name = classes[prediction[0]]

    background_tasks.add_task(log_prediction, int(prediction[0]), LOGISTIC_MODEL)
    return {"prediction label": int(prediction[0]), "name": name}

# Route: Prediction using random_forest
@app.post("/prediction/random_forest")
async def predict_rd(input_data: IrisData, background_tasks: BackgroundTasks, api_key_valid: bool = Depends(verify_api_key)):
    """
    Prediction endpoint using random forest model.
    Args:
        input_data (IrisInput): `IrisInput` object containing list of observations.
        Eg.: `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}`
    Returns:
        dict: Prediction result with label and name.
    """
    if not api_key_valid:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    await asyncio.sleep(1)
    model = models["rf_model"]
    features = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
    prediction = model.predict(np.array(features))
    name = classes[prediction[0]]

    background_tasks.add_task(log_prediction, int(prediction[0]), RF_MODEL)
    return {"prediction label": int(prediction[0]), "name": name}

# Route to to list the models
@app.get("/models")
async def available_models(api_key_valid: bool = Depends(verify_api_key)):
    if not api_key_valid:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"available_models": list(models.keys())}


async def log_prediction(prediction: int, model_name: str):
    """
    Background task to log the prediction.
    Simulates a long-running task with sleep.
    """
    await asyncio.sleep(5)  
    print(f"Logged prediction: {prediction} using model {model_name} at {asyncio.get_event_loop().time()}")


# fastapi dev main.py