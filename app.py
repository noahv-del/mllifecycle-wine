"""
Wine Quality Prediction API
Deploys the trained Random Forest model for wine quality classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Literal
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predicts wine quality (good/not good) based on physicochemical properties",
    version="1.0.0"
)

# Load the trained pipeline
try:
    pipeline = joblib.load("pipeline_base.pkl")
    print("Model pipeline loaded successfully")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None


# Define input schema
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., description="Fixed acidity", ge=0)
    volatile_acidity: float = Field(..., description="Volatile acidity", ge=0)
    citric_acid: float = Field(..., description="Citric acid", ge=0)
    residual_sugar: float = Field(..., description="Residual sugar", ge=0)
    chlorides: float = Field(..., description="Chlorides", ge=0)
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide", ge=0)
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide", ge=0)
    density: float = Field(..., description="Density", ge=0)
    pH: float = Field(..., description="pH level", ge=0, le=14)
    sulphates: float = Field(..., description="Sulphates", ge=0)
    alcohol: float = Field(..., description="Alcohol percentage", ge=0)
    color: Literal["red", "white"] = Field(..., description="Wine color")

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
                "color": "red"
            }
        }


# Define output schema
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Quality prediction (0=not good, 1=good)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: float = Field(..., description="Confidence probability")


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "message": "Wine Quality Prediction API is running",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Check if model is loaded and ready"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_wine_quality(wine: WineFeatures):
    """
    Predict wine quality based on physicochemical properties

    Returns:
    - prediction: 0 (not good) or 1 (good quality, score >= 7)
    - prediction_label: Human-readable label
    - probability: Confidence of the prediction
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([{
            "fixed acidity": wine.fixed_acidity,
            "volatile acidity": wine.volatile_acidity,
            "citric acid": wine.citric_acid,
            "residual sugar": wine.residual_sugar,
            "chlorides": wine.chlorides,
            "free sulfur dioxide": wine.free_sulfur_dioxide,
            "total sulfur dioxide": wine.total_sulfur_dioxide,
            "density": wine.density,
            "pH": wine.pH,
            "sulphates": wine.sulphates,
            "alcohol": wine.alcohol,
            "color": wine.color
        }])

        # Engineer the sulfur_ratio feature (as done in training)
        input_data["sulfur_ratio"] = (
            input_data["free sulfur dioxide"] /
            (input_data["total sulfur dioxide"] + 1e-6)
        )
        input_data["sulfur_ratio"] = input_data["sulfur_ratio"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.25)  # Use a default median value

        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        confidence = float(max(probabilities))

        # Create response
        prediction_label = "Good Quality (≥7)" if prediction == 1 else "Not Good Quality (<7)"

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(wines: list[WineFeatures]):
    """Batch prediction endpoint for multiple wines"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")

    try:
        predictions = []
        for wine in wines:
            result = predict_wine_quality(wine)
            predictions.append(result)

        return {"predictions": predictions, "count": len(predictions)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


if __name__ == "__main__":
    print("Starting Wine Quality Prediction API...")
    print("Access the API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
