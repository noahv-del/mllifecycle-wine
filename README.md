# Wine Quality Prediction API - Deployment Guide

This document describes how to deploy and use the Wine Quality Prediction API.

## Overview

The API exposes the trained Random Forest classification model via a RESTful endpoint using FastAPI. It predicts whether a wine is "good quality" (quality score ≥ 7) or "not good quality" (quality score < 7) based on its physicochemical properties.

## Prerequisites

- Python 3.7+
- Required packages (already installed):
  - fastapi
  - uvicorn
  - scikit-learn
  - pandas
  - numpy
  - joblib
  - pydantic

## Files

- `app.py` - Main FastAPI application
- `test_api.py` - Test script for the API
- `pipeline_base.pkl` - Trained model pipeline (preprocessing + classifier)
- `DEPLOYMENT.md` - This file

## Running the API

### Option 1: Direct Python Execution

```bash
python app.py
```

### Option 2: Using Uvicorn

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will start on `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL**: `GET /`
- **Description**: Health check and API information
- **Response**: JSON with status and available endpoints

### 2. Health Check
- **URL**: `GET /health`
- **Description**: Check if model is loaded and ready
- **Response**: `{"status": "healthy", "model_loaded": true}`

### 3. Single Prediction
- **URL**: `POST /predict`
- **Description**: Predict quality for a single wine
- **Request Body**: JSON with wine features
- **Response**: Prediction, label, and confidence probability

#### Example Request:
```json
{
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
```

#### Example Response:
```json
{
  "prediction": 0,
  "prediction_label": "Not Good Quality (<7)",
  "probability": 0.8523
}
```

### 4. Batch Prediction
- **URL**: `POST /predict/batch`
- **Description**: Predict quality for multiple wines
- **Request Body**: JSON array of wine features
- **Response**: Array of predictions with count

## Interactive API Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- View all endpoints
- See request/response schemas
- Test the API interactively

## Testing the API

### Using the Test Script

Run the provided test script:
```bash
python test_api.py
```

This will test:
- Health check endpoint
- Single prediction
- High quality wine prediction
- Batch prediction

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python Requests

```python
import requests

# Single prediction
wine_data = {
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

response = requests.post(
    "http://localhost:8000/predict",
    json=wine_data
)

print(response.json())
```

## Input Features

The model requires 12 features:

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| fixed_acidity | float | Fixed acidity | ≥ 0 |
| volatile_acidity | float | Volatile acidity | ≥ 0 |
| citric_acid | float | Citric acid | ≥ 0 |
| residual_sugar | float | Residual sugar | ≥ 0 |
| chlorides | float | Chlorides | ≥ 0 |
| free_sulfur_dioxide | float | Free SO₂ | ≥ 0 |
| total_sulfur_dioxide | float | Total SO₂ | ≥ 0 |
| density | float | Density | ≥ 0 |
| pH | float | pH level | 0-14 |
| sulphates | float | Sulphates | ≥ 0 |
| alcohol | float | Alcohol % | ≥ 0 |
| color | string | "red" or "white" | - |

## Model Information

- **Algorithm**: Random Forest Classifier
- **Target**: Binary classification (good/not good quality)
- **Threshold**: Quality ≥ 7 = Good
- **Preprocessing**: Log transformation, standardization, PCA, one-hot encoding
- **Training Data**: Combined red and white wine dataset (6497 samples)

## Troubleshooting

### Port Already in Use
If port 8000 is already in use, specify a different port:
```bash
uvicorn app:app --port 8001
```

### Model Not Loading
Ensure `pipeline_base.pkl` is in the same directory as `app.py`

### Connection Errors
Check that the API is running and accessible at the correct URL

## For Oral Examination

To demonstrate the deployment:

1. Start the API:
   ```bash
   python app.py
   ```

2. Show the interactive docs:
   - Open browser: http://localhost:8000/docs

3. Run test script:
   ```bash
   python test_api.py
   ```

4. Make a live prediction via the Swagger UI or using curl/Python

## Performance Considerations

- The API loads the model once at startup for efficiency
- Each prediction request is independent (stateless)
- Batch endpoint available for multiple predictions
- Response times typically < 100ms for single predictions

## Security Notes

- Currently configured for local development (`0.0.0.0`)
- For production deployment, consider:
  - Authentication/authorization
  - HTTPS/TLS encryption
  - Rate limiting
  - Input validation enhancements
  - CORS configuration

## Next Steps for Production

1. Add authentication (e.g., API keys, OAuth)
2. Containerize with Docker
3. Deploy to cloud platform (AWS, Azure, GCP)
4. Add monitoring and logging
5. Implement CI/CD pipeline
6. Add model versioning
7. Set up A/B testing for model updates
