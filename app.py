import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging

from src.mlProject.pipeline.stage_06_prediction import PredictionPipeline
from src.mlProject import logger

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Gemstone Price Prediction API",
    description="API for predicting gemstone prices using CatBoost model",
    version="1.0.0"
)

# Initialize prediction pipeline
prediction_pipeline = PredictionPipeline()


# ============= Pydantic Models =============
class GemstoneFeatures(BaseModel):
    """Input features for gemstone price prediction"""
    carat: float = Field(..., gt=0, description="Weight of the gemstone in carats")
    cut: str = Field(..., description="Cut quality: Fair, Good, Very Good, Premium, or Ideal")
    color: str = Field(..., description="Color grade: D-J (D is best)")
    clarity: str = Field(..., description="Clarity: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, or IF")
    depth: float = Field(..., gt=0, le=100, description="Depth percentage (0-100)")
    table: float = Field(..., gt=0, le=100, description="Table percentage (0-100)")
    x: float = Field(..., ge=0, description="Length in mm")
    y: float = Field(..., ge=0, description="Width in mm")
    z: float = Field(..., ge=0, description="Height in mm")

    class Config:
        schema_extra = {
            "example": {
                "carat": 0.5,
                "cut": "Ideal",
                "color": "E",
                "clarity": "SI1",
                "depth": 61.5,
                "table": 55.0,
                "x": 5.1,
                "y": 5.2,
                "z": 3.2
            }
        }


class PredictionResponse(BaseModel):
    """Response model for price prediction"""
    predicted_price: float = Field(..., description="Predicted price in dollars")
    message: str = Field("Prediction successful", description="Status message")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


# ============= Routes =============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Gemstone Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(features: GemstoneFeatures):
    """
    Predict gemstone price based on its characteristics
    
    **Example Input:**
    ```
    {
        "carat": 0.5,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI1",
        "depth": 61.5,
        "table": 55.0,
        "x": 5.1,
        "y": 5.2,
        "z": 3.2
    }
    ```
    """
    try:
        # Convert input to dictionary
        input_data = features.model_dump()
        
        # Make prediction
        predicted_price = prediction_pipeline.predict(input_data)
        
        logger.info(f"Prediction successful for input: {input_data}")
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            message="Prediction successful"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except FileNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(features_list: list[GemstoneFeatures]):
    """
    Predict prices for multiple gemstones in batch
    
    **Example Input:**
    ```
    [
        {
            "carat": 0.5,
            "cut": "Ideal",
            "color": "E",
            "clarity": "SI1",
            "depth": 61.5,
            "table": 55.0,
            "x": 5.1,
            "y": 5.2,
            "z": 3.2
        },
        {
            "carat": 1.0,
            "cut": "Premium",
            "color": "F",
            "clarity": "VS1",
            "depth": 62.0,
            "table": 56.0,
            "x": 6.5,
            "y": 6.6,
            "z": 4.1
        }
    ]
    ```
    """
    try:
        predictions = []
        
        for features in features_list:
            input_data = features.model_dump()
            predicted_price = prediction_pipeline.predict(input_data)
            predictions.append({
                "input": input_data,
                "predicted_price": round(predicted_price, 2)
            })
        
        logger.info(f"Batch prediction successful for {len(predictions)} items")
        
        return {
            "total_predictions": len(predictions),
            "predictions": predictions,
            "message": "Batch prediction successful"
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/info", tags=["Information"])
async def model_info():
    """Get information about the model and expected features"""
    return {
        "model_type": "CatBoost Regressor",
        "task": "Gemstone Price Prediction",
        "input_features": {
            "carat": {
                "type": "float",
                "description": "Weight of the gemstone in carats",
                "constraints": "greater than 0"
            },
            "cut": {
                "type": "string",
                "description": "Cut quality",
                "valid_values": ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            },
            "color": {
                "type": "string",
                "description": "Color grade",
                "valid_values": ["D", "E", "F", "G", "H", "I", "J"]
            },
            "clarity": {
                "type": "string",
                "description": "Clarity level",
                "valid_values": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
            },
            "depth": {
                "type": "float",
                "description": "Depth percentage",
                "constraints": "0 < depth <= 100"
            },
            "table": {
                "type": "float",
                "description": "Table percentage",
                "constraints": "0 < table <= 100"
            },
            "x": {
                "type": "float",
                "description": "Length in mm",
                "constraints": "greater than or equal to 0"
            },
            "y": {
                "type": "float",
                "description": "Width in mm",
                "constraints": "greater than or equal to 0"
            },
            "z": {
                "type": "float",
                "description": "Height in mm",
                "constraints": "greater than or equal to 0"
            }
        },
        "output": {
            "type": "float",
            "description": "Predicted price in dollars"
        }
    }


@app.get("/docs", include_in_schema=False)
async def swagger_docs():
    """Swagger UI documentation"""
    # FastAPI automatically serves this at /docs
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
