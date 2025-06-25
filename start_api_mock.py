#!/usr/bin/env python3
"""Startup script for the API with mock model."""

import os
import sys
import asyncio
import uvicorn
from contextlib import asynccontextmanager

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

# Import after path setup
from fastapi import FastAPI
from src.app.models import MockModel, ModelManager
from src.inference.preprocess import InferencePreprocessor

# Global variables
model_manager = None
preprocessor = None
request_queue = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with mock model."""
    global model_manager, preprocessor, request_queue
    
    print("🚀 Starting up sentiment analysis API with mock model...")
    
    # Initialize mock model manager
    model_manager = ModelManager("./test_models")
    model_manager.model = MockModel()
    await model_manager.model.load()
    model_manager.model_type = "Mock"
    
    # Initialize mock preprocessor
    class MockTokenizer:
        def __call__(self, texts, **kwargs):
            import torch
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": torch.randint(0, 1000, (len(texts), 10)),
                "attention_mask": torch.ones(len(texts), 10, dtype=torch.long)
            }
        
        def save_pretrained(self, path):
            pass
    
    class MockPreprocessor:
        def __init__(self):
            self.max_length = 512
            
        def preprocess(self, text):
            import torch
            return {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
                "features": {"length": len(text), "word_count": len(text.split())}
            }
        
        async def preprocess_batch(self, texts):
            import torch
            results = []
            for text in texts:
                results.append({
                    "input_ids": torch.randint(0, 1000, (1, 10)),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long),
                    "features": {"length": len(text), "word_count": len(text.split())}
                })
            return results
    
    preprocessor = MockPreprocessor()
    
    # Initialize request queue
    request_queue = asyncio.Queue(maxsize=1000)
    
    print("✅ Mock model startup complete!")
    
    yield
    
    print("🔄 Shutting down...")

# Create the FastAPI app with mock components
app = FastAPI(
    title="RT Sentiment API (Mock Mode)",
    description="Real-time sentiment analysis API with mock model for testing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import endpoints
from src.app.schemas import PredictionRequest, PredictionResponse, HealthResponse, BatchPredictionRequest
from fastapi import HTTPException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        if model_manager and model_manager.model and model_manager.model.is_loaded:
            return HealthResponse(status="healthy", model_loaded=True)
        else:
            return HealthResponse(status="unhealthy", model_loaded=False)
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """Predict sentiment for a single text."""
    try:
        if not model_manager or not model_manager.model or not model_manager.model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess
        preprocessed = preprocessor.preprocess(request.text)
        
        # Predict
        prediction = await model_manager.model.predict(
            preprocessed["input_ids"], 
            preprocessed["attention_mask"]
        )
        
        return PredictionResponse(
            label=prediction["label"],
            score=prediction["score"],
            confidence=prediction["confidence"]
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts."""
    try:
        if not model_manager or not model_manager.model or not model_manager.model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate batch size
        max_batch_size = 32
        if len(request.texts) > max_batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size {len(request.texts)} exceeds maximum {max_batch_size}"
            )
        
        # Preprocess batch
        preprocessed = await preprocessor.preprocess_batch(request.texts)
        
        # Predict
        predictions = await model_manager.model.predict_batch(
            [p["input_ids"] for p in preprocessed],
            [p["attention_mask"] for p in preprocessed]
        )
        
        # Format responses
        responses = []
        for prediction in predictions:
            responses.append(PredictionResponse(
                label=prediction["label"],
                score=prediction["score"],
                confidence=prediction["confidence"]
            ))
        
        return responses
        
    except Exception as e:
        print(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if not model_manager or not model_manager.model or not model_manager.model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Mock",
        "model_path": "./test_models",
        "labels": ["negative", "neutral", "positive"],
        "max_length": 512
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        loop="asyncio"
    )