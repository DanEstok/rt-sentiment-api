"""FastAPI application for sentiment analysis."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .models import SentimentModel, ModelManager
from .schemas import PredictionRequest, PredictionResponse, HealthResponse, BatchPredictionRequest
from ..inference.preprocess import InferencePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
from prometheus_client import REGISTRY

# Clear existing metrics to avoid conflicts
try:
    REGISTRY.unregister(REGISTRY._collector_to_names.get('sentiment_requests_total'))
    REGISTRY.unregister(REGISTRY._collector_to_names.get('sentiment_request_duration_seconds'))
    REGISTRY.unregister(REGISTRY._collector_to_names.get('sentiment_prediction_duration_seconds'))
except:
    pass

REQUEST_COUNT = Counter('sentiment_requests_total', 'Total sentiment analysis requests')
REQUEST_LATENCY = Histogram('sentiment_request_duration_seconds', 'Request latency')
PREDICTION_LATENCY = Histogram('sentiment_prediction_duration_seconds', 'Model prediction latency')

# Global variables
model_manager: ModelManager = None
preprocessor: InferencePreprocessor = None
request_queue: asyncio.Queue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_manager, preprocessor, request_queue
    
    logger.info("Starting up sentiment analysis API...")
    
    # Initialize model manager
    model_path = os.getenv("MODEL_PATH", "./models")
    model_manager = ModelManager(model_path)
    await model_manager.load_model()
    
    # Initialize preprocessor
    preprocessor = InferencePreprocessor(model_path)
    
    # Initialize request queue for batching
    request_queue = asyncio.Queue(maxsize=1000)
    
    # Start background batch processor
    asyncio.create_task(batch_processor())
    
    logger.info("Startup complete!")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="RT Sentiment API",
    description="Real-time sentiment analysis API with sub-100ms latency",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def batch_processor():
    """Background task to process requests in batches."""
    batch_size = int(os.getenv("BATCH_SIZE", "8"))
    batch_timeout = float(os.getenv("BATCH_TIMEOUT", "0.01"))  # 10ms
    
    while True:
        batch = []
        start_time = time.time()
        
        try:
            # Collect requests for batch
            while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
                try:
                    request_item = await asyncio.wait_for(
                        request_queue.get(), 
                        timeout=batch_timeout - (time.time() - start_time)
                    )
                    batch.append(request_item)
                except asyncio.TimeoutError:
                    break
            
            # Process batch if we have requests
            if batch:
                await process_batch(batch)
                
        except Exception as e:
            logger.error(f"Error in batch processor: {e}")
            # Mark all requests in batch as failed
            for request_item in batch:
                if not request_item["future"].done():
                    request_item["future"].set_exception(e)
        
        # Small delay to prevent busy waiting
        if not batch:
            await asyncio.sleep(0.001)


async def process_batch(batch: List[Dict[str, Any]]):
    """Process a batch of requests."""
    try:
        texts = [item["text"] for item in batch]
        
        # Preprocess batch
        preprocessed = await preprocessor.preprocess_batch(texts)
        
        # Run inference
        with PREDICTION_LATENCY.time():
            predictions = await model_manager.predict_batch([p["input_ids"] for p in preprocessed],
                                                          [p["attention_mask"] for p in preprocessed])
        
        # Set results
        for i, (request_item, prediction) in enumerate(zip(batch, predictions)):
            if not request_item["future"].done():
                request_item["future"].set_result(prediction)
                
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        for request_item in batch:
            if not request_item["future"].done():
                request_item["future"].set_exception(e)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Quick model check
        if model_manager and model_manager.is_loaded():
            return HealthResponse(status="healthy", model_loaded=True)
        else:
            return HealthResponse(status="unhealthy", model_loaded=False)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """Predict sentiment for a single text."""
    REQUEST_COUNT.inc()
    
    with REQUEST_LATENCY.time():
        try:
            # Validate model is loaded
            if not model_manager or not model_manager.is_loaded():
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # For single requests, we can process directly for lower latency
            if len(request.text) < 100:  # Short texts get fast path
                # Preprocess
                preprocessed = preprocessor.preprocess(request.text)
                
                # Predict
                with PREDICTION_LATENCY.time():
                    prediction = await model_manager.predict_single(
                        preprocessed["input_ids"], 
                        preprocessed["attention_mask"]
                    )
                
                return PredictionResponse(
                    label=prediction["label"],
                    score=prediction["score"],
                    confidence=prediction["confidence"]
                )
            
            else:  # Longer texts go through batch queue
                # Create future for result
                future = asyncio.Future()
                
                # Add to queue
                request_item = {
                    "text": request.text,
                    "future": future
                }
                
                try:
                    request_queue.put_nowait(request_item)
                except asyncio.QueueFull:
                    raise HTTPException(status_code=503, detail="Server overloaded")
                
                # Wait for result
                prediction = await asyncio.wait_for(future, timeout=5.0)
                
                return PredictionResponse(
                    label=prediction["label"],
                    score=prediction["score"],
                    confidence=prediction["confidence"]
                )
                
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts."""
    REQUEST_COUNT.inc(len(request.texts))
    
    with REQUEST_LATENCY.time():
        try:
            # Validate model is loaded
            if not model_manager or not model_manager.is_loaded():
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Validate batch size
            max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "32"))
            if len(request.texts) > max_batch_size:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Batch size {len(request.texts)} exceeds maximum {max_batch_size}"
                )
            
            # Preprocess batch
            preprocessed = await preprocessor.preprocess_batch(request.texts)
            
            # Predict
            with PREDICTION_LATENCY.time():
                predictions = await model_manager.predict_batch(
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
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if not model_manager or not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_manager.model_type,
        "model_path": model_manager.model_path,
        "labels": model_manager.get_labels(),
        "max_length": getattr(preprocessor, 'max_length', 512) if preprocessor else 512
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "src.app.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,  # Single worker for model sharing
        loop="asyncio"
    )