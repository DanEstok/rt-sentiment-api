"""Pydantic schemas for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request schema for sentiment prediction."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for sentiment"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text input."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class BatchPredictionRequest(BaseModel):
    """Request schema for batch sentiment prediction."""
    
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to analyze for sentiment"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate text inputs."""
        if not v:
            raise ValueError("Texts list cannot be empty")
        
        validated_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or only whitespace")
            if len(text) > 10000:
                raise ValueError(f"Text at index {i} exceeds maximum length of 10000 characters")
            validated_texts.append(text.strip())
        
        return validated_texts


class PredictionResponse(BaseModel):
    """Response schema for sentiment prediction."""
    
    label: str = Field(
        ...,
        description="Predicted sentiment label (positive, negative, neutral)"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the prediction"
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(
        ...,
        description="Service status (healthy, unhealthy)"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    timestamp: Optional[str] = Field(
        None,
        description="Timestamp of the health check"
    )


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    model_type: str = Field(
        ...,
        description="Type of the loaded model"
    )
    model_path: str = Field(
        ...,
        description="Path to the model files"
    )
    labels: List[str] = Field(
        ...,
        description="Available sentiment labels"
    )
    max_length: int = Field(
        ...,
        description="Maximum input sequence length"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    code: Optional[str] = Field(
        None,
        description="Error code"
    )


class MetricsResponse(BaseModel):
    """Response schema for metrics."""
    
    total_requests: int = Field(
        ...,
        description="Total number of requests processed"
    )
    average_latency_ms: float = Field(
        ...,
        description="Average request latency in milliseconds"
    )
    model_load_time_ms: float = Field(
        ...,
        description="Model loading time in milliseconds"
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds"
    )


class BenchmarkRequest(BaseModel):
    """Request schema for benchmarking."""
    
    num_requests: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of requests to send"
    )
    concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of concurrent requests"
    )
    text: Optional[str] = Field(
        None,
        description="Custom text to use for benchmarking"
    )


class BenchmarkResponse(BaseModel):
    """Response schema for benchmark results."""
    
    total_requests: int = Field(
        ...,
        description="Total number of requests sent"
    )
    successful_requests: int = Field(
        ...,
        description="Number of successful requests"
    )
    failed_requests: int = Field(
        ...,
        description="Number of failed requests"
    )
    total_time_seconds: float = Field(
        ...,
        description="Total time taken for all requests"
    )
    requests_per_second: float = Field(
        ...,
        description="Requests per second"
    )
    average_latency_ms: float = Field(
        ...,
        description="Average latency in milliseconds"
    )
    median_latency_ms: float = Field(
        ...,
        description="Median latency in milliseconds"
    )
    p95_latency_ms: float = Field(
        ...,
        description="95th percentile latency in milliseconds"
    )
    p99_latency_ms: float = Field(
        ...,
        description="99th percentile latency in milliseconds"
    )
    min_latency_ms: float = Field(
        ...,
        description="Minimum latency in milliseconds"
    )
    max_latency_ms: float = Field(
        ...,
        description="Maximum latency in milliseconds"
    )