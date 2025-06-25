# Architecture Documentation

## Overview

The RT Sentiment API is a high-performance, real-time sentiment analysis service built with FastAPI and optimized for sub-100ms latency. The system uses fine-tuned DistilBERT models and supports both PyTorch and ONNX inference backends.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Monitoring    │
│    (nginx)      │    │   (FastAPI)     │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Inference     │    │   Batch Queue   │    │   Model Cache   │
│   Service       │    │   (asyncio)     │    │   (Memory)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   ML Models     │
                    │ (PyTorch/ONNX)  │
                    └─────────────────┘
```

## Components

### 1. API Layer (FastAPI)

- **Framework**: FastAPI with async/await support
- **Endpoints**: 
  - `/predict` - Single text prediction
  - `/predict/batch` - Batch predictions
  - `/healthz` - Health checks
  - `/metrics` - Prometheus metrics
- **Features**:
  - Request validation with Pydantic v2
  - Automatic OpenAPI documentation
  - CORS support
  - Error handling and logging

### 2. Model Management

- **Model Loading**: Supports both PyTorch and ONNX models
- **Model Types**: 
  - PyTorch: Full model with GPU support
  - ONNX: Optimized for CPU inference
  - Mock: For testing and development
- **Features**:
  - Lazy loading on startup
  - Model format auto-detection
  - Graceful fallback between formats

### 3. Preprocessing Pipeline

- **Text Cleaning**: URL removal, normalization, whitespace handling
- **Tokenization**: DistilBERT tokenizer with padding/truncation
- **Batch Processing**: Efficient batching for multiple requests
- **Validation**: Input length and format validation

### 4. Inference Engine

- **Async Processing**: Non-blocking inference with thread pools
- **Batch Optimization**: Dynamic batching for improved throughput
- **Model Backends**:
  - PyTorch: Direct model inference
  - ONNX Runtime: Optimized CPU/GPU execution
- **Performance**: Target <100ms median latency

### 5. Request Queue System

- **Queue Management**: asyncio.Queue for request batching
- **Background Processing**: Dedicated batch processor task
- **Load Balancing**: Automatic request distribution
- **Timeout Handling**: Request timeout and error recovery

## Data Flow

### Single Request Flow

1. **Request Reception**: FastAPI receives POST request
2. **Validation**: Pydantic validates request schema
3. **Preprocessing**: Text cleaning and tokenization
4. **Model Selection**: Choose fast path vs batch queue
5. **Inference**: Model prediction with timing
6. **Response**: JSON response with label and confidence

### Batch Request Flow

1. **Batch Reception**: Multiple texts in single request
2. **Validation**: Validate all texts in batch
3. **Preprocessing**: Batch tokenization
4. **Inference**: Efficient batch model execution
5. **Response**: Array of predictions

### Background Batch Processing

1. **Queue Monitoring**: Continuous queue polling
2. **Batch Assembly**: Collect requests up to batch size/timeout
3. **Batch Inference**: Process multiple requests together
4. **Result Distribution**: Return results to waiting requests

## Performance Optimizations

### Model Optimizations

- **ONNX Export**: Optimized model format for inference
- **Model Quantization**: Reduced precision for faster inference
- **Graph Optimization**: ONNX Runtime optimizations
- **Memory Management**: Efficient tensor operations

### API Optimizations

- **Async Processing**: Non-blocking request handling
- **Connection Pooling**: Efficient HTTP connection reuse
- **Response Caching**: Cache frequent predictions
- **Batch Processing**: Amortize model overhead

### Infrastructure Optimizations

- **Container Optimization**: Multi-stage Docker builds
- **Resource Allocation**: CPU/memory tuning
- **Load Balancing**: Distribute requests across instances
- **Monitoring**: Real-time performance tracking

## Scalability Considerations

### Horizontal Scaling

- **Stateless Design**: No shared state between instances
- **Load Balancer**: Distribute traffic across replicas
- **Auto-scaling**: Scale based on CPU/memory/latency metrics
- **Health Checks**: Automatic unhealthy instance removal

### Vertical Scaling

- **CPU Optimization**: Multi-core utilization
- **Memory Management**: Efficient model loading
- **GPU Support**: CUDA acceleration when available
- **Batch Size Tuning**: Optimize for hardware

### Database Considerations

- **Model Storage**: Efficient model artifact storage
- **Metrics Storage**: Time-series data for monitoring
- **Logging**: Structured logging for debugging
- **Caching**: Redis for prediction caching

## Security

### API Security

- **Input Validation**: Strict request validation
- **Rate Limiting**: Prevent abuse and DoS
- **Authentication**: API key or JWT token support
- **HTTPS**: TLS encryption for data in transit

### Container Security

- **Minimal Images**: Distroless or slim base images
- **Non-root User**: Run containers as non-root
- **Secret Management**: Secure credential handling
- **Vulnerability Scanning**: Regular security scans

### Model Security

- **Model Integrity**: Verify model checksums
- **Access Control**: Restrict model file access
- **Audit Logging**: Track model usage and changes
- **Privacy**: No sensitive data in logs

## Monitoring and Observability

### Metrics

- **Request Metrics**: Count, latency, error rate
- **Model Metrics**: Inference time, batch size
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Prediction distribution

### Logging

- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: Debug, info, warning, error
- **Request Tracing**: End-to-end request tracking
- **Error Tracking**: Detailed error information

### Alerting

- **Latency Alerts**: P95 latency > threshold
- **Error Rate Alerts**: Error rate > threshold
- **Health Alerts**: Service health degradation
- **Resource Alerts**: High CPU/memory usage

## Deployment Architecture

### Development Environment

- **Local Development**: Docker Compose setup
- **Mock Models**: Fast testing with mock inference
- **Hot Reload**: Automatic code reloading
- **Debug Mode**: Detailed error information

### Staging Environment

- **Production-like**: Similar to production setup
- **Integration Testing**: End-to-end test suite
- **Performance Testing**: Load testing and benchmarks
- **Canary Deployment**: Gradual rollout testing

### Production Environment

- **High Availability**: Multi-zone deployment
- **Load Balancing**: Traffic distribution
- **Auto-scaling**: Dynamic resource allocation
- **Monitoring**: Comprehensive observability

## Technology Stack

### Core Technologies

- **Python 3.11**: Latest Python with performance improvements
- **FastAPI**: Modern, fast web framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **ONNX Runtime**: Optimized inference engine

### Infrastructure

- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **nginx**: Load balancing and reverse proxy

### Development Tools

- **Ruff**: Fast Python linter and formatter
- **Black**: Code formatting
- **pytest**: Testing framework
- **mypy**: Type checking
- **GitHub Actions**: CI/CD pipeline