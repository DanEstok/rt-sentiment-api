# API Reference

## Overview

The RT Sentiment API provides real-time sentiment analysis with sub-100ms latency. This document covers all available endpoints, request/response formats, and usage examples.

## Base URL

```
Production: https://api.rt-sentiment.com
Staging: https://staging-api.rt-sentiment.com
Local: http://localhost:8000
```

## Authentication

Currently, the API supports public access. Future versions will include API key authentication.

```bash
# Future authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.rt-sentiment.com/predict
```

## Endpoints

### Health Check

Check the health status of the API service.

#### `GET /healthz`

**Response**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

**Example**

```bash
curl -X GET http://localhost:8000/healthz
```

```python
import requests

response = requests.get("http://localhost:8000/healthz")
print(response.json())
```

### Single Prediction

Analyze sentiment for a single text input.

#### `POST /predict`

**Request Body**

```json
{
  "text": "I love this product! It's amazing!"
}
```

**Response**

```json
{
  "label": "positive",
  "score": 0.9234,
  "confidence": 0.9234
}
```

**Parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to analyze (1-10,000 characters) |

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Predicted sentiment: "positive", "negative", or "neutral" |
| `score` | float | Confidence score for the predicted label (0.0-1.0) |
| `confidence` | float | Overall model confidence (0.0-1.0) |

**Status Codes**
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Model not loaded
- `504 Gateway Timeout`: Request timeout

**Examples**

```bash
# cURL
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'
```

```python
# Python requests
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This product is amazing!"}
)
result = response.json()
print(f"Sentiment: {result['label']} (confidence: {result['confidence']:.2f})")
```

```javascript
// JavaScript fetch
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'This product is amazing!'
  })
});

const result = await response.json();
console.log(`Sentiment: ${result.label} (confidence: ${result.confidence})`);
```

### Batch Prediction

Analyze sentiment for multiple texts in a single request.

#### `POST /predict/batch`

**Request Body**

```json
{
  "texts": [
    "I love this product!",
    "This is terrible quality.",
    "It's okay, nothing special."
  ]
}
```

**Response**

```json
[
  {
    "label": "positive",
    "score": 0.9234,
    "confidence": 0.9234
  },
  {
    "label": "negative",
    "score": 0.8756,
    "confidence": 0.8756
  },
  {
    "label": "neutral",
    "score": 0.7123,
    "confidence": 0.7123
  }
]
```

**Parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | array[string] | Yes | Array of texts to analyze (1-100 items) |

**Response**

Array of prediction objects, same format as single prediction.

**Status Codes**
- `200 OK`: Successful batch prediction
- `400 Bad Request`: Batch size exceeds limit or invalid input
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Model not loaded

**Examples**

```bash
# cURL
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Great product!",
         "Poor quality.",
         "Average experience."
       ]
     }'
```

```python
# Python requests
import requests

texts = [
    "Great product!",
    "Poor quality.", 
    "Average experience."
]

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": texts}
)

results = response.json()
for i, result in enumerate(results):
    print(f"Text {i+1}: {result['label']} (confidence: {result['confidence']:.2f})")
```

### Model Information

Get information about the loaded model.

#### `GET /model/info`

**Response**

```json
{
  "model_type": "PyTorch",
  "model_path": "/app/models",
  "labels": ["negative", "neutral", "positive"],
  "max_length": 512
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | Type of loaded model (PyTorch, ONNX, Mock) |
| `model_path` | string | Path to model files |
| `labels` | array[string] | Available sentiment labels |
| `max_length` | integer | Maximum input sequence length |

**Status Codes**
- `200 OK`: Model information retrieved
- `503 Service Unavailable`: Model not loaded

**Example**

```bash
curl -X GET http://localhost:8000/model/info
```

### Metrics

Get Prometheus metrics for monitoring.

#### `GET /metrics`

**Response**

Prometheus-formatted metrics in plain text.

```
# HELP sentiment_requests_total Total sentiment analysis requests
# TYPE sentiment_requests_total counter
sentiment_requests_total 1234.0

# HELP sentiment_request_duration_seconds Request latency
# TYPE sentiment_request_duration_seconds histogram
sentiment_request_duration_seconds_bucket{le="0.01"} 100.0
sentiment_request_duration_seconds_bucket{le="0.05"} 800.0
sentiment_request_duration_seconds_bucket{le="0.1"} 950.0
sentiment_request_duration_seconds_bucket{le="0.5"} 1000.0
sentiment_request_duration_seconds_bucket{le="+Inf"} 1000.0
sentiment_request_duration_seconds_count 1000.0
sentiment_request_duration_seconds_sum 45.67
```

**Content-Type**: `text/plain; version=0.0.4; charset=utf-8`

**Example**

```bash
curl -X GET http://localhost:8000/metrics
```

## Error Handling

### Error Response Format

All errors return a consistent JSON format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

#### 400 Bad Request

```json
{
  "detail": "Batch size 150 exceeds maximum 100"
}
```

#### 422 Unprocessable Entity

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length",
      "ctx": {"limit_value": 1}
    }
  ]
}
```

#### 500 Internal Server Error

```json
{
  "detail": "Model inference failed"
}
```

#### 503 Service Unavailable

```json
{
  "detail": "Model not loaded"
}
```

#### 504 Gateway Timeout

```json
{
  "detail": "Request timeout"
}
```

## Rate Limiting

Current implementation does not enforce rate limits, but production deployments should implement:

- **Per-IP limits**: 1000 requests/hour
- **Per-API-key limits**: 10000 requests/hour
- **Burst limits**: 100 requests/minute

## Request/Response Examples

### Positive Sentiment

**Request**
```json
{
  "text": "I absolutely love this product! The quality is outstanding and delivery was super fast. Highly recommended!"
}
```

**Response**
```json
{
  "label": "positive",
  "score": 0.9567,
  "confidence": 0.9567
}
```

### Negative Sentiment

**Request**
```json
{
  "text": "Terrible experience. The product broke after one day and customer service was unhelpful. Complete waste of money."
}
```

**Response**
```json
{
  "label": "negative",
  "score": 0.9234,
  "confidence": 0.9234
}
```

### Neutral Sentiment

**Request**
```json
{
  "text": "The product is okay. It does what it's supposed to do, nothing more, nothing less. Average quality for the price."
}
```

**Response**
```json
{
  "label": "neutral",
  "score": 0.7845,
  "confidence": 0.7845
}
```

### Mixed/Ambiguous Text

**Request**
```json
{
  "text": "The product quality is good but the price is too high. Mixed feelings about this purchase."
}
```

**Response**
```json
{
  "label": "neutral",
  "score": 0.6234,
  "confidence": 0.6234
}
```

## SDK Examples

### Python SDK

```python
import requests
from typing import List, Dict, Any

class SentimentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        response = requests.post(
            f"{self.base_url}/predict",
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts."""
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"texts": texts},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        response = requests.get(f"{self.base_url}/healthz")
        response.raise_for_status()
        return response.json()

# Usage
client = SentimentClient()

# Single prediction
result = client.predict("I love this product!")
print(f"Sentiment: {result['label']}")

# Batch prediction
results = client.predict_batch([
    "Great product!",
    "Poor quality.",
    "Average experience."
])

for i, result in enumerate(results):
    print(f"Text {i+1}: {result['label']}")
```

### JavaScript SDK

```javascript
class SentimentClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async predict(text) {
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async predictBatch(texts) {
    const response = await fetch(`${this.baseUrl}/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ texts }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/healthz`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

// Usage
const client = new SentimentClient();

// Single prediction
try {
  const result = await client.predict('I love this product!');
  console.log(`Sentiment: ${result.label}`);
} catch (error) {
  console.error('Prediction failed:', error);
}

// Batch prediction
try {
  const results = await client.predictBatch([
    'Great product!',
    'Poor quality.',
    'Average experience.'
  ]);
  
  results.forEach((result, index) => {
    console.log(`Text ${index + 1}: ${result.label}`);
  });
} catch (error) {
  console.error('Batch prediction failed:', error);
}
```

## Performance Considerations

### Latency Optimization

- **Single requests**: Target <100ms median latency
- **Batch requests**: More efficient for multiple texts
- **Text length**: Shorter texts process faster
- **Concurrent requests**: API handles multiple requests efficiently

### Best Practices

1. **Use batch endpoint** for multiple texts
2. **Keep texts under 500 characters** for optimal speed
3. **Implement client-side caching** for repeated texts
4. **Handle timeouts gracefully** with retry logic
5. **Monitor response times** and adjust accordingly

### Limits

- **Text length**: 1-10,000 characters
- **Batch size**: 1-100 texts per request
- **Request timeout**: 30 seconds for single, 60 seconds for batch
- **Concurrent requests**: No hard limit, but performance may degrade

## Monitoring and Debugging

### Health Monitoring

```python
import time
import requests

def monitor_health(url: str, interval: int = 30):
    """Monitor API health continuously."""
    while True:
        try:
            response = requests.get(f"{url}/healthz", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Service healthy: {data}")
            else:
                print(f"❌ Service unhealthy: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        
        time.sleep(interval)

# Monitor every 30 seconds
monitor_health("http://localhost:8000")
```

### Performance Monitoring

```python
import time
import statistics
from typing import List

def benchmark_api(url: str, texts: List[str], num_requests: int = 100):
    """Benchmark API performance."""
    latencies = []
    
    for i in range(num_requests):
        text = texts[i % len(texts)]
        
        start_time = time.perf_counter()
        response = requests.post(
            f"{url}/predict",
            json={"text": text},
            timeout=30
        )
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        if i % 10 == 0:
            print(f"Progress: {i}/{num_requests}")
    
    # Calculate statistics
    if latencies:
        print(f"Requests: {len(latencies)}")
        print(f"Mean latency: {statistics.mean(latencies):.2f}ms")
        print(f"Median latency: {statistics.median(latencies):.2f}ms")
        print(f"Min latency: {min(latencies):.2f}ms")
        print(f"Max latency: {max(latencies):.2f}ms")

# Benchmark with sample texts
sample_texts = [
    "Great product!",
    "Poor quality.",
    "Average experience.",
    "I love this!",
    "Not recommended."
]

benchmark_api("http://localhost:8000", sample_texts)
```