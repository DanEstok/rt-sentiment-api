# Benchmark Documentation

## Overview

This document covers benchmarking and performance testing for the RT Sentiment API. The system is designed to achieve sub-100ms median latency with high throughput and reliability.

## Performance Targets

### Latency Requirements

- **Primary Target**: <100ms median latency
- **Hard Requirement**: <120ms median latency on CPU
- **P95 Target**: <150ms
- **P99 Target**: <300ms

### Throughput Targets

- **Single Instance**: 100+ requests/second
- **Batch Processing**: 500+ texts/second
- **Concurrent Users**: 50+ simultaneous users

### Reliability Targets

- **Success Rate**: >99.5%
- **Uptime**: >99.9%
- **Error Rate**: <0.5%

## Benchmarking Tools

### 1. Built-in Benchmark Script

The `scripts/benchmark.py` provides comprehensive benchmarking capabilities:

```bash
# Quick benchmark
python scripts/benchmark.py --requests 100 --concurrency 10

# Comprehensive benchmark
python scripts/benchmark.py \
  --requests 1000 \
  --concurrency 20 \
  --users 15 \
  --duration 120 \
  --test all \
  --output results.json
```

### 2. Load Testing with Locust

```python
# Install Locust
pip install locust

# Run load test
locust -f scripts/benchmark.py --host http://localhost:8000
```

### 3. HTTP Benchmarking with wrk

```bash
# Install wrk
sudo apt-get install wrk

# Single endpoint test
wrk -t12 -c400 -d30s --script=post.lua http://localhost:8000/predict

# post.lua script
wrk.method = "POST"
wrk.body = '{"text": "This is a test message for benchmarking"}'
wrk.headers["Content-Type"] = "application/json"
```

## Benchmark Scenarios

### 1. Single Request Latency

Test individual request performance:

```python
import time
import requests
import statistics

def benchmark_single_requests(url: str, num_requests: int = 100):
    """Benchmark single request latency."""
    latencies = []
    
    for i in range(num_requests):
        start_time = time.perf_counter()
        
        response = requests.post(
            f"{url}/predict",
            json={"text": f"Test message {i} for latency benchmarking"},
            timeout=30
        )
        
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
    
    return {
        "count": len(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "p99": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
    }

# Run benchmark
results = benchmark_single_requests("http://localhost:8000")
print(f"Median latency: {results['median']:.2f}ms")
```

### 2. Concurrent Request Testing

Test performance under concurrent load:

```python
import asyncio
import aiohttp
import time

async def concurrent_benchmark(url: str, num_requests: int = 100, concurrency: int = 10):
    """Benchmark concurrent request handling."""
    
    async def make_request(session, request_id):
        start_time = time.perf_counter()
        
        async with session.post(
            f"{url}/predict",
            json={"text": f"Concurrent test message {request_id}"},
            timeout=30
        ) as response:
            end_time = time.perf_counter()
            
            return {
                "request_id": request_id,
                "latency_ms": (end_time - start_time) * 1000,
                "status": response.status,
                "success": response.status == 200
            }
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(session, request_id):
        async with semaphore:
            return await make_request(session, request_id)
    
    # Run concurrent requests
    async with aiohttp.ClientSession() as session:
        tasks = [
            bounded_request(session, i) 
            for i in range(num_requests)
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    latencies = [r["latency_ms"] for r in successful_results]
    
    return {
        "total_requests": num_requests,
        "successful_requests": len(successful_results),
        "failed_requests": num_requests - len(successful_results),
        "total_time": end_time - start_time,
        "requests_per_second": num_requests / (end_time - start_time),
        "latencies": latencies
    }

# Run concurrent benchmark
results = asyncio.run(concurrent_benchmark("http://localhost:8000", 200, 20))
print(f"RPS: {results['requests_per_second']:.2f}")
```

### 3. Batch Processing Performance

Test batch endpoint efficiency:

```python
def benchmark_batch_processing(url: str, batch_sizes: list = [1, 5, 10, 20, 50]):
    """Benchmark batch processing performance."""
    results = {}
    
    for batch_size in batch_sizes:
        texts = [f"Batch test message {i}" for i in range(batch_size)]
        latencies = []
        
        for _ in range(20):  # 20 iterations per batch size
            start_time = time.perf_counter()
            
            response = requests.post(
                f"{url}/predict/batch",
                json={"texts": texts},
                timeout=60
            )
            
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                total_latency = (end_time - start_time) * 1000
                per_item_latency = total_latency / batch_size
                latencies.append(per_item_latency)
        
        if latencies:
            results[batch_size] = {
                "mean_per_item_ms": statistics.mean(latencies),
                "median_per_item_ms": statistics.median(latencies),
                "throughput_items_per_second": 1000 / statistics.median(latencies)
            }
    
    return results

# Run batch benchmark
batch_results = benchmark_batch_processing("http://localhost:8000")
for batch_size, stats in batch_results.items():
    print(f"Batch size {batch_size}: {stats['median_per_item_ms']:.2f}ms per item")
```

### 4. Sustained Load Testing

Test performance under sustained load:

```python
async def sustained_load_test(url: str, duration_seconds: int = 300, rps: float = 10.0):
    """Test sustained load over time."""
    
    async def make_sustained_requests():
        latencies = []
        request_count = 0
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration_seconds:
                request_start = time.perf_counter()
                
                async with session.post(
                    f"{url}/predict",
                    json={"text": f"Sustained load test {request_count}"},
                    timeout=30
                ) as response:
                    request_end = time.perf_counter()
                    
                    if response.status == 200:
                        latency_ms = (request_end - request_start) * 1000
                        latencies.append(latency_ms)
                
                request_count += 1
                
                # Wait to maintain target RPS
                await asyncio.sleep(1.0 / rps)
        
        return latencies
    
    latencies = await make_sustained_requests()
    
    # Analyze latency over time
    window_size = len(latencies) // 10  # 10 time windows
    windows = []
    
    for i in range(0, len(latencies), window_size):
        window = latencies[i:i + window_size]
        if window:
            windows.append({
                "window": i // window_size,
                "median_latency": statistics.median(window),
                "mean_latency": statistics.mean(window)
            })
    
    return windows

# Run sustained load test
windows = asyncio.run(sustained_load_test("http://localhost:8000", 120, 5.0))
for window in windows:
    print(f"Window {window['window']}: {window['median_latency']:.2f}ms median")
```

## Performance Analysis

### Latency Distribution Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_latency_distribution(latencies: list):
    """Analyze and visualize latency distribution."""
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 99, 99.9]
    values = np.percentile(latencies, percentiles)
    
    print("Latency Percentiles:")
    for p, v in zip(percentiles, values):
        print(f"P{p}: {v:.2f}ms")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution')
    plt.axvline(np.median(latencies), color='red', linestyle='--', label=f'Median: {np.median(latencies):.2f}ms')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(percentiles, values, 'bo-')
    plt.xlabel('Percentile')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Percentiles')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('latency_analysis.png')
    plt.show()

# Example usage
latencies = [45.2, 52.1, 48.9, 67.3, 51.8, 49.2, 55.4, 62.1, 47.8, 53.6]
analyze_latency_distribution(latencies)
```

### Throughput Analysis

```python
def analyze_throughput(results: dict):
    """Analyze throughput characteristics."""
    
    print("Throughput Analysis:")
    print(f"Total Requests: {results['total_requests']}")
    print(f"Successful Requests: {results['successful_requests']}")
    print(f"Success Rate: {results['successful_requests']/results['total_requests']*100:.2f}%")
    print(f"Requests per Second: {results['requests_per_second']:.2f}")
    print(f"Total Time: {results['total_time']:.2f}s")
    
    if results['latencies']:
        latencies = results['latencies']
        print(f"Mean Latency: {statistics.mean(latencies):.2f}ms")
        print(f"Median Latency: {statistics.median(latencies):.2f}ms")
        print(f"P95 Latency: {np.percentile(latencies, 95):.2f}ms")

# Example usage
throughput_results = {
    'total_requests': 1000,
    'successful_requests': 995,
    'total_time': 45.2,
    'requests_per_second': 22.1,
    'latencies': [45.2, 52.1, 48.9, 67.3, 51.8]
}
analyze_throughput(throughput_results)
```

## Benchmark Results Format

### Standard Results Table

```markdown
## Benchmark Results

| Test Type | Requests | Success Rate | RPS | Avg Latency | Median | P95 | P99 | Min | Max |
|-----------|----------|--------------|-----|-------------|--------|-----|-----|-----|-----|
| Single    | 1000     | 99.8%        | 45.2| 52.3ms      | 48.1ms | 89.2ms | 156.7ms | 23.4ms | 234.5ms |
| Concurrent| 1000     | 99.5%        | 67.8| 67.8ms      | 62.3ms | 125.4ms | 189.2ms | 34.1ms | 287.9ms |
| Batch-5   | 200      | 100.0%       | 234.5| 21.3ms     | 19.8ms | 34.2ms | 45.6ms | 15.2ms | 67.8ms |
| Sustained | 1500     | 99.7%        | 25.0| 58.9ms      | 54.2ms | 98.7ms | 145.3ms | 28.9ms | 198.4ms |

### Performance Analysis

**Single Requests:**
✅ Median latency under 100ms target
✅ Excellent success rate
⚠️ P99 latency could be improved

**Concurrent Requests:**
✅ Good throughput under load
✅ Latency remains acceptable
✅ High success rate maintained

**Batch Processing:**
✅ Excellent per-item latency
✅ High throughput for batch operations
✅ Perfect success rate

**Sustained Load:**
✅ Stable performance over time
✅ Consistent latency distribution
✅ Reliable under continuous load
```

## Performance Optimization

### Model Optimizations

1. **ONNX Runtime**: Use optimized ONNX models for inference
2. **Quantization**: Reduce model precision for faster inference
3. **Batch Processing**: Process multiple requests together
4. **Model Caching**: Keep models in memory

### API Optimizations

1. **Async Processing**: Use async/await for non-blocking operations
2. **Connection Pooling**: Reuse HTTP connections
3. **Request Batching**: Combine multiple requests
4. **Response Caching**: Cache frequent predictions

### Infrastructure Optimizations

1. **CPU Optimization**: Use optimized CPU instances
2. **Memory Management**: Sufficient RAM for model loading
3. **Network Optimization**: Low-latency networking
4. **Load Balancing**: Distribute requests across instances

## Monitoring and Alerting

### Key Metrics to Monitor

```python
# Prometheus metrics
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')
REQUEST_COUNT = Counter('requests_total', 'Total requests')
ERROR_RATE = Counter('errors_total', 'Total errors')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')
```

### Alert Thresholds

```yaml
# Alerting rules
groups:
  - name: sentiment-api
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, request_latency_seconds) > 0.15
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) / rate(requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: LowThroughput
        expr: rate(requests_total[5m]) < 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low request throughput"
```

## Continuous Benchmarking

### Automated Performance Testing

```yaml
# GitHub Actions workflow for performance testing
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  push:
    branches: [main]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Start API
        run: |
          python src/app/main.py &
          sleep 30
          
      - name: Run benchmarks
        run: |
          python scripts/benchmark.py \
            --requests 500 \
            --concurrency 10 \
            --output performance-results.json
            
      - name: Check performance thresholds
        run: |
          python -c "
          import json
          with open('performance-results.json') as f:
              results = json.load(f)
          
          # Check latency threshold
          median_latency = results['results']['Async']['median_latency_ms']
          assert median_latency < 120, f'Median latency {median_latency}ms exceeds 120ms threshold'
          
          # Check success rate
          success_rate = results['results']['Async']['successful_requests'] / results['results']['Async']['total_requests']
          assert success_rate > 0.995, f'Success rate {success_rate} below 99.5% threshold'
          
          print('✅ All performance thresholds met')
          "
          
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: performance-results.json
```

### Performance Regression Detection

```python
def detect_performance_regression(current_results: dict, baseline_results: dict, threshold: float = 0.1):
    """Detect performance regressions compared to baseline."""
    
    regressions = []
    
    # Check latency regression
    current_latency = current_results['median_latency_ms']
    baseline_latency = baseline_results['median_latency_ms']
    
    if current_latency > baseline_latency * (1 + threshold):
        regressions.append(f"Latency regression: {current_latency:.2f}ms vs {baseline_latency:.2f}ms baseline")
    
    # Check throughput regression
    current_rps = current_results['requests_per_second']
    baseline_rps = baseline_results['requests_per_second']
    
    if current_rps < baseline_rps * (1 - threshold):
        regressions.append(f"Throughput regression: {current_rps:.2f} RPS vs {baseline_rps:.2f} RPS baseline")
    
    # Check success rate regression
    current_success = current_results['successful_requests'] / current_results['total_requests']
    baseline_success = baseline_results['successful_requests'] / baseline_results['total_requests']
    
    if current_success < baseline_success * (1 - threshold/10):  # Stricter threshold for success rate
        regressions.append(f"Success rate regression: {current_success:.3f} vs {baseline_success:.3f} baseline")
    
    return regressions

# Example usage
current = {'median_latency_ms': 65.2, 'requests_per_second': 42.1, 'successful_requests': 995, 'total_requests': 1000}
baseline = {'median_latency_ms': 58.7, 'requests_per_second': 45.3, 'successful_requests': 998, 'total_requests': 1000}

regressions = detect_performance_regression(current, baseline)
if regressions:
    print("⚠️ Performance regressions detected:")
    for regression in regressions:
        print(f"  - {regression}")
else:
    print("✅ No performance regressions detected")
```

## Best Practices

### Benchmarking Best Practices

1. **Warm-up Period**: Allow system to warm up before measuring
2. **Multiple Runs**: Average results across multiple benchmark runs
3. **Realistic Data**: Use production-like test data
4. **Environment Consistency**: Test in consistent environments
5. **Load Patterns**: Test various load patterns and scenarios

### Performance Testing Guidelines

1. **Baseline Establishment**: Establish performance baselines
2. **Regular Testing**: Run performance tests regularly
3. **Regression Detection**: Automatically detect performance regressions
4. **Threshold Monitoring**: Set and monitor performance thresholds
5. **Continuous Improvement**: Continuously optimize based on results