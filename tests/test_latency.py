"""Latency tests for the sentiment analysis API."""

import asyncio
import statistics
import time
from typing import List
import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.app.main import app
from src.app.models import MockModel, ModelManager


class LatencyTester:
    """Utility class for latency testing."""
    
    def __init__(self, client: TestClient):
        self.client = client
        self.latencies: List[float] = []
    
    def measure_request_latency(self, endpoint: str, payload: dict) -> float:
        """Measure latency of a single request."""
        start_time = time.perf_counter()
        response = self.client.post(endpoint, json=payload)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        self.latencies.append(latency_ms)
        
        # Ensure request was successful
        assert response.status_code == 200
        
        return latency_ms
    
    async def measure_async_latency(self, endpoint: str, payload: dict, base_url: str = "http://testserver") -> float:
        """Measure latency of an async request."""
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            start_time = time.perf_counter()
            response = await client.post(endpoint, json=payload)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            self.latencies.append(latency_ms)
            
            # Ensure request was successful
            assert response.status_code == 200
            
            return latency_ms
    
    def get_statistics(self) -> dict:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        return {
            "count": len(self.latencies),
            "mean": statistics.mean(self.latencies),
            "median": statistics.median(self.latencies),
            "min": min(self.latencies),
            "max": max(self.latencies),
            "p95": statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else max(self.latencies),
            "p99": statistics.quantiles(self.latencies, n=100)[98] if len(self.latencies) >= 100 else max(self.latencies),
        }
    
    def reset(self):
        """Reset latency measurements."""
        self.latencies.clear()


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
async def mock_setup():
    """Setup mock model for testing."""
    mock_manager = ModelManager("./test_models")
    mock_manager.model = MockModel()
    await mock_manager.model.load()
    
    mock_preprocessor = AsyncMock()
    mock_preprocessor.preprocess.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]],
        "features": {"length": 20, "word_count": 4}
    }
    mock_preprocessor.preprocess_batch = AsyncMock(return_value=[
        {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "features": {"length": 20, "word_count": 4}
        }
    ])
    
    with patch('src.app.main.model_manager', mock_manager), \
         patch('src.app.main.preprocessor', mock_preprocessor):
        yield


class TestSingleRequestLatency:
    """Test latency for single requests."""
    
    @pytest.mark.asyncio
    async def test_single_request_latency_target(self, client, mock_setup):
        """Test that single requests meet the <100ms target."""
        tester = LatencyTester(client)
        
        # Test payload
        payload = {"text": "This is a test message for latency testing."}
        
        # Warm up (first request might be slower)
        tester.measure_request_latency("/predict", payload)
        tester.reset()
        
        # Measure actual latencies
        num_requests = 50
        for _ in range(num_requests):
            latency = tester.measure_request_latency("/predict", payload)
            print(f"Request latency: {latency:.2f}ms")
        
        stats = tester.get_statistics()
        print(f"Latency statistics: {stats}")
        
        # Assert median latency is under 100ms
        assert stats["median"] < 100, f"Median latency {stats['median']:.2f}ms exceeds 100ms target"
        
        # Assert 95th percentile is reasonable
        assert stats["p95"] < 120, f"P95 latency {stats['p95']:.2f}ms is too high"
    
    @pytest.mark.asyncio
    async def test_short_text_latency(self, client, mock_setup):
        """Test latency for short text inputs."""
        tester = LatencyTester(client)
        
        # Short text payload
        payload = {"text": "Good"}
        
        # Warm up
        tester.measure_request_latency("/predict", payload)
        tester.reset()
        
        # Measure latencies for short texts
        for _ in range(30):
            tester.measure_request_latency("/predict", payload)
        
        stats = tester.get_statistics()
        print(f"Short text latency statistics: {stats}")
        
        # Short texts should be even faster
        assert stats["median"] < 50, f"Short text median latency {stats['median']:.2f}ms is too high"
    
    @pytest.mark.asyncio
    async def test_long_text_latency(self, client, mock_setup):
        """Test latency for longer text inputs."""
        tester = LatencyTester(client)
        
        # Longer text payload
        long_text = "This is a much longer text that contains multiple sentences and should take more time to process. " * 10
        payload = {"text": long_text}
        
        # Warm up
        tester.measure_request_latency("/predict", payload)
        tester.reset()
        
        # Measure latencies for long texts
        for _ in range(20):
            tester.measure_request_latency("/predict", payload)
        
        stats = tester.get_statistics()
        print(f"Long text latency statistics: {stats}")
        
        # Even long texts should meet the target
        assert stats["median"] < 120, f"Long text median latency {stats['median']:.2f}ms exceeds acceptable limit"


class TestBatchRequestLatency:
    """Test latency for batch requests."""
    
    @pytest.mark.asyncio
    async def test_batch_request_latency(self, client, mock_setup):
        """Test latency for batch requests."""
        tester = LatencyTester(client)
        
        # Batch payload
        payload = {
            "texts": [
                "This is test message 1",
                "This is test message 2", 
                "This is test message 3",
                "This is test message 4",
                "This is test message 5"
            ]
        }
        
        # Warm up
        tester.measure_request_latency("/predict/batch", payload)
        tester.reset()
        
        # Measure batch latencies
        for _ in range(20):
            tester.measure_request_latency("/predict/batch", payload)
        
        stats = tester.get_statistics()
        print(f"Batch latency statistics: {stats}")
        
        # Batch requests should be efficient per item
        per_item_latency = stats["median"] / len(payload["texts"])
        assert per_item_latency < 50, f"Per-item batch latency {per_item_latency:.2f}ms is too high"


class TestConcurrentRequestLatency:
    """Test latency under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_request_latency(self, mock_setup):
        """Test latency with concurrent requests."""
        async def make_request(session_id: int) -> float:
            async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
                payload = {"text": f"Test message from session {session_id}"}
                
                start_time = time.perf_counter()
                response = await client.post("/predict", json=payload)
                end_time = time.perf_counter()
                
                assert response.status_code == 200
                return (end_time - start_time) * 1000
        
        # Test with multiple concurrent requests
        num_concurrent = 10
        num_rounds = 5
        all_latencies = []
        
        for round_num in range(num_rounds):
            print(f"Round {round_num + 1}/{num_rounds}")
            
            # Create concurrent tasks
            tasks = [make_request(i) for i in range(num_concurrent)]
            
            # Execute concurrently
            round_latencies = await asyncio.gather(*tasks)
            all_latencies.extend(round_latencies)
            
            print(f"Round {round_num + 1} median latency: {statistics.median(round_latencies):.2f}ms")
        
        # Calculate overall statistics
        overall_stats = {
            "count": len(all_latencies),
            "mean": statistics.mean(all_latencies),
            "median": statistics.median(all_latencies),
            "min": min(all_latencies),
            "max": max(all_latencies),
            "p95": statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies),
        }
        
        print(f"Concurrent request statistics: {overall_stats}")
        
        # Even under concurrent load, median should be reasonable
        assert overall_stats["median"] < 120, f"Concurrent median latency {overall_stats['median']:.2f}ms is too high"


class TestLatencyRequirements:
    """Test specific latency requirements."""
    
    @pytest.mark.asyncio
    async def test_cpu_latency_requirement(self, client, mock_setup):
        """Test that median latency on CPU is under 120ms (requirement)."""
        tester = LatencyTester(client)
        
        # Standard test payload
        payload = {"text": "This is a standard test message for latency validation."}
        
        # Warm up requests
        for _ in range(5):
            tester.measure_request_latency("/predict", payload)
        tester.reset()
        
        # Measure production-like latencies
        num_requests = 100
        for i in range(num_requests):
            latency = tester.measure_request_latency("/predict", payload)
            if i % 20 == 0:
                print(f"Progress: {i}/{num_requests}, Current latency: {latency:.2f}ms")
        
        stats = tester.get_statistics()
        print(f"Final latency statistics: {stats}")
        
        # This is the critical requirement test
        assert stats["median"] < 120, f"REQUIREMENT FAILED: Median latency {stats['median']:.2f}ms exceeds 120ms limit"
        
        # Additional checks for good performance
        if stats["median"] < 100:
            print("✅ Excellent: Median latency under 100ms target")
        elif stats["median"] < 120:
            print("✅ Good: Median latency under 120ms requirement")
        
        # Check that most requests are fast
        fast_requests = sum(1 for lat in tester.latencies if lat < 100)
        fast_percentage = (fast_requests / len(tester.latencies)) * 100
        print(f"Requests under 100ms: {fast_percentage:.1f}%")
        
        assert fast_percentage > 70, f"Only {fast_percentage:.1f}% of requests under 100ms"
    
    @pytest.mark.asyncio
    async def test_health_check_latency(self, client, mock_setup):
        """Test that health check is very fast."""
        tester = LatencyTester(client)
        
        # Measure health check latencies
        for _ in range(20):
            start_time = time.perf_counter()
            response = self.client.get("/healthz")
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            tester.latencies.append(latency_ms)
            
            assert response.status_code == 200
        
        stats = tester.get_statistics()
        print(f"Health check latency statistics: {stats}")
        
        # Health checks should be very fast
        assert stats["median"] < 10, f"Health check median latency {stats['median']:.2f}ms is too high"


class TestLatencyUnderLoad:
    """Test latency behavior under different load conditions."""
    
    @pytest.mark.asyncio
    async def test_sustained_load_latency(self, mock_setup):
        """Test latency under sustained load."""
        async def sustained_requester(duration_seconds: int, request_rate: float) -> List[float]:
            """Make requests at a sustained rate."""
            latencies = []
            end_time = time.time() + duration_seconds
            
            async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
                while time.time() < end_time:
                    start_time = time.perf_counter()
                    
                    response = await client.post(
                        "/predict",
                        json={"text": "Sustained load test message"}
                    )
                    
                    request_end_time = time.perf_counter()
                    latency_ms = (request_end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    
                    assert response.status_code == 200
                    
                    # Wait to maintain request rate
                    await asyncio.sleep(1.0 / request_rate)
            
            return latencies
        
        # Test sustained load for 10 seconds at 5 requests/second
        print("Testing sustained load (10 seconds, 5 req/s)...")
        latencies = await sustained_requester(duration_seconds=10, request_rate=5.0)
        
        stats = {
            "count": len(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        }
        
        print(f"Sustained load statistics: {stats}")
        
        # Performance should not degrade significantly under sustained load
        assert stats["median"] < 150, f"Sustained load median latency {stats['median']:.2f}ms is too high"
        assert stats["max"] < 500, f"Sustained load max latency {stats['max']:.2f}ms indicates performance issues"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])