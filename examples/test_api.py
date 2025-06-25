#!/usr/bin/env python3
"""Example script to test the sentiment API."""

import asyncio
import json
import time
from typing import List, Dict, Any

import httpx
import requests


class SentimentAPITester:
    """Test the sentiment API with various scenarios."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint."""
        test_cases = [
            {"text": "I love this product! It's amazing!", "expected": "positive"},
            {"text": "This is terrible quality. Very disappointed.", "expected": "negative"},
            {"text": "It's okay, nothing special but does the job.", "expected": "neutral"},
            {"text": "Best purchase ever! Highly recommend! 🌟", "expected": "positive"},
            {"text": "Worst experience. Complete waste of money.", "expected": "negative"},
        ]
        
        success_count = 0
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                start_time = time.perf_counter()
                response = requests.post(
                    f"{self.base_url}/predict",
                    json={"text": test_case["text"]},
                    timeout=30
                )
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    predicted_label = data["label"]
                    confidence = data["confidence"]
                    
                    print(f"✅ Test {i}: '{test_case['text'][:50]}...'")
                    print(f"   Predicted: {predicted_label} (confidence: {confidence:.3f})")
                    print(f"   Latency: {latency_ms:.2f}ms")
                    
                    if latency_ms < 120:  # Check latency requirement
                        print(f"   ✅ Latency under 120ms requirement")
                    else:
                        print(f"   ⚠️ Latency {latency_ms:.2f}ms exceeds 120ms requirement")
                    
                    success_count += 1
                else:
                    print(f"❌ Test {i} failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"❌ Test {i} error: {e}")
        
        print(f"\nSingle prediction tests: {success_count}/{len(test_cases)} passed")
        return success_count == len(test_cases)
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint."""
        test_texts = [
            "Great product, love it!",
            "Poor quality, not recommended.",
            "Average experience, nothing special.",
            "Excellent service and delivery!",
            "Terrible customer support."
        ]
        
        try:
            start_time = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={"texts": test_texts},
                timeout=60
            )
            end_time = time.perf_counter()
            
            total_latency_ms = (end_time - start_time) * 1000
            per_item_latency = total_latency_ms / len(test_texts)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Batch prediction test passed")
                print(f"   Processed {len(test_texts)} texts")
                print(f"   Total latency: {total_latency_ms:.2f}ms")
                print(f"   Per-item latency: {per_item_latency:.2f}ms")
                
                for i, (text, result) in enumerate(zip(test_texts, data)):
                    print(f"   {i+1}. '{text[:30]}...' → {result['label']} ({result['confidence']:.3f})")
                
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint."""
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test metrics endpoint."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            if response.status_code == 200:
                metrics_text = response.text
                print(f"✅ Metrics endpoint working")
                print(f"   Content-Type: {response.headers.get('content-type')}")
                print(f"   Metrics preview: {metrics_text[:200]}...")
                return True
            else:
                print(f"❌ Metrics failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics error: {e}")
            return False
    
    async def test_concurrent_requests(self, num_requests: int = 20) -> bool:
        """Test concurrent request handling."""
        print(f"\n🔄 Testing {num_requests} concurrent requests...")
        
        async def make_request(session: httpx.AsyncClient, request_id: int) -> Dict[str, Any]:
            start_time = time.perf_counter()
            try:
                response = await session.post(
                    f"{self.base_url}/predict",
                    json={"text": f"Test message {request_id} for concurrent testing"},
                    timeout=30
                )
                end_time = time.perf_counter()
                
                return {
                    "request_id": request_id,
                    "latency_ms": (end_time - start_time) * 1000,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "request_id": request_id,
                    "latency_ms": (end_time - start_time) * 1000,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                }
        
        try:
            async with httpx.AsyncClient() as client:
                tasks = [make_request(client, i) for i in range(num_requests)]
                results = await asyncio.gather(*tasks)
            
            successful_results = [r for r in results if r["success"]]
            latencies = [r["latency_ms"] for r in successful_results]
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)
                
                print(f"✅ Concurrent test completed")
                print(f"   Successful requests: {len(successful_results)}/{num_requests}")
                print(f"   Average latency: {avg_latency:.2f}ms")
                print(f"   Min latency: {min_latency:.2f}ms")
                print(f"   Max latency: {max_latency:.2f}ms")
                
                return len(successful_results) >= num_requests * 0.95  # 95% success rate
            else:
                print(f"❌ No successful concurrent requests")
                return False
                
        except Exception as e:
            print(f"❌ Concurrent test error: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs."""
        error_test_cases = [
            {"json": {"text": ""}, "description": "empty text"},
            {"json": {"text": "a" * 15000}, "description": "text too long"},
            {"json": {"invalid": "field"}, "description": "missing text field"},
            {"json": {"texts": []}, "description": "empty batch", "endpoint": "/predict/batch"},
            {"json": {"texts": ["a"] * 150}, "description": "batch too large", "endpoint": "/predict/batch"},
        ]
        
        success_count = 0
        
        for i, test_case in enumerate(error_test_cases, 1):
            endpoint = test_case.get("endpoint", "/predict")
            
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=test_case["json"],
                    timeout=30
                )
                
                if response.status_code in [400, 422]:  # Expected error codes
                    print(f"✅ Error test {i} ({test_case['description']}): {response.status_code}")
                    success_count += 1
                else:
                    print(f"❌ Error test {i} ({test_case['description']}): Expected 400/422, got {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error test {i} exception: {e}")
        
        print(f"\nError handling tests: {success_count}/{len(error_test_cases)} passed")
        return success_count == len(error_test_cases)
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        print("🧪 Starting API Tests")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Model Info", self.test_model_info),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("Error Handling", self.test_error_handling),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name} Test")
            print("-" * 30)
            result = test_func()
            results.append((test_name, result))
        
        # Run async test separately
        print(f"\n📋 Running Concurrent Requests Test")
        print("-" * 30)
        concurrent_result = asyncio.run(self.test_concurrent_requests())
        results.append(("Concurrent Requests", concurrent_result))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<25} {status}")
            if result:
                passed += 1
        
        print("-" * 50)
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("⚠️ Some tests failed. Check the API setup.")
            return False


def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the sentiment API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--concurrent", type=int, default=20, help="Number of concurrent requests to test")
    
    args = parser.parse_args()
    
    tester = SentimentAPITester(args.url)
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())