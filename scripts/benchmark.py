#!/usr/bin/env python3
"""Benchmark script for the sentiment analysis API."""

import argparse
import asyncio
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import httpx
import requests
from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging


class SentimentAPIUser(HttpUser):
    """Locust user for load testing the sentiment API."""
    
    wait_time = between(0.1, 0.5)  # Wait 100-500ms between requests
    
    def on_start(self):
        """Initialize test data."""
        self.test_texts = [
            "I love this product! It's amazing!",
            "This is terrible, worst purchase ever.",
            "It's okay, nothing special but does the job.",
            "Absolutely fantastic service and quality!",
            "Not worth the money, very disappointed.",
            "Pretty good overall, would recommend.",
            "Horrible experience, will never buy again.",
            "Excellent quality and fast delivery!",
            "Average product, meets basic expectations.",
            "Outstanding! Exceeded all my expectations!"
        ]
    
    @task(10)
    def predict_single(self):
        """Test single prediction endpoint."""
        import random
        text = random.choice(self.test_texts)
        
        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "label" in data and "score" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def predict_batch(self):
        """Test batch prediction endpoint."""
        import random
        batch_size = random.randint(2, 5)
        texts = random.sample(self.test_texts, batch_size)
        
        with self.client.post(
            "/predict/batch",
            json={"texts": texts},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data) == len(texts) and all("label" in item for item in data):
                    response.success()
                else:
                    response.failure("Invalid batch response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/healthz", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Service not healthy")
            else:
                response.failure(f"Status code: {response.status_code}")


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_texts = [
            "I love this product! It's amazing!",
            "This is terrible, worst purchase ever.",
            "It's okay, nothing special but does the job.",
            "Absolutely fantastic service and quality!",
            "Not worth the money, very disappointed.",
            "Pretty good overall, would recommend.",
            "Horrible experience, will never buy again.",
            "Excellent quality and fast delivery!",
            "Average product, meets basic expectations.",
            "Outstanding! Exceeded all my expectations!"
        ]
    
    def check_service_health(self) -> bool:
        """Check if the service is healthy before benchmarking."""
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except Exception as e:
            print(f"Health check failed: {e}")
        return False
    
    async def async_benchmark(self, num_requests: int = 100, concurrency: int = 10) -> Dict[str, Any]:
        """Run async benchmark."""
        print(f"Running async benchmark: {num_requests} requests, {concurrency} concurrent")
        
        async def make_request(session: httpx.AsyncClient, text: str) -> Dict[str, Any]:
            start_time = time.perf_counter()
            try:
                response = await session.post(
                    "/predict",
                    json={"text": text},
                    timeout=30.0
                )
                end_time = time.perf_counter()
                
                return {
                    "latency_ms": (end_time - start_time) * 1000,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "latency_ms": (end_time - start_time) * 1000,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                }
        
        # Prepare requests
        import random
        requests_data = [random.choice(self.test_texts) for _ in range(num_requests)]
        
        # Run benchmark
        start_time = time.perf_counter()
        results = []
        
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            # Process in batches to control concurrency
            for i in range(0, num_requests, concurrency):
                batch = requests_data[i:i + concurrency]
                tasks = [make_request(client, text) for text in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, dict):
                        results.append(result)
                    else:
                        results.append({
                            "latency_ms": 0,
                            "status_code": 0,
                            "success": False,
                            "error": str(result)
                        })
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        latencies = [r["latency_ms"] for r in successful_results]
        
        if latencies:
            stats = {
                "total_requests": num_requests,
                "successful_requests": len(successful_results),
                "failed_requests": num_requests - len(successful_results),
                "total_time_seconds": total_time,
                "requests_per_second": num_requests / total_time,
                "average_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
            }
        else:
            stats = {
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": num_requests,
                "total_time_seconds": total_time,
                "requests_per_second": 0,
                "average_latency_ms": 0,
                "median_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
            }
        
        return stats
    
    def sync_benchmark(self, num_requests: int = 100, num_workers: int = 10) -> Dict[str, Any]:
        """Run synchronous benchmark with threading."""
        print(f"Running sync benchmark: {num_requests} requests, {num_workers} workers")
        
        def make_request(text: str) -> Dict[str, Any]:
            start_time = time.perf_counter()
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json={"text": text},
                    timeout=30.0
                )
                end_time = time.perf_counter()
                
                return {
                    "latency_ms": (end_time - start_time) * 1000,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "latency_ms": (end_time - start_time) * 1000,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                }
        
        # Prepare requests
        import random
        requests_data = [random.choice(self.test_texts) for _ in range(num_requests)]
        
        # Run benchmark
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(make_request, requests_data))
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        latencies = [r["latency_ms"] for r in successful_results]
        
        if latencies:
            stats = {
                "total_requests": num_requests,
                "successful_requests": len(successful_results),
                "failed_requests": num_requests - len(successful_results),
                "total_time_seconds": total_time,
                "requests_per_second": num_requests / total_time,
                "average_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
            }
        else:
            stats = {
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": num_requests,
                "total_time_seconds": total_time,
                "requests_per_second": 0,
                "average_latency_ms": 0,
                "median_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
            }
        
        return stats
    
    def locust_benchmark(self, num_users: int = 10, spawn_rate: float = 1.0, run_time: int = 60) -> Dict[str, Any]:
        """Run Locust-based load test."""
        print(f"Running Locust benchmark: {num_users} users, {spawn_rate} spawn rate, {run_time}s duration")
        
        # Setup Locust environment
        env = Environment(user_classes=[SentimentAPIUser])
        env.create_local_runner()
        
        # Start load test
        env.runner.start(user_count=num_users, spawn_rate=spawn_rate)
        
        # Run for specified duration
        import gevent
        gevent.spawn_later(run_time, lambda: env.runner.quit())
        env.runner.greenlet.join()
        
        # Get statistics
        stats = env.stats.total
        
        return {
            "total_requests": stats.num_requests,
            "successful_requests": stats.num_requests - stats.num_failures,
            "failed_requests": stats.num_failures,
            "requests_per_second": stats.total_rps,
            "average_latency_ms": stats.avg_response_time,
            "median_latency_ms": stats.median_response_time,
            "min_latency_ms": stats.min_response_time,
            "max_latency_ms": stats.max_response_time,
            "p95_latency_ms": stats.get_response_time_percentile(0.95),
            "p99_latency_ms": stats.get_response_time_percentile(0.99),
        }
    
    def print_markdown_table(self, results: List[Dict[str, Any]], test_names: List[str]) -> None:
        """Print results in markdown table format."""
        print("\n## Benchmark Results\n")
        
        # Table header
        print("| Test | Requests | Success Rate | RPS | Avg Latency | Median | P95 | P99 | Min | Max |")
        print("|------|----------|--------------|-----|-------------|--------|-----|-----|-----|-----|")
        
        # Table rows
        for test_name, result in zip(test_names, results):
            success_rate = (result["successful_requests"] / result["total_requests"]) * 100 if result["total_requests"] > 0 else 0
            
            print(f"| {test_name} | {result['total_requests']} | {success_rate:.1f}% | "
                  f"{result['requests_per_second']:.1f} | {result['average_latency_ms']:.1f}ms | "
                  f"{result['median_latency_ms']:.1f}ms | {result['p95_latency_ms']:.1f}ms | "
                  f"{result['p99_latency_ms']:.1f}ms | {result['min_latency_ms']:.1f}ms | "
                  f"{result['max_latency_ms']:.1f}ms |")
        
        print("\n### Performance Analysis\n")
        
        for test_name, result in zip(test_names, results):
            print(f"**{test_name}:**")
            
            # Check latency targets
            if result["median_latency_ms"] < 100:
                print("✅ Median latency under 100ms target")
            elif result["median_latency_ms"] < 120:
                print("⚠️ Median latency under 120ms requirement but above 100ms target")
            else:
                print("❌ Median latency exceeds 120ms requirement")
            
            # Check success rate
            success_rate = (result["successful_requests"] / result["total_requests"]) * 100 if result["total_requests"] > 0 else 0
            if success_rate >= 99:
                print("✅ Excellent success rate")
            elif success_rate >= 95:
                print("⚠️ Good success rate")
            else:
                print("❌ Poor success rate")
            
            print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark the sentiment analysis API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests for sync/async tests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--users", type=int, default=10, help="Number of users for Locust test")
    parser.add_argument("--spawn-rate", type=float, default=1.0, help="Spawn rate for Locust test")
    parser.add_argument("--duration", type=int, default=60, help="Duration for Locust test (seconds)")
    parser.add_argument("--test", choices=["async", "sync", "locust", "all"], default="all", help="Test type to run")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.url)
    
    # Check service health
    print("Checking service health...")
    if not runner.check_service_health():
        print("❌ Service is not healthy. Please start the API server first.")
        return 1
    print("✅ Service is healthy")
    
    results = []
    test_names = []
    
    # Run tests based on selection
    if args.test in ["async", "all"]:
        print("\n" + "="*50)
        print("ASYNC BENCHMARK")
        print("="*50)
        async_result = asyncio.run(runner.async_benchmark(args.requests, args.concurrency))
        results.append(async_result)
        test_names.append("Async")
    
    if args.test in ["sync", "all"]:
        print("\n" + "="*50)
        print("SYNC BENCHMARK")
        print("="*50)
        sync_result = runner.sync_benchmark(args.requests, args.concurrency)
        results.append(sync_result)
        test_names.append("Sync")
    
    if args.test in ["locust", "all"]:
        try:
            print("\n" + "="*50)
            print("LOCUST BENCHMARK")
            print("="*50)
            locust_result = runner.locust_benchmark(args.users, args.spawn_rate, args.duration)
            results.append(locust_result)
            test_names.append("Locust")
        except ImportError:
            print("⚠️ Locust not available, skipping Locust benchmark")
        except Exception as e:
            print(f"⚠️ Locust benchmark failed: {e}")
    
    # Print results
    if results:
        runner.print_markdown_table(results, test_names)
        
        # Save to file if requested
        if args.output:
            output_data = {
                "timestamp": time.time(),
                "config": vars(args),
                "results": dict(zip(test_names, results))
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n📁 Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())