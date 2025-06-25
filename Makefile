.PHONY: help install install-dev test lint format clean docker-build docker-run train benchmark deploy

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev,benchmark,training]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-performance: ## Run performance tests
	pytest tests/test_latency.py -v -s

lint: ## Run linting
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format: ## Format code
	black src/ tests/ scripts/
	ruff check src/ tests/ scripts/ --fix

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

docker-build: ## Build Docker images
	docker build -f docker/Dockerfile.training -t sentiment-training .
	docker build -f docker/Dockerfile.inference -t sentiment-api .

docker-run: ## Run API with Docker
	docker run -p 8000:8000 -v $(PWD)/models:/app/models:ro sentiment-api

docker-compose-up: ## Start services with Docker Compose
	docker-compose up -d

docker-compose-down: ## Stop services with Docker Compose
	docker-compose down

train: ## Train model with default settings
	python src/training/train.py --dataset twitter --epochs 3 --export

train-gpu: ## Train model with GPU using Docker
	docker run --gpus all -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data sentiment-training python src/training/train.py --dataset twitter --epochs 3 --export

benchmark: ## Run benchmark tests
	python scripts/benchmark.py --requests 100 --concurrency 10

benchmark-full: ## Run comprehensive benchmark
	python scripts/benchmark.py --requests 1000 --concurrency 20 --users 15 --duration 120 --test all --output benchmark_results.json

dev-server: ## Start development server
	uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

health-check: ## Check API health
	curl -f http://localhost:8000/healthz || echo "API not responding"

test-api: ## Test API endpoints
	curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "I love this product!"}'

deploy-k8s: ## Deploy to Kubernetes
	kubectl apply -f deploy/k8s/

deploy-helm: ## Deploy with Helm
	helm upgrade --install sentiment-api ./deploy/helm/sentiment-api --namespace sentiment-api --create-namespace

undeploy-helm: ## Remove Helm deployment
	helm uninstall sentiment-api --namespace sentiment-api

logs: ## Show application logs (Docker Compose)
	docker-compose logs -f api

logs-k8s: ## Show Kubernetes logs
	kubectl logs -f deployment/sentiment-api -n sentiment-api

setup: install-dev ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make train' to train a model, then 'make dev-server' to start the API"

ci: lint test ## Run CI checks locally
	@echo "All CI checks passed!"