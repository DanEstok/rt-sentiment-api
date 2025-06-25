# Contributing Guide

## Welcome

Thank you for your interest in contributing to the RT Sentiment API! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Security Guidelines](#security-guidelines)
- [Release Process](#release-process)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at team@yourorg.com.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- 8GB+ RAM (for model training/testing)
- Basic knowledge of:
  - FastAPI and async Python
  - Machine Learning concepts
  - Docker and containerization
  - Testing with pytest

### First Contribution

1. **Find an issue**: Look for issues labeled `good first issue` or `help wanted`
2. **Ask questions**: Don't hesitate to ask for clarification on issues
3. **Start small**: Begin with documentation fixes or small bug fixes
4. **Learn the codebase**: Read through the existing code and documentation

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rt-sentiment-api.git
cd rt-sentiment-api

# Add upstream remote
git remote add upstream https://github.com/original-org/rt-sentiment-api.git
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### 3. Development Configuration

```bash
# Create .env file for development
cat > .env << EOF
MODEL_PATH=./models
LOG_LEVEL=DEBUG
BATCH_SIZE=4
BATCH_TIMEOUT=0.01
MAX_BATCH_SIZE=16
PYTHONPATH=.
EOF
```

### 4. Verify Setup

```bash
# Run tests to verify setup
pytest tests/ -v

# Start development server
python src/app/main.py

# In another terminal, test the API
curl http://localhost:8000/healthz
```

## Contributing Process

### 1. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Changes

- Write clean, readable code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add batch prediction endpoint

- Implement batch processing for multiple texts
- Add input validation for batch requests
- Include comprehensive tests
- Update API documentation

Closes #123"
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
# Fill out the PR template completely
```

### 5. Address Review Feedback

- Respond to review comments promptly
- Make requested changes
- Push updates to the same branch
- Re-request review when ready

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications. Use the provided tools for consistency:

```bash
# Format code with black
black src/ tests/ scripts/

# Lint with ruff
ruff check src/ tests/ scripts/

# Type checking with mypy
mypy src/ --ignore-missing-imports
```

### Code Style Rules

#### 1. Imports

```python
# Standard library imports first
import asyncio
import logging
from typing import Dict, List, Optional

# Third-party imports
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports
from .models import SentimentModel
from .schemas import PredictionRequest
```

#### 2. Type Hints

Always use type hints for function parameters and return values:

```python
def process_text(text: str, max_length: int = 512) -> Dict[str, Any]:
    """Process text with proper type hints."""
    pass

async def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """Async function with type hints."""
    pass
```

#### 3. Docstrings

Use Google-style docstrings:

```python
def train_model(dataset: str, epochs: int = 3) -> None:
    """Train sentiment analysis model.
    
    Args:
        dataset: Name of the dataset to use for training.
        epochs: Number of training epochs.
        
    Returns:
        None
        
    Raises:
        ValueError: If dataset is not supported.
        RuntimeError: If training fails.
        
    Example:
        >>> train_model("twitter", epochs=5)
    """
    pass
```

#### 4. Error Handling

```python
# Use specific exception types
try:
    result = model.predict(text)
except ModelNotLoadedError:
    logger.error("Model not loaded")
    raise HTTPException(status_code=503, detail="Model not loaded")
except ValidationError as e:
    logger.warning(f"Validation error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

#### 5. Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General information about program execution")
logger.warning("Something unexpected happened")
logger.error("A serious error occurred")
logger.critical("A very serious error occurred")

# Include context in log messages
logger.info(f"Processing batch of {len(texts)} texts")
logger.error(f"Model inference failed for text length {len(text)}: {error}")
```

### API Design Guidelines

#### 1. Endpoint Design

```python
# Use clear, RESTful endpoints
@app.post("/predict", response_model=PredictionResponse)
@app.post("/predict/batch", response_model=List[PredictionResponse])
@app.get("/healthz", response_model=HealthResponse)
@app.get("/model/info", response_model=ModelInfoResponse)
```

#### 2. Request/Response Models

```python
class PredictionRequest(BaseModel):
    """Request model with validation."""
    text: str = Field(..., min_length=1, max_length=10000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class PredictionResponse(BaseModel):
    """Response model with clear field descriptions."""
    label: str = Field(..., description="Predicted sentiment label")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
```

## Testing Guidelines

### Test Structure

```
tests/
├── test_app.py              # API endpoint tests
├── test_latency.py          # Performance tests
├── test_models.py           # Model tests
├── test_preprocessing.py    # Preprocessing tests
├── test_training.py         # Training pipeline tests
├── conftest.py             # Shared fixtures
└── fixtures/               # Test data
    ├── sample_texts.json
    └── mock_responses.json
```

### Writing Tests

#### 1. Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_text_preprocessing():
    """Test text preprocessing functionality."""
    from src.inference.preprocess import TextPreprocessor
    
    preprocessor = TextPreprocessor(mock_tokenizer, max_length=512)
    
    # Test normal text
    result = preprocessor.clean_text("This is a test message!")
    assert result == "This is a test message!"
    
    # Test text with URLs
    result = preprocessor.clean_text("Check out https://example.com")
    assert "https://example.com" not in result
    
    # Test empty text
    with pytest.raises(ValueError):
        preprocessor.preprocess("")

@pytest.mark.asyncio
async def test_prediction_endpoint():
    """Test prediction endpoint."""
    from fastapi.testclient import TestClient
    from src.app.main import app
    
    client = TestClient(app)
    
    with patch('src.app.main.model_manager') as mock_manager:
        mock_manager.is_loaded.return_value = True
        mock_manager.predict_single.return_value = {
            "label": "positive",
            "score": 0.9,
            "confidence": 0.9
        }
        
        response = client.post("/predict", json={"text": "Great product!"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "positive"
        assert 0 <= data["score"] <= 1
```

#### 2. Integration Tests

```python
@pytest.mark.integration
async def test_end_to_end_prediction():
    """Test complete prediction pipeline."""
    # This test requires a real model to be loaded
    from src.app.models import ModelManager
    from src.inference.preprocess import InferencePreprocessor
    
    # Load real model (or mock for CI)
    model_manager = ModelManager("./test_models")
    await model_manager.load_model()
    
    preprocessor = InferencePreprocessor("./test_models")
    
    # Test prediction
    text = "This product is amazing!"
    preprocessed = preprocessor.preprocess(text)
    result = await model_manager.predict_single(
        preprocessed["input_ids"], 
        preprocessed["attention_mask"]
    )
    
    assert result["label"] in ["positive", "negative", "neutral"]
    assert 0 <= result["score"] <= 1
```

#### 3. Performance Tests

```python
@pytest.mark.performance
def test_prediction_latency():
    """Test that predictions meet latency requirements."""
    import time
    
    # Warm up
    for _ in range(5):
        client.post("/predict", json={"text": "Warm up message"})
    
    # Measure latency
    latencies = []
    for _ in range(50):
        start_time = time.perf_counter()
        response = client.post("/predict", json={"text": "Test message"})
        end_time = time.perf_counter()
        
        assert response.status_code == 200
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    # Check latency requirements
    median_latency = statistics.median(latencies)
    assert median_latency < 120, f"Median latency {median_latency}ms exceeds 120ms requirement"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not performance"  # Skip performance tests
pytest -m integration        # Only integration tests
pytest tests/test_app.py     # Specific test file

# Run with coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest tests/test_latency.py -v -s
```

### Test Fixtures

```python
# conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]]
    }
    return tokenizer

@pytest.fixture
async def mock_model_manager():
    """Mock model manager for testing."""
    manager = Mock()
    manager.is_loaded.return_value = True
    manager.predict_single.return_value = {
        "label": "positive",
        "score": 0.8,
        "confidence": 0.8
    }
    return manager

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "I love this product!",
        "This is terrible quality.",
        "It's okay, nothing special.",
        "Amazing service and quality!",
        "Not worth the money."
    ]
```

## Documentation

### Code Documentation

- Use clear, descriptive variable and function names
- Add docstrings to all public functions and classes
- Include type hints for all function parameters and returns
- Add inline comments for complex logic

### API Documentation

- Update OpenAPI schemas when adding new endpoints
- Include examples in request/response models
- Document error responses and status codes
- Keep API reference documentation up to date

### User Documentation

- Update relevant documentation when making changes
- Include examples and use cases
- Keep deployment guides current
- Add troubleshooting information for common issues

## Performance Considerations

### Latency Optimization

When contributing code that affects API latency:

1. **Measure performance impact** of your changes
2. **Run latency tests** to ensure requirements are met
3. **Consider async/await** for I/O operations
4. **Optimize hot paths** in the prediction pipeline
5. **Use profiling tools** to identify bottlenecks

```python
# Example: Optimizing a function
import cProfile
import time

def profile_function():
    """Profile a function for performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your function here
    result = expensive_function()
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
    
    return result
```

### Memory Management

- **Monitor memory usage** when loading models
- **Use generators** for large datasets
- **Clean up resources** properly
- **Avoid memory leaks** in long-running processes

### Scalability

- **Design for horizontal scaling**
- **Avoid shared state** between requests
- **Use efficient data structures**
- **Consider caching** for expensive operations

## Security Guidelines

### Input Validation

```python
# Always validate user input
from pydantic import BaseModel, validator

class SecureRequest(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        # Check length
        if len(v) > 10000:
            raise ValueError("Text too long")
        
        # Check for malicious content (example)
        if any(keyword in v.lower() for keyword in ['<script>', 'javascript:']):
            raise ValueError("Invalid content")
        
        return v
```

### Error Handling

```python
# Don't expose internal details in error messages
try:
    result = process_sensitive_data(data)
except InternalError as e:
    logger.error(f"Internal error: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Dependencies

- **Keep dependencies updated** regularly
- **Review security advisories** for dependencies
- **Use specific version pins** in production
- **Scan for vulnerabilities** with tools like `safety`

```bash
# Check for security vulnerabilities
pip install safety
safety check

# Update dependencies
pip-review --auto
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version numbers** in relevant files
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** including performance tests
4. **Update documentation** if needed
5. **Create release PR** for review
6. **Tag release** after merge
7. **Deploy to staging** for final testing
8. **Deploy to production** after validation

### Creating a Release

```bash
# Update version
echo "1.2.0" > VERSION

# Update changelog
git add CHANGELOG.md VERSION
git commit -m "chore: bump version to 1.2.0"

# Create tag
git tag -a v1.2.0 -m "Release version 1.2.0"

# Push tag
git push origin v1.2.0
```

## Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: team@yourorg.com for private matters

### Asking Questions

When asking for help:

1. **Search existing issues** first
2. **Provide context** about what you're trying to do
3. **Include error messages** and stack traces
4. **Share relevant code** (use code blocks)
5. **Describe your environment** (OS, Python version, etc.)

### Reporting Bugs

Use the bug report template and include:

- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, dependencies)
- **Error messages** and logs
- **Minimal code example** if possible

### Suggesting Features

Use the feature request template and include:

- **Clear description** of the feature
- **Use case** and motivation
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to RT Sentiment API! 🎉