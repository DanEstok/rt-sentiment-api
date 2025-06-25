"""Unit tests for the FastAPI application."""

import asyncio
import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.main import app
from src.app.models import MockModel, ModelManager
from src.inference.preprocess import InferencePreprocessor


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
async def mock_model_manager():
    """Mock model manager fixture."""
    manager = ModelManager("./test_models")
    manager.model = MockModel()
    await manager.model.load()
    return manager


@pytest.fixture
async def mock_preprocessor():
    """Mock preprocessor fixture."""
    with patch('src.app.main.InferencePreprocessor') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.preprocess.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "features": {"length": 10, "word_count": 2}
        }
        mock_instance.preprocess_batch = AsyncMock(return_value=[
            {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
                "features": {"length": 10, "word_count": 2}
            }
        ])
        yield mock_instance


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_healthy(self, client):
        """Test health check when service is healthy."""
        with patch('src.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            
            response = client.get("/healthz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
    
    def test_health_check_unhealthy(self, client):
        """Test health check when model is not loaded."""
        with patch('src.app.main.model_manager', None):
            response = client.get("/healthz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    @pytest.mark.asyncio
    async def test_predict_success(self, client, mock_model_manager, mock_preprocessor):
        """Test successful prediction."""
        with patch('src.app.main.model_manager', mock_model_manager), \
             patch('src.app.main.preprocessor', mock_preprocessor):
            
            response = client.post(
                "/predict",
                json={"text": "This is a test message"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "label" in data
            assert "score" in data
            assert "confidence" in data
            assert data["label"] in ["positive", "negative", "neutral"]
            assert 0 <= data["score"] <= 1
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_empty_text(self, client):
        """Test prediction with empty text."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_long_text(self, client):
        """Test prediction with very long text."""
        long_text = "a" * 15000  # Exceeds max length
        response = client.post(
            "/predict",
            json={"text": long_text}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_model_not_loaded(self, client):
        """Test prediction when model is not loaded."""
        with patch('src.app.main.model_manager', None):
            response = client.post(
                "/predict",
                json={"text": "Test message"}
            )
            assert response.status_code == 503


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint."""
    
    @pytest.mark.asyncio
    async def test_batch_predict_success(self, client, mock_model_manager, mock_preprocessor):
        """Test successful batch prediction."""
        with patch('src.app.main.model_manager', mock_model_manager), \
             patch('src.app.main.preprocessor', mock_preprocessor):
            
            response = client.post(
                "/predict/batch",
                json={"texts": ["Test message 1", "Test message 2"]}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            
            for prediction in data:
                assert "label" in prediction
                assert "score" in prediction
                assert "confidence" in prediction
                assert prediction["label"] in ["positive", "negative", "neutral"]
                assert 0 <= prediction["score"] <= 1
                assert 0 <= prediction["confidence"] <= 1
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_too_many_texts(self, client):
        """Test batch prediction with too many texts."""
        texts = ["test"] * 150  # Exceeds max batch size
        response = client.post(
            "/predict/batch",
            json={"texts": texts}
        )
        assert response.status_code == 400


class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    def test_model_info_success(self, client, mock_model_manager):
        """Test successful model info retrieval."""
        with patch('src.app.main.model_manager', mock_model_manager), \
             patch('src.app.main.preprocessor') as mock_preprocessor:
            mock_preprocessor.max_length = 512
            
            response = client.get("/model/info")
            assert response.status_code == 200
            data = response.json()
            assert "model_type" in data
            assert "model_path" in data
            assert "labels" in data
            assert "max_length" in data
    
    def test_model_info_not_loaded(self, client):
        """Test model info when model is not loaded."""
        with patch('src.app.main.model_manager', None):
            response = client.get("/model/info")
            assert response.status_code == 503


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


class TestInputValidation:
    """Test input validation."""
    
    def test_text_validation(self):
        """Test text validation logic."""
        from src.app.schemas import PredictionRequest
        
        # Valid text
        request = PredictionRequest(text="Valid text")
        assert request.text == "Valid text"
        
        # Text with whitespace
        request = PredictionRequest(text="  Text with spaces  ")
        assert request.text == "Text with spaces"
        
        # Empty text should raise validation error
        with pytest.raises(ValueError):
            PredictionRequest(text="")
        
        # Whitespace only should raise validation error
        with pytest.raises(ValueError):
            PredictionRequest(text="   ")
    
    def test_batch_validation(self):
        """Test batch validation logic."""
        from src.app.schemas import BatchPredictionRequest
        
        # Valid batch
        request = BatchPredictionRequest(texts=["Text 1", "Text 2"])
        assert len(request.texts) == 2
        
        # Empty list should raise validation error
        with pytest.raises(ValueError):
            BatchPredictionRequest(texts=[])
        
        # Text with empty string should raise validation error
        with pytest.raises(ValueError):
            BatchPredictionRequest(texts=["Valid text", ""])


class TestAsyncBehavior:
    """Test async behavior and concurrency."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_model_manager, mock_preprocessor):
        """Test handling of concurrent requests."""
        with patch('src.app.main.model_manager', mock_model_manager), \
             patch('src.app.main.preprocessor', mock_preprocessor):
            
            from src.app.main import predict_sentiment
            from src.app.schemas import PredictionRequest
            
            # Create multiple concurrent requests
            tasks = []
            for i in range(10):
                request = PredictionRequest(text=f"Test message {i}")
                task = predict_sentiment(request)
                tasks.append(task)
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all requests completed successfully
            assert len(results) == 10
            for result in results:
                assert result.label in ["positive", "negative", "neutral"]
                assert 0 <= result.score <= 1


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_model_error_handling(self, client):
        """Test handling of model errors."""
        with patch('src.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.predict_single = AsyncMock(side_effect=Exception("Model error"))
            
            with patch('src.app.main.preprocessor') as mock_preprocessor:
                mock_preprocessor.preprocess.return_value = {
                    "input_ids": torch.randint(0, 1000, (1, 10)),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long)
                }
                
                response = client.post(
                    "/predict",
                    json={"text": "Test message"}
                )
                assert response.status_code == 500
    
    def test_timeout_handling(self, client):
        """Test timeout handling."""
        with patch('src.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.predict_single = AsyncMock(side_effect=asyncio.TimeoutError())
            
            with patch('src.app.main.preprocessor') as mock_preprocessor:
                mock_preprocessor.preprocess.return_value = {
                    "input_ids": torch.randint(0, 1000, (1, 10)),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long)
                }
                
                response = client.post(
                    "/predict",
                    json={"text": "Test message"}
                )
                assert response.status_code == 504


if __name__ == "__main__":
    pytest.main([__file__])