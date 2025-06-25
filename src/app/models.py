"""Model management and inference utilities."""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

logger = logging.getLogger(__name__)


class SentimentModel:
    """Base class for sentiment analysis models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.labels = {0: "negative", 1: "neutral", 2: "positive"}
        self.is_loaded = False
        
    async def load(self) -> None:
        """Load the model."""
        raise NotImplementedError
        
    async def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Make a prediction."""
        raise NotImplementedError
        
    async def predict_batch(self, input_ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        raise NotImplementedError


class PyTorchModel(SentimentModel):
    """PyTorch-based sentiment model."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load(self) -> None:
        """Load PyTorch model."""
        logger.info(f"Loading PyTorch model from {self.model_path}")
        start_time = time.time()
        
        def _load_model():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _load_model)
        
        self.is_loaded = True
        load_time = (time.time() - start_time) * 1000
        logger.info(f"PyTorch model loaded in {load_time:.2f}ms")
        
    def _predict_sync(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Synchronous prediction."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1).values.item()
            
            return {
                "label": self.labels[predicted_class],
                "score": probabilities[0][predicted_class].item(),
                "confidence": confidence,
                "probabilities": probabilities[0].cpu().numpy().tolist()
            }
    
    async def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Make async prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, input_ids, attention_mask)
        
    def _predict_batch_sync(self, input_ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Synchronous batch prediction."""
        # Stack tensors for batch processing
        batch_input_ids = torch.cat(input_ids_list, dim=0).to(self.device)
        batch_attention_mask = torch.cat(attention_mask_list, dim=0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            results = []
            for i in range(len(input_ids_list)):
                predicted_class = torch.argmax(probabilities[i], dim=-1).item()
                confidence = torch.max(probabilities[i], dim=-1).item()
                
                results.append({
                    "label": self.labels[predicted_class],
                    "score": probabilities[i][predicted_class].item(),
                    "confidence": confidence,
                    "probabilities": probabilities[i].cpu().numpy().tolist()
                })
            
            return results
    
    async def predict_batch(self, input_ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Make async batch prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._predict_batch_sync, input_ids_list, attention_mask_list)


class ONNXModel(SentimentModel):
    """ONNX-based sentiment model."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.session = None
        self.tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load(self) -> None:
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
            
        logger.info(f"Loading ONNX model from {self.model_path}")
        start_time = time.time()
        
        def _load_model():
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load ONNX model
            onnx_path = os.path.join(self.model_path, "exported", "model.onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
                
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
        # Load in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _load_model)
        
        self.is_loaded = True
        load_time = (time.time() - start_time) * 1000
        logger.info(f"ONNX model loaded in {load_time:.2f}ms")
        
    def _predict_sync(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Synchronous prediction."""
        # Convert to numpy
        input_ids_np = input_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()
        
        # Run inference
        ort_inputs = {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np
        }
        ort_outputs = self.session.run(None, ort_inputs)
        logits = ort_outputs[0]
        
        # Apply softmax
        probabilities = self._softmax(logits[0])
        predicted_class = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        return {
            "label": self.labels[predicted_class],
            "score": float(probabilities[predicted_class]),
            "confidence": float(confidence),
            "probabilities": probabilities.tolist()
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to numpy array."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    async def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Make async prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, input_ids, attention_mask)
        
    def _predict_batch_sync(self, input_ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Synchronous batch prediction."""
        # Stack tensors for batch processing
        batch_input_ids = torch.cat(input_ids_list, dim=0)
        batch_attention_mask = torch.cat(attention_mask_list, dim=0)
        
        # Convert to numpy
        input_ids_np = batch_input_ids.cpu().numpy()
        attention_mask_np = batch_attention_mask.cpu().numpy()
        
        # Run inference
        ort_inputs = {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np
        }
        ort_outputs = self.session.run(None, ort_inputs)
        logits = ort_outputs[0]
        
        results = []
        for i in range(len(input_ids_list)):
            probabilities = self._softmax(logits[i])
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            results.append({
                "label": self.labels[predicted_class],
                "score": float(probabilities[predicted_class]),
                "confidence": float(confidence),
                "probabilities": probabilities.tolist()
            })
        
        return results
    
    async def predict_batch(self, input_ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Make async batch prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._predict_batch_sync, input_ids_list, attention_mask_list)


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self, model_path: str, prefer_onnx: bool = True):
        self.model_path = model_path
        self.prefer_onnx = prefer_onnx and ONNX_AVAILABLE
        self.model: Optional[SentimentModel] = None
        self.model_type = None
        
    async def load_model(self) -> None:
        """Load the best available model."""
        # Try ONNX first if preferred and available
        if self.prefer_onnx:
            try:
                onnx_path = os.path.join(self.model_path, "exported", "model.onnx")
                if os.path.exists(onnx_path):
                    self.model = ONNXModel(self.model_path)
                    await self.model.load()
                    self.model_type = "ONNX"
                    logger.info("Loaded ONNX model")
                    return
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}")
        
        # Fallback to PyTorch
        try:
            self.model = PyTorchModel(self.model_path)
            await self.model.load()
            self.model_type = "PyTorch"
            logger.info("Loaded PyTorch model")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise RuntimeError(f"Could not load any model from {self.model_path}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.model.is_loaded
    
    async def predict_single(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Make a single prediction."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return await self.model.predict(input_ids, attention_mask)
    
    async def predict_batch(self, input_ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return await self.model.predict_batch(input_ids_list, attention_mask_list)
    
    def get_labels(self) -> List[str]:
        """Get available labels."""
        if self.model:
            return list(self.model.labels.values())
        return ["negative", "neutral", "positive"]


class MockModel(SentimentModel):
    """Mock model for testing."""
    
    def __init__(self, model_path: str = "mock"):
        super().__init__(model_path)
        
    async def load(self) -> None:
        """Mock load."""
        await asyncio.sleep(0.001)  # Simulate loading time
        self.is_loaded = True
        logger.info("Mock model loaded")
        
    async def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Mock prediction."""
        await asyncio.sleep(0.001)  # Simulate inference time
        
        # Simple mock logic based on input
        text_length = input_ids.shape[1]
        if text_length > 100:
            label = "positive"
            score = 0.8
        elif text_length > 50:
            label = "neutral"
            score = 0.6
        else:
            label = "negative"
            score = 0.7
            
        return {
            "label": label,
            "score": score,
            "confidence": score,
            "probabilities": [0.2, 0.3, 0.5] if label == "positive" else [0.7, 0.2, 0.1]
        }
        
    async def predict_batch(self, input_ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Mock batch prediction."""
        results = []
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            result = await self.predict(input_ids, attention_mask)
            results.append(result)
        return results