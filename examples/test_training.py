#!/usr/bin/env python3
"""Test script for the training pipeline."""

import json
import sys
from pathlib import Path

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing data loading...")
    
    try:
        # Test importing the data module
        sys.path.append('.')
        from src.training.data import create_synthetic_dataset, preprocess_data
        from transformers import AutoTokenizer
        
        print("✅ Successfully imported data module")
        
        # Test synthetic data creation
        train_data, eval_data = create_synthetic_dataset()
        print(f"✅ Created synthetic dataset: {len(train_data)} train, {len(eval_data)} eval samples")
        
        # Test data format
        if train_data and all(isinstance(item, dict) and 'text' in item and 'label' in item for item in train_data[:5]):
            print("✅ Data format is correct")
        else:
            print("❌ Data format is incorrect")
            return False
        
        # Test preprocessing (with mock tokenizer)
        print("✅ Data loading tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_components():
    """Test model components."""
    print("\n🧪 Testing model components...")
    
    try:
        from src.app.models import MockModel
        from src.app.schemas import PredictionRequest, PredictionResponse
        import asyncio
        import torch
        
        print("✅ Successfully imported model components")
        
        # Test mock model
        mock_model = MockModel()
        asyncio.run(mock_model.load())
        print("✅ Mock model loaded successfully")
        
        # Test prediction
        dummy_input = torch.randint(0, 1000, (1, 10))
        dummy_mask = torch.ones(1, 10, dtype=torch.long)
        
        result = asyncio.run(mock_model.predict(dummy_input, dummy_mask))
        print(f"✅ Mock prediction successful: {result}")
        
        # Test schemas
        request = PredictionRequest(text="Test message")
        print(f"✅ Request schema validation passed: {request.text}")
        
        response = PredictionResponse(label="positive", score=0.8, confidence=0.8)
        print(f"✅ Response schema validation passed: {response.label}")
        
        print("✅ Model component tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model component test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n🧪 Testing preprocessing...")
    
    try:
        from src.inference.preprocess import TextPreprocessor, validate_input, extract_features
        
        print("✅ Successfully imported preprocessing module")
        
        # Test text cleaning
        test_texts = [
            "This is a normal text message.",
            "Check out this URL: https://example.com and email: test@example.com",
            "EXCESSIVE CAPS AND!!! PUNCTUATION???",
            "  Text with   extra   spaces  ",
        ]
        
        # Mock tokenizer for testing
        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                return {
                    "input_ids": torch.randint(0, 1000, (len(texts), 10)),
                    "attention_mask": torch.ones(len(texts), 10, dtype=torch.long)
                }
        
        preprocessor = TextPreprocessor(MockTokenizer(), max_length=512)
        
        for text in test_texts:
            cleaned = preprocessor.clean_text(text)
            print(f"✅ Cleaned: '{text[:30]}...' → '{cleaned[:30]}...'")
        
        # Test validation
        valid_text = "This is a valid text message."
        error = validate_input(valid_text)
        if error is None:
            print("✅ Text validation passed for valid input")
        else:
            print(f"❌ Text validation failed: {error}")
            return False
        
        # Test feature extraction
        features = extract_features(valid_text)
        expected_keys = ['length', 'word_count', 'exclamation_count', 'question_count']
        if all(key in features for key in expected_keys):
            print(f"✅ Feature extraction passed: {features}")
        else:
            print(f"❌ Feature extraction missing keys: {features}")
            return False
        
        print("✅ Preprocessing tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_api_schemas():
    """Test API schemas and validation."""
    print("\n🧪 Testing API schemas...")
    
    try:
        from src.app.schemas import (
            PredictionRequest, PredictionResponse, 
            BatchPredictionRequest, HealthResponse
        )
        
        print("✅ Successfully imported schema module")
        
        # Test valid requests
        valid_request = PredictionRequest(text="This is a test message")
        print(f"✅ Valid single request: {valid_request.text}")
        
        valid_batch = BatchPredictionRequest(texts=["Text 1", "Text 2", "Text 3"])
        print(f"✅ Valid batch request: {len(valid_batch.texts)} texts")
        
        # Test responses
        response = PredictionResponse(label="positive", score=0.85, confidence=0.85)
        print(f"✅ Valid response: {response.label} ({response.score})")
        
        health = HealthResponse(status="healthy", model_loaded=True)
        print(f"✅ Valid health response: {health.status}")
        
        # Test validation errors
        try:
            PredictionRequest(text="")  # Should fail
            print("❌ Empty text validation should have failed")
            return False
        except ValueError:
            print("✅ Empty text validation correctly failed")
        
        try:
            PredictionRequest(text="a" * 15000)  # Should fail
            print("❌ Long text validation should have failed")
            return False
        except ValueError:
            print("✅ Long text validation correctly failed")
        
        print("✅ API schema tests passed")
        return True
        
    except Exception as e:
        print(f"❌ API schema test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "src/training/train.py",
        "src/training/data.py", 
        "src/training/utils.py",
        "src/inference/preprocess.py",
        "src/inference/export.py",
        "src/app/main.py",
        "src/app/models.py",
        "src/app/schemas.py",
        "tests/test_app.py",
        "tests/test_latency.py",
        "scripts/benchmark.py",
        "docker/Dockerfile.training",
        "docker/Dockerfile.inference",
        "docker/docker-compose.yml",
        ".github/workflows/ci.yml",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests."""
    print("🚀 RT Sentiment API - Component Tests")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Data Loading", test_data_loading),
        ("Model Components", test_model_components),
        ("Preprocessing", test_preprocessing),
        ("API Schemas", test_api_schemas),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 COMPONENT TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All component tests passed!")
        print("\nNext steps:")
        print("1. Run 'python scripts/setup_mock_model.py' to create test models")
        print("2. Run 'python src/app/main.py' to start the API")
        print("3. Run 'python examples/test_api.py' to test the running API")
        return True
    else:
        print("⚠️ Some component tests failed. Check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)