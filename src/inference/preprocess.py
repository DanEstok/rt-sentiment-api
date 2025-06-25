"""Preprocessing utilities for inference."""

import logging
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing for sentiment analysis."""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove excessive capitalization (keep some for emphasis)
        words = text.split()
        cleaned_words = []
        for word in words:
            if len(word) > 2 and word.isupper() and not word.startswith('@') and not word.startswith('#'):
                # Convert to title case if all caps
                cleaned_words.append(word.title())
            else:
                cleaned_words.append(word)
        
        text = ' '.join(cleaned_words)
        
        return text.strip()
    
    def preprocess_single(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess a single text for inference."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "original_text": text,
            "cleaned_text": cleaned_text
        }
    
    def preprocess_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of texts for inference."""
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Tokenize batch
        encodings = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "original_texts": texts,
            "cleaned_texts": cleaned_texts
        }


class BatchProcessor:
    """Process batches of requests efficiently."""
    
    def __init__(self, preprocessor: TextPreprocessor, batch_size: int = 32):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        
    async def process_batch(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of texts."""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_result = self.preprocessor.preprocess_batch(batch_texts)
            
            # Split batch result back into individual results
            for j in range(len(batch_texts)):
                result = {
                    "input_ids": batch_result["input_ids"][j:j+1],
                    "attention_mask": batch_result["attention_mask"][j:j+1],
                    "original_text": batch_result["original_texts"][j],
                    "cleaned_text": batch_result["cleaned_texts"][j]
                }
                results.append(result)
        
        return results


def validate_input(text: str, max_length: int = 10000) -> Optional[str]:
    """Validate input text."""
    if not text or not isinstance(text, str):
        return "Text must be a non-empty string"
    
    if len(text.strip()) == 0:
        return "Text cannot be empty or only whitespace"
    
    if len(text) > max_length:
        return f"Text too long. Maximum length is {max_length} characters"
    
    return None


def extract_features(text: str) -> Dict[str, Any]:
    """Extract additional features from text."""
    features = {
        "length": len(text),
        "word_count": len(text.split()),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        "has_url": bool(re.search(r'http[s]?://', text)),
        "has_mention": bool(re.search(r'@\w+', text)),
        "has_hashtag": bool(re.search(r'#\w+', text)),
    }
    
    return features


class InferencePreprocessor:
    """Main preprocessor for inference pipeline."""
    
    def __init__(self, model_path: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_preprocessor = TextPreprocessor(self.tokenizer, max_length)
        self.batch_processor = BatchProcessor(self.text_preprocessor)
        
    def preprocess(self, text: str) -> Dict[str, Any]:
        """Preprocess text for inference."""
        # Validate input
        error = validate_input(text)
        if error:
            raise ValueError(error)
        
        # Preprocess text
        result = self.text_preprocessor.preprocess_single(text)
        
        # Extract additional features
        features = extract_features(text)
        result["features"] = features
        
        return result
    
    async def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Preprocess batch of texts for inference."""
        # Validate all inputs
        for i, text in enumerate(texts):
            error = validate_input(text)
            if error:
                raise ValueError(f"Text {i}: {error}")
        
        # Process batch
        results = await self.batch_processor.process_batch(texts)
        
        # Add features to each result
        for i, result in enumerate(results):
            features = extract_features(texts[i])
            result["features"] = features
        
        return results