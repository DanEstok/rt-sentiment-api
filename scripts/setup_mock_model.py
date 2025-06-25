#!/usr/bin/env python3
"""Setup script to create a mock model for testing."""

import json
import os
from pathlib import Path

def create_mock_model_files():
    """Create mock model files for testing."""
    model_dir = Path("./test_models")
    model_dir.mkdir(exist_ok=True)
    
    # Create mock tokenizer files
    tokenizer_config = {
        "do_lower_case": True,
        "model_max_length": 512,
        "tokenizer_class": "DistilBertTokenizer"
    }
    
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Create mock model config
    model_config = {
        "architectures": ["DistilBertForSequenceClassification"],
        "model_type": "distilbert",
        "num_labels": 3,
        "id2label": {
            "0": "negative",
            "1": "neutral", 
            "2": "positive"
        },
        "label2id": {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Create mock vocab file
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "the": 5,
        "a": 6,
        "an": 7,
        "and": 8,
        "or": 9,
        "but": 10,
        "in": 11,
        "on": 12,
        "at": 13,
        "to": 14,
        "for": 15,
        "of": 16,
        "with": 17,
        "by": 18,
        "this": 19,
        "that": 20,
        "is": 21,
        "was": 22,
        "are": 23,
        "were": 24,
        "be": 25,
        "been": 26,
        "have": 27,
        "has": 28,
        "had": 29,
        "do": 30,
        "does": 31,
        "did": 32,
        "will": 33,
        "would": 34,
        "could": 35,
        "should": 36,
        "may": 37,
        "might": 38,
        "can": 39,
        "love": 40,
        "like": 41,
        "good": 42,
        "great": 43,
        "excellent": 44,
        "amazing": 45,
        "wonderful": 46,
        "fantastic": 47,
        "awesome": 48,
        "perfect": 49,
        "hate": 50,
        "dislike": 51,
        "bad": 52,
        "terrible": 53,
        "awful": 54,
        "horrible": 55,
        "worst": 56,
        "disappointing": 57,
        "poor": 58,
        "okay": 59,
        "average": 60,
        "decent": 61,
        "fine": 62,
        "normal": 63,
        "standard": 64,
        "product": 65,
        "service": 66,
        "quality": 67,
        "price": 68,
        "delivery": 69,
        "customer": 70,
        "support": 71,
        "experience": 72,
        "recommend": 73,
        "buy": 74,
        "purchase": 75,
        "order": 76,
        "item": 77,
        "company": 78,
        "website": 79,
        "fast": 80,
        "slow": 81,
        "quick": 82,
        "easy": 83,
        "difficult": 84,
        "hard": 85,
        "simple": 86,
        "complex": 87,
        "cheap": 88,
        "expensive": 89,
        "worth": 90,
        "money": 91,
        "value": 92,
        "satisfied": 93,
        "happy": 94,
        "pleased": 95,
        "disappointed": 96,
        "frustrated": 97,
        "angry": 98,
        "upset": 99
    }
    
    # Add more tokens to reach a reasonable vocab size
    for i in range(100, 1000):
        vocab[f"token_{i}"] = i
    
    with open(model_dir / "vocab.txt", "w") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    # Create special tokens map
    special_tokens = {
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]"
    }
    
    with open(model_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    # Create a simple tokenizer.json (minimal version)
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True
        },
        "pre_tokenizer": {
            "type": "BertPreTokenizer"
        },
        "post_processor": {
            "type": "BertProcessing",
            "sep": ["[SEP]", 3],
            "cls": ["[CLS]", 2]
        },
        "decoder": {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": True
        },
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab
        }
    }
    
    with open(model_dir / "tokenizer.json", "w") as f:
        json.dump(tokenizer_json, f, indent=2)
    
    print(f"✅ Mock model files created in {model_dir}")
    print("Files created:")
    for file in model_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    create_mock_model_files()