"""Data loading and preprocessing utilities."""

import logging
from typing import Dict, List, Tuple, Any

import torch
from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load and prepare dataset for training."""
    if dataset_name == "twitter":
        return load_twitter_dataset()
    elif dataset_name == "trustpilot":
        return load_trustpilot_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_twitter_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load Twitter sentiment dataset."""
    try:
        # Use a public Twitter sentiment dataset
        dataset = hf_load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english")
        
        train_data = []
        eval_data = []
        
        # Convert to our format
        for split, data_list in [("train", train_data), ("validation", eval_data)]:
            if split in dataset:
                for item in dataset[split]:
                    data_list.append({
                        "text": item["text"],
                        "label": item["label"]  # 0: negative, 1: neutral, 2: positive
                    })
        
        # If no validation split, create one from train
        if not eval_data and train_data:
            split_idx = int(len(train_data) * 0.8)
            eval_data = train_data[split_idx:]
            train_data = train_data[:split_idx]
        
        logger.info(f"Loaded Twitter dataset: {len(train_data)} train, {len(eval_data)} eval")
        return train_data, eval_data
        
    except Exception as e:
        logger.warning(f"Failed to load Twitter dataset: {e}")
        # Fallback to synthetic data
        return create_synthetic_dataset()


def load_trustpilot_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load Trustpilot sentiment dataset."""
    try:
        # Use a public review dataset as proxy
        dataset = hf_load_dataset("amazon_polarity")
        
        train_data = []
        eval_data = []
        
        # Take a subset and convert labels
        for i, item in enumerate(dataset["train"]):
            if i >= 10000:  # Limit size
                break
            
            # Convert binary labels to 3-class
            label = item["label"]  # 0: negative, 1: positive
            if label == 0:
                new_label = 0  # negative
            else:
                new_label = 2  # positive
            
            data_point = {
                "text": item["content"][:500],  # Truncate long reviews
                "label": new_label
            }
            
            if i < 8000:
                train_data.append(data_point)
            else:
                eval_data.append(data_point)
        
        logger.info(f"Loaded Trustpilot dataset: {len(train_data)} train, {len(eval_data)} eval")
        return train_data, eval_data
        
    except Exception as e:
        logger.warning(f"Failed to load Trustpilot dataset: {e}")
        # Fallback to synthetic data
        return create_synthetic_dataset()


def create_synthetic_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create synthetic dataset for testing."""
    logger.info("Creating synthetic dataset")
    
    positive_texts = [
        "I love this product! It's amazing!",
        "Great service and fast delivery.",
        "Excellent quality, highly recommended!",
        "Perfect! Exactly what I needed.",
        "Outstanding customer support.",
    ]
    
    negative_texts = [
        "Terrible product, waste of money.",
        "Poor quality and bad service.",
        "I hate this, very disappointing.",
        "Awful experience, would not recommend.",
        "Complete garbage, asking for refund.",
    ]
    
    neutral_texts = [
        "It's okay, nothing special.",
        "Average product, does the job.",
        "Not bad, but could be better.",
        "Decent quality for the price.",
        "It works as expected.",
    ]
    
    train_data = []
    eval_data = []
    
    # Create training data
    for _ in range(100):
        for texts, label in [(positive_texts, 2), (negative_texts, 0), (neutral_texts, 1)]:
            for text in texts:
                train_data.append({"text": text, "label": label})
    
    # Create eval data (smaller subset)
    for texts, label in [(positive_texts, 2), (negative_texts, 0), (neutral_texts, 1)]:
        for text in texts[:2]:
            eval_data.append({"text": text, "label": label})
    
    return train_data, eval_data


def preprocess_data(
    data: List[Dict[str, Any]], 
    tokenizer: AutoTokenizer, 
    max_length: int = 512
) -> Dataset:
    """Preprocess data for training."""
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    # Tokenize texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long)
    }
    
    return Dataset.from_dict(dataset_dict)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }