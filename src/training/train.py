#!/usr/bin/env python3
"""Training script for sentiment analysis model."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset

from .data import load_dataset, preprocess_data
from .utils import compute_metrics, ModelArguments
from ..inference.export import export_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["twitter", "trustpilot"],
        default="twitter",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base model name",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./models", help="Output directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Number of warmup steps"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export model after training"
    )
    
    return parser.parse_args()


def setup_model_and_tokenizer(
    model_name: str, num_labels: int = 3
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Setup model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def train_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    args: argparse.Namespace,
) -> Trainer:
    """Train the model."""
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=None,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    return trainer


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset: {args.dataset}")
    train_data, eval_data = load_dataset(args.dataset)
    
    logger.info("Setting up model and tokenizer")
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    logger.info("Preprocessing data")
    train_dataset = preprocess_data(train_data, tokenizer, args.max_length)
    eval_dataset = preprocess_data(eval_data, tokenizer, args.max_length)
    
    logger.info("Starting training")
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, args)
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate final model
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Export model if requested
    if args.export:
        logger.info("Exporting model to TorchScript and ONNX")
        export_model(args.output_dir, args.max_length)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()