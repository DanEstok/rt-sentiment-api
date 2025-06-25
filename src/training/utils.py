"""Training utilities and helper functions."""

import logging
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = "distilbert-base-uncased"
    num_labels: int = 3
    cache_dir: str = None
    use_fast_tokenizer: bool = True
    model_revision: str = "main"
    use_auth_token: bool = False


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1, 2]
    )
    
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "f1_negative": f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        "f1_neutral": f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        "f1_positive": f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
        "precision_negative": precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
        "precision_neutral": precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
        "precision_positive": precision_per_class[2] if len(precision_per_class) > 2 else 0.0,
        "recall_negative": recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
        "recall_neutral": recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
        "recall_positive": recall_per_class[2] if len(recall_per_class) > 2 else 0.0,
    }
    
    return metrics


def get_label_names() -> Dict[int, str]:
    """Get mapping from label indices to names."""
    return {
        0: "negative",
        1: "neutral", 
        2: "positive"
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for logging."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")
    return ", ".join(formatted)


class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self):
        self.metrics_history = []
        
    def add_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Add metrics for a training step."""
        metrics_with_step = {"step": step, **metrics}
        self.metrics_history.append(metrics_with_step)
        
    def get_best_metrics(self, metric_name: str = "eval_f1") -> Dict[str, Any]:
        """Get the best metrics based on a specific metric."""
        if not self.metrics_history:
            return {}
            
        best_metrics = max(
            self.metrics_history, 
            key=lambda x: x.get(metric_name, 0)
        )
        return best_metrics
        
    def log_summary(self) -> None:
        """Log a summary of training metrics."""
        if not self.metrics_history:
            logger.info("No metrics to summarize")
            return
            
        best_metrics = self.get_best_metrics()
        logger.info(f"Best metrics: {format_metrics(best_metrics)}")
        
        final_metrics = self.metrics_history[-1]
        logger.info(f"Final metrics: {format_metrics(final_metrics)}")


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb