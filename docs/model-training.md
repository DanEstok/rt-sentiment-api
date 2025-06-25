# Model Training Guide

## Overview

This guide covers training sentiment analysis models using the RT Sentiment API training pipeline. The system supports fine-tuning DistilBERT on custom datasets with automatic export to production-ready formats.

## Quick Start

### Basic Training

```bash
# Train on Twitter dataset with default settings
python src/training/train.py --dataset twitter --epochs 3

# Train on Trustpilot dataset with custom parameters
python src/training/train.py \
  --dataset trustpilot \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --export
```

### Docker Training

```bash
# Build training image
docker build -f docker/Dockerfile.training -t sentiment-training .

# Run training with GPU
docker run --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  sentiment-training \
  python src/training/train.py --dataset twitter --epochs 3 --export
```

## Training Configuration

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | twitter | Dataset to use (twitter, trustpilot) |
| `--epochs` | int | 3 | Number of training epochs |
| `--batch_size` | int | 16 | Training batch size |
| `--learning_rate` | float | 2e-5 | Learning rate |
| `--model_name` | str | distilbert-base-uncased | Base model |
| `--output_dir` | str | ./models | Output directory |
| `--max_length` | int | 512 | Maximum sequence length |
| `--warmup_steps` | int | 500 | Number of warmup steps |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--save_steps` | int | 500 | Save checkpoint every N steps |
| `--eval_steps` | int | 500 | Evaluate every N steps |
| `--export` | flag | False | Export model after training |

### Environment Variables

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Training settings
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/app/cache
export HF_DATASETS_CACHE=/app/cache/datasets
```

## Datasets

### Supported Datasets

#### Twitter Sentiment Dataset

- **Source**: CardiffNLP Tweet Sentiment Multilingual (English)
- **Labels**: 0 (negative), 1 (neutral), 2 (positive)
- **Size**: ~100K tweets
- **Features**: Short text, informal language, hashtags, mentions

```python
# Load Twitter dataset
from src.training.data import load_dataset
train_data, eval_data = load_dataset("twitter")
```

#### Trustpilot Dataset

- **Source**: Amazon Polarity (as proxy for reviews)
- **Labels**: 0 (negative), 2 (positive), 1 (neutral - synthetic)
- **Size**: ~10K reviews (subset)
- **Features**: Longer text, formal language, product reviews

```python
# Load Trustpilot dataset
from src.training.data import load_dataset
train_data, eval_data = load_dataset("trustpilot")
```

### Custom Datasets

To add a custom dataset, implement a loader function in `src/training/data.py`:

```python
def load_custom_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load custom dataset."""
    train_data = []
    eval_data = []
    
    # Load your data
    for item in your_data_source:
        data_point = {
            "text": item["text"],
            "label": item["label"]  # 0: negative, 1: neutral, 2: positive
        }
        train_data.append(data_point)
    
    return train_data, eval_data
```

Then update the `load_dataset` function to include your custom dataset.

## Model Architecture

### Base Model: DistilBERT

- **Architecture**: Transformer encoder (6 layers)
- **Parameters**: 66M parameters
- **Vocabulary**: 30K WordPiece tokens
- **Max Length**: 512 tokens
- **Speed**: ~2x faster than BERT-base

### Fine-tuning Setup

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,  # negative, neutral, positive
    problem_type="single_label_classification"
)
```

### Classification Head

- **Input**: Hidden states from DistilBERT [CLS] token
- **Architecture**: Linear layer (768 → 3)
- **Activation**: Softmax for probability distribution
- **Loss**: Cross-entropy loss

## Training Process

### 1. Data Preprocessing

```python
# Text cleaning and normalization
def preprocess_text(text: str) -> str:
    # Remove URLs, normalize whitespace, etc.
    return cleaned_text

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
)
```

### 2. Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    fp16=True,  # Mixed precision training
)
```

### 3. Training Loop

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Start training
trainer.train()
```

### 4. Model Export

After training, models are exported to multiple formats:

```python
from src.inference.export import export_model

# Export to TorchScript and ONNX
torchscript_path, onnx_path = export_model("./models", max_length=512)
```

## Evaluation Metrics

### Primary Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 across all classes
- **Precision**: Weighted precision across all classes
- **Recall**: Weighted recall across all classes

### Per-Class Metrics

- **F1 per class**: F1 for negative, neutral, positive
- **Precision per class**: Precision for each sentiment
- **Recall per class**: Recall for each sentiment

### Evaluation Function

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
```

## Hyperparameter Tuning

### Learning Rate

- **Range**: 1e-5 to 5e-5
- **Default**: 2e-5
- **Schedule**: Linear warmup + linear decay

```python
# Learning rate scheduling
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

### Batch Size

- **Range**: 8 to 64 (depending on GPU memory)
- **Default**: 16
- **Considerations**: Larger batches = more stable gradients

### Epochs

- **Range**: 2 to 10
- **Default**: 3
- **Early Stopping**: Stop if no improvement for 3 evaluations

### Weight Decay

- **Range**: 0.0 to 0.1
- **Default**: 0.01
- **Purpose**: Regularization to prevent overfitting

## Advanced Training Techniques

### Mixed Precision Training

```python
# Enable FP16 for faster training
training_args = TrainingArguments(
    fp16=True,
    fp16_opt_level="O1",
    dataloader_num_workers=4
)
```

### Gradient Accumulation

```python
# Simulate larger batch sizes
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size = 32
)
```

### Learning Rate Scheduling

```python
# Cosine annealing with restarts
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,  # Steps for first restart
    T_mult=2   # Multiply period by 2 after each restart
)
```

## Monitoring Training

### TensorBoard Integration

```python
# Enable TensorBoard logging
training_args = TrainingArguments(
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard"
)

# View logs
# tensorboard --logdir ./logs
```

### Weights & Biases Integration

```python
# Enable W&B logging
import wandb

training_args = TrainingArguments(
    report_to="wandb",
    run_name="sentiment-training"
)

wandb.init(project="rt-sentiment-api")
```

### Custom Callbacks

```python
class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        if logs:
            print(f"Evaluation metrics: {logs}")
            # Log to external service
            log_metrics(logs)

trainer.add_callback(MetricsCallback())
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 8

# Enable gradient checkpointing
--gradient_checkpointing

# Use FP16
--fp16
```

#### Slow Training

```bash
# Increase batch size
--batch_size 32

# Use multiple GPUs
--per_device_train_batch_size 16

# Optimize data loading
--dataloader_num_workers 8
```

#### Poor Performance

```bash
# Increase learning rate
--learning_rate 3e-5

# Train longer
--epochs 5

# Reduce weight decay
--weight_decay 0.005
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data
print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")
print(f"Sample: {train_dataset[0]}")

# Check model
print(f"Model parameters: {model.num_parameters()}")
print(f"Model device: {next(model.parameters()).device}")
```

## Production Deployment

### Model Validation

Before deploying, validate the trained model:

```python
# Test inference
from src.inference.preprocess import InferencePreprocessor

preprocessor = InferencePreprocessor("./models")
result = preprocessor.preprocess("This is a test message")
print(f"Prediction: {result}")

# Verify exports
from src.inference.export import verify_exported_models
success = verify_exported_models("./models")
assert success, "Model export verification failed"
```

### Performance Testing

```python
# Benchmark inference speed
import time

texts = ["Test message"] * 100
start_time = time.time()

for text in texts:
    result = model.predict(text)

end_time = time.time()
avg_latency = (end_time - start_time) / len(texts) * 1000
print(f"Average latency: {avg_latency:.2f}ms")
```

### Model Versioning

```bash
# Tag model versions
git tag v1.0.0
git push origin v1.0.0

# Save model with version
cp -r ./models ./models-v1.0.0

# Update model registry
echo "v1.0.0" > ./models/VERSION
```

## Best Practices

### Data Quality

- **Clean Text**: Remove noise, normalize formatting
- **Balanced Labels**: Ensure reasonable class distribution
- **Quality Control**: Manual review of samples
- **Data Augmentation**: Increase dataset size with variations

### Training Efficiency

- **Warm Start**: Start from pre-trained checkpoints
- **Early Stopping**: Prevent overfitting
- **Validation Split**: Hold out data for evaluation
- **Reproducibility**: Set random seeds

### Model Quality

- **Cross Validation**: Multiple train/validation splits
- **Error Analysis**: Analyze misclassified examples
- **Bias Testing**: Test on diverse demographics
- **A/B Testing**: Compare model versions in production