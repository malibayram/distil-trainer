# Distil Trainer

A comprehensive knowledge distillation training framework for transformer models.

[![PyPI version](https://badge.fury.io/py/distil-trainer.svg)](https://badge.fury.io/py/distil-trainer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **7 Distillation Strategies**:

  - Classical Embedding Distillation (MSE/Cosine loss)
  - Layer Reduction (Depth Pruning)
  - Width Pruning (Hidden size, Attention heads, MLP)
  - Combined Depth-Width Pruning
  - Multilingual Model Extension
  - LLM to Embedding Model Conversion
  - Reasoning/Chain-of-Thought Distillation

- **Flexible Architecture**:

  - Support for SentenceTransformers and HuggingFace models
  - Multiple loss functions (MSE, KL Divergence, Cosine, Ranking)
  - Configurable importance estimation for pruning
  - PCA projection for dimension reduction

- **Production Ready**:
  - Export to HuggingFace Hub
  - ONNX export support
  - Distributed training with Accelerate
  - Comprehensive evaluation framework

## Installation

```bash
pip install distil-trainer
```

With optional dependencies:

```bash
# For experiment tracking
pip install distil-trainer[tracking]

# For model export
pip install distil-trainer[export]

# For MTEB evaluation
pip install distil-trainer[evaluation]

# For all features
pip install distil-trainer[all]
```

## Quick Start

### Basic Embedding Distillation

Distill knowledge from a large teacher model to a smaller student model:

```python
from distil_trainer import DistilTrainer, DistilTrainerConfig, DistillationConfig

config = DistilTrainerConfig(
    teacher_model="sentence-transformers/all-mpnet-base-v2",
    student_model="sentence-transformers/paraphrase-TinyBERT-L6-v2",
    distillation_config=DistillationConfig(
        loss_type="mse",  # Options: mse, kl_divergence, cosine, ranking, combined
        use_pca_projection=True,  # When student dim < teacher dim
    ),
    output_dir="./distilled_model",
)

trainer = DistilTrainer(config)
trainer.load_data(train_data="sentence-transformers/all-nli")
trainer.train()
trainer.save_model("./final_model")
trainer.save_model("./final_model")
```

### Custom Dataset Columns

If your dataset uses different column names (e.g., "text" instead of "sentence"), you can specify this when loading data:

```python
# Load dataset with custom text column
trainer.load_data(
    train_data="alibayram/cosmos-corpus-00-5",
    text_column="text"
)
```

### Layer Reduction (Depth Pruning)

Reduce model depth by keeping only selected layers:

```python
from distil_trainer import DistilTrainer, DistilTrainerConfig
from distil_trainer.core.config import LayerReductionConfig

config = DistilTrainerConfig(
    teacher_model="mixedbread-ai/mxbai-embed-large-v1",
    student_init_strategy="layer_reduction",
    pruning_config=LayerReductionConfig(
        # Explicitly specify layers to keep (0-indexed)
        layers_to_keep=[0, 3, 6, 9, 12, 15, 18, 21],
        # Or use automatic selection
        # num_layers_to_keep=8,
        # layer_selection="importance",  # Options: first, last, even, importance, custom
    ),
    output_dir="./layer_reduced_model",
)

trainer = DistilTrainer(config)
trainer.train()
```

### Width Pruning

Reduce hidden dimensions, attention heads, and intermediate sizes:

```python
from distil_trainer import DistilTrainer, DistilTrainerConfig
from distil_trainer.core.config import WidthPruningConfig

config = DistilTrainerConfig(
    teacher_model="Qwen/Qwen3-8B",
    student_init_strategy="width_pruning",
    pruning_config=WidthPruningConfig(
        # Target absolute dimensions
        target_hidden_size=3072,
        target_intermediate_size=9216,
        target_num_attention_heads=24,
        target_num_key_value_heads=4,
        # Or use ratios (alternative to absolute values)
        # hidden_size_ratio=0.75,
        # intermediate_size_ratio=0.75,
        # Importance estimation method
        importance_method="activation",  # Options: activation, gradient, taylor, wanda, cosine_similarity
        calibration_samples=1024,
    ),
    output_dir="./width_pruned_model",
)

trainer = DistilTrainer(config)
trainer.train()
```

### Combined Depth-Width Pruning

Apply both depth and width pruning for maximum compression:

```python
from distil_trainer import DistilTrainer, DistilTrainerConfig
from distil_trainer.core.config import (
    CombinedPruningConfig,
    LayerReductionConfig,
    WidthPruningConfig,
)

config = DistilTrainerConfig(
    teacher_model="meta-llama/Llama-3.2-3B",
    student_init_strategy="combined_pruning",
    pruning_config=CombinedPruningConfig(
        depth_config=LayerReductionConfig(
            num_layers_to_keep=16,
            layer_selection="importance",
        ),
        width_config=WidthPruningConfig(
            hidden_size_ratio=0.75,
            intermediate_size_ratio=0.75,
        ),
        pruning_order="depth_first",  # Options: depth_first, width_first, interleaved
        num_iterations=1,
    ),
    output_dir="./compressed_model",
)

trainer = DistilTrainer(config)
trainer.train()
```

### Multilingual Model Extension

Extend a monolingual model to support multiple languages:

```python
from distil_trainer.core.config import MultilingualConfig
from distil_trainer.distillation import MultilingualDistillationStrategy

config = MultilingualConfig(
    # Teacher understands these languages
    source_languages=["en"],
    # Student should learn these languages
    target_languages=["de", "es", "fr", "it", "pt", "zh", "ja", "ko"],
    # Student model (multilingual encoder)
    student_model="xlm-roberta-base",
    student_max_seq_length=128,
    # Parallel sentence datasets for training
    parallel_datasets=[
        "sentence-transformers/parallel-sentences-talks",
        "sentence-transformers/parallel-sentences-tatoeba",
    ],
    max_sentences_per_language=500000,
    # Training settings
    num_train_epochs=5,
    evaluation_steps=5000,
)

strategy = MultilingualDistillationStrategy(
    teacher_model="paraphrase-distilroberta-base-v2",
    config=config,
)
strategy.train()
```

## Configuration Reference

### DistilTrainerConfig

Main configuration class for distillation training:

| Parameter               | Type               | Default             | Description                               |
| ----------------------- | ------------------ | ------------------- | ----------------------------------------- |
| `teacher_model`         | str/Model          | Required            | Teacher model name or instance            |
| `student_model`         | str/Model          | None                | Student model (None creates from teacher) |
| `student_init_strategy` | str                | "from_pretrained"   | How to initialize student                 |
| `pruning_config`        | PruningConfig      | None                | Pruning configuration                     |
| `distillation_config`   | DistillationConfig | Default             | Distillation loss settings                |
| `training_config`       | TrainingConfig     | Default             | Training hyperparameters                  |
| `output_dir`            | str                | "./distilled_model" | Output directory                          |
| `device`                | str                | "auto"              | Device to use                             |
| `precision`             | str                | "bf16"              | Training precision                        |

### DistillationConfig

Configuration for distillation losses:

| Parameter                       | Type  | Default | Description                                                  |
| ------------------------------- | ----- | ------- | ------------------------------------------------------------ |
| `loss_type`                     | str   | "mse"   | Loss function: mse, kl_divergence, cosine, ranking, combined |
| `logit_loss_weight`             | float | 1.0     | Weight for logit distillation loss                           |
| `embedding_loss_weight`         | float | 1.0     | Weight for embedding distillation loss                       |
| `intermediate_loss_weight`      | float | 0.0     | Weight for intermediate layer loss                           |
| `temperature`                   | float | 1.0     | Temperature for KL divergence                                |
| `use_pca_projection`            | bool  | True    | Use PCA when student dim < teacher dim                       |
| `precompute_teacher_embeddings` | bool  | True    | Cache teacher embeddings                                     |

### TrainingConfig

Training hyperparameters:

| Parameter                     | Type  | Default  | Description               |
| ----------------------------- | ----- | -------- | ------------------------- |
| `num_train_epochs`            | int   | 1        | Number of training epochs |
| `per_device_train_batch_size` | int   | 64       | Batch size per device     |
| `learning_rate`               | float | 1e-4     | Initial learning rate     |
| `lr_scheduler_type`           | str   | "cosine" | LR scheduler type         |
| `warmup_ratio`                | float | 0.1      | Warmup ratio              |
| `weight_decay`                | float | 0.01     | Weight decay              |
| `max_grad_norm`               | float | 1.0      | Gradient clipping         |
| `eval_strategy`               | str   | "steps"  | Evaluation strategy       |
| `eval_steps`                  | int   | 500      | Evaluation frequency      |

## Evaluation

The framework includes comprehensive evaluation tools:

```python
from distil_trainer.evaluation import (
    EmbeddingSimilarityEvaluator,
    MSEEvaluator,
    BenchmarkRunner,
)

# Evaluate embedding similarity between teacher and student
evaluator = EmbeddingSimilarityEvaluator(
    teacher_model=teacher,
    student_model=student,
)
results = evaluator.evaluate(test_sentences)

# Run MTEB benchmarks (requires distil-trainer[evaluation])
runner = BenchmarkRunner(
    model="./distilled_model",
    tasks=["STS12", "STS13", "STS14", "STS15", "STS16"],
)
benchmark_results = runner.run()
```

## Data Loading

Support for multiple data formats:

```python
from distil_trainer.data import (
    DistillationDataModule,
    SentenceDistillationDataset,
    TripletDataset,
    ParallelSentencesDataset,
)

# Load from HuggingFace datasets
data_module = DistillationDataModule(
    train_data="sentence-transformers/all-nli",
    eval_data="sentence-transformers/stsb",
    text_column="sentence",
    max_seq_length=512,
    num_workers=4,
)

# Or use triplet format for ranking loss
triplet_dataset = TripletDataset(
    data_path="path/to/triplets.jsonl",
    query_column="query",
    positive_column="positive",
    negative_column="negative",
)
```

## Best Practices

Based on NVIDIA Minitron research:

1. **Sizing**: Train largest model first, then prune and distill iteratively
2. **Pruning**: Prefer width over depth pruning for better accuracy
3. **Retraining**: Use distillation loss exclusively (not conventional training)
4. **Loss Selection**:
   - Logit + intermediate + embedding when depth is reduced significantly
   - Logit-only when depth isn't reduced significantly

### Recommended Workflow

```python
# Step 1: Start with importance estimation
from distil_trainer.pruning import ImportanceEstimator

estimator = ImportanceEstimator(model, method="activation")
importance_scores = estimator.estimate(calibration_data)

# Step 2: Apply pruning based on importance
from distil_trainer.pruning import WidthPruner

pruner = WidthPruner(model, importance_scores)
pruned_model = pruner.prune(target_hidden_size=2048)

# Step 3: Distill knowledge from teacher
config = DistilTrainerConfig(
    teacher_model=original_model,
    student_model=pruned_model,
    distillation_config=DistillationConfig(
        loss_type="combined",
        logit_loss_weight=1.0,
        embedding_loss_weight=0.5,
    ),
)
trainer = DistilTrainer(config)
trainer.train()
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{distil_trainer,
  title = {Distil Trainer: A Comprehensive Knowledge Distillation Framework},
  author = {Ali Bayram},
  year = {2025},
  url = {https://github.com/malibayram/distil-trainer}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
