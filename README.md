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

# For all features
pip install distil-trainer[all]
```

## Quick Start

### Embedding Distillation

```python
from distil_trainer import DistilTrainer, DistilTrainerConfig

config = DistilTrainerConfig(
    teacher_model="sentence-transformers/all-mpnet-base-v2",
    student_model="sentence-transformers/paraphrase-TinyBERT-L6-v2",
    output_dir="./distilled_model"
)

trainer = DistilTrainer(config)
trainer.load_data(train_data="sentence-transformers/all-nli")
trainer.train()
trainer.save_model("./final_model")
```

### Layer Reduction

```python
from distil_trainer import DistilTrainer, DistilTrainerConfig
from distil_trainer.pruning import LayerReductionConfig

config = DistilTrainerConfig(
    teacher_model="mixedbread-ai/mxbai-embed-large-v1",
    student_init_strategy="layer_reduction",
    pruning_config=LayerReductionConfig(
        layers_to_keep=[0, 3, 6, 9, 12, 15, 18, 21],
    ),
    output_dir="./distilled_model"
)

trainer = DistilTrainer(config)
trainer.train()
```

### Width Pruning

```python
from distil_trainer import DistilTrainer, DistilTrainerConfig
from distil_trainer.pruning import WidthPruningConfig

config = DistilTrainerConfig(
    teacher_model="Qwen/Qwen3-8B",
    student_init_strategy="width_pruning",
    pruning_config=WidthPruningConfig(
        target_hidden_size=3072,
        target_intermediate_size=9216,
    ),
    output_dir="./distilled_model"
)

trainer = DistilTrainer(config)
trainer.train()
```

### Multilingual Extension

```python
from distil_trainer import MultilingualDistilTrainer
from distil_trainer.distillation import MultilingualConfig

config = MultilingualConfig(
    teacher_model="paraphrase-distilroberta-base-v2",
    student_model="xlm-roberta-base",
    source_languages=["en"],
    target_languages=["de", "es", "fr", "it"],
)

trainer = MultilingualDistilTrainer(config)
trainer.train()
```

## Documentation

For detailed documentation, see the [implementation plan](docs/implementation_plan.md).

## Best Practices

Based on NVIDIA Minitron research:

1. **Sizing**: Train largest model first, then prune and distill iteratively
2. **Pruning**: Prefer width over depth pruning for better accuracy
3. **Retraining**: Use distillation loss exclusively (not conventional training)
4. **Loss Selection**:
   - Logit + intermediate + embedding when depth is reduced significantly
   - Logit-only when depth isn't reduced significantly

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{distil_trainer,
  title = {Distil Trainer: A Comprehensive Knowledge Distillation Framework},
  author = {Ali Bayram},
  year = {2024},
  url = {https://github.com/malibayram/distil-trainer}
}
```
