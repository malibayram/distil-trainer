# Embedding Distillation Pipeline

This guide covers the complete pipeline for knowledge distillation using precomputed teacher embeddings.

## Overview

The pipeline consists of two main components:

1. **TeacherEmbeddingsGenerator** - Precompute and store teacher embeddings
2. **EmbeddingDistillationTrainer** - Train student models using stored embeddings

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Source Dataset │────▶│    Generator    │────▶│ Dataset + Embs  │
│    (text)       │     │ (teacher model) │     │  (HuggingFace)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Distilled Model │◀────│     Trainer     │◀────│ Dataset + Embs  │
│   (student)     │     │ (student model) │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Part 1: TeacherEmbeddingsGenerator

Generate and store teacher model embeddings for later use in distillation.

### Installation

```bash
pip install distil-trainer
```

### Basic Usage

```python
from distil_trainer import TeacherEmbeddingsGenerator
from huggingface_hub import login

login(token="hf_cB***")

generator = TeacherEmbeddingsGenerator(
    teacher_model="google/embeddinggemma-300m",
    batch_size=64,
    use_bf16=True,
    compile_model=True,
    use_flash_attention=True,
)

# Generate embeddings
dataset = generator.generate_and_push(
    source_dataset="alibayram/cosmos-corpus-00-5",
    text_column="text",
    output_type="final",
    push_every=20000,
    token="hf_cB***",
)
```

### Constructor Parameters

| Parameter             | Type   | Default  | Description                                  |
| --------------------- | ------ | -------- | -------------------------------------------- |
| `teacher_model`       | `str`  | required | HuggingFace model ID or SentenceTransformer  |
| `device`              | `str`  | `"auto"` | Device: `"auto"`, `"cuda"`, `"mps"`, `"cpu"` |
| `batch_size`          | `int`  | `32`     | Batch size for encoding                      |
| `use_fp16`            | `bool` | `False`  | Use float16 precision                        |
| `use_bf16`            | `bool` | `False`  | Use bfloat16 precision (A100/H100)           |
| `compile_model`       | `bool` | `False`  | Use torch.compile()                          |
| `use_flash_attention` | `bool` | `False`  | Enable Flash Attention 2                     |

### Output Types

#### 1. Final Embeddings (Default)

Generates both `teacher_embedding_final` and `teacher_embedding_pre_dense` columns.

```python
dataset = generator.generate(
    source_dataset="your-dataset",
    output_type="final",
)
# Columns: ['text', 'teacher_embedding_final', 'teacher_embedding_pre_dense']
```

#### 2. Pre-Dense Embeddings Only

Generates embeddings before the Dense layer (if present).

```python
dataset = generator.generate(
    source_dataset="your-dataset",
    output_type="pre_dense",
)
# Columns: ['text', 'teacher_embedding_pre_dense']
```

#### 3. Hidden States (Multi-Layer)

Generates per-layer hidden states for advanced distillation.

```python
dataset = generator.generate(
    source_dataset="your-dataset",
    output_type="hidden_states",
    layer_indices=[0, 6, 12],  # None = all layers
)
# Columns: ['text', 'hidden_state_layer_0', 'hidden_state_layer_6', 'hidden_state_layer_12']
```

### Iterative Push (Large Datasets)

For large datasets, push checkpoints every N samples:

```python
generator.generate_and_push(
    source_dataset="your-dataset",
    repo_id="your-username/embeddings-dataset",
    push_every=10000,  # Push every 10k samples
    token="your-hf-token",
)
```

### Performance Optimization

```python
# Optimized for A100/H100
generator = TeacherEmbeddingsGenerator(
    teacher_model="sentence-transformers/all-mpnet-base-v2",
    batch_size=128,
    use_bf16=True,           # 2x faster
    compile_model=True,      # 10-30% faster
    use_flash_attention=True, # 20-50% faster (needs flash-attn)
)
```

---

## Part 2: EmbeddingDistillationTrainer

Train student models using precomputed teacher embeddings.

### Basic Usage

```python
from distil_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

config = EmbeddingTrainerConfig(
    student_model="sentence-transformers/all-MiniLM-L6-v2",
    target_type="final",
    num_epochs=3,
    batch_size=32,
)

trainer = EmbeddingDistillationTrainer(config)
trainer.train("your-username/corpus-with-embeddings")
```

### Configuration Options

#### Model Settings

| Parameter       | Type  | Default   | Description                |
| --------------- | ----- | --------- | -------------------------- |
| `student_model` | `str` | required  | Student model to train     |
| `target_type`   | `str` | `"final"` | `"final"` or `"pre_dense"` |

#### Training Settings

| Parameter       | Type    | Default | Description           |
| --------------- | ------- | ------- | --------------------- |
| `learning_rate` | `float` | `2e-5`  | Learning rate         |
| `num_epochs`    | `int`   | `3`     | Number of epochs      |
| `batch_size`    | `int`   | `32`    | Training batch size   |
| `warmup_ratio`  | `float` | `0.1`   | Warmup steps ratio    |
| `weight_decay`  | `float` | `0.01`  | Weight decay          |
| `max_grad_norm` | `float` | `1.0`   | Gradient clipping     |
| `loss_type`     | `str`   | `"mse"` | `"mse"` or `"cosine"` |

#### Column Settings

| Parameter                    | Type  | Default                         | Description            |
| ---------------------------- | ----- | ------------------------------- | ---------------------- |
| `text_column`                | `str` | `"text"`                        | Text column name       |
| `final_embedding_column`     | `str` | `"teacher_embedding_final"`     | Final embedding column |
| `pre_dense_embedding_column` | `str` | `"teacher_embedding_pre_dense"` | Pre-dense column       |

#### Output Settings

| Parameter       | Type  | Default             | Description         |
| --------------- | ----- | ------------------- | ------------------- |
| `output_dir`    | `str` | `"./trained_model"` | Save directory      |
| `save_steps`    | `int` | `1000`              | Checkpoint interval |
| `logging_steps` | `int` | `100`               | Logging interval    |

#### Hub Settings

| Parameter      | Type   | Default | Description             |
| -------------- | ------ | ------- | ----------------------- |
| `push_to_hub`  | `bool` | `False` | Push to HuggingFace Hub |
| `hub_model_id` | `str`  | `None`  | Repository ID           |
| `hub_token`    | `str`  | `None`  | HuggingFace token       |

#### Optimization Settings

| Parameter                | Type   | Default | Description         |
| ------------------------ | ------ | ------- | ------------------- |
| `use_fp16`               | `bool` | `False` | Float16 precision   |
| `use_bf16`               | `bool` | `False` | Bfloat16 precision  |
| `compile_model`          | `bool` | `False` | torch.compile()     |
| `use_flash_attention`    | `bool` | `False` | Flash Attention 2   |
| `gradient_checkpointing` | `bool` | `False` | Memory optimization |

### Target Types

#### Final Mode

Train the complete model (including Dense layer) to match teacher's final embeddings.

```python
config = EmbeddingTrainerConfig(
    student_model="your-model",
    target_type="final",
    final_embedding_column="teacher_embedding_final",
)
```

#### Pre-Dense Mode

Train only the transformer backbone (before Dense layer). Useful for:

- Training a vanilla transformer without Dense layer
- More granular control over model architecture

```python
config = EmbeddingTrainerConfig(
    student_model="your-model",
    target_type="pre_dense",
    pre_dense_embedding_column="teacher_embedding_pre_dense",
)
```

### Full Example

```python
from distil_trainer import (
    TeacherEmbeddingsGenerator,
    EmbeddingDistillationTrainer,
    EmbeddingTrainerConfig,
)

# Step 1: Generate embeddings
generator = TeacherEmbeddingsGenerator(
    teacher_model="sentence-transformers/all-mpnet-base-v2",
    batch_size=64,
    use_bf16=True,
)

generator.generate_and_push(
    source_dataset="your-username/corpus",
    repo_id="your-username/corpus-embeddings",
    push_every=50000,
    token="your-hf-token",
)

# Step 2: Train student
config = EmbeddingTrainerConfig(
    student_model="alibayram/cloned_sentence_transformer",
    target_type="final",
    num_epochs=1,
    batch_size=64,
    learning_rate=2e-5,
    use_bf16=True,
    push_to_hub=True,
    hub_model_id="alibayram/distilled-sentence-transformer",
    hub_token="hf_cB***",
)

trainer = EmbeddingDistillationTrainer(config)
metrics = trainer.train("alibayram/cosmos-corpus-0-05-with-embeddings")

print(f"Final loss: {metrics['train_loss']:.4f}")
```

---

## Optimization Guide

### GPU Recommendations

| GPU       | Recommended Settings                                          |
| --------- | ------------------------------------------------------------- |
| A100/H100 | `use_bf16=True, compile_model=True, use_flash_attention=True` |
| RTX 4090  | `use_bf16=True, compile_model=True`                           |
| RTX 3090  | `use_fp16=True, compile_model=True`                           |
| V100      | `use_fp16=True`                                               |
| MPS (Mac) | `use_fp16=True` (no compile)                                  |

### Memory Optimization

For large models or batch sizes:

```python
config = EmbeddingTrainerConfig(
    ...,
    gradient_checkpointing=True,  # Trades speed for memory
    batch_size=16,  # Reduce if OOM
)
```

### Flash Attention Installation

```bash
pip install flash-attn --no-build-isolation
```

> **Note**: Flash Attention requires CUDA and Ampere+ GPUs (A100, RTX 30xx+)
