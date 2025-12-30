"""Configuration dataclasses for distillation training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from transformers import PreTrainedModel, PreTrainedTokenizer

from sentence_transformers import SentenceTransformer


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    # Basic Training
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 means use epochs
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Learning Rate
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01

    # Scheduler
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0

    # Optimization
    optimizer: Literal["adamw", "adam", "sgd", "adafactor"] = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Logging & Evaluation
    logging_steps: int = 100
    eval_strategy: Literal["steps", "epoch", "no"] = "steps"
    eval_steps: int = 500

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Tracking
    run_name: str | None = None
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class DistillationConfig:
    """Configuration for distillation losses and strategies."""

    # Loss Type
    loss_type: Literal[
        "mse",  # Mean Squared Error on embeddings
        "kl_divergence",  # KL divergence on logits
        "cosine",  # Cosine similarity loss
        "ranking",  # Ranking loss for embedding models
        "combined",  # Combination of multiple losses
    ] = "mse"

    # Loss Weights (for combined loss)
    logit_loss_weight: float = 1.0
    embedding_loss_weight: float = 1.0
    intermediate_loss_weight: float = 0.0
    attention_loss_weight: float = 0.0

    # Temperature for KL divergence
    temperature: float = 1.0

    # Embedding Distillation Options
    use_pca_projection: bool = True  # When student dim < teacher dim
    pca_num_samples: int = 20000

    # Intermediate Layer Mapping
    layer_mapping: dict[int, int] | None = None

    # Teacher Inference Settings
    teacher_inference_batch_size: int = 128
    precompute_teacher_embeddings: bool = True
    cache_teacher_embeddings: bool = True
    teacher_embeddings_cache_dir: str | None = None

    # Ranking Loss Options
    in_batch_negatives: bool = True
    hard_negatives_per_sample: int = 5
    margin: float = 0.5


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Dataset paths or names
    train_data: str | None = None
    eval_data: str | None = None
    test_data: str | None = None

    # Dataset options
    dataset_name: str | None = None
    dataset_config: str | None = None
    text_column: str = "sentence"
    max_samples: int | None = None

    # Preprocessing
    max_seq_length: int = 512
    num_workers: int = 4
    remove_columns: list[str] | None = None

    # Data format
    data_format: Literal["single", "pair", "triplet"] = "single"


@dataclass
class PruningConfig:
    """Base configuration for pruning."""

    # Pruning method
    method: Literal["depth", "width", "combined"] = "depth"

    # Importance estimation
    importance_method: Literal[
        "activation", "gradient", "taylor", "wanda", "cosine_similarity"
    ] = "activation"
    calibration_samples: int = 1024


@dataclass
class LayerReductionConfig(PruningConfig):
    """Configuration for layer reduction (depth pruning)."""

    method: Literal["depth", "width", "combined"] = "depth"

    # Layers to keep (0-indexed)
    layers_to_keep: list[int] | None = None

    # Alternative: specify how many layers to keep
    num_layers_to_keep: int | None = None

    # Alternative: specify layers to drop
    layers_to_drop: list[int] | None = None

    # Layer selection strategy
    layer_selection: Literal[
        "first", "last", "even", "importance", "custom"
    ] = "importance"


@dataclass
class WidthPruningConfig(PruningConfig):
    """Configuration for width-based pruning."""

    method: Literal["depth", "width", "combined"] = "width"

    # Target dimensions (set to None to keep original)
    target_hidden_size: int | None = None
    target_intermediate_size: int | None = None
    target_num_attention_heads: int | None = None
    target_num_key_value_heads: int | None = None

    # Alternative: specify reduction ratios
    hidden_size_ratio: float | None = None
    intermediate_size_ratio: float | None = None
    attention_head_ratio: float | None = None


@dataclass
class CombinedPruningConfig(PruningConfig):
    """Configuration for combined depth and width pruning."""

    method: Literal["depth", "width", "combined"] = "combined"

    # Target model size (parameters)
    target_params: int | None = None

    # Individual configs
    depth_config: LayerReductionConfig | None = None
    width_config: WidthPruningConfig | None = None

    # Pruning order
    pruning_order: Literal["depth_first", "width_first", "interleaved"] = "depth_first"

    # Iterative pruning
    num_iterations: int = 1
    prune_ratio_per_iteration: float = 0.5


@dataclass
class MultilingualConfig:
    """Configuration for multilingual knowledge distillation."""

    # Source languages (teacher understands these)
    source_languages: list[str] = field(default_factory=lambda: ["en"])

    # Target languages (student should learn these)
    target_languages: list[str] = field(default_factory=list)

    # Parallel sentence datasets
    parallel_datasets: list[str] = field(
        default_factory=lambda: [
            "sentence-transformers/parallel-sentences-talks",
            "sentence-transformers/parallel-sentences-tatoeba",
        ]
    )

    # Maximum sentences per language pair
    max_sentences_per_language: int = 500000

    # Student model configuration
    student_model: str = "xlm-roberta-base"
    student_max_seq_length: int = 128

    # Training settings
    num_train_epochs: int = 5
    evaluation_steps: int = 5000


@dataclass
class EmbeddingConversionConfig:
    """Configuration for converting LLM to embedding model."""

    # Source LLM
    source_model: str = ""

    # Attention modification
    convert_to_bidirectional: bool = True

    # Pooling strategy
    pooling_mode: Literal["mean", "cls", "last_token", "weighted_mean"] = "mean"

    # Training data format
    data_format: Literal["triplet", "pair", "single"] = "triplet"

    # Prefix configuration
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    # Loss function
    loss_type: Literal["ranking", "contrastive", "cosine"] = "ranking"
    in_batch_negatives: bool = True
    hard_negatives_count: int = 5


@dataclass
class ReasoningDistillationConfig:
    """Configuration for reasoning/CoT distillation."""

    # Teacher model (reasoning model)
    teacher_model: str = ""

    # Student model
    student_model: str = ""

    # Data generation
    generate_reasoning_data: bool = True
    reasoning_api: str | None = None
    num_reasoning_samples: int = 10000

    # Reasoning format
    reasoning_format: Literal["cot", "step_by_step", "scratchpad"] = "cot"

    # Special tokens
    thought_start_token: str = "<|begin_of_thought|>"
    thought_end_token: str = "<|end_of_thought|>"
    solution_start_token: str = "<|begin_of_solution|>"
    solution_end_token: str = "<|end_of_solution|>"

    # Training settings
    max_seq_length: int = 16384
    mask_reasoning: bool = False


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    project: str = "distil-trainer"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    group: str | None = None
    notes: str | None = None


@dataclass
class HubConfig:
    """Configuration for HuggingFace Hub integration."""
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = None
    hub_private_repo: bool = False
    push_to_hub_interval: Literal["every_save", "end"] = "end"


@dataclass
class DistilTrainerConfig:
    """Main configuration for distillation training."""

    # Model Configuration
    teacher_model: str | SentenceTransformer | PreTrainedModel = ""
    student_model: str | SentenceTransformer | PreTrainedModel | None = None
    student_model_name: str | None = None

    # Student Initialization Strategy
    student_init_strategy: Literal[
        "from_pretrained",
        "layer_reduction",
        "width_pruning",
        "depth_pruning",
        "combined_pruning",
    ] = "from_pretrained"

    # Pruning Configuration (if applicable)
    pruning_config: PruningConfig | LayerReductionConfig | WidthPruningConfig | CombinedPruningConfig | None = None

    # Distillation Configuration
    distillation_config: DistillationConfig = field(default_factory=DistillationConfig)

    # Training Configuration
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # Data Configuration
    data_config: DataConfig = field(default_factory=DataConfig)

    # Output Configuration
    output_dir: str = "./distilled_model"
    save_strategy: Literal["steps", "epoch", "best"] = "best"
    save_total_limit: int = 2

    # Hardware Configuration
    device: str = "auto"
    precision: Literal["fp32", "fp16", "bf16", "int8"] = "bf16"
    distributed: bool = False
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Logging
    logging_dir: str | None = None
    seed: int = 42

    # Integrations
    wandb_config: WandbConfig = field(default_factory=WandbConfig)
    hub_config: HubConfig = field(default_factory=HubConfig)
