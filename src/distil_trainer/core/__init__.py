"""Core module for distillation training."""

from distil_trainer.core.config import (
    DistilTrainerConfig,
    DistillationConfig,
    TrainingConfig,
)
from distil_trainer.core.trainer import DistilTrainer
from distil_trainer.core.callbacks import (
    DistillationCallback,
    EvaluationCallback,
    LoggingCallback,
)

__all__ = [
    "DistilTrainer",
    "DistilTrainerConfig",
    "DistillationConfig",
    "TrainingConfig",
    "DistillationCallback",
    "EvaluationCallback",
    "LoggingCallback",
]
