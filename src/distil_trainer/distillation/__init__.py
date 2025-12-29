"""Distillation module for knowledge transfer."""

from distil_trainer.distillation.losses import (
    DistillationLosses,
    CombinedDistillationLoss,
)
from distil_trainer.distillation.strategies import (
    EmbeddingDistillationStrategy,
    LogitDistillationStrategy,
)
from distil_trainer.distillation.multilingual import (
    MultilingualDistillationStrategy,
)

__all__ = [
    "DistillationLosses",
    "CombinedDistillationLoss",
    "EmbeddingDistillationStrategy",
    "LogitDistillationStrategy",
    "MultilingualDistillationStrategy",
]
