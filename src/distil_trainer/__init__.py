"""
Distil Trainer - A comprehensive knowledge distillation training framework.

This package provides tools for:
- Classical embedding distillation
- Model pruning (depth and width)
- Multilingual model extension
- LLM to embedding model conversion
- Reasoning/Chain-of-thought distillation
"""

from distil_trainer.core.config import (
    DistilTrainerConfig,
    DistillationConfig,
    TrainingConfig,
)
from distil_trainer.core.trainer import DistilTrainer

__version__ = "0.1.9"
__author__ = "Ali Bayram"
__email__ = "malibayram@gmail.com"

__all__ = [
    # Core
    "DistilTrainer",
    "DistilTrainerConfig",
    "DistillationConfig",
    "TrainingConfig",
    # Version
    "__version__",
]
