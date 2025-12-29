"""Pruning module for model compression."""

from distil_trainer.pruning.depth_pruning import DepthPruner
from distil_trainer.pruning.width_pruning import WidthPruner
from distil_trainer.pruning.combined_pruning import CombinedPruner
from distil_trainer.pruning.importance import ImportanceEstimator

__all__ = [
    "DepthPruner",
    "WidthPruner",
    "CombinedPruner",
    "ImportanceEstimator",
]
