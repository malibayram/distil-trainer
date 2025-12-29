"""Data module for distillation training."""

from distil_trainer.data.datamodule import DistillationDataModule
from distil_trainer.data.datasets import (
    SentenceDistillationDataset,
    TripletDataset,
    ParallelSentencesDataset,
)
from distil_trainer.data.collators import DistillationCollator
from distil_trainer.data.loaders import DatasetLoaders

__all__ = [
    "DistillationDataModule",
    "SentenceDistillationDataset",
    "TripletDataset",
    "ParallelSentencesDataset",
    "DistillationCollator",
    "DatasetLoaders",
]
