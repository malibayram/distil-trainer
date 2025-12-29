"""Multilingual distillation for extending models to new languages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from distil_trainer.distillation.losses import DistillationLosses

logger = logging.getLogger(__name__)


@dataclass
class MultilingualDistillationStrategy:
    """
    Strategy for extending a monolingual model to multiple languages.

    Uses parallel sentences to train a multilingual student where:
    - Student source language embeddings match teacher source embeddings
    - Student target language embeddings match teacher source embeddings

    Example:
        >>> strategy = MultilingualDistillationStrategy(
        ...     source_languages=["en"],
        ...     target_languages=["de", "es", "fr"]
        ... )
        >>> strategy.prepare(teacher, student, train_data)
    """

    # Source languages (teacher understands these)
    source_languages: list[str] = field(default_factory=lambda: ["en"])

    # Target languages (student should learn these)
    target_languages: list[str] = field(default_factory=list)

    # Parallel sentence datasets
    parallel_datasets: list[str] = field(
        default_factory=lambda: [
            "sentence-transformers/parallel-sentences-talks",
        ]
    )

    # Maximum sentences per language pair
    max_sentences_per_language: int = 500000

    # Loss function
    loss_fn: str = "mse"

    def __post_init__(self):
        self.teacher = None
        self.student = None
        self._train_datasets = None
        self._eval_datasets = None

    def prepare(
        self,
        teacher: SentenceTransformer,
        student: SentenceTransformer,
        batch_size: int = 64,
    ) -> None:
        """
        Prepare for multilingual distillation.

        Args:
            teacher: Teacher model (monolingual).
            student: Student model (multilingual).
            batch_size: Batch size for encoding.
        """
        self.teacher = teacher
        self.student = student
        self.batch_size = batch_size

    def load_parallel_data(self) -> tuple[DatasetDict, DatasetDict]:
        """
        Load parallel sentence data for training.

        Returns:
            Tuple of (train_datasets, eval_datasets) as DatasetDict.
        """
        train_datasets = DatasetDict()
        eval_datasets = DatasetDict()

        for source_lang in self.source_languages:
            for target_lang in self.target_languages:
                subset = f"{source_lang}-{target_lang}"

                for dataset_name in self.parallel_datasets:
                    try:
                        train_data = load_dataset(dataset_name, subset, split="train")

                        # Limit size
                        if len(train_data) > self.max_sentences_per_language:
                            train_data = train_data.select(range(self.max_sentences_per_language))

                        # Try to get eval split
                        try:
                            eval_data = load_dataset(dataset_name, subset, split="dev")
                            if len(eval_data) > 1000:
                                eval_data = eval_data.select(range(1000))
                        except Exception:
                            # Split from train
                            split_data = train_data.train_test_split(test_size=1000, shuffle=True)
                            train_data = split_data["train"]
                            eval_data = split_data["test"]

                        train_datasets[subset] = train_data
                        eval_datasets[subset] = eval_data

                        logger.info(f"Loaded {len(train_data)} training samples for {subset}")

                    except Exception as e:
                        logger.warning(f"Could not load {dataset_name}/{subset}: {e}")

        self._train_datasets = train_datasets
        self._eval_datasets = eval_datasets

        return train_datasets, eval_datasets

    def prepare_dataset(
        self,
        dataset: Any,
        source_col: str = "english",
        target_col: str = "non_english",
    ) -> Any:
        """
        Prepare dataset with teacher embeddings.

        Args:
            dataset: Dataset with parallel sentences.
            source_col: Column name for source language sentences.
            target_col: Column name for target language sentences.

        Returns:
            Dataset with teacher embeddings added.
        """

        def add_teacher_embeddings(batch):
            source_sentences = batch[source_col]
            with torch.no_grad():
                embeddings = self.teacher.encode(
                    source_sentences,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            return {"label": embeddings.tolist()}

        return dataset.map(add_teacher_embeddings, batched=True, batch_size=10000)

    def get_loss_function(self):
        """Get the loss function."""
        if self.loss_fn == "mse":
            return DistillationLosses.mse_loss
        elif self.loss_fn == "cosine":
            return DistillationLosses.cosine_loss
        else:
            return DistillationLosses.mse_loss

    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """
        Compute multilingual distillation loss.

        The loss encourages:
        1. Student source embeddings to match teacher source embeddings
        2. Student target embeddings to match teacher source embeddings

        Args:
            student_output: Student embeddings (for source or target text).
            teacher_output: Teacher embeddings (for source text).
            batch: Batch containing source and target sentences.

        Returns:
            Loss value.
        """
        loss_fn = self.get_loss_function()
        return loss_fn(student_output, teacher_output)


class MultilingualDistilTrainer:
    """
    Trainer for multilingual distillation.

    Extends a monolingual teacher to multiple languages via parallel sentence training.

    Example:
        >>> trainer = MultilingualDistilTrainer(
        ...     teacher_model="paraphrase-distilroberta-base-v2",
        ...     student_model="xlm-roberta-base",
        ... )
        >>> trainer.add_languages(["de", "es", "fr"])
        >>> trainer.train()
    """

    def __init__(
        self,
        teacher_model: str | SentenceTransformer,
        student_model: str | SentenceTransformer,
        source_languages: list[str] | None = None,
        target_languages: list[str] | None = None,
        output_dir: str = "./multilingual_model",
    ):
        """
        Initialize the trainer.

        Args:
            teacher_model: Teacher model (monolingual, e.g., English).
            student_model: Student model (multilingual base).
            source_languages: Languages the teacher understands.
            target_languages: Languages to extend to.
            output_dir: Output directory for the trained model.
        """
        if isinstance(teacher_model, str):
            self.teacher = SentenceTransformer(teacher_model)
        else:
            self.teacher = teacher_model

        if isinstance(student_model, str):
            self.student = SentenceTransformer(student_model)
        else:
            self.student = student_model

        self.source_languages = source_languages or ["en"]
        self.target_languages = target_languages or []
        self.output_dir = output_dir

        self.strategy = MultilingualDistillationStrategy(
            source_languages=self.source_languages,
            target_languages=self.target_languages,
        )

    def add_languages(self, languages: list[str]) -> None:
        """Add target languages to extend to."""
        self.target_languages.extend(languages)
        self.strategy.target_languages = self.target_languages

    def train(
        self,
        num_epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 2e-5,
    ) -> None:
        """
        Train the multilingual model.

        Args:
            num_epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
        """
        logger.info("Starting multilingual distillation training...")

        # Prepare strategy
        self.strategy.prepare(self.teacher, self.student, batch_size)

        # Load data
        train_datasets, eval_datasets = self.strategy.load_parallel_data()

        # Prepare datasets with teacher embeddings
        for subset in train_datasets:
            logger.info(f"Preparing {subset} with teacher embeddings...")
            train_datasets[subset] = self.strategy.prepare_dataset(train_datasets[subset])

        logger.info("Training prepared. Use DistilTrainer for actual training.")

        # Save the strategy and datasets for use with DistilTrainer
        self._train_datasets = train_datasets
        self._eval_datasets = eval_datasets

    def save_model(self, path: str | None = None) -> None:
        """Save the trained model."""
        save_path = path or self.output_dir
        self.student.save(save_path)
        logger.info(f"Model saved to {save_path}")
