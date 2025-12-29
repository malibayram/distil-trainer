"""Base data module for distillation training."""

from __future__ import annotations

import logging
from typing import Any, Callable

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

from distil_trainer.data.collators import DistillationCollator

logger = logging.getLogger(__name__)


class DistillationDataModule:
    """
    Base data module for distillation training.

    Handles data loading, preprocessing, and DataLoader creation.

    Example:
        >>> datamodule = DistillationDataModule(
        ...     train_data="sentence-transformers/all-nli",
        ...     text_column="sentence",
        ...     batch_size=32,
        ... )
        >>> datamodule.prepare_data()
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        train_data: str | Dataset | None = None,
        eval_data: str | Dataset | None = None,
        test_data: str | Dataset | None = None,
        tokenizer: Any = None,
        text_column: str = "sentence",
        max_seq_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        preprocessing_fn: Callable | None = None,
        max_samples: int | None = None,
    ):
        """
        Initialize the data module.

        Args:
            train_data: Training data path/name or Dataset.
            eval_data: Evaluation data path/name or Dataset.
            test_data: Test data path/name or Dataset.
            tokenizer: Tokenizer for encoding text.
            text_column: Name of the text column in the dataset.
            max_seq_length: Maximum sequence length for tokenization.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for data loading.
            preprocessing_fn: Optional preprocessing function.
            max_samples: Maximum number of samples to use.
        """
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_fn = preprocessing_fn
        self.max_samples = max_samples

        # Datasets
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        # Collator
        self.collator = DistillationCollator(
            tokenizer=tokenizer,
            max_length=max_seq_length,
            text_column=text_column,
        )

    def prepare_data(self) -> None:
        """Download and prepare data. Called once on a single process."""
        # Download datasets if they're specified as strings
        if isinstance(self.train_data, str):
            logger.info(f"Downloading training data: {self.train_data}")
            _ = load_dataset(self.train_data, split="train")

        if isinstance(self.eval_data, str):
            logger.info(f"Downloading eval data: {self.eval_data}")
            _ = load_dataset(self.eval_data, split="validation")

    def setup(self, stage: str | None = None) -> None:
        """
        Set up datasets for training/evaluation.

        Args:
            stage: "fit", "validate", "test", or None for all.
        """
        if stage in (None, "fit"):
            self.train_dataset = self._load_dataset(self.train_data, "train")
            if self.eval_data is not None:
                self.eval_dataset = self._load_dataset(self.eval_data, "validation")

        if stage in (None, "validate"):
            if self.eval_dataset is None and self.eval_data is not None:
                self.eval_dataset = self._load_dataset(self.eval_data, "validation")

        if stage in (None, "test"):
            if self.test_data is not None:
                self.test_dataset = self._load_dataset(self.test_data, "test")

    def _load_dataset(self, data: str | Dataset, split: str = "train") -> Dataset:
        """Load and preprocess a dataset."""
        if data is None:
            return None

        if isinstance(data, Dataset):
            dataset = data
        else:
            try:
                dataset = load_dataset(data, split=split)
            except Exception:
                # Try loading without split
                loaded = load_dataset(data)
                if isinstance(loaded, DatasetDict):
                    if split in loaded:
                        dataset = loaded[split]
                    else:
                        dataset = list(loaded.values())[0]
                else:
                    dataset = loaded

        # Limit samples if specified
        if self.max_samples is not None and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))

        # Apply preprocessing
        if self.preprocessing_fn is not None:
            dataset = dataset.map(
                self.preprocessing_fn,
                batched=True,
                remove_columns=dataset.column_names,
            )

        logger.info(f"Loaded dataset with {len(dataset)} samples")
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Training dataset not loaded. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Return validation DataLoader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader | None:
        """Return test DataLoader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )
