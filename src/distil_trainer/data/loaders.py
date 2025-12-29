"""Built-in dataset loaders."""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class DatasetLoaders:
    """Built-in loaders for common datasets."""

    @staticmethod
    def load_allnli(split: str = "train", config: str = "pair-score") -> Dataset:
        """
        Load AllNLI dataset for sentence distillation.

        Args:
            split: Dataset split ("train", "dev", "test").
            config: Configuration name.

        Returns:
            Dataset with sentence pairs.
        """
        logger.info(f"Loading AllNLI dataset ({split})")
        return load_dataset("sentence-transformers/all-nli", config, split=split)

    @staticmethod
    def load_wikipedia_sentences(
        language: str = "en",
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load Wikipedia sentences dataset.

        Args:
            language: Language code (only "en" currently supported).
            max_samples: Maximum number of samples.

        Returns:
            Dataset with sentences.
        """
        logger.info("Loading Wikipedia sentences dataset")
        dataset = load_dataset(
            "sentence-transformers/wikipedia-en-sentences",
            split="train",
        )

        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset

    @staticmethod
    def load_stsb(split: str = "validation") -> Dataset:
        """
        Load STS Benchmark for evaluation.

        Args:
            split: Dataset split ("train", "validation", "test").

        Returns:
            Dataset with sentence pairs and similarity scores.
        """
        logger.info(f"Loading STS Benchmark ({split})")
        return load_dataset("sentence-transformers/stsb", split=split)

    @staticmethod
    def load_parallel_sentences(
        source_lang: str = "en",
        target_lang: str = "de",
        dataset: str = "talks",
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load parallel sentences for multilingual training.

        Args:
            source_lang: Source language code.
            target_lang: Target language code.
            dataset: Dataset name (talks, europarl, tatoeba, etc.).
            split: Dataset split.
            max_samples: Maximum number of samples.

        Returns:
            Dataset with parallel sentences.
        """
        dataset_name = f"sentence-transformers/parallel-sentences-{dataset}"
        subset = f"{source_lang}-{target_lang}"

        logger.info(f"Loading parallel sentences: {dataset_name}/{subset}")

        try:
            data = load_dataset(dataset_name, subset, split=split)
        except Exception:
            # Try reversed language pair
            subset = f"{target_lang}-{source_lang}"
            data = load_dataset(dataset_name, subset, split=split)

        if max_samples is not None and len(data) > max_samples:
            data = data.select(range(max_samples))

        return data

    @staticmethod
    def load_specter() -> tuple[Dataset, Dataset, Dataset]:
        """
        Load Specter triplet dataset for retrieval training.

        Returns:
            Tuple of (train, validation, test) datasets.
        """
        logger.info("Loading Specter dataset")
        train = load_dataset("allenai/specter", split="train")
        val = load_dataset("allenai/specter", split="validation")
        test = load_dataset("allenai/specter", split="test")
        return train, val, test

    @staticmethod
    def load_bespoke_stratos(max_samples: int | None = None) -> Dataset:
        """
        Load Bespoke-Stratos reasoning dataset.

        Args:
            max_samples: Maximum number of samples.

        Returns:
            Dataset with reasoning chains.
        """
        logger.info("Loading Bespoke-Stratos dataset")
        dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")

        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset

    @staticmethod
    def load_msmarco(
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load MS MARCO passage ranking dataset.

        Args:
            split: Dataset split.
            max_samples: Maximum number of samples.

        Returns:
            Dataset with queries and passages.
        """
        logger.info(f"Loading MS MARCO dataset ({split})")
        dataset = load_dataset("ms_marco", "v2.1", split=split)

        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset
