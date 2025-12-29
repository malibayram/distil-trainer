"""Data collators for distillation training."""

from __future__ import annotations

from typing import Any

import torch


class DistillationCollator:
    """
    Collator for distillation training data.

    Handles tokenization and batching of text data.

    Example:
        >>> collator = DistillationCollator(tokenizer, max_length=512)
        >>> batch = collator([{"sentence": "Hello"}, {"sentence": "World"}])
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_length: int = 512,
        text_column: str = "sentence",
        padding: bool = True,
        truncation: bool = True,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
            text_column: Name of the text column.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.padding = padding
        self.truncation = truncation

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Collated batch dictionary.
        """
        # Extract text
        texts = [sample.get(self.text_column, sample.get("sentence", "")) for sample in batch]

        # Tokenize if tokenizer is available
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt",
            )
            result = dict(encoded)
        else:
            result = {self.text_column: texts}

        # Add labels if present
        if "label" in batch[0]:
            labels = [sample["label"] for sample in batch]
            if isinstance(labels[0], torch.Tensor):
                result["label"] = torch.stack(labels)
            else:
                result["label"] = torch.tensor(labels)

        return result


class TripletCollator:
    """
    Collator for triplet data (query, positive, negatives).

    Example:
        >>> collator = TripletCollator(tokenizer)
        >>> batch = collator([{"query": "q1", "positive": "p1", "negatives": ["n1"]}])
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of triplet samples."""
        queries = [sample["query"] for sample in batch]
        positives = [sample["positive"] for sample in batch]

        result = {}

        if self.tokenizer is not None:
            # Tokenize queries
            query_encoded = self.tokenizer(
                queries,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt",
            )
            result["query_input_ids"] = query_encoded["input_ids"]
            result["query_attention_mask"] = query_encoded["attention_mask"]

            # Tokenize positives
            pos_encoded = self.tokenizer(
                positives,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt",
            )
            result["positive_input_ids"] = pos_encoded["input_ids"]
            result["positive_attention_mask"] = pos_encoded["attention_mask"]

            # Tokenize negatives if present
            if "negatives" in batch[0]:
                all_negatives = []
                for sample in batch:
                    all_negatives.extend(sample.get("negatives", []))

                if all_negatives:
                    neg_encoded = self.tokenizer(
                        all_negatives,
                        padding=self.padding,
                        truncation=self.truncation,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    result["negative_input_ids"] = neg_encoded["input_ids"]
                    result["negative_attention_mask"] = neg_encoded["attention_mask"]

        else:
            result["query"] = queries
            result["positive"] = positives
            if "negatives" in batch[0]:
                result["negatives"] = [sample.get("negatives", []) for sample in batch]

        return result


class ParallelSentenceCollator:
    """
    Collator for parallel sentence data (multilingual).

    Example:
        >>> collator = ParallelSentenceCollator(tokenizer)
        >>> batch = collator([{"english": "Hello", "non_english": "Hallo"}])
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_length: int = 128,
        padding: bool = True,
        truncation: bool = True,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of parallel sentence samples."""
        english = [sample["english"] for sample in batch]
        non_english = [sample["non_english"] for sample in batch]

        result = {}

        if self.tokenizer is not None:
            # Tokenize English
            en_encoded = self.tokenizer(
                english,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt",
            )
            result["english_input_ids"] = en_encoded["input_ids"]
            result["english_attention_mask"] = en_encoded["attention_mask"]

            # Tokenize non-English
            ne_encoded = self.tokenizer(
                non_english,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt",
            )
            result["non_english_input_ids"] = ne_encoded["input_ids"]
            result["non_english_attention_mask"] = ne_encoded["attention_mask"]

        else:
            result["english"] = english
            result["non_english"] = non_english

        # Add labels if present
        if "label" in batch[0]:
            labels = [sample["label"] for sample in batch]
            if isinstance(labels[0], torch.Tensor):
                result["label"] = torch.stack(labels)
            else:
                result["label"] = torch.tensor(labels)

        return result
