"""Dataset classes for distillation training."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset


class SentenceDistillationDataset(TorchDataset):
    """
    Dataset for sentence embedding distillation.

    Example:
        >>> dataset = SentenceDistillationDataset(
        ...     sentences=["Hello world", "How are you?"],
        ...     teacher_embeddings=teacher_embeddings,
        ...     tokenizer=tokenizer,
        ... )
        >>> item = dataset[0]
    """

    def __init__(
        self,
        sentences: list[str],
        teacher_embeddings: torch.Tensor | list | None = None,
        tokenizer: Any = None,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            sentences: List of sentences to encode.
            teacher_embeddings: Precomputed teacher embeddings.
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
        """
        self.sentences = sentences
        self.teacher_embeddings = teacher_embeddings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {"sentence": self.sentences[idx]}

        if self.teacher_embeddings is not None:
            if isinstance(self.teacher_embeddings, torch.Tensor):
                item["label"] = self.teacher_embeddings[idx]
            else:
                item["label"] = torch.tensor(self.teacher_embeddings[idx])

        return item


class TripletDataset(TorchDataset):
    """
    Dataset for triplet (query, positive, negative) training.

    Used for contrastive learning and retrieval model training.

    Example:
        >>> dataset = TripletDataset(
        ...     queries=["What is Python?"],
        ...     positive_docs=[["Python is a programming language..."]],
        ...     negative_docs=[["Java is a programming language..."]]
        ... )
    """

    def __init__(
        self,
        queries: list[str],
        positive_docs: list[list[str]],
        negative_docs: list[list[str]] | None = None,
        tokenizer: Any = None,
        max_length: int = 512,
        num_negatives: int = 5,
    ):
        """
        Initialize the dataset.

        Args:
            queries: List of query strings.
            positive_docs: List of positive document lists (multiple per query).
            negative_docs: List of negative document lists.
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
            num_negatives: Number of negatives to sample per query.
        """
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        query = self.queries[idx]
        positives = self.positive_docs[idx]

        # Select first positive (or random)
        positive = positives[0] if positives else ""

        item = {
            "query": query,
            "positive": positive,
        }

        # Add negatives if available
        if self.negative_docs is not None and idx < len(self.negative_docs):
            negatives = self.negative_docs[idx]
            # Limit to num_negatives
            item["negatives"] = negatives[: self.num_negatives]

        return item


class ParallelSentencesDataset(TorchDataset):
    """
    Dataset for parallel sentences (multilingual training).

    Example:
        >>> dataset = ParallelSentencesDataset(
        ...     source_sentences=["Hello"],
        ...     target_sentences=["Hallo"],
        ...     teacher_embeddings=embeddings,
        ... )
    """

    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        teacher_embeddings: torch.Tensor | list | None = None,
        tokenizer: Any = None,
        max_length: int = 128,
    ):
        """
        Initialize the dataset.

        Args:
            source_sentences: Source language sentences.
            target_sentences: Target language sentences.
            teacher_embeddings: Teacher embeddings for source sentences.
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
        """
        assert len(source_sentences) == len(target_sentences)

        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.teacher_embeddings = teacher_embeddings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "english": self.source_sentences[idx],
            "non_english": self.target_sentences[idx],
        }

        if self.teacher_embeddings is not None:
            if isinstance(self.teacher_embeddings, torch.Tensor):
                item["label"] = self.teacher_embeddings[idx]
            else:
                item["label"] = torch.tensor(self.teacher_embeddings[idx])

        return item


class ReasoningDataset(TorchDataset):
    """
    Dataset for chain-of-thought/reasoning distillation.

    Example:
        >>> dataset = ReasoningDataset(
        ...     questions=["What is 2+2?"],
        ...     reasoning_chains=["Let me think... 2+2=4"],
        ...     answers=["4"],
        ... )
    """

    def __init__(
        self,
        questions: list[str],
        reasoning_chains: list[str],
        answers: list[str],
        system_prompts: list[str] | None = None,
        tokenizer: Any = None,
        max_length: int = 16384,
        chat_format: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            questions: List of questions.
            reasoning_chains: List of reasoning chains.
            answers: List of final answers.
            system_prompts: Optional system prompts.
            tokenizer: Tokenizer for encoding.
            max_length: Maximum sequence length.
            chat_format: Whether to format as chat.
        """
        self.questions = questions
        self.reasoning_chains = reasoning_chains
        self.answers = answers
        self.system_prompts = system_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_format = chat_format

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        question = self.questions[idx]
        reasoning = self.reasoning_chains[idx]
        answer = self.answers[idx]

        item = {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
        }

        if self.system_prompts is not None and idx < len(self.system_prompts):
            item["system"] = self.system_prompts[idx]

        if self.chat_format:
            # Format as conversation
            full_response = f"<|begin_of_thought|>\n{reasoning}\n<|end_of_thought|>\n\n<|begin_of_solution|>\n{answer}\n<|end_of_solution|>"
            item["full_response"] = full_response

        return item
