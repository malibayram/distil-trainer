"""Distillation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer

from distil_trainer.distillation.losses import DistillationLosses


class DistillationStrategy(ABC):
    """Base class for distillation strategies."""

    @abstractmethod
    def prepare(self, teacher: Any, student: Any, data: Any) -> None:
        """Prepare for distillation."""
        pass

    @abstractmethod
    def get_loss_function(self) -> Callable:
        """Get the loss function for this strategy."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Compute the distillation loss."""
        pass


@dataclass
class EmbeddingDistillationStrategy(DistillationStrategy):
    """
    Strategy for distilling embedding models.

    The student learns to produce embeddings similar to the teacher.
    Uses MSE or cosine loss between student and teacher embeddings.

    Example:
        >>> strategy = EmbeddingDistillationStrategy(loss_fn="mse")
        >>> strategy.prepare(teacher, student, train_data)
        >>> loss_fn = strategy.get_loss_function()
    """

    # Loss function
    loss_fn: str = "mse"  # "mse", "cosine", "combined"

    # Whether to precompute teacher embeddings
    precompute_embeddings: bool = True

    # PCA for dimension reduction
    use_pca: bool = True
    pca_components: int | None = None

    # Dataset requirements
    required_columns: list[str] = field(default_factory=lambda: ["sentence"])

    def prepare(self, teacher: Any, student: Any, data: Any) -> None:
        """Prepare for embedding distillation."""
        self.teacher = teacher
        self.student = student

    def get_loss_function(self) -> Callable:
        """Get the loss function."""
        if self.loss_fn == "mse":
            return DistillationLosses.mse_loss
        elif self.loss_fn == "cosine":
            return DistillationLosses.cosine_loss
        elif self.loss_fn == "combined":
            def combined_loss(student_out, teacher_out):
                mse = DistillationLosses.mse_loss(student_out, teacher_out)
                cosine = DistillationLosses.cosine_loss(student_out, teacher_out)
                return 0.5 * mse + 0.5 * cosine
            return combined_loss
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Compute embedding distillation loss."""
        loss_fn = self.get_loss_function()
        return loss_fn(student_output, teacher_output)


@dataclass
class LogitDistillationStrategy(DistillationStrategy):
    """
    Strategy for distilling via logit matching.

    Uses KL divergence between softmax distributions of student and teacher.

    Example:
        >>> strategy = LogitDistillationStrategy(temperature=2.0)
        >>> loss = strategy.compute_loss(student_logits, teacher_logits, batch)
    """

    # Temperature for softmax
    temperature: float = 1.0

    # Whether to also include hard label loss
    alpha: float = 0.5  # Weight for soft targets vs hard targets

    def prepare(self, teacher: Any, student: Any, data: Any) -> None:
        """Prepare for logit distillation."""
        self.teacher = teacher
        self.student = student

    def get_loss_function(self) -> Callable:
        """Get the loss function."""
        def loss_fn(student_logits, teacher_logits):
            return DistillationLosses.kl_divergence_loss(
                student_logits, teacher_logits, self.temperature
            )
        return loss_fn

    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Compute logit distillation loss."""
        soft_loss = DistillationLosses.kl_divergence_loss(
            student_output, teacher_output, self.temperature
        )

        # If hard labels available, combine with cross-entropy
        if "labels" in batch and self.alpha < 1.0:
            import torch.nn.functional as F
            hard_loss = F.cross_entropy(student_output, batch["labels"])
            return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return soft_loss


@dataclass
class RankingDistillationStrategy(DistillationStrategy):
    """
    Strategy for distilling retrieval models with ranking loss.

    Example:
        >>> strategy = RankingDistillationStrategy(in_batch_negatives=True)
        >>> loss = strategy.compute_loss(query_emb, pos_emb, batch)
    """

    # Whether to use in-batch negatives
    in_batch_negatives: bool = True

    # Number of hard negatives per sample
    hard_negatives: int = 5

    # Margin for triplet loss
    margin: float = 0.5

    def prepare(self, teacher: Any, student: Any, data: Any) -> None:
        """Prepare for ranking distillation."""
        self.teacher = teacher
        self.student = student

    def get_loss_function(self) -> Callable:
        """Get the loss function."""
        def loss_fn(query_emb, pos_emb, neg_emb=None):
            return DistillationLosses.ranking_loss(
                query_emb, pos_emb, neg_emb,
                margin=self.margin,
                in_batch_negatives=self.in_batch_negatives,
            )
        return loss_fn

    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Compute ranking distillation loss."""
        # For ranking, student_output contains query embeddings
        # teacher_output contains positive embeddings (from teacher)
        query_embeddings = student_output

        # Get positive embeddings from batch or compute
        if "positive_embeddings" in batch:
            positive_embeddings = batch["positive_embeddings"]
        else:
            # Use teacher embeddings as targets
            positive_embeddings = teacher_output

        # Get negative embeddings if available
        negative_embeddings = batch.get("negative_embeddings", None)

        return DistillationLosses.ranking_loss(
            query_embeddings,
            positive_embeddings,
            negative_embeddings,
            margin=self.margin,
            in_batch_negatives=self.in_batch_negatives,
        )
