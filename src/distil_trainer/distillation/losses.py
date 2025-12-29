"""Distillation loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class DistillationLosses:
    """Collection of distillation loss functions."""

    @staticmethod
    def mse_loss(
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean Squared Error loss between embeddings.

        Args:
            student_output: Student model embeddings [batch_size, dim].
            teacher_output: Teacher model embeddings [batch_size, dim].

        Returns:
            MSE loss value.
        """
        return F.mse_loss(student_output, teacher_output)

    @staticmethod
    def cosine_loss(
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine similarity loss (1 - cosine_similarity).

        Args:
            student_output: Student model embeddings [batch_size, dim].
            teacher_output: Teacher model embeddings [batch_size, dim].

        Returns:
            Cosine loss value.
        """
        similarity = F.cosine_similarity(student_output, teacher_output, dim=-1)
        return (1 - similarity).mean()

    @staticmethod
    def kl_divergence_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        KL divergence loss on softmax distributions.

        Args:
            student_logits: Student model logits.
            teacher_logits: Teacher model logits.
            temperature: Temperature for softmax (higher = softer).

        Returns:
            KL divergence loss value.
        """
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        return loss * (temperature ** 2)

    @staticmethod
    def ranking_loss(
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None = None,
        margin: float = 0.5,
        in_batch_negatives: bool = True,
    ) -> torch.Tensor:
        """
        Ranking loss for retrieval models.

        Args:
            query_embeddings: Query embeddings [batch_size, dim].
            positive_embeddings: Positive document embeddings [batch_size, dim].
            negative_embeddings: Negative document embeddings [batch_size, num_neg, dim].
            margin: Margin for triplet loss.
            in_batch_negatives: Whether to use in-batch negatives.

        Returns:
            Ranking loss value.
        """
        batch_size = query_embeddings.size(0)

        # Positive scores
        pos_scores = F.cosine_similarity(query_embeddings, positive_embeddings, dim=-1)

        if in_batch_negatives:
            # Use other positives in batch as negatives
            # Similarity matrix: [batch_size, batch_size]
            sim_matrix = torch.mm(
                F.normalize(query_embeddings, p=2, dim=-1),
                F.normalize(positive_embeddings, p=2, dim=-1).t()
            )

            # Diagonal contains positive similarities
            # Off-diagonal contains negative similarities
            labels = torch.arange(batch_size, device=query_embeddings.device)

            # Cross-entropy loss with in-batch negatives
            loss = F.cross_entropy(sim_matrix / 0.05, labels)  # Temperature-scaled

        elif negative_embeddings is not None:
            # Use provided negatives
            num_negatives = negative_embeddings.size(1)

            # Compute negative scores
            neg_scores = torch.bmm(
                query_embeddings.unsqueeze(1),
                negative_embeddings.transpose(1, 2)
            ).squeeze(1)  # [batch_size, num_neg]

            # Triplet margin loss
            pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, num_negatives)
            loss = F.relu(margin - pos_scores_expanded + neg_scores).mean()

        else:
            raise ValueError("Either in_batch_negatives or negative_embeddings required")

        return loss

    @staticmethod
    def contrastive_loss(
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Contrastive loss for pairs.

        Args:
            embeddings1: First set of embeddings [batch_size, dim].
            embeddings2: Second set of embeddings [batch_size, dim].
            labels: Binary labels (1 = similar, 0 = dissimilar) [batch_size].
            margin: Margin for dissimilar pairs.

        Returns:
            Contrastive loss value.
        """
        distances = F.pairwise_distance(embeddings1, embeddings2)

        # Similar pairs: minimize distance
        # Dissimilar pairs: maximize distance (up to margin)
        loss = labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2)

        return loss.mean()

    @staticmethod
    def intermediate_layer_loss(
        student_hidden_states: tuple[torch.Tensor, ...],
        teacher_hidden_states: tuple[torch.Tensor, ...],
        layer_mapping: dict[int, int],
    ) -> torch.Tensor:
        """
        Loss on intermediate layer representations.

        Args:
            student_hidden_states: Tuple of student hidden states per layer.
            teacher_hidden_states: Tuple of teacher hidden states per layer.
            layer_mapping: Maps student layer indices to teacher layer indices.

        Returns:
            Intermediate layer loss value.
        """
        total_loss = 0.0
        num_layers = 0

        for student_idx, teacher_idx in layer_mapping.items():
            if student_idx < len(student_hidden_states) and teacher_idx < len(teacher_hidden_states):
                student_hidden = student_hidden_states[student_idx]
                teacher_hidden = teacher_hidden_states[teacher_idx]

                # Handle dimension mismatch with projection
                if student_hidden.size(-1) != teacher_hidden.size(-1):
                    # Simple mean pooling to match dimensions
                    if student_hidden.size(-1) < teacher_hidden.size(-1):
                        teacher_hidden = teacher_hidden[..., :student_hidden.size(-1)]
                    else:
                        student_hidden = student_hidden[..., :teacher_hidden.size(-1)]

                total_loss += F.mse_loss(student_hidden, teacher_hidden)
                num_layers += 1

        return total_loss / max(num_layers, 1)

    @staticmethod
    def attention_transfer_loss(
        student_attentions: tuple[torch.Tensor, ...],
        teacher_attentions: tuple[torch.Tensor, ...],
        layer_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Loss on attention patterns.

        Args:
            student_attentions: Tuple of student attention weights per layer.
            teacher_attentions: Tuple of teacher attention weights per layer.
            layer_mapping: Maps student layer indices to teacher layer indices.

        Returns:
            Attention transfer loss value.
        """
        if layer_mapping is None:
            # Default: map corresponding layers
            min_layers = min(len(student_attentions), len(teacher_attentions))
            layer_mapping = {i: i for i in range(min_layers)}

        total_loss = 0.0
        num_layers = 0

        for student_idx, teacher_idx in layer_mapping.items():
            if student_idx < len(student_attentions) and teacher_idx < len(teacher_attentions):
                student_attn = student_attentions[student_idx]
                teacher_attn = teacher_attentions[teacher_idx]

                # Attention matrices: [batch, heads, seq, seq]
                # Average over heads if different number of heads
                if student_attn.size(1) != teacher_attn.size(1):
                    student_attn = student_attn.mean(dim=1, keepdim=True)
                    teacher_attn = teacher_attn.mean(dim=1, keepdim=True)

                total_loss += F.mse_loss(student_attn, teacher_attn)
                num_layers += 1

        return total_loss / max(num_layers, 1)


class CombinedDistillationLoss(nn.Module):
    """Combine multiple distillation losses with weights."""

    def __init__(
        self,
        logit_weight: float = 1.0,
        embedding_weight: float = 1.0,
        intermediate_weight: float = 0.0,
        attention_weight: float = 0.0,
        temperature: float = 1.0,
        layer_mapping: dict[int, int] | None = None,
    ):
        """
        Initialize combined loss.

        Args:
            logit_weight: Weight for logit distillation loss.
            embedding_weight: Weight for embedding MSE loss.
            intermediate_weight: Weight for intermediate layer loss.
            attention_weight: Weight for attention transfer loss.
            temperature: Temperature for KL divergence.
            layer_mapping: Layer mapping for intermediate/attention losses.
        """
        super().__init__()
        self.logit_weight = logit_weight
        self.embedding_weight = embedding_weight
        self.intermediate_weight = intermediate_weight
        self.attention_weight = attention_weight
        self.temperature = temperature
        self.layer_mapping = layer_mapping or {}

    def forward(
        self,
        student_output: torch.Tensor | dict,
        teacher_output: torch.Tensor | dict,
    ) -> torch.Tensor:
        """
        Compute weighted combination of losses.

        Args:
            student_output: Student model output (tensor or dict with multiple outputs).
            teacher_output: Teacher model output (tensor or dict with multiple outputs).

        Returns:
            Combined loss value.
        """
        total_loss = torch.tensor(0.0, device=self._get_device(student_output))

        # Handle simple tensor outputs (embedding distillation)
        if isinstance(student_output, torch.Tensor) and isinstance(teacher_output, torch.Tensor):
            if self.embedding_weight > 0:
                total_loss = total_loss + self.embedding_weight * DistillationLosses.mse_loss(
                    student_output, teacher_output
                )
            return total_loss

        # Handle dict outputs with multiple components
        if isinstance(student_output, dict) and isinstance(teacher_output, dict):
            # Logit loss
            if self.logit_weight > 0 and "logits" in student_output and "logits" in teacher_output:
                total_loss = total_loss + self.logit_weight * DistillationLosses.kl_divergence_loss(
                    student_output["logits"],
                    teacher_output["logits"],
                    self.temperature,
                )

            # Embedding loss
            if self.embedding_weight > 0:
                if "last_hidden_state" in student_output and "last_hidden_state" in teacher_output:
                    total_loss = total_loss + self.embedding_weight * DistillationLosses.mse_loss(
                        student_output["last_hidden_state"],
                        teacher_output["last_hidden_state"],
                    )
                elif "embeddings" in student_output and "embeddings" in teacher_output:
                    total_loss = total_loss + self.embedding_weight * DistillationLosses.mse_loss(
                        student_output["embeddings"],
                        teacher_output["embeddings"],
                    )

            # Intermediate layer loss
            if self.intermediate_weight > 0 and self.layer_mapping:
                if "hidden_states" in student_output and "hidden_states" in teacher_output:
                    total_loss = total_loss + self.intermediate_weight * DistillationLosses.intermediate_layer_loss(
                        student_output["hidden_states"],
                        teacher_output["hidden_states"],
                        self.layer_mapping,
                    )

            # Attention loss
            if self.attention_weight > 0:
                if "attentions" in student_output and "attentions" in teacher_output:
                    total_loss = total_loss + self.attention_weight * DistillationLosses.attention_transfer_loss(
                        student_output["attentions"],
                        teacher_output["attentions"],
                        self.layer_mapping,
                    )

        return total_loss

    def _get_device(self, output) -> torch.device:
        """Get device from output."""
        if isinstance(output, torch.Tensor):
            return output.device
        if isinstance(output, dict):
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v.device
        return torch.device("cpu")
