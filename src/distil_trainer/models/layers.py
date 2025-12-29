"""Custom layers for distillation models."""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseProjection(nn.Module):
    """
    Dense projection layer for dimension reduction.

    Used to project teacher embeddings to student dimension via PCA weights.

    Example:
        >>> projection = DenseProjection(in_features=768, out_features=256)
        >>> reduced = projection(embeddings)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weights: torch.Tensor | None = None,
    ):
        """
        Initialize the projection layer.

        Args:
            in_features: Input dimension (teacher).
            out_features: Output dimension (student).
            bias: Whether to include bias term.
            weights: Optional initial weights (e.g., from PCA).
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if weights is not None:
            with torch.no_grad():
                self.linear.weight.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply projection."""
        return self.linear(x)


class PoolingLayer(nn.Module):
    """
    Pooling layer for converting token embeddings to sentence embeddings.

    Supports multiple pooling strategies.

    Example:
        >>> pooler = PoolingLayer(pooling_mode="mean")
        >>> sentence_embedding = pooler(token_embeddings, attention_mask)
    """

    def __init__(
        self,
        pooling_mode: str = "mean",
    ):
        """
        Initialize the pooling layer.

        Args:
            pooling_mode: One of "mean", "cls", "max", "weighted_mean".
        """
        super().__init__()
        self.pooling_mode = pooling_mode

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply pooling to token embeddings.

        Args:
            token_embeddings: Token embeddings [batch, seq_len, dim].
            attention_mask: Attention mask [batch, seq_len].

        Returns:
            Pooled embeddings [batch, dim].
        """
        if self.pooling_mode == "cls":
            return token_embeddings[:, 0]

        elif self.pooling_mode == "max":
            # Mask out padding tokens
            token_embeddings = token_embeddings.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float("-inf")
            )
            return token_embeddings.max(dim=1).values

        elif self.pooling_mode == "weighted_mean":
            # Position-weighted mean
            weights = torch.arange(
                1, token_embeddings.size(1) + 1,
                device=token_embeddings.device,
                dtype=token_embeddings.dtype,
            )
            weights = weights.unsqueeze(0).unsqueeze(-1) * attention_mask.unsqueeze(-1)
            return (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1)

        else:  # mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
