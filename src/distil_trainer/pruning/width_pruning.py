"""Width pruning for transformer models."""

from __future__ import annotations

import copy
import logging
from typing import Literal

import torch
from torch import nn
from transformers import PreTrainedModel

from sentence_transformers import SentenceTransformer

from distil_trainer.core.config import WidthPruningConfig
from distil_trainer.pruning.importance import ImportanceEstimator

logger = logging.getLogger(__name__)


class WidthPruner:
    """
    Handles width reduction in transformer models.

    Width pruning reduces:
    - Hidden size (embedding dimension)
    - Intermediate size (MLP hidden dimension)
    - Number of attention heads

    Example:
        >>> config = WidthPruningConfig(
        ...     target_hidden_size=3072,
        ...     target_intermediate_size=9216,
        ... )
        >>> pruner = WidthPruner(model)
        >>> pruned_model = pruner.prune(config)
    """

    def __init__(
        self,
        model: SentenceTransformer | PreTrainedModel,
        calibration_data: list[str] | None = None,
    ):
        """
        Initialize the WidthPruner.

        Args:
            model: The model to prune.
            calibration_data: Optional calibration data for importance estimation.
        """
        self.model = model
        self.calibration_data = calibration_data
        self._config = self._get_model_config()

    def _get_model_config(self):
        """Get the model configuration."""
        if isinstance(self.model, SentenceTransformer):
            transformer = self.model._first_module()
            if hasattr(transformer, "auto_model"):
                return transformer.auto_model.config
            return transformer.config
        return self.model.config

    def compute_importance(
        self,
        dimension: Literal["hidden", "intermediate", "heads"],
        calibration_data: list[str] | None = None,
        method: Literal["activation", "magnitude", "taylor", "wanda"] = "activation",
        num_samples: int = 1024,
    ) -> torch.Tensor:
        """
        Compute importance scores for the specified dimension.

        Args:
            dimension: Which dimension to compute importance for.
            calibration_data: Sentences to use for calibration.
            method: Importance estimation method.
            num_samples: Number of samples to use.

        Returns:
            Tensor of importance scores.
        """
        data = calibration_data or self.calibration_data

        estimator = ImportanceEstimator(self.model)

        if dimension == "hidden":
            return estimator.hidden_dimension_importance(data[:num_samples], method=method)
        elif dimension == "intermediate":
            return estimator.intermediate_dimension_importance(data[:num_samples], method=method)
        elif dimension == "heads":
            return estimator.attention_head_importance(data[:num_samples], method=method)
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

    def prune(
        self,
        config: WidthPruningConfig,
        calibration_data: list[str] | None = None,
    ) -> SentenceTransformer | PreTrainedModel:
        """
        Apply width pruning based on configuration.

        Args:
            config: Width pruning configuration.
            calibration_data: Optional calibration data.

        Returns:
            New model with reduced width.
        """
        data = calibration_data or self.calibration_data

        # Determine target dimensions
        current_hidden = getattr(self._config, "hidden_size", None)
        current_intermediate = getattr(self._config, "intermediate_size", None)
        current_heads = getattr(self._config, "num_attention_heads", None)

        target_hidden = config.target_hidden_size
        target_intermediate = config.target_intermediate_size
        target_heads = config.target_num_attention_heads

        # Use ratios if absolute values not specified
        if target_hidden is None and config.hidden_size_ratio is not None:
            target_hidden = int(current_hidden * config.hidden_size_ratio)
        if target_intermediate is None and config.intermediate_size_ratio is not None:
            target_intermediate = int(current_intermediate * config.intermediate_size_ratio)
        if target_heads is None and config.attention_head_ratio is not None:
            target_heads = int(current_heads * config.attention_head_ratio)

        logger.info(f"Width pruning targets:")
        if target_hidden:
            logger.info(f"  Hidden size: {current_hidden} -> {target_hidden}")
        if target_intermediate:
            logger.info(f"  Intermediate size: {current_intermediate} -> {target_intermediate}")
        if target_heads:
            logger.info(f"  Attention heads: {current_heads} -> {target_heads}")

        # Create a deep copy of the model
        pruned_model = copy.deepcopy(self.model)

        # Compute importance scores if we have calibration data
        hidden_importance = None
        intermediate_importance = None
        head_importance = None

        if data is not None:
            estimator = ImportanceEstimator(self.model)
            if target_hidden and target_hidden < current_hidden:
                hidden_importance = estimator.hidden_dimension_importance(
                    data[:config.calibration_samples],
                    method=config.importance_method,
                )
            if target_intermediate and target_intermediate < current_intermediate:
                intermediate_importance = estimator.intermediate_dimension_importance(
                    data[:config.calibration_samples],
                    method=config.importance_method,
                )
            if target_heads and target_heads < current_heads:
                head_importance = estimator.attention_head_importance(
                    data[:config.calibration_samples],
                    method=config.importance_method,
                )

        # Apply pruning
        if target_hidden and target_hidden < current_hidden:
            pruned_model = self._prune_hidden_size(
                pruned_model, target_hidden, hidden_importance
            )

        if target_intermediate and target_intermediate < current_intermediate:
            pruned_model = self._prune_intermediate_size(
                pruned_model, target_intermediate, intermediate_importance
            )

        if target_heads and target_heads < current_heads:
            pruned_model = self._prune_attention_heads(
                pruned_model, target_heads, head_importance
            )

        # Log statistics
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        reduction = 1 - (pruned_params / original_params)
        logger.info(f"Parameter reduction: {original_params:,} -> {pruned_params:,} ({reduction:.1%})")

        return pruned_model

    def _prune_hidden_size(
        self,
        model: SentenceTransformer | PreTrainedModel,
        target_size: int,
        importance: torch.Tensor | None = None,
    ) -> SentenceTransformer | PreTrainedModel:
        """Prune the hidden dimension size."""
        logger.info(f"Pruning hidden size to {target_size}")

        # Get auto_model
        if isinstance(model, SentenceTransformer):
            transformer = model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
        else:
            auto_model = model

        config = auto_model.config
        current_size = config.hidden_size

        # Determine which dimensions to keep
        if importance is not None:
            # Keep highest importance dimensions
            _, indices = torch.topk(importance, target_size)
            keep_indices = indices.sort().values
        else:
            # Keep first N dimensions
            keep_indices = torch.arange(target_size)

        # Update embedding layer
        if hasattr(auto_model, "embeddings"):
            embeddings = auto_model.embeddings
            if hasattr(embeddings, "word_embeddings"):
                old_weight = embeddings.word_embeddings.weight.data
                new_weight = old_weight[:, keep_indices]
                embeddings.word_embeddings = nn.Embedding(
                    old_weight.size(0), target_size
                )
                embeddings.word_embeddings.weight.data = new_weight

            if hasattr(embeddings, "position_embeddings"):
                old_weight = embeddings.position_embeddings.weight.data
                new_weight = old_weight[:, keep_indices]
                embeddings.position_embeddings = nn.Embedding(
                    old_weight.size(0), target_size
                )
                embeddings.position_embeddings.weight.data = new_weight

            if hasattr(embeddings, "LayerNorm"):
                old_ln = embeddings.LayerNorm
                embeddings.LayerNorm = nn.LayerNorm(target_size, eps=old_ln.eps)
                embeddings.LayerNorm.weight.data = old_ln.weight.data[keep_indices]
                embeddings.LayerNorm.bias.data = old_ln.bias.data[keep_indices]

        # Update encoder layers
        encoder_layers = self._get_encoder_layers(auto_model)
        for layer in encoder_layers:
            self._prune_layer_hidden(layer, keep_indices, target_size)

        # Update config
        config.hidden_size = target_size

        return model

    def _prune_intermediate_size(
        self,
        model: SentenceTransformer | PreTrainedModel,
        target_size: int,
        importance: torch.Tensor | None = None,
    ) -> SentenceTransformer | PreTrainedModel:
        """Prune the MLP intermediate dimension."""
        logger.info(f"Pruning intermediate size to {target_size}")

        # Get auto_model
        if isinstance(model, SentenceTransformer):
            transformer = model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
        else:
            auto_model = model

        config = auto_model.config
        current_size = config.intermediate_size

        # Determine which dimensions to keep
        if importance is not None:
            _, indices = torch.topk(importance, target_size)
            keep_indices = indices.sort().values
        else:
            keep_indices = torch.arange(target_size)

        # Update encoder layers
        encoder_layers = self._get_encoder_layers(auto_model)
        for layer in encoder_layers:
            self._prune_layer_intermediate(layer, keep_indices, target_size)

        # Update config
        config.intermediate_size = target_size

        return model

    def _prune_attention_heads(
        self,
        model: SentenceTransformer | PreTrainedModel,
        target_heads: int,
        importance: torch.Tensor | None = None,
    ) -> SentenceTransformer | PreTrainedModel:
        """Prune the number of attention heads."""
        logger.info(f"Pruning attention heads to {target_heads}")

        # Get auto_model
        if isinstance(model, SentenceTransformer):
            transformer = model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
        else:
            auto_model = model

        config = auto_model.config
        current_heads = config.num_attention_heads
        head_dim = config.hidden_size // current_heads

        # Determine which heads to keep
        if importance is not None:
            _, indices = torch.topk(importance, target_heads)
            keep_head_indices = indices.sort().values
        else:
            keep_head_indices = torch.arange(target_heads)

        # Update encoder layers
        encoder_layers = self._get_encoder_layers(auto_model)
        for layer in encoder_layers:
            self._prune_layer_heads(layer, keep_head_indices, head_dim)

        # Update config
        config.num_attention_heads = target_heads

        return model

    def _get_encoder_layers(self, model) -> nn.ModuleList:
        """Get encoder layers from the model."""
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Could not find encoder layers")

    def _prune_layer_hidden(
        self,
        layer: nn.Module,
        keep_indices: torch.Tensor,
        target_size: int,
    ) -> None:
        """Prune hidden dimension in a single layer."""
        # This is model-specific and may need customization for different architectures
        # Here we provide a generic implementation for BERT-like models

        # Prune attention
        if hasattr(layer, "attention"):
            attention = layer.attention
            if hasattr(attention, "self"):
                self_attn = attention.self
                # Query, Key, Value projections
                for proj_name in ["query", "key", "value"]:
                    if hasattr(self_attn, proj_name):
                        proj = getattr(self_attn, proj_name)
                        new_weight = proj.weight.data[:, keep_indices]
                        new_proj = nn.Linear(target_size, proj.out_features, bias=proj.bias is not None)
                        new_proj.weight.data = new_weight
                        if proj.bias is not None:
                            new_proj.bias.data = proj.bias.data
                        setattr(self_attn, proj_name, new_proj)

            if hasattr(attention, "output"):
                output = attention.output
                if hasattr(output, "dense"):
                    old_dense = output.dense
                    new_dense = nn.Linear(old_dense.in_features, target_size, bias=old_dense.bias is not None)
                    new_dense.weight.data = old_dense.weight.data[keep_indices]
                    if old_dense.bias is not None:
                        new_dense.bias.data = old_dense.bias.data[keep_indices]
                    output.dense = new_dense

                if hasattr(output, "LayerNorm"):
                    old_ln = output.LayerNorm
                    output.LayerNorm = nn.LayerNorm(target_size, eps=old_ln.eps)
                    output.LayerNorm.weight.data = old_ln.weight.data[keep_indices]
                    output.LayerNorm.bias.data = old_ln.bias.data[keep_indices]

        # Prune intermediate/MLP
        if hasattr(layer, "intermediate"):
            intermediate = layer.intermediate
            if hasattr(intermediate, "dense"):
                old_dense = intermediate.dense
                new_dense = nn.Linear(target_size, old_dense.out_features, bias=old_dense.bias is not None)
                new_dense.weight.data = old_dense.weight.data[:, keep_indices]
                if old_dense.bias is not None:
                    new_dense.bias.data = old_dense.bias.data
                intermediate.dense = new_dense

        if hasattr(layer, "output"):
            output = layer.output
            if hasattr(output, "dense"):
                old_dense = output.dense
                new_dense = nn.Linear(old_dense.in_features, target_size, bias=old_dense.bias is not None)
                new_dense.weight.data = old_dense.weight.data[keep_indices]
                if old_dense.bias is not None:
                    new_dense.bias.data = old_dense.bias.data[keep_indices]
                output.dense = new_dense

            if hasattr(output, "LayerNorm"):
                old_ln = output.LayerNorm
                output.LayerNorm = nn.LayerNorm(target_size, eps=old_ln.eps)
                output.LayerNorm.weight.data = old_ln.weight.data[keep_indices]
                output.LayerNorm.bias.data = old_ln.bias.data[keep_indices]

    def _prune_layer_intermediate(
        self,
        layer: nn.Module,
        keep_indices: torch.Tensor,
        target_size: int,
    ) -> None:
        """Prune intermediate dimension in a single layer."""
        if hasattr(layer, "intermediate"):
            intermediate = layer.intermediate
            if hasattr(intermediate, "dense"):
                old_dense = intermediate.dense
                new_dense = nn.Linear(old_dense.in_features, target_size, bias=old_dense.bias is not None)
                new_dense.weight.data = old_dense.weight.data[keep_indices]
                if old_dense.bias is not None:
                    new_dense.bias.data = old_dense.bias.data[keep_indices]
                intermediate.dense = new_dense

        if hasattr(layer, "output"):
            output = layer.output
            if hasattr(output, "dense"):
                old_dense = output.dense
                new_dense = nn.Linear(target_size, old_dense.out_features, bias=old_dense.bias is not None)
                new_dense.weight.data = old_dense.weight.data[:, keep_indices]
                if old_dense.bias is not None:
                    new_dense.bias.data = old_dense.bias.data
                output.dense = new_dense

    def _prune_layer_heads(
        self,
        layer: nn.Module,
        keep_head_indices: torch.Tensor,
        head_dim: int,
    ) -> None:
        """Prune attention heads in a single layer."""
        if not hasattr(layer, "attention"):
            return

        attention = layer.attention
        if not hasattr(attention, "self"):
            return

        self_attn = attention.self
        target_heads = len(keep_head_indices)
        new_head_size = target_heads * head_dim

        # Compute dimension indices for keeping specific heads
        dim_indices = []
        for head_idx in keep_head_indices:
            start = head_idx * head_dim
            end = start + head_dim
            dim_indices.extend(range(start, end))
        dim_indices = torch.tensor(dim_indices)

        # Prune Query, Key, Value projections
        for proj_name in ["query", "key", "value"]:
            if hasattr(self_attn, proj_name):
                proj = getattr(self_attn, proj_name)
                new_weight = proj.weight.data[dim_indices]
                new_proj = nn.Linear(proj.in_features, new_head_size, bias=proj.bias is not None)
                new_proj.weight.data = new_weight
                if proj.bias is not None:
                    new_proj.bias.data = proj.bias.data[dim_indices]
                setattr(self_attn, proj_name, new_proj)

        # Update num_attention_heads attribute
        if hasattr(self_attn, "num_attention_heads"):
            self_attn.num_attention_heads = target_heads
        if hasattr(self_attn, "all_head_size"):
            self_attn.all_head_size = new_head_size
