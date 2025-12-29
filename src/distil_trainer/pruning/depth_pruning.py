"""Depth pruning (layer reduction) for transformer models."""

from __future__ import annotations

import copy
import logging
from typing import Literal

import torch
from torch import nn
from transformers import PreTrainedModel

from sentence_transformers import SentenceTransformer

from distil_trainer.pruning.importance import ImportanceEstimator

logger = logging.getLogger(__name__)


class DepthPruner:
    """
    Handles layer removal from transformer models.

    Depth pruning removes entire transformer layers to reduce model size
    while preserving the overall architecture.

    Example:
        >>> pruner = DepthPruner(model)
        >>> importance = pruner.compute_layer_importance(calibration_data)
        >>> layers_to_keep = pruner.select_layers_to_keep(num_layers=8)
        >>> pruned_model = pruner.prune(layers_to_keep=layers_to_keep)
    """

    def __init__(
        self,
        model: SentenceTransformer | PreTrainedModel,
        calibration_data: list[str] | None = None,
    ):
        """
        Initialize the DepthPruner.

        Args:
            model: The model to prune.
            calibration_data: Optional calibration data for importance estimation.
        """
        self.model = model
        self.calibration_data = calibration_data
        self._transformer = self._get_transformer_module()
        self._original_num_layers = self._get_num_layers()

    def _get_transformer_module(self) -> nn.Module:
        """Get the underlying transformer module."""
        if isinstance(self.model, SentenceTransformer):
            # SentenceTransformer wraps a transformer model
            for module in self.model.modules():
                if hasattr(module, "encoder") and hasattr(module.encoder, "layer"):
                    return module
                if hasattr(module, "layers"):
                    return module
            # Try the first module which is typically the transformer
            return self.model._first_module()
        else:
            return self.model

    def _get_encoder_layers(self) -> nn.ModuleList:
        """Get the encoder layers from the transformer."""
        transformer = self._transformer

        # Try different attribute names used by different models
        if hasattr(transformer, "encoder") and hasattr(transformer.encoder, "layer"):
            return transformer.encoder.layer
        if hasattr(transformer, "encoder") and hasattr(transformer.encoder, "layers"):
            return transformer.encoder.layers
        if hasattr(transformer, "layers"):
            return transformer.layers
        if hasattr(transformer, "auto_model"):
            auto_model = transformer.auto_model
            if hasattr(auto_model, "encoder") and hasattr(auto_model.encoder, "layer"):
                return auto_model.encoder.layer
            if hasattr(auto_model, "layers"):
                return auto_model.layers

        raise ValueError("Could not find encoder layers in model")

    def _get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        return len(self._get_encoder_layers())

    def compute_layer_importance(
        self,
        calibration_data: list[str] | None = None,
        method: Literal["activation", "gradient", "cosine_similarity", "lm_loss"] = "activation",
        num_samples: int = 1024,
    ) -> dict[int, float]:
        """
        Compute importance scores for each layer.

        Args:
            calibration_data: Sentences to use for calibration.
            method: Importance estimation method.
            num_samples: Number of samples to use.

        Returns:
            Dictionary mapping layer index to importance score.
        """
        data = calibration_data or self.calibration_data
        if data is None:
            raise ValueError("Calibration data required for importance estimation")

        estimator = ImportanceEstimator(self.model)

        if method == "activation":
            importance = estimator.activation_based_layer_importance(data[:num_samples])
        elif method == "cosine_similarity":
            importance = estimator.drop_layer_importance(data[:num_samples])
        else:
            importance = estimator.activation_based_layer_importance(data[:num_samples])

        return importance

    def select_layers_to_keep(
        self,
        num_layers: int | None = None,
        ratio: float | None = None,
        strategy: Literal["first", "last", "even", "importance"] = "importance",
        importance_scores: dict[int, float] | None = None,
    ) -> list[int]:
        """
        Select which layers to keep based on the strategy.

        Args:
            num_layers: Number of layers to keep.
            ratio: Ratio of layers to keep (alternative to num_layers).
            strategy: Layer selection strategy.
            importance_scores: Precomputed importance scores (for 'importance' strategy).

        Returns:
            List of layer indices to keep.
        """
        total_layers = self._original_num_layers

        if num_layers is None and ratio is not None:
            num_layers = int(total_layers * ratio)
        if num_layers is None:
            raise ValueError("Either num_layers or ratio must be specified")

        num_layers = min(num_layers, total_layers)

        if strategy == "first":
            return list(range(num_layers))

        elif strategy == "last":
            return list(range(total_layers - num_layers, total_layers))

        elif strategy == "even":
            # Evenly distribute layers
            if num_layers == 1:
                return [0]
            step = (total_layers - 1) / (num_layers - 1)
            return [int(round(i * step)) for i in range(num_layers)]

        elif strategy == "importance":
            if importance_scores is None:
                if self.calibration_data is None:
                    # Fall back to even distribution
                    logger.warning("No importance scores or calibration data, using even distribution")
                    return self.select_layers_to_keep(num_layers=num_layers, strategy="even")
                importance_scores = self.compute_layer_importance()

            # Sort layers by importance and keep the most important ones
            sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            layers_to_keep = sorted([layer_idx for layer_idx, _ in sorted_layers[:num_layers]])
            return layers_to_keep

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def prune(
        self,
        layers_to_keep: list[int] | None = None,
        num_layers_to_keep: int | None = None,
        layers_to_drop: list[int] | None = None,
        layer_selection: Literal["first", "last", "even", "importance", "custom"] = "custom",
    ) -> SentenceTransformer | PreTrainedModel:
        """
        Create a new model with only the specified layers.

        Args:
            layers_to_keep: Explicit list of layer indices to keep.
            num_layers_to_keep: Number of layers to keep (alternative).
            layers_to_drop: Layers to drop (alternative).
            layer_selection: Strategy for selecting layers when using num_layers_to_keep.

        Returns:
            New model with reduced layers.
        """
        # Determine which layers to keep
        if layers_to_keep is not None:
            final_layers_to_keep = layers_to_keep
        elif layers_to_drop is not None:
            all_layers = set(range(self._original_num_layers))
            final_layers_to_keep = sorted(all_layers - set(layers_to_drop))
        elif num_layers_to_keep is not None:
            if layer_selection == "custom":
                layer_selection = "importance"
            final_layers_to_keep = self.select_layers_to_keep(
                num_layers=num_layers_to_keep,
                strategy=layer_selection,
            )
        else:
            raise ValueError("Must specify layers_to_keep, num_layers_to_keep, or layers_to_drop")

        logger.info(f"Keeping layers: {final_layers_to_keep}")
        logger.info(f"Reducing from {self._original_num_layers} to {len(final_layers_to_keep)} layers")

        # Create a deep copy of the model
        pruned_model = copy.deepcopy(self.model)

        # Get the encoder layers
        if isinstance(pruned_model, SentenceTransformer):
            transformer = pruned_model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
        else:
            auto_model = pruned_model

        # Find and replace the layers
        encoder_layers = None
        if hasattr(auto_model, "encoder") and hasattr(auto_model.encoder, "layer"):
            encoder_layers = auto_model.encoder.layer
            parent = auto_model.encoder
            attr_name = "layer"
        elif hasattr(auto_model, "encoder") and hasattr(auto_model.encoder, "layers"):
            encoder_layers = auto_model.encoder.layers
            parent = auto_model.encoder
            attr_name = "layers"
        elif hasattr(auto_model, "layers"):
            encoder_layers = auto_model.layers
            parent = auto_model
            attr_name = "layers"

        if encoder_layers is None:
            raise ValueError("Could not find encoder layers in model")

        # Create new layer list with only kept layers
        new_layers = nn.ModuleList([encoder_layers[i] for i in final_layers_to_keep])
        setattr(parent, attr_name, new_layers)

        # Update config if available
        if hasattr(auto_model, "config"):
            auto_model.config.num_hidden_layers = len(final_layers_to_keep)

        # Log statistics
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        reduction = 1 - (pruned_params / original_params)
        logger.info(f"Parameter reduction: {original_params:,} -> {pruned_params:,} ({reduction:.1%})")

        return pruned_model
