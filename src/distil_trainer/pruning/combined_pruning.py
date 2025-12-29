"""Combined depth and width pruning."""

from __future__ import annotations

import logging
from typing import Literal

from transformers import PreTrainedModel

from sentence_transformers import SentenceTransformer

from distil_trainer.core.config import CombinedPruningConfig, LayerReductionConfig, WidthPruningConfig
from distil_trainer.pruning.depth_pruning import DepthPruner
from distil_trainer.pruning.width_pruning import WidthPruner

logger = logging.getLogger(__name__)


class CombinedPruner:
    """
    Combines depth and width pruning for maximum compression.

    Example:
        >>> config = CombinedPruningConfig(
        ...     depth_config=LayerReductionConfig(num_layers_to_keep=8),
        ...     width_config=WidthPruningConfig(target_hidden_size=3072),
        ...     pruning_order="depth_first"
        ... )
        >>> pruner = CombinedPruner(model)
        >>> pruned_model = pruner.prune(config)
    """

    def __init__(
        self,
        model: SentenceTransformer | PreTrainedModel,
        calibration_data: list[str] | None = None,
    ):
        """
        Initialize the CombinedPruner.

        Args:
            model: The model to prune.
            calibration_data: Optional calibration data for importance estimation.
        """
        self.model = model
        self.calibration_data = calibration_data

    def prune(
        self,
        config: CombinedPruningConfig,
        calibration_data: list[str] | None = None,
    ) -> SentenceTransformer | PreTrainedModel:
        """
        Apply combined pruning based on configuration.

        Args:
            config: Combined pruning configuration.
            calibration_data: Optional calibration data.

        Returns:
            New model with both depth and width reduced.
        """
        data = calibration_data or self.calibration_data
        model = self.model

        if config.pruning_order == "depth_first":
            model = self._apply_depth_pruning(model, config.depth_config, data)
            model = self._apply_width_pruning(model, config.width_config, data)
        elif config.pruning_order == "width_first":
            model = self._apply_width_pruning(model, config.width_config, data)
            model = self._apply_depth_pruning(model, config.depth_config, data)
        elif config.pruning_order == "interleaved":
            # Apply in iterations, alternating
            for i in range(config.num_iterations):
                logger.info(f"Pruning iteration {i + 1}/{config.num_iterations}")
                if i % 2 == 0:
                    model = self._apply_depth_pruning(model, config.depth_config, data)
                else:
                    model = self._apply_width_pruning(model, config.width_config, data)
        else:
            raise ValueError(f"Unknown pruning order: {config.pruning_order}")

        # Log final statistics
        original_params = sum(p.numel() for p in self.model.parameters())
        final_params = sum(p.numel() for p in model.parameters())
        reduction = 1 - (final_params / original_params)
        logger.info(f"Total parameter reduction: {original_params:,} -> {final_params:,} ({reduction:.1%})")

        return model

    def _apply_depth_pruning(
        self,
        model: SentenceTransformer | PreTrainedModel,
        config: LayerReductionConfig | None,
        calibration_data: list[str] | None,
    ) -> SentenceTransformer | PreTrainedModel:
        """Apply depth pruning if configured."""
        if config is None:
            return model

        logger.info("Applying depth pruning...")
        pruner = DepthPruner(model, calibration_data)
        return pruner.prune(
            layers_to_keep=config.layers_to_keep,
            num_layers_to_keep=config.num_layers_to_keep,
            layers_to_drop=config.layers_to_drop,
            layer_selection=config.layer_selection,
        )

    def _apply_width_pruning(
        self,
        model: SentenceTransformer | PreTrainedModel,
        config: WidthPruningConfig | None,
        calibration_data: list[str] | None,
    ) -> SentenceTransformer | PreTrainedModel:
        """Apply width pruning if configured."""
        if config is None:
            return model

        logger.info("Applying width pruning...")
        pruner = WidthPruner(model, calibration_data)
        return pruner.prune(config, calibration_data)
