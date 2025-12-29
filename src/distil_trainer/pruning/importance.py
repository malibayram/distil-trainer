"""Importance estimation for pruning."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ImportanceEstimator:
    """
    Methods for estimating component importance.

    Supports importance estimation for:
    - Layers (for depth pruning)
    - Hidden dimensions (for width pruning)
    - Intermediate dimensions (for MLP pruning)
    - Attention heads (for head pruning)

    Example:
        >>> estimator = ImportanceEstimator(model)
        >>> layer_importance = estimator.activation_based_layer_importance(sentences)
        >>> hidden_importance = estimator.hidden_dimension_importance(sentences)
    """

    def __init__(self, model: SentenceTransformer | PreTrainedModel):
        """
        Initialize the ImportanceEstimator.

        Args:
            model: The model to estimate importance for.
        """
        self.model = model
        self.device = next(model.parameters()).device

    def activation_based_layer_importance(
        self,
        sentences: list[str],
        batch_size: int = 32,
    ) -> dict[int, float]:
        """
        Estimate layer importance based on activation magnitudes.

        Higher activation magnitude suggests higher importance.

        Args:
            sentences: Sentences to use for estimation.
            batch_size: Batch size for inference.

        Returns:
            Dictionary mapping layer index to importance score.
        """
        self.model.eval()

        # Get the transformer model
        if isinstance(self.model, SentenceTransformer):
            transformer = self.model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
            tokenizer = self.model.tokenizer
        else:
            auto_model = self.model
            tokenizer = None

        # Get encoder layers
        encoder_layers = self._get_encoder_layers(auto_model)
        num_layers = len(encoder_layers)

        # Track activations per layer
        layer_activations = {i: 0.0 for i in range(num_layers)}
        num_samples = 0

        # Register hooks to capture activations
        handles = []
        activation_sums = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                activation_sums[layer_idx] = activation_sums.get(layer_idx, 0) + hidden_states.abs().mean().item()
            return hook

        for i, layer in enumerate(encoder_layers):
            handle = layer.register_forward_hook(make_hook(i))
            handles.append(handle)

        try:
            with torch.no_grad():
                for i in range(0, len(sentences), batch_size):
                    batch_sentences = sentences[i:i + batch_size]

                    if isinstance(self.model, SentenceTransformer):
                        self.model.encode(batch_sentences, convert_to_tensor=True)
                    else:
                        if tokenizer:
                            inputs = tokenizer(
                                batch_sentences,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                            ).to(self.device)
                            self.model(**inputs)

                    num_samples += 1

            # Average the activation sums
            for layer_idx in range(num_layers):
                layer_activations[layer_idx] = activation_sums.get(layer_idx, 0) / max(num_samples, 1)

        finally:
            for handle in handles:
                handle.remove()

        return layer_activations

    def drop_layer_importance(
        self,
        sentences: list[str],
        batch_size: int = 32,
    ) -> dict[int, float]:
        """
        Estimate layer importance by measuring embedding similarity when layer is dropped.

        Lower similarity when dropped = higher importance.

        Args:
            sentences: Sentences to use for estimation.
            batch_size: Batch size for inference.

        Returns:
            Dictionary mapping layer index to importance score.
        """
        if not isinstance(self.model, SentenceTransformer):
            raise ValueError("drop_layer_importance only works with SentenceTransformer models")

        # Get reference embeddings
        with torch.no_grad():
            reference_embeddings = self.model.encode(
                sentences[:min(len(sentences), 500)],
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # Get transformer model
        transformer = self.model._first_module()
        if hasattr(transformer, "auto_model"):
            auto_model = transformer.auto_model
        else:
            auto_model = transformer

        encoder_layers = self._get_encoder_layers(auto_model)
        num_layers = len(encoder_layers)

        importance = {}

        for layer_idx in range(num_layers):
            # Temporarily remove the layer
            original_layer = encoder_layers[layer_idx]

            # Replace with identity
            encoder_layers[layer_idx] = nn.Identity()

            try:
                with torch.no_grad():
                    dropped_embeddings = self.model.encode(
                        sentences[:min(len(sentences), 500)],
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )

                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    reference_embeddings, dropped_embeddings, dim=1
                ).mean().item()

                # Lower similarity = higher importance
                importance[layer_idx] = 1 - similarity

            finally:
                # Restore the layer
                encoder_layers[layer_idx] = original_layer

        return importance

    def hidden_dimension_importance(
        self,
        sentences: list[str],
        method: Literal["activation", "magnitude", "taylor"] = "activation",
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Estimate importance of each hidden dimension.

        Args:
            sentences: Sentences to use for estimation.
            method: Importance estimation method.
            batch_size: Batch size for inference.

        Returns:
            Tensor of importance scores for each hidden dimension.
        """
        self.model.eval()

        if isinstance(self.model, SentenceTransformer):
            transformer = self.model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
        else:
            auto_model = self.model

        hidden_size = auto_model.config.hidden_size
        importance = torch.zeros(hidden_size, device=self.device)

        if method == "activation":
            # Track output activations
            num_samples = 0

            with torch.no_grad():
                for i in range(0, len(sentences), batch_size):
                    batch_sentences = sentences[i:i + batch_size]

                    if isinstance(self.model, SentenceTransformer):
                        embeddings = self.model.encode(
                            batch_sentences,
                            convert_to_tensor=True,
                            show_progress_bar=False,
                        )
                        importance += embeddings.abs().mean(dim=0)
                    else:
                        # For regular models, would need tokenizer
                        pass

                    num_samples += 1

            importance = importance / max(num_samples, 1)

        elif method == "magnitude":
            # Use weight magnitudes from final layer
            encoder_layers = self._get_encoder_layers(auto_model)
            last_layer = encoder_layers[-1]

            # Sum up weight magnitudes
            for name, param in last_layer.named_parameters():
                if param.dim() >= 2:
                    # Aggregate across the hidden dimension
                    importance += param.abs().sum(dim=0)[:hidden_size] if param.size(-1) >= hidden_size else torch.zeros(hidden_size, device=self.device)

        return importance

    def intermediate_dimension_importance(
        self,
        sentences: list[str],
        method: Literal["activation", "magnitude"] = "activation",
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Estimate importance of each intermediate (MLP) dimension.

        Args:
            sentences: Sentences to use for estimation.
            method: Importance estimation method.
            batch_size: Batch size for inference.

        Returns:
            Tensor of importance scores for each intermediate dimension.
        """
        self.model.eval()

        if isinstance(self.model, SentenceTransformer):
            transformer = self.model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
        else:
            auto_model = self.model

        intermediate_size = auto_model.config.intermediate_size
        importance = torch.zeros(intermediate_size, device=self.device)

        if method == "magnitude":
            encoder_layers = self._get_encoder_layers(auto_model)

            for layer in encoder_layers:
                if hasattr(layer, "intermediate") and hasattr(layer.intermediate, "dense"):
                    weight = layer.intermediate.dense.weight
                    importance += weight.abs().mean(dim=1)

            importance = importance / len(encoder_layers)

        return importance

    def attention_head_importance(
        self,
        sentences: list[str],
        method: Literal["activation", "attention_entropy"] = "activation",
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Estimate importance of each attention head.

        Args:
            sentences: Sentences to use for estimation.
            method: Importance estimation method.
            batch_size: Batch size for inference.

        Returns:
            Tensor of importance scores for each attention head.
        """
        self.model.eval()

        if isinstance(self.model, SentenceTransformer):
            transformer = self.model._first_module()
            if hasattr(transformer, "auto_model"):
                auto_model = transformer.auto_model
            else:
                auto_model = transformer
        else:
            auto_model = self.model

        num_heads = auto_model.config.num_attention_heads
        importance = torch.zeros(num_heads, device=self.device)

        if method == "activation":
            encoder_layers = self._get_encoder_layers(auto_model)
            head_dim = auto_model.config.hidden_size // num_heads

            for layer in encoder_layers:
                if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                    self_attn = layer.attention.self
                    if hasattr(self_attn, "query"):
                        weight = self_attn.query.weight
                        # Reshape to [num_heads, head_dim, hidden_size]
                        weight_per_head = weight.view(num_heads, head_dim, -1)
                        importance += weight_per_head.abs().mean(dim=(1, 2))

            importance = importance / len(encoder_layers)

        return importance

    def _get_encoder_layers(self, model) -> nn.ModuleList:
        """Get encoder layers from the model."""
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Could not find encoder layers")
