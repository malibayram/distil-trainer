"""Unit tests for distil_trainer package."""

import pytest
import torch

from distil_trainer import DistilTrainer, DistilTrainerConfig


class TestConfig:
    """Test configuration classes."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = DistilTrainerConfig(
            teacher_model="sentence-transformers/all-MiniLM-L6-v2",
            output_dir="./test_output",
        )
        assert config.teacher_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.student_init_strategy == "from_pretrained"

    def test_layer_reduction_config(self):
        """Test layer reduction configuration."""
        from distil_trainer.core.config import LayerReductionConfig

        config = LayerReductionConfig(layers_to_keep=[0, 2, 4])
        assert config.layers_to_keep == [0, 2, 4]
        assert config.method == "depth"


class TestLosses:
    """Test loss functions."""

    def test_mse_loss(self):
        """Test MSE loss."""
        from distil_trainer.distillation.losses import DistillationLosses

        student = torch.randn(32, 768)
        teacher = torch.randn(32, 768)

        loss = DistillationLosses.mse_loss(student, teacher)
        assert loss.item() >= 0

    def test_cosine_loss(self):
        """Test cosine loss."""
        from distil_trainer.distillation.losses import DistillationLosses

        student = torch.randn(32, 768)
        teacher = torch.randn(32, 768)

        loss = DistillationLosses.cosine_loss(student, teacher)
        assert 0 <= loss.item() <= 2

    def test_kl_divergence_loss(self):
        """Test KL divergence loss."""
        from distil_trainer.distillation.losses import DistillationLosses

        student_logits = torch.randn(32, 1000)
        teacher_logits = torch.randn(32, 1000)

        loss = DistillationLosses.kl_divergence_loss(
            student_logits, teacher_logits, temperature=2.0
        )
        assert loss.item() >= 0


class TestDatasets:
    """Test dataset classes."""

    def test_sentence_distillation_dataset(self):
        """Test SentenceDistillationDataset."""
        from distil_trainer.data.datasets import SentenceDistillationDataset

        sentences = ["Hello world", "How are you?"]
        dataset = SentenceDistillationDataset(sentences=sentences)

        assert len(dataset) == 2
        assert dataset[0]["sentence"] == "Hello world"


class TestCollators:
    """Test collator classes."""

    def test_distillation_collator(self):
        """Test DistillationCollator without tokenizer."""
        from distil_trainer.data.collators import DistillationCollator

        collator = DistillationCollator(text_column="sentence")
        batch = [{"sentence": "Hello"}, {"sentence": "World"}]

        result = collator(batch)
        assert result["sentence"] == ["Hello", "World"]


class TestLayers:
    """Test model layers."""

    def test_dense_projection(self):
        """Test DenseProjection layer."""
        from distil_trainer.models.layers import DenseProjection

        projection = DenseProjection(in_features=768, out_features=256)
        embeddings = torch.randn(32, 768)

        output = projection(embeddings)
        assert output.shape == (32, 256)

    def test_pooling_layer_mean(self):
        """Test mean pooling."""
        from distil_trainer.models.layers import PoolingLayer

        pooler = PoolingLayer(pooling_mode="mean")
        token_embeddings = torch.randn(4, 16, 768)
        attention_mask = torch.ones(4, 16)

        output = pooler(token_embeddings, attention_mask)
        assert output.shape == (4, 768)

    def test_pooling_layer_cls(self):
        """Test CLS pooling."""
        from distil_trainer.models.layers import PoolingLayer

        pooler = PoolingLayer(pooling_mode="cls")
        token_embeddings = torch.randn(4, 16, 768)
        attention_mask = torch.ones(4, 16)

        output = pooler(token_embeddings, attention_mask)
        assert output.shape == (4, 768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
