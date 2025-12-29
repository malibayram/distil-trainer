"""Metrics for evaluating distillation quality."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel

from sentence_transformers import SentenceTransformer


class DistillationMetrics:
    """Metrics for evaluating distillation quality."""

    @staticmethod
    def embedding_mse(
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> float:
        """Mean squared error between embeddings."""
        return torch.nn.functional.mse_loss(student_embeddings, teacher_embeddings).item()

    @staticmethod
    def embedding_cosine_similarity(
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> float:
        """Average cosine similarity between embeddings."""
        return torch.nn.functional.cosine_similarity(
            student_embeddings, teacher_embeddings, dim=-1
        ).mean().item()

    @staticmethod
    def compression_ratio(
        student_model: SentenceTransformer | PreTrainedModel,
        teacher_model: SentenceTransformer | PreTrainedModel,
    ) -> float:
        """Parameter count ratio (teacher / student)."""
        student_params = sum(p.numel() for p in student_model.parameters())
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        return teacher_params / student_params

    @staticmethod
    def speedup_factor(
        student_model: SentenceTransformer | PreTrainedModel,
        teacher_model: SentenceTransformer | PreTrainedModel,
        input_batch: dict | list[str],
        num_runs: int = 10,
    ) -> float:
        """Inference speedup of student over teacher."""
        import time

        # Warm up
        if isinstance(student_model, SentenceTransformer):
            student_model.encode(input_batch[:1])
            teacher_model.encode(input_batch[:1])

        # Time student
        start = time.time()
        for _ in range(num_runs):
            if isinstance(student_model, SentenceTransformer):
                student_model.encode(input_batch)
            else:
                student_model(**input_batch)
        student_time = time.time() - start

        # Time teacher
        start = time.time()
        for _ in range(num_runs):
            if isinstance(teacher_model, SentenceTransformer):
                teacher_model.encode(input_batch)
            else:
                teacher_model(**input_batch)
        teacher_time = time.time() - start

        return teacher_time / student_time
