"""Evaluators for distillation quality assessment."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingSimilarityEvaluator:
    """
    Evaluate sentence embedding quality on STS tasks.

    Computes correlation between model similarity and human scores.

    Example:
        >>> evaluator = EmbeddingSimilarityEvaluator(
        ...     sentences1=["Hello", "Hi"],
        ...     sentences2=["World", "Earth"],
        ...     scores=[0.5, 0.8],
        ... )
        >>> results = evaluator(model)
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        batch_size: int = 32,
        name: str = "sts-eval",
        main_similarity: str = "cosine",
        show_progress_bar: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            sentences1: First set of sentences.
            sentences2: Second set of sentences.
            scores: Ground truth similarity scores.
            batch_size: Batch size for encoding.
            name: Name for logging.
            main_similarity: Similarity function to use.
            show_progress_bar: Whether to show progress bar.
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.batch_size = batch_size
        self.name = name
        self.main_similarity = main_similarity
        self.show_progress_bar = show_progress_bar

    def __call__(self, model: SentenceTransformer) -> dict[str, float]:
        """
        Evaluate the model.

        Args:
            model: Model to evaluate.

        Returns:
            Dictionary with correlation scores.
        """
        logger.info(f"Running {self.name} evaluation")

        # Encode sentences
        embeddings1 = model.encode(
            self.sentences1,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )
        embeddings2 = model.encode(
            self.sentences2,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

        # Compute similarities
        if self.main_similarity == "cosine":
            similarities = torch.nn.functional.cosine_similarity(
                embeddings1, embeddings2
            ).cpu().numpy()
        else:
            # Euclidean distance (convert to similarity)
            distances = torch.pairwise_distance(embeddings1, embeddings2).cpu().numpy()
            similarities = 1 / (1 + distances)

        # Compute correlations
        pearson = pearsonr(similarities, self.scores)[0]
        spearman = spearmanr(similarities, self.scores)[0]

        results = {
            f"{self.name}_pearson_{self.main_similarity}": pearson,
            f"{self.name}_spearman_{self.main_similarity}": spearman,
        }

        logger.info(f"{self.name}: Pearson={pearson:.4f}, Spearman={spearman:.4f}")

        return results


class MSEEvaluator:
    """
    Evaluate MSE between student and teacher embeddings.

    Example:
        >>> evaluator = MSEEvaluator(
        ...     source_sentences=["Hello", "World"],
        ...     target_sentences=["Hello", "World"],
        ...     teacher_model=teacher,
        ... )
        >>> results = evaluator(student_model)
    """

    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        teacher_model: SentenceTransformer,
        batch_size: int = 32,
        name: str = "mse-eval",
        show_progress_bar: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            source_sentences: Source sentences.
            target_sentences: Target sentences (same as source for distillation).
            teacher_model: Teacher model for comparison.
            batch_size: Batch size for encoding.
            name: Name for logging.
            show_progress_bar: Whether to show progress bar.
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.teacher_model = teacher_model
        self.batch_size = batch_size
        self.name = name
        self.show_progress_bar = show_progress_bar

        # Precompute teacher embeddings
        with torch.no_grad():
            self.teacher_embeddings = teacher_model.encode(
                source_sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=True,
            )

    def __call__(self, model: SentenceTransformer) -> dict[str, float]:
        """
        Evaluate the model.

        Args:
            model: Student model to evaluate.

        Returns:
            Dictionary with MSE score.
        """
        logger.info(f"Running {self.name} evaluation")

        # Encode with student
        student_embeddings = model.encode(
            self.target_sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

        # Compute MSE
        mse = torch.nn.functional.mse_loss(
            student_embeddings, self.teacher_embeddings
        ).item()

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            student_embeddings, self.teacher_embeddings
        ).mean().item()

        results = {
            f"{self.name}_mse": mse,
            f"{self.name}_cosine_similarity": cosine_sim,
        }

        logger.info(f"{self.name}: MSE={mse:.6f}, Cosine={cosine_sim:.4f}")

        return results


class TranslationEvaluator:
    """
    Evaluate multilingual alignment via translation retrieval.

    Checks if source[i] embedding is closest to target[i].

    Example:
        >>> evaluator = TranslationEvaluator(
        ...     source_sentences=["Hello", "World"],
        ...     target_sentences=["Hallo", "Welt"],
        ... )
        >>> results = evaluator(model)
    """

    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        batch_size: int = 32,
        name: str = "translation-eval",
        show_progress_bar: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            source_sentences: Source language sentences.
            target_sentences: Target language sentences (parallel).
            batch_size: Batch size for encoding.
            name: Name for logging.
            show_progress_bar: Whether to show progress bar.
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.batch_size = batch_size
        self.name = name
        self.show_progress_bar = show_progress_bar

    def __call__(self, model: SentenceTransformer) -> dict[str, float]:
        """
        Evaluate the model.

        Args:
            model: Model to evaluate.

        Returns:
            Dictionary with retrieval accuracy scores.
        """
        logger.info(f"Running {self.name} evaluation")

        # Encode both sets
        source_embeddings = model.encode(
            self.source_sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )
        target_embeddings = model.encode(
            self.target_sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

        # Normalize for cosine similarity
        source_embeddings = torch.nn.functional.normalize(source_embeddings, p=2, dim=1)
        target_embeddings = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(source_embeddings, target_embeddings.t())

        # Get rankings
        correct_indices = torch.arange(len(self.source_sentences), device=similarity_matrix.device)
        rankings = (similarity_matrix.argsort(dim=1, descending=True) == correct_indices.unsqueeze(1)).nonzero()[:, 1]

        # Compute metrics
        acc_at_1 = (rankings == 0).float().mean().item()
        acc_at_5 = (rankings < 5).float().mean().item()
        acc_at_10 = (rankings < 10).float().mean().item()
        mrr = (1.0 / (rankings.float() + 1)).mean().item()

        results = {
            f"{self.name}_accuracy@1": acc_at_1,
            f"{self.name}_accuracy@5": acc_at_5,
            f"{self.name}_accuracy@10": acc_at_10,
            f"{self.name}_mrr": mrr,
        }

        logger.info(f"{self.name}: Acc@1={acc_at_1:.4f}, MRR={mrr:.4f}")

        return results


class SequentialEvaluator:
    """
    Run multiple evaluators sequentially.

    Example:
        >>> evaluator = SequentialEvaluator([evaluator1, evaluator2])
        >>> results = evaluator(model)
    """

    def __init__(
        self,
        evaluators: list[Any],
        main_score_function: Callable[[dict], float] | None = None,
    ):
        """
        Initialize the sequential evaluator.

        Args:
            evaluators: List of evaluators to run.
            main_score_function: Function to compute main score from results.
        """
        self.evaluators = evaluators
        self.main_score_function = main_score_function or (
            lambda x: np.mean([v for v in x.values() if isinstance(v, (int, float))])
        )

    def __call__(self, model: SentenceTransformer) -> dict[str, float]:
        """
        Run all evaluators.

        Args:
            model: Model to evaluate.

        Returns:
            Combined dictionary of all evaluation results.
        """
        all_results = {}

        for evaluator in self.evaluators:
            try:
                results = evaluator(model)
                all_results.update(results)
            except Exception as e:
                logger.warning(f"Evaluator {evaluator} failed: {e}")

        # Compute main score
        all_results["main_score"] = self.main_score_function(all_results)

        return all_results
