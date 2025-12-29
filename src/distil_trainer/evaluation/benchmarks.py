"""Benchmark runners for model evaluation."""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run standard benchmarks for evaluation."""

    def __init__(self, model: SentenceTransformer | Any):
        """
        Initialize the benchmark runner.

        Args:
            model: Model to evaluate.
        """
        self.model = model

    def run_stsb(self, split: str = "test") -> dict[str, float]:
        """Run STS Benchmark evaluation."""
        from datasets import load_dataset
        from distil_trainer.evaluation.evaluators import EmbeddingSimilarityEvaluator

        dataset = load_dataset("sentence-transformers/stsb", split=split)

        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=dataset["sentence1"],
            sentences2=dataset["sentence2"],
            scores=dataset["score"],
            name=f"stsb-{split}",
        )

        return evaluator(self.model)

    def run_mteb(
        self,
        tasks: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Run MTEB benchmark suite.

        Requires mteb package: pip install mteb
        """
        try:
            from mteb import MTEB
        except ImportError:
            logger.warning("MTEB not installed. Install with: pip install mteb")
            return {}

        if tasks is None:
            tasks = ["STS12", "STS13", "STS14", "STS15", "STS16"]

        evaluation = MTEB(tasks=tasks)
        results = evaluation.run(self.model, output_folder=None)

        # Flatten results
        flat_results = {}
        for task_result in results:
            task_name = task_result.task_name
            for metric, value in task_result.scores.items():
                flat_results[f"{task_name}_{metric}"] = value

        return flat_results

    def run_all(self) -> dict[str, dict[str, float]]:
        """Run all available benchmarks."""
        results = {}

        try:
            results["stsb"] = self.run_stsb()
        except Exception as e:
            logger.warning(f"STSB evaluation failed: {e}")

        try:
            results["mteb"] = self.run_mteb()
        except Exception as e:
            logger.warning(f"MTEB evaluation failed: {e}")

        return results
