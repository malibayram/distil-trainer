"""Training callbacks for distillation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class DistillationCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass

    @abstractmethod
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_step_begin(self, trainer: Any, step: int) -> None:
        """Called at the beginning of each training step."""
        pass

    @abstractmethod
    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        """Called at the end of each training step."""
        pass


class LoggingCallback(DistillationCallback):
    """Callback for logging training progress."""

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_begin(self, trainer: Any) -> None:
        logger.info("Training started")

    def on_train_end(self, trainer: Any) -> None:
        logger.info("Training completed")

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        logger.info(f"Epoch {epoch + 1} started")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        logger.info(f"Epoch {epoch + 1} completed: {metrics}")

    def on_step_begin(self, trainer: Any, step: int) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if step % self.log_every_n_steps == 0:
            logger.info(f"Step {step}: loss = {loss:.4f}")


class EvaluationCallback(DistillationCallback):
    """Callback for running evaluation during training."""

    def __init__(self, eval_every_n_steps: int = 500):
        self.eval_every_n_steps = eval_every_n_steps

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        eval_metrics = trainer.evaluate()
        logger.info(f"End of epoch {epoch + 1} evaluation: {eval_metrics}")

    def on_step_begin(self, trainer: Any, step: int) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if step % self.eval_every_n_steps == 0:
            eval_metrics = trainer.evaluate()
            logger.info(f"Step {step} evaluation: {eval_metrics}")


class CheckpointCallback(DistillationCallback):
    """Callback for saving checkpoints during training."""

    def __init__(self, save_every_n_steps: int = 500, save_total_limit: int = 3):
        self.save_every_n_steps = save_every_n_steps
        self.save_total_limit = save_total_limit
        self.saved_checkpoints: list[str] = []

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        trainer.save_model()

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        pass

    def on_step_begin(self, trainer: Any, step: int) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if step % self.save_every_n_steps == 0:
            checkpoint_name = f"checkpoint-{step}"
            trainer._save_checkpoint(checkpoint_name)
            self.saved_checkpoints.append(checkpoint_name)

            # Remove old checkpoints
            while len(self.saved_checkpoints) > self.save_total_limit:
                old_checkpoint = self.saved_checkpoints.pop(0)
                # Could add logic to delete the old checkpoint directory


class EarlyStoppingCallback(DistillationCallback):
    """Callback for early stopping based on evaluation metrics."""

    def __init__(
        self,
        metric_name: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def on_train_begin(self, trainer: Any) -> None:
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        if self.metric_name not in metrics:
            return

        current_value = metrics[self.metric_name]

        if self.mode == "min":
            is_improvement = current_value < self.best_value - self.min_delta
        else:
            is_improvement = current_value > self.best_value + self.min_delta

        if is_improvement:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")

    def on_step_begin(self, trainer: Any, step: int) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        pass
