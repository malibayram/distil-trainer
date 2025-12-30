"""Main DistilTrainer class for knowledge distillation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)

from sentence_transformers import SentenceTransformer

from distil_trainer.core.config import (
    DistilTrainerConfig,
    DistillationConfig,
    LayerReductionConfig,
    WidthPruningConfig,
)

logger = logging.getLogger(__name__)


class DistilTrainer:
    """
    Main trainer class for knowledge distillation.

    Supports multiple distillation strategies:
    - Classical embedding distillation (MSE/Cosine loss)
    - Layer reduction (depth pruning)
    - Width pruning
    - Combined pruning

    Example:
        >>> config = DistilTrainerConfig(
        ...     teacher_model="sentence-transformers/all-mpnet-base-v2",
        ...     student_model="sentence-transformers/paraphrase-TinyBERT-L6-v2",
        ...     output_dir="./distilled_model"
        ... )
        >>> trainer = DistilTrainer(config)
        >>> trainer.load_data(train_data="sentence-transformers/all-nli")
        >>> trainer.train()
        >>> trainer.save_model("./final_model")
    """

    def __init__(self, config: DistilTrainerConfig):
        """
        Initialize the DistilTrainer.

        Args:
            config: Configuration for distillation training.
        """
        self.config = config
        self.device = self._get_device()

        # Initialize models
        self.teacher_model = self._load_teacher_model()
        self.student_model = self._initialize_student_model()

        # PCA for dimension reduction (if needed)
        self.pca = None
        self.teacher_projection = None

        # Data
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_metric = float("inf")

        logger.info(f"Initialized DistilTrainer with device: {self.device}")
        logger.info(f"Teacher model: {self._get_model_info(self.teacher_model)}")
        logger.info(f"Student model: {self._get_model_info(self.student_model)}")

    def _get_device(self) -> torch.device:
        """Determine the device to use for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_teacher_model(self) -> SentenceTransformer | PreTrainedModel:
        """Load the teacher model."""
        teacher = self.config.teacher_model

        if isinstance(teacher, str):
            logger.info(f"Loading teacher model from: {teacher}")
            teacher = SentenceTransformer(teacher)

        teacher.to(self.device)
        teacher.eval()
        return teacher

    def _initialize_student_model(self) -> SentenceTransformer | PreTrainedModel:
        """Initialize the student model based on the configured strategy."""
        strategy = self.config.student_init_strategy

        if strategy == "from_pretrained":
            return self._load_pretrained_student()
        elif strategy in ("layer_reduction", "depth_pruning"):
            return self._create_layer_reduced_student()
        elif strategy == "width_pruning":
            return self._create_width_pruned_student()
        elif strategy == "combined_pruning":
            return self._create_combined_pruned_student()
        else:
            raise ValueError(f"Unknown student initialization strategy: {strategy}")

    def _load_pretrained_student(self) -> SentenceTransformer | PreTrainedModel:
        """Load a pretrained student model."""
        student = self.config.student_model

        if student is None:
            raise ValueError("student_model must be provided for 'from_pretrained' strategy")

        if isinstance(student, str):
            logger.info(f"Loading student model from: {student}")
            student = SentenceTransformer(student)

        student.to(self.device)
        return student

    def _create_layer_reduced_student(self) -> SentenceTransformer | PreTrainedModel:
        """Create a student model by removing layers from the teacher."""
        from distil_trainer.pruning import DepthPruner

        pruning_config = self.config.pruning_config
        if not isinstance(pruning_config, LayerReductionConfig):
            raise ValueError("pruning_config must be LayerReductionConfig for layer reduction")

        logger.info("Creating layer-reduced student from teacher")

        # Clone teacher model
        student = SentenceTransformer(self.config.teacher_model)

        # Apply depth pruning
        pruner = DepthPruner(student)
        student = pruner.prune(
            layers_to_keep=pruning_config.layers_to_keep,
            num_layers_to_keep=pruning_config.num_layers_to_keep,
            layers_to_drop=pruning_config.layers_to_drop,
            layer_selection=pruning_config.layer_selection,
        )

        student.to(self.device)
        return student

    def _create_width_pruned_student(self) -> SentenceTransformer | PreTrainedModel:
        """Create a student model by pruning width dimensions."""
        from distil_trainer.pruning import WidthPruner

        pruning_config = self.config.pruning_config
        if not isinstance(pruning_config, WidthPruningConfig):
            raise ValueError("pruning_config must be WidthPruningConfig for width pruning")

        logger.info("Creating width-pruned student from teacher")

        # Clone teacher model
        student = SentenceTransformer(self.config.teacher_model)

        # Apply width pruning
        pruner = WidthPruner(student)
        student = pruner.prune(pruning_config)

        student.to(self.device)
        return student

    def _create_combined_pruned_student(self) -> SentenceTransformer | PreTrainedModel:
        """Create a student model using both depth and width pruning."""
        from distil_trainer.pruning import CombinedPruner

        pruning_config = self.config.pruning_config

        logger.info("Creating combined-pruned student from teacher")

        # Clone teacher model
        student = SentenceTransformer(self.config.teacher_model)

        # Apply combined pruning
        pruner = CombinedPruner(student)
        student = pruner.prune(pruning_config)

        student.to(self.device)
        return student

    def _get_model_info(self, model: SentenceTransformer | PreTrainedModel) -> str:
        """Get a string representation of model info."""
        if isinstance(model, SentenceTransformer):
            num_params = sum(p.numel() for p in model.parameters())
            embedding_dim = model.get_sentence_embedding_dimension()
            return f"SentenceTransformer(params={num_params:,}, dim={embedding_dim})"
        else:
            num_params = sum(p.numel() for p in model.parameters())
            return f"PreTrainedModel(params={num_params:,})"

    def load_data(
        self,
        train_data: str | Dataset | None = None,
        eval_data: str | Dataset | None = None,
        test_data: str | Dataset | None = None,
        text_column: str | None = None,
    ) -> None:
        """
        Load training, evaluation, and test datasets.

        Args:
            train_data: Path or name of training dataset, or Dataset object.
            eval_data: Path or name of evaluation dataset, or Dataset object.
            test_data: Path or name of test dataset, or Dataset object.
            text_column: Name of the column containing text/sentences.
                        Overrides config.data_config.text_column if provided.
        """
        if text_column is not None:
            self.config.data_config.text_column = text_column
            logger.info(f"Set text_column to: {text_column}")

        if train_data is not None:
            self.train_dataset = self._load_dataset(train_data, "train")
            logger.info(f"Loaded training dataset: {len(self.train_dataset)} samples")

        if eval_data is not None:
            self.eval_dataset = self._load_dataset(eval_data, "validation")
            logger.info(f"Loaded evaluation dataset: {len(self.eval_dataset)} samples")

        if test_data is not None:
            self.test_dataset = self._load_dataset(test_data, "test")
            logger.info(f"Loaded test dataset: {len(self.test_dataset)} samples")

    def _load_dataset(self, data: str | Dataset, split: str = "train") -> Dataset:
        """Load a dataset from a path or name."""
        if isinstance(data, Dataset):
            dataset = data
        else:
            logger.info(f"Loading dataset: {data}")

            try:
                dataset = load_dataset(data, split=split)
            except Exception:
                # Try loading as a DatasetDict and getting the split
                dataset_dict = load_dataset(data)
                if isinstance(dataset_dict, DatasetDict):
                    if split in dataset_dict:
                        dataset = dataset_dict[split]
                    else:
                        # Use the first available split
                        dataset = list(dataset_dict.values())[0]
                else:
                    dataset = dataset_dict

        # Apply max_samples limit if configured
        max_samples = self.config.data_config.max_samples
        if max_samples is not None and max_samples > 0:
            original_size = len(dataset)
            if max_samples < original_size:
                dataset = dataset.select(range(max_samples))
                logger.info(f"Limited dataset from {original_size} to {max_samples} samples")

        return dataset

    def setup_pca_projection(self) -> None:
        """Set up PCA projection if student dimension is smaller than teacher."""
        if not isinstance(self.teacher_model, SentenceTransformer):
            return
        if not isinstance(self.student_model, SentenceTransformer):
            return

        teacher_dim = self.teacher_model.get_sentence_embedding_dimension()
        student_dim = self.student_model.get_sentence_embedding_dimension()

        if student_dim >= teacher_dim:
            logger.info("Student dimension >= teacher dimension, no PCA needed")
            return

        logger.info(f"Setting up PCA projection: {teacher_dim} -> {student_dim}")

        # Collect sample sentences for PCA
        if self.train_dataset is None:
            raise ValueError("Training dataset required for PCA projection")

        text_column = self.config.data_config.text_column
        num_samples = min(
            self.config.distillation_config.pca_num_samples,
            len(self.train_dataset),
        )

        sample_sentences = self.train_dataset[:num_samples][text_column]

        # Compute teacher embeddings
        logger.info(f"Computing teacher embeddings for {num_samples} samples")
        with torch.no_grad():
            embeddings = self.teacher_model.encode(
                sample_sentences,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=self.config.distillation_config.teacher_inference_batch_size,
            )

        # Fit PCA
        logger.info("Fitting PCA...")
        self.pca = PCA(n_components=student_dim)
        self.pca.fit(embeddings)

        # Create projection layer for teacher
        from distil_trainer.models import DenseProjection

        self.teacher_projection = DenseProjection(
            in_features=teacher_dim,
            out_features=student_dim,
            weights=torch.tensor(self.pca.components_, dtype=torch.float32),
        )
        self.teacher_projection.to(self.device)

        logger.info(f"PCA projection ready: explained variance ratio = {sum(self.pca.explained_variance_ratio_):.4f}")

    def precompute_teacher_embeddings(self) -> None:
        """Precompute teacher embeddings for the training dataset."""
        if not self.config.distillation_config.precompute_teacher_embeddings:
            return

        if self.train_dataset is None:
            raise ValueError("Training dataset required")

        logger.info("Precomputing teacher embeddings...")

        text_column = self.config.data_config.text_column
        sentences = self.train_dataset[text_column]

        batch_size = self.config.distillation_config.teacher_inference_batch_size

        with torch.no_grad():
            embeddings = self.teacher_model.encode(
                sentences,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
            )

            # Apply projection if needed
            if self.teacher_projection is not None:
                embeddings = self.teacher_projection(embeddings)

        # Add embeddings to dataset
        if isinstance(embeddings, torch.Tensor):
            embeddings_list = embeddings.cpu().tolist()
        else:
            # Already a list (e.g., when encode returns list directly)
            embeddings_list = embeddings
        self.train_dataset = self.train_dataset.add_column("label", embeddings_list)

        logger.info("Teacher embeddings precomputed and cached")

    def train(self) -> dict[str, float]:
        """
        Run the distillation training.

        Returns:
            Dictionary of training metrics.
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset required. Call load_data() first.")

        logger.info("Starting distillation training...")

        # Setup WandB
        is_wandb_avail = False
        if "wandb" in self.config.training_config.report_to or (self.config.wandb_config.project is not None):
            try:
                import wandb
                from dataclasses import asdict
                
                # Check if already initialized
                if wandb.run is None:
                    wandb.init(
                        project=self.config.wandb_config.project,
                        entity=self.config.wandb_config.entity,
                        name=self.config.wandb_config.name or self.config.training_config.run_name,
                        tags=self.config.wandb_config.tags,
                        group=self.config.wandb_config.group,
                        notes=self.config.wandb_config.notes,
                        config=asdict(self.config),
                    )
                is_wandb_avail = True
            except ImportError:
                logger.warning("wandb not installed, skipping logging")

        # Setup PCA if needed
        if self.config.distillation_config.use_pca_projection:
            self.setup_pca_projection()

        # Precompute teacher embeddings if enabled
        self.precompute_teacher_embeddings()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Get loss function
        loss_fn = self._get_loss_function()

        # Create data loader
        train_dataloader = self._create_dataloader(self.train_dataset, shuffle=True)

        # Training loop
        training_config = self.config.training_config
        num_epochs = training_config.num_train_epochs
        total_steps = len(train_dataloader) * num_epochs

        if training_config.max_steps > 0:
            total_steps = min(total_steps, training_config.max_steps)

        logger.info(f"Training for {num_epochs} epochs, {total_steps} total steps")

        self.student_model.train()
        self.global_step = 0

        avg_epoch_loss = 0.0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=False,
            )

            for batch in progress_bar:
                loss = self._training_step(batch, loss_fn)

                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                current_loss = loss.item()
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

                # Logging
                if self.global_step % training_config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(f"Step {self.global_step}: loss = {avg_loss:.4f}")
                    
                    if is_wandb_avail:
                        wandb.log(
                            {
                                "train/loss": current_loss,
                                "train/avg_loss": avg_loss,
                                "train/epoch": epoch + (num_batches / len(train_dataloader)),
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                            },
                            step=self.global_step
                        )

                # Evaluation
                if (
                    training_config.eval_strategy == "steps"
                    and self.global_step % training_config.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step}: {eval_metrics}")
                    
                    if is_wandb_avail:
                        wandb_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        wandb.log(wandb_metrics, step=self.global_step)

                    # Save best model
                    if self._is_better_metric(eval_metrics):
                        self._save_checkpoint("best")

                # Save checkpoint
                if self.global_step % training_config.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")

                # Check max steps
                if training_config.max_steps > 0 and self.global_step >= training_config.max_steps:
                    break

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} completed: avg_loss = {avg_epoch_loss:.4f}")

            if is_wandb_avail:
                wandb.log({"train/epoch_loss": avg_epoch_loss}, step=self.global_step)

            if training_config.max_steps > 0 and self.global_step >= training_config.max_steps:
                break

        logger.info("Training completed!")

        # Load best model if configured
        if training_config.load_best_model_at_end:
            self._load_checkpoint("best")
        
        # Push to Hub at end if configured
        if self.config.hub_config.push_to_hub:
             self._push_to_hub_with_config()

        if is_wandb_avail:
            wandb.finish()

        return {"train_loss": avg_epoch_loss}

    def _setup_optimizer(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        training_config = self.config.training_config

        # Optimizer
        if training_config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=training_config.learning_rate,
                betas=(training_config.adam_beta1, training_config.adam_beta2),
                eps=training_config.adam_epsilon,
                weight_decay=training_config.weight_decay,
            )
        elif training_config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.student_model.parameters(),
                lr=training_config.learning_rate,
            )
        elif training_config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.student_model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config.optimizer}")

        # Scheduler
        num_training_steps = self._get_num_training_steps()
        warmup_steps = training_config.warmup_steps
        if warmup_steps == 0 and training_config.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * training_config.warmup_ratio)

        self.scheduler = get_scheduler(
            training_config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _get_num_training_steps(self) -> int:
        """Calculate the total number of training steps."""
        if self.train_dataset is None:
            return 0

        training_config = self.config.training_config
        num_batches = len(self.train_dataset) // training_config.per_device_train_batch_size
        total_steps = num_batches * training_config.num_train_epochs

        if training_config.max_steps > 0:
            total_steps = min(total_steps, training_config.max_steps)

        return total_steps

    def _get_loss_function(self) -> Callable:
        """Get the loss function based on configuration."""
        from distil_trainer.distillation import DistillationLosses

        loss_type = self.config.distillation_config.loss_type

        if loss_type == "mse":
            return DistillationLosses.mse_loss
        elif loss_type == "cosine":
            return DistillationLosses.cosine_loss
        elif loss_type == "kl_divergence":
            temperature = self.config.distillation_config.temperature
            return lambda s, t: DistillationLosses.kl_divergence_loss(s, t, temperature)
        elif loss_type == "combined":
            from distil_trainer.distillation import CombinedDistillationLoss

            return CombinedDistillationLoss(
                logit_weight=self.config.distillation_config.logit_loss_weight,
                embedding_weight=self.config.distillation_config.embedding_loss_weight,
                intermediate_weight=self.config.distillation_config.intermediate_loss_weight,
                attention_weight=self.config.distillation_config.attention_loss_weight,
                temperature=self.config.distillation_config.temperature,
                layer_mapping=self.config.distillation_config.layer_mapping,
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader from a dataset."""
        from distil_trainer.data import DistillationCollator

        batch_size = self.config.training_config.per_device_train_batch_size

        # Get tokenizer from student model
        tokenizer = None
        if isinstance(self.student_model, SentenceTransformer):
            tokenizer = self.student_model.tokenizer

        collator = DistillationCollator(
            tokenizer=tokenizer,
            max_length=self.config.data_config.max_seq_length,
            text_column=self.config.data_config.text_column,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=self.config.data_config.num_workers,
            pin_memory=True,
        )

    def _training_step(self, batch: dict[str, torch.Tensor], loss_fn: Callable) -> torch.Tensor:
        """Perform a single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Get student embeddings
        student_output = self.student_model(batch)

        # Get teacher embeddings (from precomputed or compute on-the-fly)
        if "label" in batch:
            teacher_output = batch["label"]
        else:
            with torch.no_grad():
                teacher_output = self.teacher_model(batch)
                if self.teacher_projection is not None:
                    teacher_output = self.teacher_projection(teacher_output)

        # Compute loss
        loss = loss_fn(student_output, teacher_output)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.training_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                self.config.training_config.max_grad_norm,
            )

        self.optimizer.step()
        self.scheduler.step()

        return loss

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the student model.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}

        self.student_model.eval()
        eval_dataloader = self._create_dataloader(self.eval_dataset, shuffle=False)

        total_loss = 0.0
        num_batches = 0
        loss_fn = self._get_loss_function()

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                student_output = self.student_model(batch)

                if "label" in batch:
                    teacher_output = batch["label"]
                else:
                    teacher_output = self.teacher_model(batch)
                    if self.teacher_projection is not None:
                        teacher_output = self.teacher_projection(teacher_output)

                loss = loss_fn(student_output, teacher_output)
                total_loss += loss.item()
                num_batches += 1

        self.student_model.train()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"eval_loss": avg_loss}

    def _is_better_metric(self, metrics: dict[str, float]) -> bool:
        """Check if current metrics are better than the best."""
        metric_name = self.config.training_config.metric_for_best_model
        if metric_name not in metrics:
            return False

        current_value = metrics[metric_name]
        is_better = current_value < self.best_metric

        if not self.config.training_config.greater_is_better:
            is_better = current_value < self.best_metric
        else:
            is_better = current_value > self.best_metric

        if is_better:
            self.best_metric = current_value

        return is_better

    def _save_checkpoint(self, name: str) -> None:
        """Save a checkpoint."""
        output_dir = Path(self.config.output_dir) / "checkpoints" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.student_model, SentenceTransformer):
            self.student_model.save(str(output_dir))
        else:
            self.student_model.save_pretrained(output_dir)

        logger.info(f"Saved checkpoint: {output_dir}")

        # Push to Hub Logic
        if self.config.hub_config.push_to_hub and self.config.hub_config.push_to_hub_interval == "every_save":
             self._push_to_hub_with_config(commit_message=f"Upload checkpoint {name}")

    def _push_to_hub_with_config(self, commit_message: str = "Upload distilled model") -> None:
        """Helper to push to hub using config settings."""
        if not self.config.hub_config.push_to_hub:
            return
            
        repo_id = self.config.hub_config.hub_model_id
        if not repo_id:
             logger.warning("push_to_hub is True but hub_model_id is not set. Skipping push.")
             return
             
        try:
            url = self.push_to_hub(
                repo_id=repo_id,
                private=self.config.hub_config.hub_private_repo,
                commit_message=commit_message,
            )
            logger.info(f"Pushed model to Hub: {url}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")

    def _load_checkpoint(self, name: str) -> None:
        """Load a checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints" / name

        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_dir}")
            return

        if isinstance(self.student_model, SentenceTransformer):
            self.student_model = SentenceTransformer(str(checkpoint_dir))
        else:
            self.student_model = self.student_model.__class__.from_pretrained(checkpoint_dir)

        self.student_model.to(self.device)
        logger.info(f"Loaded checkpoint: {checkpoint_dir}")

    def save_model(self, output_path: str | None = None) -> None:
        """
        Save the trained student model.

        Args:
            output_path: Path to save the model. Defaults to output_dir/final.
        """
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "final")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.student_model, SentenceTransformer):
            self.student_model.save(str(output_dir))
        else:
            self.student_model.save_pretrained(output_dir)

        logger.info(f"Model saved to: {output_dir}")

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Upload distilled model",
    ) -> str:
        """
        Push the model to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub.
            private: Whether the repository should be private.
            commit_message: Commit message for the upload.

        Returns:
            URL of the uploaded model.
        """
        if isinstance(self.student_model, SentenceTransformer):
            return self.student_model.push_to_hub(
                repo_id=repo_id,
                private=private,
                commit_message=commit_message,
            )
        else:
            return self.student_model.push_to_hub(
                repo_id=repo_id,
                private=private,
                commit_message=commit_message,
            )
