"""Minimal Embedding Distillation Trainer using precomputed teacher embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TargetType = Literal["pre_dense", "final"]


@dataclass
class EmbeddingTrainerConfig:
    """Configuration for embedding distillation training."""
    
    # Model
    student_model: str
    
    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Loss
    loss_type: Literal["mse", "cosine"] = "mse"
    
    # Target
    target_type: TargetType = "final"
    
    # Columns
    text_column: str = "text"
    final_embedding_column: str = "teacher_embedding_final"
    pre_dense_embedding_column: str = "teacher_embedding_pre_dense"
    
    # Output
    output_dir: str = "./trained_model"
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = None
    
    # Device
    device: str = "auto"
    
    # Optimization
    use_fp16: bool = False
    use_bf16: bool = False
    compile_model: bool = False
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    
    # WandB
    use_wandb: bool = False
    wandb_project: str = "distillation"
    wandb_run_name: str | None = None


class EmbeddingDistillationTrainer:
    """
    Minimal trainer for distillation using precomputed teacher embeddings.
    
    Supports two training modes:
    - "pre_dense": Train to match pre-Dense layer output (trains transformer only)
    - "final": Train to match final output (trains full model including Dense)
    
    Example:
        >>> config = EmbeddingTrainerConfig(
        ...     student_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     target_type="final",
        ...     use_bf16=True,
        ...     compile_model=True,
        ... )
        >>> trainer = EmbeddingDistillationTrainer(config)
        >>> trainer.train("your-username/corpus-with-embeddings")
    """
    
    def __init__(self, config: EmbeddingTrainerConfig):
        self.config = config
        self.device = self._get_device()
        self._original_model = None  # Store uncompiled model for internal access
        self.student_model = self._load_student()
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        
    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _is_flash_attention_available(self) -> bool:
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _load_student(self) -> SentenceTransformer:
        logger.info(f"Loading student: {self.config.student_model}")
        
        # Flash attention
        model_kwargs = {}
        if self.config.use_flash_attention:
            if self._is_flash_attention_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Enabling Flash Attention 2")
            else:
                logger.warning("flash-attn not installed, skipping Flash Attention")
        
        model = SentenceTransformer(self.config.student_model, model_kwargs=model_kwargs)
        model.to(self.device)
        
        # Precision
        if self.config.use_bf16 and self.device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)
            logger.info("Using bfloat16 precision")
        elif self.config.use_fp16:
            model = model.half()
            logger.info("Using float16 precision")
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Store uncompiled model reference for operations that need internal access
        self._original_model = model
        
        # Compile
        if self.config.compile_model:
            try:
                compiled = torch.compile(model)
                logger.info("Model compiled with torch.compile()")
                return compiled
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}")
        
        return model
    
    def _get_loss_fn(self):
        if self.config.loss_type == "mse":
            return torch.nn.MSELoss()
        elif self.config.loss_type == "cosine":
            return torch.nn.CosineEmbeddingLoss()
        raise ValueError(f"Unknown loss: {self.config.loss_type}")
    
    def _get_target_column(self) -> str:
        if self.config.target_type == "pre_dense":
            return self.config.pre_dense_embedding_column
        return self.config.final_embedding_column
    
    def _has_dense_layer(self) -> bool:
        for module in self._original_model:
            if module.__class__.__name__ == 'Dense':
                return True
        return False
    
    def _get_pre_dense_index(self) -> int:
        for idx, module in enumerate(self._original_model):
            if module.__class__.__name__ == 'Dense':
                return idx
        return len(self._original_model)
    
    def train(
        self,
        dataset: str | Dataset,
        split: str = "train",
        eval_dataset: str | Dataset | None = None,
    ) -> dict:
        """
        Train the student model.
        
        Args:
            dataset: HuggingFace dataset with precomputed embeddings.
            split: Dataset split to use.
            eval_dataset: Optional evaluation dataset.
            
        Returns:
            Training metrics.
        """
        # Load dataset
        if isinstance(dataset, str):
            logger.info(f"Loading dataset: {dataset}")
            train_ds = load_dataset(dataset, split=split)
        else:
            train_ds = dataset
            
        target_column = self._get_target_column()
        text_column = self.config.text_column
        
        logger.info(f"Training with target: {target_column}")
        logger.info(f"Dataset size: {len(train_ds)}")
        
        # Initialize WandB
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config={
                        "student_model": self.config.student_model,
                        "target_type": self.config.target_type,
                        "learning_rate": self.config.learning_rate,
                        "num_epochs": self.config.num_epochs,
                        "batch_size": self.config.batch_size,
                        "loss_type": self.config.loss_type,
                    }
                )
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self._wandb = None
        else:
            self._wandb = None
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup scheduler
        total_steps = (len(train_ds) // self.config.batch_size) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Loss function
        loss_fn = self._get_loss_fn()
        
        # Training loop
        self.student_model.train()
        self.global_step = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches
            indices = list(range(len(train_ds)))
            
            progress = tqdm(
                range(0, len(train_ds), self.config.batch_size),
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            
            for start_idx in progress:
                end_idx = min(start_idx + self.config.batch_size, len(train_ds))
                batch_indices = indices[start_idx:end_idx]
                batch = train_ds.select(batch_indices)
                
                # Get texts and targets
                texts = batch[text_column]
                targets = torch.tensor(batch[target_column]).to(self.device)
                
                # Forward pass with gradient tracking
                if self.config.target_type == "pre_dense" and self._has_dense_layer():
                    student_output = self._forward_pre_dense(texts)
                else:
                    student_output = self._forward_with_gradients(texts)
                
                # Match dtype (for bf16/fp16)
                targets = targets.to(dtype=student_output.dtype)
                
                # Compute loss
                if self.config.loss_type == "cosine":
                    labels = torch.ones(len(texts), dtype=student_output.dtype).to(self.device)
                    loss = loss_fn(student_output, targets, labels)
                else:
                    loss = loss_fn(student_output, targets)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {self.global_step}: loss = {avg_loss:.4f}, lr = {lr:.2e}")
                    
                    if self._wandb:
                        self._wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch + 1,
                        }, step=self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_model(f"{self.config.output_dir}/checkpoint-{self.global_step}")
                    # Push to hub at every checkpoint
                    if self.config.push_to_hub and self.config.hub_model_id:
                        logger.info(f"Pushing checkpoint {self.global_step} to hub...")
                        self.push_to_hub()
            
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}: avg_loss = {avg_epoch_loss:.4f}")
        
        # Save final model
        self.save_model(self.config.output_dir)
        
        # Push to hub
        if self.config.push_to_hub and self.config.hub_model_id:
            self.push_to_hub()
        
        return {"train_loss": avg_epoch_loss}
    
    def _forward_pre_dense(self, texts: list[str]) -> torch.Tensor:
        """Forward pass through model up to Dense layer."""
        # Use original model for internal access (compiled model wraps it)
        model = self._original_model
        pre_dense_idx = self._get_pre_dense_index()
        
        features = model.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        
        for idx in range(pre_dense_idx):
            features = model[idx](features)
        
        embedding = features.get('sentence_embedding')
        if embedding is None:
            embedding = features.get('token_embeddings')
            if embedding is not None and len(embedding.shape) == 3:
                embedding = embedding.mean(dim=1)
        
        return embedding
    
    def _forward_with_gradients(self, texts: list[str]) -> torch.Tensor:
        """Forward pass through full model with gradient tracking."""
        # Use original model for internal access (compiled model wraps it)
        model = self._original_model
        features = model.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        
        # Run through all modules
        for module in model:
            features = module(features)
        
        embedding = features.get('sentence_embedding')
        if embedding is None:
            embedding = features.get('token_embeddings')
            if embedding is not None and len(embedding.shape) == 3:
                embedding = embedding.mean(dim=1)
        
        return embedding
    
    def save_model(self, path: str) -> None:
        """Save the student model."""
        # Use original model for save (compiled model doesn't have .save())
        self._original_model.save(path)
        logger.info(f"Model saved: {path}")
    
    def push_to_hub(self) -> str:
        """Push model to HuggingFace Hub."""
        # Use original model for push (compiled model doesn't have .push_to_hub())
        url = self._original_model.push_to_hub(
            repo_id=self.config.hub_model_id,
            token=self.config.hub_token,
            exist_ok=True,
        )
        logger.info(f"Pushed to: {url}")
        return url
