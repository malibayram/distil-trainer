#!/usr/bin/env python3
"""
Example script for training with precomputed teacher embeddings.

Usage:
    python train_with_embeddings.py
"""

from distil_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

# Configuration
config = EmbeddingTrainerConfig(
    # Student model to train
    student_model="sentence-transformers/all-MiniLM-L6-v2",
    
    # Target type: "final" or "pre_dense"
    # - "final": Train full model to match final embeddings
    # - "pre_dense": Train transformer only (before Dense layer)
    target_type="final",
    
    # Dataset columns (must match your generated dataset)
    text_column="text",
    final_embedding_column="teacher_embedding_final",
    pre_dense_embedding_column="teacher_embedding_pre_dense",
    
    # Training params
    learning_rate=2e-5,
    num_epochs=3,
    batch_size=32,
    loss_type="mse",  # or "cosine"
    
    # Optimization options (for faster training on GPU):
    # use_bf16=True,            # bfloat16 (best for A100/H100)
    # use_fp16=True,            # float16 (faster, less memory)
    # compile_model=True,       # torch.compile() for PyTorch 2.0+
    # use_flash_attention=True, # Flash Attention 2 (if installed)
    # gradient_checkpointing=True,  # Reduce memory at cost of speed
    
    # Output
    output_dir="./trained_student",
    save_steps=1000,
    logging_steps=100,
    
    # Optional: Push to Hub
    # push_to_hub=True,
    # hub_model_id="your-username/distilled-model",
    # hub_token="your-token",
)

# Initialize trainer
trainer = EmbeddingDistillationTrainer(config)

# Train using precomputed embeddings dataset
# This dataset should have been created by TeacherEmbeddingsGenerator
metrics = trainer.train(
    dataset="your-username/corpus-with-embeddings",  # or local Dataset object
    split="train",
)

print(f"Training complete! Loss: {metrics['train_loss']:.4f}")
