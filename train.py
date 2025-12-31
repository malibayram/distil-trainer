#!/usr/bin/env python3
"""Training script for distillation using precomputed embeddings."""

import os
os.environ["WANDB_API_KEY"] = "afbbb4f"

from huggingface_hub import hf_hub_download, HfApi
import json

# First, fix the tokenizer_config.json issue
print("Fixing tokenizer_config.json...")
try:
    path = hf_hub_download('alibayram/cloned_sentence_transformer', 'tokenizer_config.json')
    with open(path) as f:
        config = json.load(f)

    if isinstance(config.get('extra_special_tokens'), list):
        config['extra_special_tokens'] = {}
        with open('/tmp/tokenizer_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj='/tmp/tokenizer_config.json',
            path_in_repo='tokenizer_config.json',
            repo_id='alibayram/cloned_sentence_transformer',
            token='hf_cBjjmmV'
        )
        print("Tokenizer config fixed and uploaded!")
    else:
        print("Tokenizer config already correct.")
except Exception as e:
    print(f"Warning: Could not fix tokenizer: {e}")

# Now run training
from distil_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

config = EmbeddingTrainerConfig(
    student_model="alibayram/cloned_sentence_transformer",
    target_type="final",
    num_epochs=1,
    batch_size=16,
    learning_rate=2e-5,
    use_bf16=True,
    gradient_checkpointing=True,
    
    # Dataset columns
    text_column="text",
    final_embedding_column="teacher_embedding_final",
    
    # Output
    output_dir="./trained_model",
    save_steps=1000,
    logging_steps=100,
    
    # WandB
    use_wandb=True,
    wandb_project="distillation",
    wandb_run_name="cloned-distillation-run1",
    
    # Push to Hub
    push_to_hub=True,
    hub_model_id="alibayram/distilled-sentence-transformer",
    hub_token="hf_cBjjmmVC",
)

trainer = EmbeddingDistillationTrainer(config)
metrics = trainer.train("alibayram/cosmos-corpus-0-05-with-embeddings")

print(f"Final loss: {metrics['train_loss']:.4f}")
