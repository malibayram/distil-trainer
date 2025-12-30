import json
import subprocess
import sys
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from datasets import load_dataset
from distil_trainer import DistilTrainer, DistilTrainerConfig, DistillationConfig, TrainingConfig
from distil_trainer.core.config import HubConfig, WandbConfig

# Load environment variables from .env file
load_dotenv()

# 1. Ensure protobuf is installed (Fixes ImportError)
try:
    import google.protobuf
except ImportError:
    print("Installing missing dependency: protobuf...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf"])

# 2. Helper: Download and patch the student model config (Fixes AttributeError)
def get_patched_student_model(model_id, local_dir="./patched_student_model"):
    """
    Downloads the model and fixes known issues in tokenizer_config.json.
    """
    print(f"Downloading and patching model: {model_id}")
    
    # Download model to a local directory (only downloads changed files)
    local_path = snapshot_download(repo_id=model_id, local_dir=local_dir)
    
    # Patch tokenizer_config.json
    config_path = Path(local_path) / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # FIX: 'extra_special_tokens' should be a dict or not present, but sometimes it's a list causing AttributeError
        if "extra_special_tokens" in data and isinstance(data["extra_special_tokens"], list):
            print(f"Fixing malformed 'extra_special_tokens' in {config_path}")
            
            # Backup original
            shutil.copy(config_path, config_path.with_suffix('.json.bak'))
            
            # Remove the malformed field (Tokenizer will fallback to defaults which is usually correct)
            del data["extra_special_tokens"]
            
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
            print("Fixed: 'extra_special_tokens' removed.")
    
    return local_path

# 3. Main Training Setup
def main():
    # Use the local patched model path
    student_model_id = "alibayram/cloned_sentence_transformer"
    local_student_path = get_patched_student_model(student_model_id)

    # Configure Trainer with memory-saving settings
    config = DistilTrainerConfig(
        teacher_model="google/embeddinggemma-300m",
        student_model=local_student_path,  # Use local path with fix
        output_dir="./distilled_model",
        device="cuda",
        # Reduce batch sizes to prevent CUDA OOM
        distillation_config=DistillationConfig(
            teacher_inference_batch_size=16,  # Default is 128
        ),
        training_config=TrainingConfig(
            per_device_train_batch_size=8,    # Default is 64
            per_device_eval_batch_size=8,     # Default is 64
            gradient_accumulation_steps=8,    # Compensate with gradient accumulation
            report_to=["wandb"],              # Enable wandb logging
        ),
        # Push to HuggingFace Hub after training
        hub_config=HubConfig(
            push_to_hub=True,
            hub_model_id="alibayram/distilled-sentence-transformer",
            hub_token=os.getenv("HF_TOKEN"),
        ),
        # Weights & Biases logging
        wandb_config=WandbConfig(
            project="distil-trainer",
            name="embedding-distillation",
        ),
    )

    trainer = DistilTrainer(config)

    # Load the larger gated dataset and split it
    print("Loading dataset: alibayram/cosmos-corpus-0-05")
    dataset = load_dataset("alibayram/cosmos-corpus-0-05", split="train")
    print(f"Total samples: {len(dataset)}")
    
    # Split into 99% train and 1% eval
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    trainer.load_data(
        train_data=train_dataset,
        eval_data=eval_dataset,
        text_column="text"
    )
    trainer.train()

if __name__ == "__main__":
    main()
