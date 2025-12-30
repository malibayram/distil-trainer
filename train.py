import json
import subprocess
import sys
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from distil_trainer import DistilTrainer, DistilTrainerConfig, DataConfig

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

    # Configure Trainer
    config = DistilTrainerConfig(
        teacher_model="google/embeddinggemma-300m",
        student_model=local_student_path,  # Use local path with fix
        output_dir="./distilled_model",
        device="cuda",
        
        # Configure data directly in config
        data_config=DataConfig(
            train_data="alibayram/cosmos-corpus-00-5",
            text_column="text",
            batch_size=32 # Adjust based on GPU memory
        )
    )

    trainer = DistilTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
