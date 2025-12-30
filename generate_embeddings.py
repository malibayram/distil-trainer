#!/usr/bin/env python3
"""
Example script for generating teacher embeddings and pushing to HuggingFace.

Usage:
    python generate_embeddings.py
"""

from distil_trainer import TeacherEmbeddingsGenerator

# Initialize the generator
generator = TeacherEmbeddingsGenerator(
    teacher_model="google/embeddinggemma-300m",
    batch_size=32,  # Adjust based on your GPU memory
)

# Option 1: Final embeddings (generates both pre_dense AND final when Dense layer exists)
dataset = generator.generate(
    source_dataset="alibayram/cosmos-corpus-00-5",
    text_column="text",
    split="train",
    output_type="final",
    max_samples=100,  # Remove this for full dataset
)
print(f"Final embeddings: {dataset.column_names}")

# Option 2: Pre-dense embeddings (before Dense layer, if present)
# dataset = generator.generate(
#     source_dataset="alibayram/cosmos-corpus-00-5",
#     text_column="text",
#     output_type="pre_dense",
#     max_samples=1000,
# )

# Option 3: Hidden states from specific layers (for multi-layer distillation)
# dataset = generator.generate(
#     source_dataset="alibayram/cosmos-corpus-00-5",
#     text_column="text",
#     output_type="hidden_states",
#     layer_indices=[0, 6, 12],  # None = all layers
#     max_samples=1000,
# )

# Push to HuggingFace Hub (all at once)
# generator.push_to_hub(
#     repo_id="your-username/corpus-with-embeddings",
#     token="your-hf-token",  # Or set HF_TOKEN env variable
# )

# Option 4: Generate and push iteratively (for large datasets)
# Pushes checkpoint every 10,000 samples to avoid losing progress
# generator.generate_and_push(
#     source_dataset="alibayram/cosmos-corpus-00-5",
#     repo_id="your-username/corpus-with-embeddings",
#     text_column="text",
#     output_type="final",
#     push_every=10000,  # Push checkpoint every N samples
#     token="your-hf-token",
# )

# Or save locally
generator.save_to_disk("./embeddings_dataset")

print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Find the embedding column (varies by output_type)
embedding_cols = [c for c in dataset.column_names if 'teacher_embedding' in c or 'hidden_state' in c]
if embedding_cols:
    print(f"First embedding shape: {len(dataset[0][embedding_cols[0]])}")
