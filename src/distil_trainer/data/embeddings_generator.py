"""Teacher Embeddings Generator for precomputing and storing teacher embeddings."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

OutputType = Literal["final", "pre_dense", "hidden_states"]


class TeacherEmbeddingsGenerator:
    """
    Generate and store teacher embeddings for later use in distillation.
    
    This class precomputes teacher model embeddings and saves them as a 
    HuggingFace dataset, eliminating the need to run the teacher model 
    during student training.
    
    Supports multiple output types:
    - "final": Final sentence embedding (after all layers including Dense)
    - "pre_dense": Output before the Dense layer (if present)
    - "hidden_states": All transformer layer hidden states
    
    Example:
        >>> generator = TeacherEmbeddingsGenerator(
        ...     teacher_model="sentence-transformers/all-mpnet-base-v2"
        ... )
        >>> dataset = generator.generate(
        ...     source_dataset="your-username/your-corpus",
        ...     text_column="text",
        ...     output_type="final"
        ... )
        >>> generator.push_to_hub("your-username/corpus-with-embeddings")
    """

    def __init__(
        self,
        teacher_model: str | SentenceTransformer,
        device: str = "auto",
        batch_size: int = 32,
        use_fp16: bool = False,
        use_bf16: bool = False,
        compile_model: bool = False,
        use_flash_attention: bool = False,
    ):
        """
        Initialize the generator.
        
        Args:
            teacher_model: Teacher model ID or SentenceTransformer instance.
            device: Device to use ('auto', 'cuda', 'mps', 'cpu').
            batch_size: Batch size for encoding.
            use_fp16: Use float16 precision (faster, less memory).
            use_bf16: Use bfloat16 precision (better for A100/H100).
            compile_model: Use torch.compile() for faster inference (PyTorch 2.0+).
            use_flash_attention: Enable Flash Attention if available.
        """
        self.batch_size = batch_size
        self.device = self._get_device(device)
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.compile_model = compile_model
        self.use_flash_attention = use_flash_attention
        self.teacher_model = self._load_model(teacher_model)
        self.dataset: Dataset | None = None
        self.teacher_model_id = teacher_model if isinstance(teacher_model, str) else "custom"
        
    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _is_flash_attention_available(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _load_model(self, model: str | SentenceTransformer) -> SentenceTransformer:
        """Load and optimize the teacher model."""
        if isinstance(model, str):
            logger.info(f"Loading teacher model: {model}")
            
            # Enable Flash Attention via model kwargs (only if available)
            model_kwargs = {}
            if self.use_flash_attention:
                if self._is_flash_attention_available():
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Enabling Flash Attention 2")
                else:
                    logger.warning("flash-attn not installed, skipping Flash Attention")
            
            model = SentenceTransformer(model, model_kwargs=model_kwargs)
        
        model.to(self.device)
        
        # Apply precision conversion
        if self.use_bf16 and self.device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)
            logger.info("Using bfloat16 precision")
        elif self.use_fp16:
            model = model.half()
            logger.info("Using float16 precision")
        
        # Compile model for faster inference
        if self.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}")
        
        model.eval()
        return model
    
    def _get_transformer_model(self):
        """Get the underlying transformer model from SentenceTransformer."""
        for module in self.teacher_model.modules():
            if hasattr(module, 'auto_model'):
                return module.auto_model
        return None
    
    def _has_dense_layer(self) -> bool:
        """Check if the model has a Dense layer after pooling."""
        for module in self.teacher_model:
            if module.__class__.__name__ == 'Dense':
                return True
        return False
    
    def _get_pre_dense_index(self) -> int:
        """Get the index of the module before Dense layer."""
        for idx, module in enumerate(self.teacher_model):
            if module.__class__.__name__ == 'Dense':
                return idx
        return len(self.teacher_model)

    def generate(
        self,
        source_dataset: str | Dataset,
        text_column: str = "text",
        split: str = "train",
        output_type: OutputType = "final",
        layer_indices: list[int] | None = None,
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Generate embeddings for the source dataset.
        
        Args:
            source_dataset: HuggingFace dataset ID or Dataset object.
            text_column: Name of the column containing text.
            split: Dataset split to use (if loading from HuggingFace).
            output_type: Type of output to generate:
                - "final": Final sentence embedding (default)
                - "pre_dense": Output before Dense layer
                - "hidden_states": Per-layer transformer hidden states
            layer_indices: For hidden_states, which layers to include.
                          None means all layers. E.g., [0, 6, 12] for specific layers.
            max_samples: Maximum number of samples to process.
            
        Returns:
            Dataset with text and embeddings.
        """
        # Load dataset
        if isinstance(source_dataset, str):
            logger.info(f"Loading dataset: {source_dataset}")
            dataset = load_dataset(source_dataset, split=split)
        else:
            dataset = source_dataset
            
        # Limit samples if requested
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")
            
        logger.info(f"Generating {output_type} embeddings for {len(dataset)} samples...")
        
        texts = dataset[text_column]
        
        if output_type == "final":
            dataset = self._generate_final_embeddings(dataset, texts)
        elif output_type == "pre_dense":
            dataset = self._generate_pre_dense_embeddings(dataset, texts)
        elif output_type == "hidden_states":
            dataset = self._generate_hidden_states(dataset, texts, layer_indices)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        self.dataset = dataset
        return self.dataset
    
    def _generate_final_embeddings(self, dataset: Dataset, texts: list[str]) -> Dataset:
        """Generate both pre-dense and final sentence embeddings."""
        # Generate final embeddings
        with torch.no_grad():
            embeddings = self.teacher_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        
        dataset = dataset.add_column("teacher_embedding_final", embeddings.tolist())
        logger.info(f"Generated final embeddings (dim={len(embeddings[0])})")
        
        # Also generate pre-dense embeddings if Dense layer exists
        if self._has_dense_layer():
            dataset = self._generate_pre_dense_embeddings(dataset, texts)
        
        return dataset
    
    def _generate_pre_dense_embeddings(self, dataset: Dataset, texts: list[str]) -> Dataset:
        """Generate embeddings before the Dense layer."""
        if not self._has_dense_layer():
            logger.info("No Dense layer found, skipping pre-dense embeddings")
            return dataset
        
        pre_dense_idx = self._get_pre_dense_index()
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating pre-dense"):
            batch_texts = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                features = self.teacher_model.tokenize(batch_texts)
                features = {k: v.to(self.device) for k, v in features.items()}
                
                for idx in range(pre_dense_idx):
                    features = self.teacher_model[idx](features)
                
                embedding = features.get('sentence_embedding', features.get('token_embeddings'))
                if embedding is not None:
                    if len(embedding.shape) == 3:
                        embedding = embedding.mean(dim=1)
                    all_embeddings.append(embedding.cpu().float().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        dataset = dataset.add_column("teacher_embedding_pre_dense", embeddings.tolist())
        logger.info(f"Generated pre-dense embeddings (dim={len(embeddings[0])})")
        return dataset
    
    def _generate_hidden_states(
        self, 
        dataset: Dataset, 
        texts: list[str],
        layer_indices: list[int] | None = None,
    ) -> Dataset:
        """Generate per-layer hidden states."""
        transformer = self._get_transformer_model()
        if transformer is None:
            raise ValueError("Could not find transformer model for hidden states extraction")
        
        transformer.config.output_hidden_states = True
        all_hidden_states = {}
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating hidden states"):
            batch_texts = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                features = self.teacher_model.tokenize(batch_texts)
                features = {k: v.to(self.device) for k, v in features.items()}
                
                outputs = transformer(**features)
                hidden_states = outputs.hidden_states
                
                if layer_indices is None:
                    indices_to_use = list(range(len(hidden_states)))
                else:
                    indices_to_use = layer_indices
                
                for layer_idx in indices_to_use:
                    layer_key = f"hidden_state_layer_{layer_idx}"
                    
                    attention_mask = features.get('attention_mask')
                    if attention_mask is not None:
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled = (hidden_states[layer_idx] * mask).sum(dim=1) / mask.sum(dim=1)
                    else:
                        pooled = hidden_states[layer_idx].mean(dim=1)
                    
                    if layer_key not in all_hidden_states:
                        all_hidden_states[layer_key] = []
                    all_hidden_states[layer_key].append(pooled.cpu().float().numpy())
        
        for layer_key, embeddings_list in all_hidden_states.items():
            embeddings = np.concatenate(embeddings_list, axis=0)
            dataset = dataset.add_column(layer_key, embeddings.tolist())
            logger.info(f"Added {layer_key} (dim={len(embeddings[0])})")
        
        transformer.config.output_hidden_states = False
        return dataset
    
    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> str:
        """
        Push the generated dataset to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID on HuggingFace Hub.
            private: Whether the repository should be private.
            token: HuggingFace Hub token for authentication.
            commit_message: Custom commit message.
            
        Returns:
            URL of the uploaded dataset.
        """
        if self.dataset is None:
            raise ValueError("No dataset to push. Call generate() first.")
            
        if commit_message is None:
            commit_message = f"Add teacher embeddings from {self.teacher_model_id}"
            
        logger.info(f"Pushing dataset to: {repo_id}")
        
        self.dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token,
            commit_message=commit_message,
        )
        
        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Dataset pushed: {url}")
        return url
    
    def save_to_disk(self, path: str) -> None:
        """
        Save the generated dataset to disk.
        
        Args:
            path: Path to save the dataset.
        """
        if self.dataset is None:
            raise ValueError("No dataset to save. Call generate() first.")
            
        self.dataset.save_to_disk(path)
        logger.info(f"Dataset saved to: {path}")
    
    def generate_and_push(
        self,
        source_dataset: str | Dataset,
        repo_id: str,
        text_column: str = "text",
        split: str = "train",
        output_type: OutputType = "final",
        layer_indices: list[int] | None = None,
        push_every: int = 10000,
        private: bool = False,
        token: str | None = None,
        max_samples: int | None = None,
    ) -> str:
        """
        Generate embeddings and push to HuggingFace Hub iteratively.
        
        This method processes the dataset in chunks and pushes updates
        to HuggingFace Hub every `push_every` examples, providing
        checkpoint-like behavior for large datasets.
        
        Args:
            source_dataset: HuggingFace dataset ID or Dataset object.
            repo_id: Repository ID on HuggingFace Hub.
            text_column: Name of the column containing text.
            split: Dataset split to use (if loading from HuggingFace).
            output_type: Type of output to generate.
            layer_indices: For hidden_states, which layers to include.
            push_every: Push to hub after this many examples (default: 10000).
            private: Whether the repository should be private.
            token: HuggingFace Hub token for authentication.
            max_samples: Maximum number of samples to process.
            
        Returns:
            URL of the uploaded dataset.
        """
        from datasets import concatenate_datasets
        
        # Load dataset
        if isinstance(source_dataset, str):
            logger.info(f"Loading dataset: {source_dataset}")
            dataset = load_dataset(source_dataset, split=split)
        else:
            dataset = source_dataset
            
        # Limit samples if requested
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")
        
        total_samples = len(dataset)
        logger.info(f"Processing {total_samples} samples, pushing every {push_every}...")
        
        processed_dataset = None
        processed_count = 0
        
        # Process in chunks
        for start_idx in range(0, total_samples, push_every):
            end_idx = min(start_idx + push_every, total_samples)
            chunk = dataset.select(range(start_idx, end_idx))
            texts = chunk[text_column]
            
            logger.info(f"Processing chunk {start_idx}-{end_idx} ({end_idx - start_idx} samples)")
            
            # Generate embeddings for this chunk
            if output_type == "final":
                chunk = self._generate_final_embeddings(chunk, texts)
            elif output_type == "pre_dense":
                chunk = self._generate_pre_dense_embeddings(chunk, texts)
            elif output_type == "hidden_states":
                chunk = self._generate_hidden_states(chunk, texts, layer_indices)
            
            # Concatenate with previous chunks
            if processed_dataset is None:
                processed_dataset = chunk
            else:
                processed_dataset = concatenate_datasets([processed_dataset, chunk])
            
            processed_count = end_idx
            
            # Push checkpoint
            commit_msg = f"Checkpoint: {processed_count}/{total_samples} samples from {self.teacher_model_id}"
            logger.info(f"Pushing checkpoint: {processed_count}/{total_samples}")
            
            processed_dataset.push_to_hub(
                repo_id=repo_id,
                private=private,
                token=token,
                commit_message=commit_msg,
            )
        
        self.dataset = processed_dataset
        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Completed! Dataset pushed: {url}")
        return url

