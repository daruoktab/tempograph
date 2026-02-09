# src/embedders/hf_embedder.py
"""
HuggingFace Embedding Models via Transformers
Mendukung berbagai model embedding dari HuggingFace Hub
"""

import asyncio
from typing import List, Optional, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor

from .base import BaseEmbedder, EmbedderType

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Embedder menggunakan model dari HuggingFace Hub.
    Mendukung model sentence-transformers dan model transformer lainnya.
    
    Models yang direkomendasikan untuk Indonesian:
    - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (dim=384)
    - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (dim=768)
    - "intfloat/multilingual-e5-small" (dim=384)
    - "intfloat/multilingual-e5-base" (dim=768)
    - "intfloat/multilingual-e5-large" (dim=1024)
    - "BAAI/bge-m3" (dim=1024, state-of-the-art multilingual)
    - "firqaaa/indo-sentence-bert-base" (dim=768, Indonesian-specific)
    - "indobenchmark/indobert-base-p1" (dim=768, Indonesian BERT)
    """
    
    # Recommended models dengan dimensi dan info
    RECOMMENDED_MODELS: Dict[str, Dict[str, Any]] = {
        # Google Gemma Embedding
        "google/embeddinggemma-300m": {
            "dimension": 768,
            "description": "Google EmbeddingGemma 300M - lightweight, high quality, supports MRL",
            "max_seq_length": 2048,
            "supports_mrl": True  # Matryoshka Representation Learning
        },
        
        # Multilingual sentence transformers
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
            "dimension": 384,
            "description": "Lightweight multilingual model, good for quick experiments",
            "max_seq_length": 128
        },
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
            "dimension": 768,
            "description": "Strong multilingual model with good performance",
            "max_seq_length": 128
        },
        
        # E5 models (state-of-the-art)
        "intfloat/multilingual-e5-small": {
            "dimension": 384,
            "description": "Efficient E5 model, good quality/speed tradeoff",
            "max_seq_length": 512,
            "query_prefix": "query: ",
            "passage_prefix": "passage: "
        },
        "intfloat/multilingual-e5-base": {
            "dimension": 768,
            "description": "Balanced E5 model for most use cases",
            "max_seq_length": 512,
            "query_prefix": "query: ",
            "passage_prefix": "passage: "
        },
        "intfloat/multilingual-e5-large": {
            "dimension": 1024,
            "description": "Highest quality E5, best for research comparison",
            "max_seq_length": 512,
            "query_prefix": "query: ",
            "passage_prefix": "passage: "
        },
        
        # BGE-M3 (state-of-the-art multilingual)
        "BAAI/bge-m3": {
            "dimension": 1024,
            "description": "State-of-the-art multilingual, supports dense/sparse/colbert",
            "max_seq_length": 8192
        },
        
        # Indonesian-specific models
        "firqaaa/indo-sentence-bert-base": {
            "dimension": 768,
            "description": "Indonesian Sentence-BERT, trained on Indonesian data",
            "max_seq_length": 512
        },
        "indobenchmark/indobert-base-p1": {
            "dimension": 768,
            "description": "IndoBERT Phase 1, pre-trained on Indonesian corpus",
            "max_seq_length": 512,
            "pooling": "mean"  # Needs manual pooling
        }
    }
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_workers: int = 2
    ):
        """
        Initialize HuggingFace embedder.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_fp16: Use half precision (faster, less memory)
            batch_size: Batch size for encoding
            max_workers: Thread pool workers for async wrapping
        """
        super().__init__(model_name=model_name, model_type=EmbedderType.HUGGINGFACE)
        
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self._model = None
        self._tokenizer = None
        self._is_sentence_transformer = False
        
        # Get model info if available
        self._model_info = self.RECOMMENDED_MODELS.get(model_name, {})
        if "dimension" in self._model_info:
            self._dimension = self._model_info["dimension"]
    
    async def initialize(self):
        """Load model from HuggingFace"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_model)
        logger.info(f"HuggingFace embedder initialized: {self.model_name} (dim={self._dimension})")
    
    def _load_model(self):
        """Synchronous model loading"""
        import torch
        
        # Determine device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try sentence-transformers first (easier API)
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            self._is_sentence_transformer = True
            
            # Get dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()
            
            # Apply fp16 if requested and on GPU
            if self.use_fp16 and self.device == "cuda":
                self._model = self._model.half()
                
            logger.info(f"Loaded as SentenceTransformer on {self.device}")
            
        except Exception as e:
            logger.info(f"SentenceTransformer failed ({e}), falling back to transformers")
            self._load_with_transformers()
    
    def _load_with_transformers(self):
        """Load using raw transformers library"""
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self._model.to(self.device)
        self._model.eval()
        
        if self.use_fp16 and self.device == "cuda":
            self._model = self._model.half()
        
        self._is_sentence_transformer = False
        
        # Get dimension from hidden size
        self._dimension = self._model.config.hidden_size
        
        logger.info(f"Loaded with transformers on {self.device}")
    
    async def _embed_impl(self, texts: List[str]) -> List[List[float]]:
        """Embed texts asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            texts
        )
    
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding"""
        import torch
        
        # Add prefixes if E5 model
        if "query_prefix" in self._model_info:
            texts = [self._model_info["passage_prefix"] + t for t in texts]
        
        if self._is_sentence_transformer:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        
        else:
            # Manual encoding with transformers
            return self._encode_with_transformers(texts)
    
    def _encode_with_transformers(self, texts: List[str]) -> List[List[float]]:
        """Encode using raw transformers"""
        import torch
        import torch.nn.functional as F
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self._model_info.get("max_seq_length", 512),
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self._model(**encoded)
                
                # Mean pooling over tokens
                attention_mask = encoded["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                ).float()
                
                embeddings = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.extend(embeddings.cpu().tolist())
        
        return all_embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (with query prefix for E5 models).
        """
        if "query_prefix" in self._model_info:
            query = self._model_info["query_prefix"] + query
        
        result = await self.embed([query])
        return result.embeddings[0] if result.embeddings else []
    
    async def close(self):
        """Cleanup resources"""
        self._model = None
        self._tokenizer = None
        self._executor.shutdown(wait=False)
    
    @classmethod
    def list_recommended_models(cls) -> Dict[str, str]:
        """List recommended models with descriptions"""
        return {
            name: info["description"]
            for name, info in cls.RECOMMENDED_MODELS.items()
        }
