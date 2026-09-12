"""P1 Semantic Embedding Engine for Crisis Reports.
Provides high-performance, normalized sentence-level representations:
- Sentence-Transformers (all-MiniLM-L6-v2, paraphrase-MiniLM-L3-v2)
- Sublinear TF-IDF baseline (scikit-learn)
"""

import logging
from typing import List, Optional, Union
import numpy as np

from src.preprocessing import preprocess_crisis_text

logger = logging.getLogger(__name__)


class BaseEmbeddingModel:
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Dense neural sentence embedding using Sentence-Transformers.
    Embeddings are L2-normalized so that inner product equals cosine similarity.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading SentenceTransformer model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        cleaned_texts = [preprocess_crisis_text(t) for t in texts]
        # normalize_embeddings=True ensures ||e||_2 = 1, enabling fast matrix multiplication for cosine similarity
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


class TfidfEmbedding(BaseEmbeddingModel):
    """
    Sparse lexical TF-IDF baseline with sublinear term-frequency scaling.
    L2-normalized for cosine similarity.
    """
    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2"
        )
        self.is_fitted = False

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        cleaned = [preprocess_crisis_text(t) for t in texts]
        if not self.is_fitted:
            matrix = self.vectorizer.fit_transform(cleaned)
            self.is_fitted = True
        else:
            matrix = self.vectorizer.transform(cleaned)
        return matrix.toarray().astype(np.float32)


def get_embedding_model(
    model_type: str = "sentence_transformer",
    model_name: str = "all-MiniLM-L6-v2"
) -> BaseEmbeddingModel:
    """Factory helper to obtain an embedding model."""
    if model_type == "sentence_transformer":
        return SentenceTransformerEmbedding(model_name=model_name)
    elif model_type == "tfidf":
        return TfidfEmbedding()
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")
