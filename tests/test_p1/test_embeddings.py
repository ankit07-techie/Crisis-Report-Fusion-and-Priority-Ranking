"""Unit tests for P1 embeddings engine."""

import numpy as np
from src.embeddings import get_embedding_model


def test_sentence_transformer_embeddings():
    model = get_embedding_model("sentence_transformer", "all-MiniLM-L6-v2")
    texts = [
        "Major flooding on River Road.",
        "River Road submerged under flood water."
    ]
    embs = model.encode(texts)
    assert embs.shape == (2, 384)

    # Verify L2 normalization
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    # Cosine similarity
    sim = float(np.dot(embs[0], embs[1]))
    assert sim > 0.70  # Highly semantically similar


def test_tfidf_embeddings():
    model = get_embedding_model("tfidf")
    texts = [
        "Emergency shelter location open at school.",
        "School gym opened as disaster evacuation shelter."
    ]
    embs = model.encode(texts)
    assert embs.shape[0] == 2
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)
