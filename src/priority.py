"""Priority scoring module for P2 — Intelligence / ML Engineer (AI-03).
Predicts continuous operational priority score bounded between 1.0 (lowest) and 5.0 (critical emergency).
Strictly dependent only on the input report text.
"""

import json
import math
import re
from pathlib import Path
import numpy as np
from src.category import predict_category, tokenize, _load_model

_PRIORITY_CACHE = None

def _load_priority_model():
    global _PRIORITY_CACHE
    if _PRIORITY_CACHE is not None:
        return _PRIORITY_CACHE

    weights_file = Path(__file__).resolve().parent.parent / "data" / "p2_models" / "model_weights.json"
    if weights_file.exists():
        with open(weights_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            weights = np.array(data["priority_weights"], dtype=np.float64)
            bias = float(data["priority_bias"])
            _PRIORITY_CACHE = (weights, bias)
            return _PRIORITY_CACHE

    _PRIORITY_CACHE = (None, 3.0)
    return _PRIORITY_CACHE

HIGH_URGENCY_TERMS = [
    "immediately", "life threatening", "trapped", "trpped", "dying", "unconscious", "explosion",
    "children", "baby", "infant", "cannot breathe", "drowning", "critical", "urgent",
    "bleeding heavily", "suffocating", "severe", "collapse", "colapse", "gas leak", "dam failure", "tornado", "starving",
    "toxic", "chemical", "gas", "odor", "leaking", "fire", "wildfire", "flood"
]

MODERATE_URGENCY_TERMS = [
    "soon", "blocked", "shortage", "damaged", "leak", "need supply", "no electricity",
    "injured", "pain", "broken", "assistance", "evacuat", "blackout", "breach"
]

def predict_priority(text: str) -> float:
    """
    Predict operational priority score for a single report.

    Args:
        text: Raw crisis message string.

    Returns:
        Float priority score bounded between 1.0 (lowest) and 5.0 (critical emergency).
    """
    if not text or not text.strip():
        return 1.0

    text_lower = text.lower()
    model_data = _load_model()
    vocab, idf = model_data[0], model_data[1]
    weights, bias = _load_priority_model()

    score = bias
    if vocab and weights is not None and len(weights) == len(vocab):
        tokens = tokenize(text)
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        vec = np.zeros(len(vocab), dtype=np.float64)
        for t, cnt in token_counts.items():
            if t in vocab:
                idx = vocab[t]
                tf = 1.0 + math.log(cnt) if cnt > 0 else 0.0
                vec[idx] = tf * idf.get(t, 1.0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            score += float(np.dot(vec, weights))

    cat = predict_category(text)
    if cat in ["SearchAndRescue", "EmergingThreats", "NewSubEvent"]:
        score += 1.5
    elif cat in ["GoodsServices", "MovePeople"]:
        score += 0.8
    elif cat in ["InformationWanted", "MultimediaShare", "Volunteer"]:
        score -= 0.4

    has_high = any(term in text_lower for term in HIGH_URGENCY_TERMS)
    has_mod = any(term in text_lower for term in MODERATE_URGENCY_TERMS)

    if has_high:
        score += 1.5
    elif has_mod:
        score += 0.6
    elif cat not in ["SearchAndRescue", "EmergingThreats", "NewSubEvent", "GoodsServices", "MovePeople"]:
        score -= 0.5

    final_score = max(1.0, min(5.0, score))
    return round(final_score, 1)
