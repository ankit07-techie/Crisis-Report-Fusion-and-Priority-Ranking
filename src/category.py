"""Category prediction module for P2 — Intelligence / ML Engineer (AI-03).
Predicts humanitarian crisis information category matching the official 11 TREC-IS Task-2 Reduced Information Categories.
Strictly dependent only on the input report text.
"""

import json
import math
import re
from pathlib import Path
import numpy as np

# Official 11 TREC-IS Task-2 Reduced Information Categories
CATEGORIES = [
    "Location",
    "EmergingThreats",
    "MultimediaShare",
    "MovePeople",
    "NewSubEvent",
    "FirstPartyObservation",
    "InformationWanted",
    "ServiceAvailable",
    "SearchAndRescue",
    "Volunteer",
    "GoodsServices"
]

CATEGORY_KEYWORD_PATTERNS = {
    "SearchAndRescue": [r"\btrap\b", r"\btrapped\b", r"\btrpped\b", r"\brubble\b", r"\bcollapse\b", r"\bcolapse\b", r"\bdrowning\b", r"\bstranded\b", r"\bmissing\b", r"\broof\b", r"\brooftop\b", r"\bdebris\b", r"\bburied\b", r"\brescue\b", r"\bhiker\b", r"\bswimmer\b", r"\blandslide\b"],
    "GoodsServices": [r"\binjur", r"\binjured\b", r"\bbleed", r"\bbleeding\b", r"\bunconscious\b", r"\bfracture\b", r"\bambulance\b", r"\bhospital\b", r"\bicu\b", r"\bdoctor\b", r"\bpatient\b", r"\bpatients\b", r"\btriage\b", r"\bheart attack\b", r"\bparamedic\b", r"\bwound\b", r"\bblood\b", r"\bcasualty\b", r"\bfood\b", r"\bwater\b", r"\bdrinking\b", r"\bstarving\b", r"\binfant formula\b", r"\bration\b", r"\bpotable\b", r"\bsupply\b", r"\bclinic\b", r"\bbandages\b"],
    "EmergingThreats": [r"\bgas leak\b", r"\bfire\b", r"\bexplosion\b", r"\btoxic\b", r"\bchemical\b", r"\bsmoke\b", r"\bflames\b", r"\bradiological\b", r"\bhazard\b", r"\bpropane\b", r"\bfumes\b", r"\bburning\b", r"\bblaze\b", r"\bchlorine\b", r"\bspill\b", r"\bwildfire\b", r"\bdam\b", r"\bbreach\b"],
    "Location": [r"\bgps\b", r"\broute\b", r"\bhighway\b", r"\bavenue\b", r"\bstreet\b", r"\bdistrict\b", r"\bsector\b", r"\bbound\b", r"\bentrance\b", r"\bkilometer\b", r"\bmarker\b", r"\bintersection\b", r"\blocated\b", r"\bcoordinates\b", r"\bcheckpoint\b"],
    "MovePeople": [r"\bevacuat", r"\bbuses\b", r"\bnational guard\b", r"\bmove to\b", r"\bclear streets\b", r"\btransportation\b", r"\bmandatory evacuation\b"],
    "FirstPartyObservation": [r"\bi can see\b", r"\bfront porch\b", r"\bour street\b", r"\bground shook\b", r"\bmy window\b", r"\bdangling\b", r"\bmy house\b", r"\bi am standing\b", r"\bmy front\b", r"\bwent out completely\b"],
    "InformationWanted": [r"\bis the\b", r"\bdoes anyone\b", r"\bwhere can\b", r"\bhas anyone\b", r"\bare buses\b", r"\bupdates on\b", r"\bknow if\b", r"\bwhat is\b", r"\bstatus of\b", r"\bopen to\b", r"\bstill open\b", r"\?\s*$"],
    "ServiceAvailable": [r"\boperating\b", r"\bhotspot\b", r"\bwi-fi\b", r"\bhot meals\b", r"\bcharging station\b", r"\bopen and accepting\b", r"\btowing service\b", r"\bdebris removal\b"],
    "Volunteer": [r"\bvolunteers needed\b", r"\bcalling all\b", r"\blooking for\b", r"\bsandbagging\b", r"\bcot setup\b", r"\bnurses and doctors\b", r"\bseeking bilingual\b"],
    "MultimediaShare": [r"\bvideo\b", r"\bphoto\b", r"\bfootage\b", r"\binfographic\b", r"http://", r"https://"],
    "NewSubEvent": [r"\bsecondary\b", r"\bupgraded\b", r"\bupgraded to\b", r"\btouchdown\b", r"\bblackout spread\b", r"\baftershock\b"]
}

_MODEL_CACHE = None

def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    weights_file = Path(__file__).resolve().parent.parent / "data" / "p2_models" / "model_weights.json"
    if weights_file.exists():
        with open(weights_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            vocab = data.get("vocab", {})
            idf = data.get("idf", {})
            cat_weights = np.array(data["cat_weights"], dtype=np.float64) if "cat_weights" in data else None
            cat_priors = np.array(data["cat_priors"], dtype=np.float64) if "cat_priors" in data else None
            centroids = {k: np.array(v, dtype=np.float64) for k, v in data.get("centroids", {}).items()}
            _MODEL_CACHE = (vocab, idf, centroids, cat_weights, cat_priors)
            return _MODEL_CACHE

    _MODEL_CACHE = ({}, {}, {}, None, None)
    return _MODEL_CACHE

def tokenize(text: str) -> list[str]:
    text_clean = text.lower()
    words = re.findall(r"\b[a-z0-9]+\b", text_clean)
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams

def predict_category(text: str) -> str:
    """
    Predict the official TREC-IS Task-2 reduced category for a single report.

    Args:
        text: Raw crisis message string.

    Returns:
        Category string matching one of 11 official TREC-IS Task-2 categories.
    """
    if not text or not text.strip():
        return "FirstPartyObservation"

    text_lower = text.lower()
    vocab, idf, centroids, cat_weights, cat_priors = _load_model()

    kw_scores = {}
    for cat, patterns in CATEGORY_KEYWORD_PATTERNS.items():
        score = sum(10.0 for pat in patterns if re.search(pat, text_lower))
        kw_scores[cat] = score

    sim_scores = {cat: 0.0 for cat in CATEGORIES}
    if vocab:
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
            if cat_weights is not None and cat_priors is not None:
                scores = np.dot(vec, cat_weights.T) + cat_priors
                for idx, cat in enumerate(CATEGORIES):
                    sim_scores[cat] = float(scores[idx])
            elif centroids:
                for cat, centroid in centroids.items():
                    if np.linalg.norm(centroid) > 0:
                        sim_scores[cat] = float(np.dot(vec, centroid))

    best_cat = "FirstPartyObservation"
    best_score = -1e9
    for cat in CATEGORIES:
        combined = sim_scores.get(cat, 0.0) + kw_scores.get(cat, 0.0)
        if combined > best_score:
            best_score = combined
            best_cat = cat

    return best_cat
