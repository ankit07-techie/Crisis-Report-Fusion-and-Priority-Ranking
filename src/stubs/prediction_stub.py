"""P2 Classification & Priority Prediction Stub.
Provides heuristic-based category and priority scoring for crisis reports.
"""

import re
from typing import List, Tuple
from src.schemas import Report
from src.config import DEFAULT_CONFIG

# Keyword signatures for category mapping
CATEGORY_KEYWORDS = {
    "Search & Rescue": [
        "trap", "trapped", "rubble", "collapse", "drowning", "stranded", "missing",
        "under debris", "buried", "save us", "help trapped", "roof", "rooftop"
    ],
    "Medical Assistance": [
        "injur", "injured", "bleed", "bleeding", "unconscious", "fracture", "ambulance",
        "hospital", "doctor", "triage", "heart attack", "paramedic", "critical condition",
        "wound", "dying", "blood"
    ],
    "Hazardous Material / Fire": [
        "gas leak", "fire", "explosion", "toxic", "chemical", "smoke", "flames",
        "radiological", "hazard", "propane", "fumes", "burning"
    ],
    "Infrastructure Damage": [
        "bridge", "road", "power line", "blackout", "pipeline", "dam", "collapsed bridge",
        "crater", "grid", "highway blocked", "culvert", "water main"
    ],
    "Food & Water Shortage": [
        "starv", "starvation", "dehydrat", "drinking water", "ration", "clean water",
        "thirst", "no food", "infant formula", "hunger"
    ],
    "Shelter & Evacuation": [
        "homeless", "tents", "camp", "displaced", "shelter", "evacuat", "nowhere to sleep",
        "need blankets", "roof blown off"
    ]
}

# Priority weighting terms
HIGH_URGENCY_TERMS = [
    "immediately", "life threatening", "trapped", "dying", "unconscious", "explosion",
    "children", "baby", "infant", "cannot breathe", "drowning", "critical", "urgent",
    "bleeding heavily", "suffocating", "severe"
]

MODERATE_URGENCY_TERMS = [
    "soon", "blocked", "shortage", "damaged", "leak", "need supply", "no electricity",
    "injured", "pain", "broken", "assistance"
]


def predict_category(text: str) -> str:
    """Predict crisis information category based on text content."""
    text_lower = text.lower()
    
    # Calculate score per category
    category_scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            category_scores[cat] = score
            
    if category_scores:
        return max(category_scores.items(), key=lambda x: x[1])[0]
        
    return "General Information"


def predict_priority(text: str) -> float:
    """Predict operational priority score from 1.0 (lowest) to 5.0 (critical emergency)."""
    text_lower = text.lower()
    
    # Base priority
    priority = 2.0
    
    # Check category implications
    cat = predict_category(text)
    if cat in ["Search & Rescue", "Hazardous Material / Fire"]:
        priority += 1.5
    elif cat in ["Medical Assistance"]:
        priority += 1.2
    elif cat in ["Infrastructure Damage", "Food & Water Shortage"]:
        priority += 0.7
    elif cat in ["Shelter & Evacuation"]:
        priority += 0.5
        
    # Check urgency terms
    for term in HIGH_URGENCY_TERMS:
        if term in text_lower:
            priority += 0.8
            break
            
    for term in MODERATE_URGENCY_TERMS:
        if term in text_lower:
            priority += 0.4
            break
            
    # Clamp priority between configured min and max
    priority = max(DEFAULT_CONFIG.min_priority, min(DEFAULT_CONFIG.max_priority, priority))
    return round(priority, 1)


def predict_category_and_priority_batch(
    reports: List[Report]
) -> List[Tuple[str, float]]:
    """Batch prediction helper for throughput efficiency."""
    results = []
    for r in reports:
        cat = predict_category(r.text)
        pri = predict_priority(r.text)
        results.append((cat, pri))
    return results
