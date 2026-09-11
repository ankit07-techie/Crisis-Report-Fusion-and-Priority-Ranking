"""Configuration settings for Crisis Report Fusion and Priority Ranking (AI-03).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for pipeline execution and thresholds."""
    
    # Clustering / Fusion parameters (P1)
    similarity_threshold: float = 0.30
    min_cluster_size: int = 1
    max_evidence_per_cluster: int = 10
    
    # Classification categories (P2)
    categories: List[str] = field(default_factory=lambda: [
        "Search & Rescue",
        "Medical Assistance",
        "Infrastructure Damage",
        "Food & Water Shortage",
        "Shelter & Evacuation",
        "Hazardous Material / Fire",
        "General Information"
    ])
    
    # Priority Scale (P2): 1.0 (Lowest) to 5.0 (Critical Emergency)
    min_priority: float = 1.0
    max_priority: float = 5.0
    
    # Priority levels mapping
    priority_levels: Dict[str, float] = field(default_factory=lambda: {
        "CRITICAL": 5.0,
        "HIGH": 4.0,
        "MEDIUM": 3.0,
        "LOW": 2.0,
        "INFORMATIONAL": 1.0
    })
    
    # Random seed for reproducibility
    random_seed: int = 42


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
