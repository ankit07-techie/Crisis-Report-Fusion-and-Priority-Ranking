"""P1 Data Loader Module.
Loads deduplicated reports for clustering while consolidating multi-annotator assessments
for local development evaluation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

from src.schemas import Report

# Actionable hierarchy for consolidating multi-annotator categories
TASK2_ACTIONABLE_PRIORITY = [
    "SearchAndRescue",
    "MovePeople",
    "EmergingThreats",
    "ServiceAvailable",
    "GoodsServices",
    "InformationWanted",
    "Volunteer",
    "NewSubEvent",
    "FirstPartyObservation",
    "Location",
    "MultimediaShare"
]

PRIORITY_ORDER = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1
}

SCORE_TO_PRIORITY = {v: k for k, v in PRIORITY_ORDER.items()}

PRIORITY_SCORE_MAP = {
    "Critical": 5.0,
    "High": 4.0,
    "Medium": 3.0,
    "Low": 1.5
}


@dataclass
class GroundTruthMeta:
    """Consolidated offline evaluation metadata (never passed to inference pipeline)."""
    item_id: str
    source_item_id: str
    event_id: str
    event_name: str
    event_type: str
    is_variant: bool
    transform_type: Optional[int]
    task2_categories: List[str]
    all_categories: List[str]
    primary_category: str
    priority: str
    priority_score: float
    num_annotator_assessments: int


def consolidate_annotator_records(records: List[Dict[str, Any]]) -> Tuple[Report, GroundTruthMeta]:
    """
    Consolidate multiple annotator records for a single item_id:
    - Text: shared across all records for this item_id
    - Categories: union of all annotated categories
    - Priority: highest urgency rating (Critical > High > Medium > Low)
    - Event: ground-truth event ID
    """
    first = records[0]
    item_id = first["item_id"]
    text = first["text"]

    all_cats_set: Set[str] = set()
    task2_cats_set: Set[str] = set()
    max_pri_rank = 0

    for r in records:
        all_cats_set.update(r.get("all_categories", []))
        task2_cats_set.update(r.get("task2_categories", []))
        p_rank = PRIORITY_ORDER.get(r.get("priority", "Low"), 1)
        if p_rank > max_pri_rank:
            max_pri_rank = p_rank

    consolidated_priority = SCORE_TO_PRIORITY.get(max_pri_rank, "Low")
    consolidated_score = PRIORITY_SCORE_MAP[consolidated_priority]

    # Select primary actionable category
    task2_list = list(task2_cats_set)
    primary = next((c for c in TASK2_ACTIONABLE_PRIORITY if c in task2_cats_set), task2_list[0] if task2_list else "Location")

    report = Report(id=item_id, text=text)
    meta = GroundTruthMeta(
        item_id=item_id,
        source_item_id=first.get("source_item_id", item_id),
        event_id=first["event_id"],
        event_name=first.get("event_name", ""),
        event_type=first.get("event_type", ""),
        is_variant=first.get("is_variant", False),
        transform_type=first.get("transform_type"),
        task2_categories=task2_list,
        all_categories=list(all_cats_set),
        primary_category=primary,
        priority=consolidated_priority,
        priority_score=consolidated_score,
        num_annotator_assessments=len(records)
    )

    return report, meta


def load_deduplicated_dataset(
    clean_path: Path = Path("data/processed/trecis2020_a_clean.jsonl"),
    variants_path: Optional[Path] = Path("data/processed/trecis2020_a_variants.jsonl"),
    include_variants: bool = True
) -> Tuple[List[Report], Dict[str, GroundTruthMeta]]:
    """
    Load crisis reports deduplicated by item_id.
    Consolidates multi-annotator assessments into single canonical reports with ground-truth metadata.
    """
    records_by_id: Dict[str, List[Dict[str, Any]]] = {}

    # Load clean records
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records_by_id.setdefault(r["item_id"], []).append(r)

    # Optionally load variants
    if include_variants and variants_path and variants_path.exists():
        with open(variants_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    records_by_id.setdefault(r["item_id"], []).append(r)

    reports: List[Report] = []
    meta_by_id: Dict[str, GroundTruthMeta] = {}

    for item_id, rec_list in records_by_id.items():
        rep, meta = consolidate_annotator_records(rec_list)
        reports.append(rep)
        meta_by_id[item_id] = meta

    return reports, meta_by_id
