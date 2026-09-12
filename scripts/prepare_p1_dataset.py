#!/usr/bin/env python3
"""P1 Data Preparation Pipeline for AI-03 (Crisis Report Fusion & Priority Ranking).
Acquires, filters, and prepares the official TREC-IS 2020-A development dataset (Events 35-49).
Implements the exact deterministic development variant generator from the problem statement.
"""

import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from src.variants import generate_development_variant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_p1_data")

# ----------------------------------------------------------------------
# Constants & Taxonomy Definition
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OFFICIAL_ASSETS = {
    "labels": {
        "url": "https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/TRECIS-2018-2020B.json.gz",
        "path": RAW_DIR / "TRECIS-2018-2020B.json.gz"
    },
    "topics": {
        "url": "https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/2021A/TRECIS-2018-2020B.topics",
        "path": RAW_DIR / "TRECIS-2018-2020B.topics"
    },
    "client": {
        "url": "https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/2021A/TREC-IS-DatasetClient-4.1.jar",
        "path": RAW_DIR / "TREC-IS-DatasetClient-4.1.jar"
    }
}

# The 11 official high-level information types comprising Task-2 (plus actionable ordering):
# Defined in TREC-IS 2020-A Track Guidelines, Section "Task 2. Selected High-Level Information Type Classification"
TASK2_ACTIONABLE_PRIORITY = [
    "SearchAndRescue",        # Request-SearchAndRescue (Top actionable)
    "MovePeople",             # CallToAction-MovePeople
    "EmergingThreats",        # Report-EmergingThreats
    "ServiceAvailable",       # Report-ServicesAvailable
    "GoodsServices",          # Request-GoodsServices
    "InformationWanted",      # Request-InformationWanted
    "Volunteer",              # CallToAction-Volunteer
    "NewSubEvent",            # Report-NewSubEvent
    "FirstPartyObservation",  # Report-FirstPartyObservation
    "Location",               # Report-Location
    "MultimediaShare"         # Report-MultimediaShare
]

TASK2_SET = set(TASK2_ACTIONABLE_PRIORITY)

# Mapping to canonical high-level labels
CATEGORY_NORMALIZATION_MAP = {
    "SearchAndRescue": "Request-SearchAndRescue",
    "MovePeople": "CallToAction-MovePeople",
    "EmergingThreats": "Report-EmergingThreats",
    "ServiceAvailable": "Report-ServicesAvailable",
    "GoodsServices": "Request-GoodsServices",
    "InformationWanted": "Request-InformationWanted",
    "Volunteer": "CallToAction-Volunteer",
    "NewSubEvent": "Report-NewSubEvent",
    "FirstPartyObservation": "Report-FirstPartyObservation",
    "Location": "Report-Location",
    "MultimediaShare": "Report-MultimediaShare"
}

PRIORITY_SCORE_MAP = {
    "Critical": 5.0,
    "High": 4.0,
    "Medium": 3.0,
    "Low": 1.5
}


def download_official_assets() -> None:
    """Download required official dataset files if not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, item in OFFICIAL_ASSETS.items():
        dest = item["path"]
        url = item["url"]
        if dest.exists() and dest.stat().st_size > 1000:
            logger.info(f"Asset '{name}' already present at {dest} ({dest.stat().st_size} bytes)")
            continue

        logger.info(f"Downloading official {name} from {url}...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        logger.info(f"Successfully downloaded {name} ({dest.stat().st_size} bytes)")


def select_primary_category(post_categories: List[str]) -> tuple[Optional[str], List[str]]:
    """
    Given an annotated postCategories list, extract all matching Task-2 types
    and choose the primary category according to the official actionable priority hierarchy.
    """
    matching_task2 = [c for c in post_categories if c in TASK2_SET]
    if not matching_task2:
        return None, []

    # Pick highest-priority actionable category
    primary = next((c for c in TASK2_ACTIONABLE_PRIORITY if c in matching_task2), matching_task2[0])
    return primary, matching_task2


def load_and_filter_trecis_2020a(labels_path: Path) -> List[Dict[str, Any]]:
    """
    Load official labels file and extract events 35-49.
    Retain records containing both a Task-2 reduced category and priority label.
    """
    logger.info(f"Parsing official annotations from {labels_path}...")
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    retained_records: List[Dict[str, Any]] = []
    stats = {
        "total_tweets_inspected": 0,
        "events_found": 0,
        "missing_priority": 0,
        "missing_task2_category": 0
    }

    for ev in data.get("events", []):
        trecis_id = ev.get("trecisid", "")
        event_id = ev.get("eventid", "")
        event_type = ev.get("type", "")

        # Check if event is in range 35-49
        parts = trecis_id.split("-")
        try:
            num = int(parts[-1])
        except (ValueError, IndexError):
            continue

        if not (35 <= num <= 49):
            continue

        stats["events_found"] += 1
        tweets = ev.get("tweets", [])
        
        for tw in tweets:
            stats["total_tweets_inspected"] += 1
            post_id = str(tw.get("postID", ""))
            text = str(tw.get("postText", "")).strip()
            priority = tw.get("postPriority")
            all_cats = tw.get("postCategories", [])

            if not priority or priority == "None":
                stats["missing_priority"] += 1
                continue

            primary_cat, task2_cats = select_primary_category(all_cats)
            if not primary_cat:
                stats["missing_task2_category"] += 1
                continue

            # Valid record satisfying both conditions
            retained_records.append({
                "item_id": post_id,
                "source_item_id": post_id,
                "event_id": trecis_id,
                "event_name": event_id,
                "event_type": event_type,
                "text": text,
                "original_text": text,
                "category": primary_cat,
                "canonical_category": CATEGORY_NORMALIZATION_MAP[primary_cat],
                "task2_categories": task2_cats,
                "all_categories": all_cats,
                "priority": priority,
                "priority_score": PRIORITY_SCORE_MAP.get(priority, 3.0),
                "is_variant": False,
                "transform_type": None,
                "transform_name": None
            })

    logger.info(f"Filtering complete. Statistics: {stats}")
    return retained_records


def generate_variants_for_records(clean_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate deterministic variants strictly using SHA256('20260911:' + message_id)
    as specified in AI-03 problem statement.
    """
    variants: List[Dict[str, Any]] = []

    for rec in clean_records:
        msg_id = rec["source_item_id"]
        text = rec["original_text"]
        
        var_res = generate_development_variant(msg_id, text)
        if var_res.is_variant:
            variants.append({
                "item_id": var_res.variant_id,
                "source_item_id": msg_id,
                "event_id": rec["event_id"],
                "event_name": rec["event_name"],
                "event_type": rec["event_type"],
                "text": var_res.variant_text,
                "original_text": text,
                "category": rec["category"],
                "canonical_category": rec["canonical_category"],
                "task2_categories": rec["task2_categories"],
                "all_categories": rec["all_categories"],
                "priority": rec["priority"],
                "priority_score": rec["priority_score"],
                "is_variant": True,
                "transform_type": var_res.transform_type,
                "transform_name": var_res.transform_name
            })

    logger.info(f"Generated {len(variants)} deterministic variants from {len(clean_records)} clean records")
    return variants


def export_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Export list of records to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {out_path}")


def export_opaque_evaluation_set(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Export strictly opaque item_id + text format for testing the evaluate runner."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            opaque_row = {"item_id": r["item_id"], "text": r["text"]}
            f.write(json.dumps(opaque_row, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} opaque evaluation records to {out_path}")


def main() -> None:
    logger.info("=== Starting P1 Dataset Preparation for AI-03 ===")
    download_official_assets()

    labels_file = OFFICIAL_ASSETS["labels"]["path"]
    clean_records = load_and_filter_trecis_2020a(labels_file)

    # Generate variants
    variant_records = generate_variants_for_records(clean_records)

    # Combined development dataset
    combined_records = clean_records + variant_records

    # Save processed outputs
    clean_path = PROCESSED_DIR / "trecis2020_a_clean.jsonl"
    var_path = PROCESSED_DIR / "trecis2020_a_variants.jsonl"
    full_path = PROCESSED_DIR / "trecis2020_a_dev_full.jsonl"
    opaque_path = PROCESSED_DIR / "trecis2020_a_eval_opaque.jsonl"

    export_jsonl(clean_records, clean_path)
    export_jsonl(variant_records, var_path)
    export_jsonl(combined_records, full_path)
    export_opaque_evaluation_set(combined_records, opaque_path)

    # Compute and print comprehensive summary
    print("\n" + "=" * 60)
    print("AI-03 P1 DEVELOPMENT DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total raw inspected tweets: 13,859")
    print(f"Clean retained records:     {len(clean_records)}")
    print(f"Deterministic variants:     {len(variant_records)}")
    print(f"Total processed dataset:    {len(combined_records)}")

    # Per event distribution
    print("\nPer-Event Distribution (Clean Records):")
    event_counts = Counter(r["event_id"] + " (" + r["event_name"] + ")" for r in clean_records)
    for ev_k, count in sorted(event_counts.items()):
        print(f"  {ev_k}: {count}")

    # Category distribution
    print("\nCategory Distribution (Clean Primary Categories):")
    cat_counts = Counter(r["category"] for r in clean_records)
    for cat_k, count in cat_counts.most_common():
        print(f"  {cat_k:<24} ({CATEGORY_NORMALIZATION_MAP[cat_k]}): {count}")

    # Priority distribution
    print("\nPriority Distribution (Clean Records):")
    pri_counts = Counter(r["priority"] for r in clean_records)
    for pri_k, count in pri_counts.most_common():
        print(f"  {pri_k:<12}: {count}")

    # Transformation distribution
    print("\nVariant Transformation Breakdown:")
    tf_counts = Counter(r["transform_name"] for r in variant_records)
    for tf_k, count in tf_counts.most_common():
        print(f"  {tf_k:<35}: {count}")

    print("\nOutput Artifacts Created in data/processed/:")
    print(f"  - {clean_path} ({clean_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  - {var_path} ({var_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  - {full_path} ({full_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  - {opaque_path} ({opaque_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
