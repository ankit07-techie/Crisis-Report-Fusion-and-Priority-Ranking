#!/usr/bin/env python3
"""Official Evaluation Runner for Crisis Report Fusion and Priority Ranking (AI-03).
Conforms strictly to Section 8 of the PRD:
- Accepts ONLY opaque item ID + report text at evaluation time.
- Does NOT expect source ID, transformation type, labels, or cluster ground truth.
- Outputs item ID, predicted cluster ID, category, priority score, and evidence IDs.
- Fully decoupled from UI.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from src.schemas import Report, EvaluationRecord
from src.pipeline import CrisisPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def detect_id_and_text_keys(row: Dict[str, Any]) -> tuple[str, str]:
    """Detect the opaque ID key and text key from possible variations (case-insensitive)."""
    id_candidates = {"item_id", "id", "report_id", "uid", "item"}
    text_candidates = {"text", "report_text", "content", "body", "message", "report"}
    
    found_id = next((k for k in row if str(k).strip().lower() in id_candidates), None)
    found_text = next((k for k in row if str(k).strip().lower() in text_candidates), None)
    
    if not found_id or not found_text:
        raise ValueError(
            f"Input row missing required ID or text field. Available keys: {list(row.keys())}. "
            f"Expected one of ID {sorted(id_candidates)} and one of text {sorted(text_candidates)}."
        )
    return found_id, found_text


def load_input_reports(input_path: Path) -> List[Report]:
    """Load opaque reports from JSONL, JSON, or CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    reports: List[Report] = []
    ext = input_path.suffix.lower()
    
    if ext == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                id_k, text_k = detect_id_and_text_keys(data)
                reports.append(Report(id=str(data[id_k]), text=str(data[text_k])))
                
    elif ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of records.")
            for row in data:
                id_k, text_k = detect_id_and_text_keys(row)
                reports.append(Report(id=str(row[id_k]), text=str(row[text_k])))
                
    elif ext == ".csv":
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV input file has no header row.")
            # Verify keys from first row
            for row in reader:
                id_k, text_k = detect_id_and_text_keys(row)
                reports.append(Report(id=str(row[id_k]), text=str(row[text_k])))
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Must be .jsonl, .json, or .csv")
        
    logger.info(f"Loaded {len(reports)} opaque records from {input_path}")
    return reports


def write_evaluation_output(records: List[EvaluationRecord], output_path: Path, out_format: str) -> None:
    """Write evaluation predictions to JSONL or CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                # Output strict JSON line
                line_data = {
                    "item_id": rec.item_id,
                    "predicted_cluster_id": rec.predicted_cluster_id,
                    "category": rec.category,
                    "predicted_information_category": rec.predicted_information_category,
                    "priority_score": rec.priority_score,
                    "evidence_ids": rec.evidence_ids
                }
                f.write(json.dumps(line_data) + "\n")
    elif out_format == "csv":
        fieldnames = ["item_id", "predicted_cluster_id", "category", "predicted_information_category", "priority_score", "evidence_ids"]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow({
                    "item_id": rec.item_id,
                    "predicted_cluster_id": rec.predicted_cluster_id,
                    "category": rec.category,
                    "predicted_information_category": rec.predicted_information_category,
                    "priority_score": rec.priority_score,
                    "evidence_ids": ";".join(rec.evidence_ids)
                })
    else:
        raise ValueError(f"Unsupported output format: {out_format}")
        
    logger.info(f"Successfully wrote {len(records)} evaluation records to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Official Evaluation Runner for Crisis Report Fusion and Priority Ranking (AI-03)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to input evaluation file (.jsonl, .json, or .csv) containing opaque item ID + text"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Path to save evaluation predictions (.jsonl or .csv)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["jsonl", "csv"],
        default=None,
        help="Output format. Defaults to inferring from output file extension"
    )
    
    args = parser.parse_args()
    
    out_format = args.format
    if not out_format:
        ext = args.output.suffix.lower()
        if ext in [".jsonl", ".json"]:
            out_format = "jsonl"
        elif ext == ".csv":
            out_format = "csv"
        else:
            out_format = "jsonl"
            
    try:
        # Load opaque reports
        reports = load_input_reports(args.input)
        if not reports:
            logger.warning("No reports found in input file.")
            write_evaluation_output([], args.output, out_format)
            return 0
            
        # Run end-to-end integration pipeline
        pipeline = CrisisPipeline()
        predictions = pipeline.process_batch(reports)
        
        # Convert to official EvaluationRecord
        records = [EvaluationRecord.from_prediction(p) for p in predictions]
        
        # Write output
        write_evaluation_output(records, args.output, out_format)
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
