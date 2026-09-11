"""Integration tests for official evaluation runner (evaluate.py)."""

import json
import csv
import subprocess
import sys
from pathlib import Path


def test_evaluate_cli_jsonl(tmp_path):
    # Prepare sample input JSONL with opaque IDs and text
    input_file = tmp_path / "eval_input.jsonl"
    output_file = tmp_path / "eval_output.jsonl"
    
    samples = [
        {"item_id": "OPAQ_001", "text": "Two people stuck in vehicle submerged in water on 5th avenue."},
        {"item_id": "OPAQ_002", "text": "Vehicle submerged under 5th avenue overpass with people inside."},
        {"item_id": "OPAQ_003", "text": "Distribution point needed for bottled water in Sector 9."}
    ]
    
    with open(input_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    # Execute evaluate.py via CLI
    cmd = [
        sys.executable,
        "evaluate.py",
        "--input", str(input_file),
        "--output", str(output_file)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"evaluate.py failed with: {res.stderr}"
    assert output_file.exists()
    
    # Verify outputs
    outputs = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            outputs.append(json.loads(line.strip()))
            
    assert len(outputs) == 3
    for rec in outputs:
        assert "item_id" in rec
        assert "predicted_cluster_id" in rec
        assert "category" in rec
        assert "priority_score" in rec
        assert "evidence_ids" in rec
        assert isinstance(rec["evidence_ids"], list)
        assert len(rec["evidence_ids"]) >= 1
        
    # Check semantic fusion of OPAQ_001 and OPAQ_002
    rec1 = next(r for r in outputs if r["item_id"] == "OPAQ_001")
    rec2 = next(r for r in outputs if r["item_id"] == "OPAQ_002")
    assert rec1["predicted_cluster_id"] == rec2["predicted_cluster_id"]
    assert "OPAQ_002" in rec1["evidence_ids"]
    assert "OPAQ_001" in rec2["evidence_ids"]


def test_evaluate_cli_csv(tmp_path):
    input_file = tmp_path / "eval_input.csv"
    output_file = tmp_path / "eval_output.csv"
    
    with open(input_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "report_text"])
        writer.writerow(["ID_101", "Chemical gas odor near factory, burning throats."])
        writer.writerow(["ID_102", "Need food supplies at church shelter."])
        
    cmd = [
        sys.executable,
        "evaluate.py",
        "--input", str(input_file),
        "--output", str(output_file),
        "--format", "csv"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"evaluate.py failed with: {res.stderr}"
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 2
    assert rows[0]["item_id"] == "ID_101"
    assert rows[0]["category"] == "Hazardous Material / Fire"
    assert "ID_101" in rows[0]["evidence_ids"]
