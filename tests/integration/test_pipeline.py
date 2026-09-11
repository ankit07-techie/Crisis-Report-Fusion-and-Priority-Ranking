"""Integration tests for end-to-end CrisisPipeline."""

import pytest
from src.schemas import Report, Prediction
from src.pipeline import CrisisPipeline


def test_pipeline_single_report():
    pipeline = CrisisPipeline()
    report = Report(id="REP_001", text="Elderly couple trapped on rooftop due to flash flood.")
    pred = pipeline.process_report(report)
    
    assert isinstance(pred, Prediction)
    assert pred.item_id == "REP_001"
    assert pred.predicted_cluster_id.startswith("INCIDENT_")
    assert pred.category == "Search & Rescue"
    assert pred.priority_score >= 4.0
    assert "REP_001" in pred.evidence_ids


def test_pipeline_batch_and_fusion():
    pipeline = CrisisPipeline()
    
    # Two differently worded reports regarding the same bridge incident
    r1 = Report(id="R1", text="The main suspension bridge on Highway 9 has collapsed under floodwaters.")
    r2 = Report(id="R2", text="Highway 9 bridge collapsed completely, traffic stopped and road blocked.")
    
    # An unrelated report regarding hospital power
    r3 = Report(id="R3", text="St. Jude Hospital backup generators failing, ICU patients in critical danger.")
    
    preds = pipeline.process_batch([r1, r2, r3])
    assert len(preds) == 3
    
    p1 = next(p for p in preds if p.item_id == "R1")
    p2 = next(p for p in preds if p.item_id == "R2")
    p3 = next(p for p in preds if p.item_id == "R3")
    
    # R1 and R2 should share the same incident cluster
    assert p1.predicted_cluster_id == p2.predicted_cluster_id
    # Both R1 and R2 should be in each other's evidence IDs
    assert "R1" in p2.evidence_ids
    assert "R2" in p1.evidence_ids
    
    # R3 should have a different incident cluster
    assert p3.predicted_cluster_id != p1.predicted_cluster_id
    assert p3.category == "Medical Assistance"
