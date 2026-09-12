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
    # Strictly validate against the official TREC-IS Task-2 category string
    assert pred.category == "SearchAndRescue"
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
    # Strictly validate against the official TREC-IS Task-2 category string
    assert p3.category == "GoodsServices"


def test_pipeline_dict_clustering_support():
    pipeline = CrisisPipeline()
    # Mock cluster function returning a raw dictionary {report_id: cluster_id}
    pipeline._cluster_fn = lambda reports, **kw: {
        "R1": "INCIDENT_CUSTOM_1",
        "R2": "INCIDENT_CUSTOM_1",
        "R3": "INCIDENT_CUSTOM_2"
    }
    r1 = Report(id="R1", text="Flood in sector 1.")
    r2 = Report(id="R2", text="Flood waters in sector 1.")
    r3 = Report(id="R3", text="Gas leak.")
    
    preds = pipeline.process_batch([r1, r2, r3])
    p1 = next(p for p in preds if p.item_id == "R1")
    p2 = next(p for p in preds if p.item_id == "R2")
    p3 = next(p for p in preds if p.item_id == "R3")
    
    assert p1.predicted_cluster_id == "INCIDENT_CUSTOM_1"
    assert p2.predicted_cluster_id == "INCIDENT_CUSTOM_1"
    assert p3.predicted_cluster_id == "INCIDENT_CUSTOM_2"
    assert "R2" in p1.evidence_ids
    assert "R1" in p2.evidence_ids


def test_pipeline_string_priority_support():
    pipeline = CrisisPipeline()
    # Mock priority function returning categorical string "CRITICAL"
    pipeline._priority_fn = lambda text: "CRITICAL"
    
    rep = Report(id="REP_001", text="Emergency situation.")
    pred = pipeline.process_report(rep)
    assert pred.priority_score == 5.0
