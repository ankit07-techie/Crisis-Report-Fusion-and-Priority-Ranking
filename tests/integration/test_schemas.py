"""Integration tests for shared schemas."""

import pytest
from pydantic import ValidationError
from src.schemas import Report, ClusterResult, Prediction, EvaluationRecord


def test_report_valid():
    report = Report(id="REP_001", text="Severe flood waters rising on Maple Street.")
    assert report.id == "REP_001"
    assert "flood waters" in report.text
    assert report.metadata == {}


def test_report_invalid_empty_text():
    with pytest.raises(ValidationError):
        Report(id="REP_002", text="")


def test_cluster_result():
    cluster = ClusterResult(
        cluster_id="INCIDENT_001",
        confidence=0.92,
        evidence_ids=["REP_001", "REP_002"]
    )
    assert cluster.cluster_id == "INCIDENT_001"
    assert len(cluster.evidence_ids) == 2


def test_prediction_creation_and_evaluation_record_conversion():
    pred = Prediction(
        item_id="REP_001",
        predicted_cluster_id="INCIDENT_001",
        category="Search & Rescue",
        priority_score=4.5,
        evidence_ids=["REP_001", "REP_002"]
    )
    assert pred.item_id == "REP_001"
    assert pred.priority_score == 4.5
    
    # Test conversion to official EvaluationRecord
    eval_rec = EvaluationRecord.from_prediction(pred)
    assert eval_rec.item_id == pred.item_id
    assert eval_rec.predicted_cluster_id == pred.predicted_cluster_id
    assert eval_rec.category == pred.category
    assert eval_rec.predicted_information_category == pred.category
    assert eval_rec.priority_score == 4.5
    assert eval_rec.evidence_ids == ["REP_001", "REP_002"]
