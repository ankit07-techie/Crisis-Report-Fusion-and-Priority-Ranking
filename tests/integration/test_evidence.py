"""Integration tests for evidence ID tracking and preservation."""

import pytest
from src.schemas import Report
from src.evidence import EvidenceRegistry, validate_evidence_ids


def test_validate_evidence_ids():
    valid_pool = {"R1", "R2", "R3"}
    incoming = ["R1", "R2", "R999", "R1", "R3", "R_FAKE"]
    
    cleaned = validate_evidence_ids(incoming, valid_pool)
    assert cleaned == ["R1", "R2", "R3"]
    assert "R999" not in cleaned
    assert "R_FAKE" not in cleaned
    # Ensure no duplicates
    assert len(cleaned) == len(set(cleaned))


def test_evidence_registry():
    registry = EvidenceRegistry()
    
    r1 = Report(id="REP_001", text="Bridge collapsed on Route 9.")
    r2 = Report(id="REP_002", text="Route 9 bridge is down, cars stopped.")
    r3 = Report(id="REP_003", text="Gas leak near downtown market.")
    
    registry.register_report(r1, "INCIDENT_001")
    registry.register_report(r2, "INCIDENT_001")
    registry.register_report(r3, "INCIDENT_002")
    
    # Evidence for cluster 1
    ev_cluster1 = registry.get_cluster_evidence("INCIDENT_001")
    assert ev_cluster1 == ["REP_001", "REP_002"]
    
    # Evidence for report 2: REP_002 should lead, followed by REP_001
    ev_rep2 = registry.get_evidence_for_report("REP_002")
    assert ev_rep2 == ["REP_002", "REP_001"]
    
    # Check related reports
    related = registry.get_related_reports("INCIDENT_001")
    assert len(related) == 2
    assert {r.id for r in related} == {"REP_001", "REP_002"}
