"""Unit tests for P1 report fusion and clustering engine."""

from src.schemas import Report
from src.clustering import cluster_reports, assign_cluster


def test_semantic_fusion_differently_worded():
    """Verify differently worded crisis reports about the same event are fused."""
    r1 = Report(id="REP_A", text="The main suspension bridge over Highway 9 has collapsed under severe floodwaters.")
    r2 = Report(id="REP_B", text="Highway 9 bridge overpass structure washed out by raging river waters, traffic halted.")
    r3 = Report(id="REP_C", text="Shelter at community gym desperately needs clean drinking water and infant milk.")

    clusters = cluster_reports([r1, r2, r3], similarity_threshold=0.55)

    # Bridge reports should fuse into one cluster, shelter in another
    bridge_cluster = next((c for c in clusters if "REP_A" in c.evidence_ids), None)
    assert bridge_cluster is not None
    assert "REP_B" in bridge_cluster.evidence_ids
    assert "REP_C" not in bridge_cluster.evidence_ids

    # Verify evidence IDs
    all_evidence = [eid for c in clusters for eid in c.evidence_ids]
    assert set(all_evidence) == {"REP_A", "REP_B", "REP_C"}


def test_assign_cluster_online():
    """Verify assign_cluster for incoming stream processing."""
    history = [
        Report(id="HIST_1", text="Chemical gas leak at industrial warehouse on 5th avenue."),
        Report(id="HIST_2", text="Families stranded on rooftops along Elm Street due to rising river.")
    ]

    # New report matching chemical leak
    new_gas = Report(id="NEW_1", text="Toxic fumes and strong chemical smell reported near 5th avenue storage.")
    res_gas = assign_cluster(new_gas, existing_reports=history, similarity_threshold=0.55)

    # Should match HIST_1's cluster
    assert "HIST_1" in res_gas.evidence_ids
    assert "NEW_1" in res_gas.evidence_ids

    # New distinct report
    new_wildfire = Report(id="NEW_2", text="Massive brushfire burning timber along North Ridge.")
    res_fire = assign_cluster(new_wildfire, existing_reports=history, similarity_threshold=0.55)

    # Should spawn a new cluster
    assert res_fire.cluster_id not in [c.cluster_id for c in cluster_reports(history)]
    assert res_fire.evidence_ids == ["NEW_2"]
