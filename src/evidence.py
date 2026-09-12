"""Evidence ID management and validation (Section 7).
Guarantees evidence IDs originate strictly from real report/item identifiers.
Never fabricates IDs or uses hidden metadata linkages.
"""

from typing import List, Set, Dict, Optional
from src.schemas import Report


def validate_evidence_ids(evidence_ids: List[str], valid_id_pool: Set[str]) -> List[str]:
    """
    Validate that all evidence IDs exist in the pool of actual ingested report IDs.
    Strips out duplicates and any IDs not present in valid_id_pool.
    """
    cleaned: List[str] = []
    seen: Set[str] = set()
    
    for eid in evidence_ids:
        if eid in valid_id_pool and eid not in seen:
            cleaned.append(eid)
            seen.add(eid)
            
    return cleaned


class EvidenceRegistry:
    """
    In-memory registry tracking which actual report IDs belong to which incident clusters.
    Ensures strict evidence traceability without synthetic ID generation.
    """
    def __init__(self):
        self._cluster_to_reports: Dict[str, List[str]] = {}
        self._report_to_cluster: Dict[str, str] = {}
        self._known_reports: Dict[str, Report] = {}

    def register_report(self, report: Report, cluster_id: str) -> None:
        """Register an ingested report and its assigned cluster."""
        self._known_reports[report.id] = report
        
        # Clean up previous cluster association if report is reassigned
        old_cluster = self._report_to_cluster.get(report.id)
        if old_cluster and old_cluster != cluster_id and old_cluster in self._cluster_to_reports:
            if report.id in self._cluster_to_reports[old_cluster]:
                self._cluster_to_reports[old_cluster].remove(report.id)
                
        self._report_to_cluster[report.id] = cluster_id
        
        if cluster_id not in self._cluster_to_reports:
            self._cluster_to_reports[cluster_id] = []
        if report.id not in self._cluster_to_reports[cluster_id]:
            self._cluster_to_reports[cluster_id].append(report.id)

    def get_cluster_evidence(self, cluster_id: str, max_items: Optional[int] = None) -> List[str]:
        """Retrieve real report IDs associated with an incident cluster."""
        reports = self._cluster_to_reports.get(cluster_id, [])
        if max_items is not None:
            return list(reports[:max_items])
        return list(reports)

    def get_evidence_for_report(self, report_id: str, max_items: Optional[int] = None) -> List[str]:
        """Retrieve corroborating report IDs for a specific report."""
        cluster_id = self._report_to_cluster.get(report_id)
        if not cluster_id:
            # If not assigned to a cluster, evidence is solely the report itself if known
            return [report_id] if report_id in self._known_reports else []
            
        cluster_reports = self.get_cluster_evidence(cluster_id)
        # Ensure the target report is first, followed by corroborating evidence IDs
        ordered = [report_id] + [r for r in cluster_reports if r != report_id]
        if max_items is not None:
            return ordered[:max_items]
        return ordered

    def get_related_reports(self, cluster_id: str) -> List[Report]:
        """Return the actual Report objects belonging to a cluster."""
        report_ids = self._cluster_to_reports.get(cluster_id, [])
        return [self._known_reports[rid] for rid in report_ids if rid in self._known_reports]

    def all_known_ids(self) -> Set[str]:
        """Return all registered report IDs."""
        return set(self._known_reports.keys())
