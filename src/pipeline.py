"""End-to-End Integration Pipeline (Section 6).
Integrates P1 (clustering), P2 (category/priority), and P3 (evidence tracking).
Dynamically delegates to teammate modules if present, falling back to stubs.
"""

import importlib
import logging
from typing import List, Optional, Callable, Any
from src.schemas import Report, Prediction, ClusterResult
from src.evidence import EvidenceRegistry
from src.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


class CrisisPipeline:
    """
    Unified AI-03 Crisis Processing Pipeline.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.evidence_registry = EvidenceRegistry()
        
        # Resolve P1 (Clustering / Fusion)
        self._cluster_fn, self._assign_fn, self.p1_source = self._resolve_p1()
        
        # Resolve P2 (Classification & Priority)
        self._category_fn, self._priority_fn, self.p2_source = self._resolve_p2()
        
        # In-memory indexed reports for online clustering
        self._indexed_reports: List[Report] = []

    def _resolve_p1(self) -> tuple[Callable, Callable, str]:
        """Dynamically resolve P1 clustering module or fall back to stub."""
        candidates = ["src.clustering", "src.fusion", "clustering", "fusion"]
        for mod_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "cluster_reports"):
                    assign_fn = getattr(mod, "assign_cluster", None)
                    logger.info(f"Loaded P1 clustering module from {mod_name}")
                    return mod.cluster_reports, assign_fn, mod_name
            except (ImportError, ModuleNotFoundError):
                continue
                
        # Fallback to stub
        from src.stubs.clustering_stub import cluster_reports, assign_cluster
        return cluster_reports, assign_cluster, "src.stubs.clustering_stub"

    def _resolve_p2(self) -> tuple[Callable, Callable, str]:
        """Dynamically resolve P2 prediction modules or fall back to stub."""
        candidates = ["src.classification", "src.priority", "src.prediction"]
        cat_fn = None
        pri_fn = None
        source_names = []
        
        for mod_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "predict_category") and not cat_fn:
                    cat_fn = mod.predict_category
                    source_names.append(f"{mod_name}:predict_category")
                if hasattr(mod, "predict_priority") and not pri_fn:
                    pri_fn = mod.predict_priority
                    source_names.append(f"{mod_name}:predict_priority")
            except (ImportError, ModuleNotFoundError):
                continue
                
        if not cat_fn or not pri_fn:
            from src.stubs.prediction_stub import predict_category, predict_priority
            cat_fn = cat_fn or predict_category
            pri_fn = pri_fn or predict_priority
            source_names.append("src.stubs.prediction_stub")
            
        return cat_fn, pri_fn, ", ".join(source_names)

    def process_report(self, report: Report) -> Prediction:
        """
        Process a single crisis report through clustering, classification, priority,
        and evidence association.
        """
        # 1. Cluster Assignment (P1)
        if self._assign_fn:
            cluster_res = self._assign_fn(
                report,
                existing_reports=self._indexed_reports,
                similarity_threshold=self.config.similarity_threshold
            )
        else:
            all_reports = self._indexed_reports + [report]
            all_clusters = self._cluster_fn(all_reports, similarity_threshold=self.config.similarity_threshold)
            cluster_res = next((c for c in all_clusters if report.id in c.evidence_ids), None)
            if not cluster_res:
                cluster_res = ClusterResult(cluster_id="INCIDENT_001", evidence_ids=[report.id])
                
        # Register into evidence registry
        self.evidence_registry.register_report(report, cluster_res.cluster_id)
        if report.id not in {r.id for r in self._indexed_reports}:
            self._indexed_reports.append(report)
            
        # 2. Category & Priority Prediction (P2)
        category = self._category_fn(report.text)
        priority = float(self._priority_fn(report.text))
        
        # 3. Evidence ID Association (P3)
        evidence_ids = self.evidence_registry.get_evidence_for_report(
            report.id, max_items=self.config.max_evidence_per_cluster
        )
        
        return Prediction(
            item_id=report.id,
            predicted_cluster_id=cluster_res.cluster_id,
            category=category,
            priority_score=priority,
            evidence_ids=evidence_ids,
            metadata={"cluster_confidence": cluster_res.confidence}
        )

    def process_batch(self, reports: List[Report]) -> List[Prediction]:
        """
        Process a collection of crisis reports in batch.
        Clusters all reports together, assigns categories/priorities, and extracts evidence.
        """
        if not reports:
            return []
            
        # Cluster all reports
        clusters: List[ClusterResult] = self._cluster_fn(
            reports, similarity_threshold=self.config.similarity_threshold
        )
        
        # Map report ID to assigned cluster
        report_to_cluster: dict[str, ClusterResult] = {}
        for c in clusters:
            for rid in c.evidence_ids:
                report_to_cluster[rid] = c
                
        # Register in evidence tracker
        for r in reports:
            c = report_to_cluster.get(r.id)
            cid = c.cluster_id if c else f"INCIDENT_{len(self.evidence_registry._cluster_to_reports) + 1:03d}"
            self.evidence_registry.register_report(r, cid)
            if r.id not in {ex.id for ex in self._indexed_reports}:
                self._indexed_reports.append(r)
                
        # Assemble predictions
        predictions: List[Prediction] = []
        for r in reports:
            cluster_res = report_to_cluster.get(r.id)
            cid = cluster_res.cluster_id if cluster_res else "INCIDENT_001"
            confidence = cluster_res.confidence if cluster_res else 1.0
            
            cat = self._category_fn(r.text)
            pri = float(self._priority_fn(r.text))
            ev_ids = self.evidence_registry.get_evidence_for_report(
                r.id, max_items=self.config.max_evidence_per_cluster
            )
            
            predictions.append(Prediction(
                item_id=r.id,
                predicted_cluster_id=cid,
                category=cat,
                priority_score=pri,
                evidence_ids=ev_ids,
                metadata={"cluster_confidence": confidence}
            ))
            
        return predictions

    def get_related_reports(self, cluster_id: str) -> List[Report]:
        """Fetch all reports associated with an incident cluster."""
        return self.evidence_registry.get_related_reports(cluster_id)

    def reset(self) -> None:
        """Reset internal indices and evidence state."""
        self._indexed_reports.clear()
        self.evidence_registry = EvidenceRegistry()
