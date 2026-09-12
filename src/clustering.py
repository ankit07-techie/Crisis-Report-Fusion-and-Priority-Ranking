"""P1 Semantic Report Fusion and Incident Clustering Engine for AI-03.
Implements the official P1 contract specified in docs/INTERFACES.md:
- cluster_reports(reports: List[Report], similarity_threshold: Optional[float] = None) -> List[ClusterResult]
- assign_cluster(report: Report, existing_reports: Optional[List[Report]] = None, similarity_threshold: Optional[float] = None) -> ClusterResult

Features:
- Normalized dense sentence embeddings via all-MiniLM-L6-v2 (with robust TF-IDF fallback)
- Agglomerative clustering with cosine distance and average linkage
- Strict evidence-ID preservation (zero synthetic IDs)
- Exemplar summary generation from centroid-closest reports
"""

import logging
from typing import List, Optional, Dict
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.schemas import Report, ClusterResult
from src.embeddings import get_embedding_model, BaseEmbeddingModel

logger = logging.getLogger(__name__)

# Default optimal similarity threshold selected from benchmark validation & audit sweep
DEFAULT_SIMILARITY_THRESHOLD = 0.54

# Global singleton embedding model to avoid reloading weights repeatedly
_GLOBAL_EMBEDDING_MODEL: Optional[BaseEmbeddingModel] = None


def get_default_embedding_model() -> BaseEmbeddingModel:
    """Retrieve or initialize the global sentence embedding model."""
    global _GLOBAL_EMBEDDING_MODEL
    if _GLOBAL_EMBEDDING_MODEL is None:
        try:
            _GLOBAL_EMBEDDING_MODEL = get_embedding_model("sentence_transformer", "all-MiniLM-L6-v2")
            logger.info("Initialized all-MiniLM-L6-v2 embedding model for P1 clustering.")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers ({e}). Falling back to TF-IDF.")
            _GLOBAL_EMBEDDING_MODEL = get_embedding_model("tfidf")
    return _GLOBAL_EMBEDDING_MODEL


def cluster_reports(
    reports: List[Report],
    similarity_threshold: Optional[float] = None
) -> List[ClusterResult]:
    """
    Cluster a collection of raw crisis reports into discrete incident groups.
    Preserves actual report IDs in evidence_ids without fabrication.

    Args:
        reports: List of Report objects (each containing 'id' and 'text').
        similarity_threshold: Cosine similarity cutoff (default 0.55).

    Returns:
        List of ClusterResult objects.
    """
    if not reports:
        return []

    if len(reports) == 1:
        r = reports[0]
        return [
            ClusterResult(
                cluster_id="INCIDENT_001",
                confidence=1.0,
                evidence_ids=[r.id],
                summary=r.text[:140]
            )
        ]

    tau = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
    distance_threshold = max(0.001, 1.0 - tau)

    # 1. Generate normalized embeddings
    model = get_default_embedding_model()
    texts = [r.text for r in reports]
    embeddings = model.encode(texts, batch_size=64)

    # 2. Agglomerative clustering (cosine metric, average linkage)
    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(embeddings)

    # 3. Group reports by assigned cluster label
    clusters_map: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters_map.setdefault(label, []).append(idx)

    # 4. Assemble ClusterResult objects
    results: List[ClusterResult] = []
    # Sort clusters by size descending for consistent, intuitive ordering
    sorted_cluster_labels = sorted(clusters_map.keys(), key=lambda k: len(clusters_map[k]), reverse=True)

    for c_idx, label in enumerate(sorted_cluster_labels, start=1):
        member_indices = clusters_map[label]
        cluster_id = f"INCIDENT_{c_idx:03d}"
        evidence_ids = [reports[i].id for i in member_indices]

        # Calculate cluster centroid and exemplar text
        member_embs = embeddings[member_indices]
        centroid = np.mean(member_embs, axis=0, keepdims=True)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Distance of members to centroid
        similarities_to_centroid = np.dot(member_embs, centroid.T).flatten()
        best_exemplar_idx = member_indices[int(np.argmax(similarities_to_centroid))]
        summary = reports[best_exemplar_idx].text[:140]

        # Confidence: average similarity to centroid
        confidence = float(np.clip(np.mean(similarities_to_centroid), 0.0, 1.0))

        results.append(
            ClusterResult(
                cluster_id=cluster_id,
                confidence=round(confidence, 4),
                evidence_ids=evidence_ids,
                summary=summary
            )
        )

    return results


def assign_cluster(
    report: Report,
    existing_reports: Optional[List[Report]] = None,
    similarity_threshold: Optional[float] = None
) -> ClusterResult:
    """
    Assign a single incoming report to either an existing incident cluster or spawn a new cluster.

    Args:
        report: Single incoming Report.
        existing_reports: History of previously indexed reports.
        similarity_threshold: Similarity cutoff (default 0.55).

    Returns:
        ClusterResult with cluster assignment and evidence report IDs.
    """
    if not existing_reports:
        return ClusterResult(
            cluster_id="INCIDENT_001",
            confidence=1.0,
            evidence_ids=[report.id],
            summary=report.text[:140]
        )

    tau = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD

    # Cluster all existing reports together
    existing_clusters = cluster_reports(existing_reports, similarity_threshold=tau)

    # Encode incoming report and all existing reports
    model = get_default_embedding_model()
    incoming_emb = model.encode([report.text], batch_size=1)[0]
    all_texts = [r.text for r in existing_reports]
    existing_embs = model.encode(all_texts, batch_size=64)

    # Calculate similarity to all existing reports
    sims = np.dot(existing_embs, incoming_emb)

    # Find closest existing report
    best_match_idx = int(np.argmax(sims))
    best_sim = float(sims[best_match_idx])
    best_report_id = existing_reports[best_match_idx].id

    if best_sim >= tau:
        # Fuse into existing cluster containing the best matching report
        target_cluster = next((c for c in existing_clusters if best_report_id in c.evidence_ids), existing_clusters[0])
        updated_evidence = [report.id] + [eid for eid in target_cluster.evidence_ids if eid != report.id]
        return ClusterResult(
            cluster_id=target_cluster.cluster_id,
            confidence=round(best_sim, 4),
            evidence_ids=updated_evidence,
            summary=target_cluster.summary or report.text[:140]
        )
    else:
        # Spawn new incident cluster
        new_cid = f"INCIDENT_{len(existing_clusters) + 1:03d}"
        return ClusterResult(
            cluster_id=new_cid,
            confidence=1.0,
            evidence_ids=[report.id],
            summary=report.text[:140]
        )
