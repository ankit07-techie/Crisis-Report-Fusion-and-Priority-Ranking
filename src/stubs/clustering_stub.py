"""P1 Clustering & Fusion Stub.
Provides fallback semantic clustering using TF-IDF and cosine similarity.
Preserves real report IDs as evidence.
"""

import math
import re
from collections import Counter
from typing import List, Optional, Dict, Set, Any
from src.schemas import Report, ClusterResult
from src.config import DEFAULT_CONFIG


STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "of", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "has", "have", "had",
    "is", "am", "are", "was", "were", "be", "been", "being"
}


def _tokenize(text: str) -> List[str]:
    """Tokenize, lowercase, and remove common stopwords."""
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
    return [t for t in raw_tokens if t not in STOP_WORDS]


def _compute_cosine_sim(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute cosine similarity between two token lists."""
    if not tokens1 or not tokens2:
        return 0.0
    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    
    sum1 = sum(val ** 2 for val in vec1.values())
    sum2 = sum(val ** 2 for val in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator


def cluster_reports(
    reports: List[Report],
    similarity_threshold: Optional[float] = None
) -> List[ClusterResult]:
    """
    Cluster a collection of reports into incident groups.
    Each report is compared to existing clusters; if similarity exceeds threshold,
    it joins the highest-similarity cluster. Otherwise, a new cluster is created.
    """
    threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_CONFIG.similarity_threshold
    
    clusters: List[Dict[str, Any]] = []  # [{cluster_id, report_ids, tokens_list, texts}]
    
    for report in reports:
        report_tokens = _tokenize(report.text)
        best_match_idx = -1
        best_sim = 0.0
        
        for idx, cluster in enumerate(clusters):
            # Compare report against all reports in this cluster and take max similarity
            cluster_sim = max(
                _compute_cosine_sim(report_tokens, t) for t in cluster["tokens_list"]
            )
            if cluster_sim > best_sim:
                best_sim = cluster_sim
                best_match_idx = idx
                
        if best_match_idx >= 0 and best_sim >= threshold:
            # Add to existing cluster
            clusters[best_match_idx]["report_ids"].append(report.id)
            clusters[best_match_idx]["tokens_list"].append(report_tokens)
            clusters[best_match_idx]["texts"].append(report.text)
        else:
            # Create new cluster
            new_cluster_id = f"INCIDENT_{len(clusters) + 1:03d}"
            clusters.append({
                "cluster_id": new_cluster_id,
                "report_ids": [report.id],
                "tokens_list": [report_tokens],
                "texts": [report.text],
            })
            
    # Build ClusterResult objects
    results = []
    for cluster in clusters:
        results.append(ClusterResult(
            cluster_id=cluster["cluster_id"],
            confidence=1.0,
            evidence_ids=list(cluster["report_ids"]),
            summary=cluster["texts"][0][:120] if cluster["texts"] else None
        ))
    return results


def assign_cluster(
    report: Report,
    existing_reports: Optional[List[Report]] = None,
    similarity_threshold: Optional[float] = None
) -> ClusterResult:
    """
    Assign a single report to a cluster given existing context reports.
    """
    threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_CONFIG.similarity_threshold
    
    if not existing_reports:
        return ClusterResult(
            cluster_id="INCIDENT_001",
            confidence=1.0,
            evidence_ids=[report.id],
            summary=report.text[:120]
        )
        
    all_reports = list(existing_reports) + [report]
    clustered = cluster_reports(all_reports, similarity_threshold=threshold)
    
    for c in clustered:
        if report.id in c.evidence_ids:
            return c
            
    return ClusterResult(
        cluster_id="INCIDENT_001",
        confidence=1.0,
        evidence_ids=[report.id],
        summary=report.text[:120]
    )
