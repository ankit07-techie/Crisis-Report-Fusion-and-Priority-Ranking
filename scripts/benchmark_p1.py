#!/usr/bin/env python3
"""Controlled P1 Semantic Embedding + Report Fusion/Clustering Benchmark.
Evaluates:
- Embedding Models: all-MiniLM-L6-v2, paraphrase-MiniLM-L3-v2, TF-IDF
- Clustering Algorithms: Agglomerative (average linkage), Graph Connected Components, DBSCAN
- Cosine Similarity Thresholds: 0.55 to 0.85
- Key Metric: Pairwise Clustering F1 (AI-03 35% scoring criteria)
- Robustness: Originals, Variants, Combined, and per-transform degradation breakdown
"""

import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_deduplicated_dataset, GroundTruthMeta
from src.schemas import Report
from src.embeddings import get_embedding_model


def compute_pairwise_f1(pred_labels: np.ndarray, true_labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Pairwise Precision, Recall, and F1 via contingency matrix.
    Runs in O(N) time without allocating an N x N matrix.
    """
    n = len(pred_labels)
    if n <= 1:
        return 1.0, 1.0, 1.0

    # Map labels to integers 0..K-1
    _, pred_idx = np.unique(pred_labels, return_inverse=True)
    _, true_idx = np.unique(true_labels, return_inverse=True)

    # Contingency table
    k_pred = pred_idx.max() + 1
    k_true = true_idx.max() + 1

    # Linear index for (pred, true)
    lin_idx = pred_idx * k_true + true_idx
    cell_counts = np.bincount(lin_idx, minlength=k_pred * k_true)

    # TP: pairs in same pred and same true
    tp = np.sum(cell_counts * (cell_counts - 1) // 2)

    # Total predicted pairs
    pred_counts = np.bincount(pred_idx, minlength=k_pred)
    total_pred = np.sum(pred_counts * (pred_counts - 1) // 2)

    # Total true pairs
    true_counts = np.bincount(true_idx, minlength=k_true)
    total_true = np.sum(true_counts * (true_counts - 1) // 2)

    precision = float(tp / total_pred) if total_pred > 0 else 1.0
    recall = float(tp / total_true) if total_true > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def cluster_agglomerative(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative clustering with cosine distance and average linkage."""
    from sklearn.cluster import AgglomerativeClustering
    distance_threshold = max(0.001, 1.0 - threshold)
    model = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    return model.fit_predict(embeddings)


def cluster_graph_components(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Thresholded graph connected components via cosine similarity."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    # embeddings are L2 normalized, so dot product = cosine similarity
    sim_matrix = np.dot(embeddings, embeddings.T)
    adj_matrix = (sim_matrix >= threshold).astype(np.int32)
    sparse_adj = csr_matrix(adj_matrix)
    _, labels = connected_components(sparse_adj, directed=False)
    return labels


def cluster_dbscan(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """DBSCAN with cosine distance."""
    from sklearn.cluster import DBSCAN
    eps = max(0.001, 1.0 - threshold)
    model = DBSCAN(metric="cosine", eps=eps, min_samples=2)
    raw_labels = model.fit_predict(embeddings)

    # Handle singletons (-1): assign unique cluster IDs
    labels = raw_labels.copy()
    max_label = labels.max() if labels.max() >= 0 else 0
    next_id = max_label + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = next_id
            next_id += 1
    return labels


def run_benchmark():
    print("=" * 80)
    print("P1 SEMANTIC EMBEDDING + REPORT FUSION BENCHMARK")
    print("=" * 80)

    # 1. Load Deduplicated Reports and Ground Truth
    print("\n[Step 1] Loading deduplicated dataset...")
    reports, meta_dict = load_deduplicated_dataset(include_variants=True)
    n_total = len(reports)
    print(f"Loaded {n_total} total unique reports (Originals + Variants).")

    texts = [r.text for r in reports]
    true_events = np.array([meta_dict[r.id].event_id for r in reports])
    is_variant = np.array([meta_dict[r.id].is_variant for r in reports])
    transform_types = np.array([meta_dict[r.id].transform_type if meta_dict[r.id].transform_type is not None else -1 for r in reports])

    # 2. Benchmark Embedding Models
    print("\n[Step 2] Generating & Benchmarking Candidate Embeddings...")
    models = {
        "all-MiniLM-L6-v2": ("sentence_transformer", "all-MiniLM-L6-v2"),
        "paraphrase-MiniLM-L3-v2": ("sentence_transformer", "paraphrase-MiniLM-L3-v2"),
        "TF-IDF Baseline": ("tfidf", "")
    }

    embeddings_cache = {}
    embedding_times = {}

    for m_label, (m_type, m_name) in models.items():
        t0 = time.time()
        emb_model = get_embedding_model(m_type, m_name)
        embs = emb_model.encode(texts, batch_size=64)
        t_elap = time.time() - t0
        embeddings_cache[m_label] = embs
        embedding_times[m_label] = t_elap
        print(f"  - {m_label:<25}: shape={embs.shape}, time={t_elap:.2f}s ({len(texts)/t_elap:.1f} sent/s)")

    # 3. Systematic Algorithm & Threshold Comparison Table
    print("\n[Step 3] Systematic Clustering Benchmark Comparison:")
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    clustering_algorithms = {
        "Agglomerative (Avg)": cluster_agglomerative,
        "Graph Connected": cluster_graph_components,
        "DBSCAN": cluster_dbscan
    }

    results = []
    header = f"| {'Approach':<20} | {'Embedding':<22} | {'Threshold':>9} | {'Clusters':>8} | {'Pairwise P':>10} | {'Pairwise R':>10} | {'Pairwise F1':>11} | {'Runtime':>7} |"
    print("\n" + header)
    print("|" + "-"*22 + "|" + "-"*24 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*13 + "|" + "-"*9 + "|")

    best_config = None
    best_f1 = -1.0

    for emb_name, embs in embeddings_cache.items():
        for alg_name, alg_fn in clustering_algorithms.items():
            for tau in thresholds:
                t0 = time.time()
                preds = alg_fn(embs, tau)
                t_elap = time.time() - t0

                num_clusters = len(np.unique(preds))
                p, r, f1 = compute_pairwise_f1(preds, true_events)

                results.append({
                    "approach": alg_name,
                    "embedding": emb_name,
                    "threshold": tau,
                    "clusters": num_clusters,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "runtime": t_elap
                })

                row = f"| {alg_name:<20} | {emb_name:<22} | {tau:>9.2f} | {num_clusters:>8} | {p:>10.4f} | {r:>10.4f} | {f1:>11.4f} | {t_elap:>6.3f}s |"
                print(row)

                if f1 > best_f1:
                    best_f1 = f1
                    best_config = (alg_name, emb_name, tau, preds)

    # 4. Detailed Breakdown of Best Performing Model
    best_alg, best_emb, best_tau, best_preds = best_config
    print("\n" + "=" * 80)
    print(f"BEST PERFORMING CONFIGURATION: {best_alg} + {best_emb} @ Threshold {best_tau:.2f}")
    print(f"Overall Combined Pairwise F1: {best_f1:.4f}")
    print("=" * 80)

    # Breakdown by Subsets
    orig_mask = ~is_variant
    var_mask = is_variant

    p_orig, r_orig, f1_orig = compute_pairwise_f1(best_preds[orig_mask], true_events[orig_mask])
    p_var, r_var, f1_var = compute_pairwise_f1(best_preds[var_mask], true_events[var_mask])

    print("\nSubset Robustness Breakdown:")
    print(f"  1. Original Reports Only (N={orig_mask.sum()}): P={p_orig:.4f}, R={r_orig:.4f}, Pairwise F1={f1_orig:.4f}")
    print(f"  2. Variants Only        (N={var_mask.sum()}): P={p_var:.4f}, R={r_var:.4f}, Pairwise F1={f1_var:.4f}")
    print(f"  3. Combined             (N={len(reports)}): P={results[0]['precision']:.4f}, R={results[0]['recall']:.4f}, Pairwise F1={best_f1:.4f}")

    # Breakdown by Transformation Families
    print("\nPer-Transformation Family Robustness:")
    tf_names = {
        0: "lowercase_remove_punctuation",
        1: "prepend_update_append_verify",
        2: "remove_final_15pct_tokens",
        3: "swap_alphanumeric_at_40_intervals"
    }
    for t_type, t_name in tf_names.items():
        t_mask = (transform_types == t_type)
        p_t, r_t, f1_t = compute_pairwise_f1(best_preds[t_mask], true_events[t_mask])
        print(f"  Transform {t_type} ({t_name:<35}): N={t_mask.sum():>4}, F1={f1_t:.4f} (P={p_t:.4f}, R={r_t:.4f})")

    # Semantic Equivalence: Variant-to-Parent Pairing
    # Check what fraction of variant-parent pairs end up in the exact same predicted cluster
    parent_map = {r.id: meta_dict[r.id].source_item_id for r in reports}
    id_to_pred = {r.id: best_preds[i] for i, r in enumerate(reports)}

    variant_parent_same_cluster = 0
    total_var_pairs = 0
    for r in reports:
        if meta_dict[r.id].is_variant:
            total_var_pairs += 1
            src_id = meta_dict[r.id].source_item_id
            if id_to_pred[r.id] == id_to_pred.get(src_id):
                variant_parent_same_cluster += 1

    print(f"\nExact Variant-to-Parent Semantic Fusion Rate: {variant_parent_same_cluster}/{total_var_pairs} ({variant_parent_same_cluster/total_var_pairs*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
