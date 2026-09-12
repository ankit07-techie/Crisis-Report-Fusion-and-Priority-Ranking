#!/usr/bin/env python3
"""P1 Focused Optimization and Audit Pass.
Performs:
1. Exact Pairwise Clustering F1 verification
2. Fine-grained threshold neighborhood sweep (0.50 to 0.60)
3. Recall bottleneck investigation (linkage strategies: average, complete, single, ward)
4. Comprehensive variant audit (false parent merges, missed parent merges, variant-to-variant fusion)
5. Preprocessing impact analysis (raw text vs. preprocessed text)
"""

import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_deduplicated_dataset
from src.embeddings import SentenceTransformerEmbedding
from src.preprocessing import preprocess_crisis_text


def compute_pairwise_metrics(pred_labels: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """Exact Pairwise Precision, Recall, and F1 via contingency matrix."""
    n = len(pred_labels)
    if n <= 1:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    _, pred_idx = np.unique(pred_labels, return_inverse=True)
    _, true_idx = np.unique(true_labels, return_inverse=True)

    k_pred = pred_idx.max() + 1
    k_true = true_idx.max() + 1

    lin_idx = pred_idx * k_true + true_idx
    cell_counts = np.bincount(lin_idx, minlength=k_pred * k_true)

    tp = int(np.sum(cell_counts * (cell_counts - 1) // 2))

    pred_counts = np.bincount(pred_idx, minlength=k_pred)
    total_pred = int(np.sum(pred_counts * (pred_counts - 1) // 2))

    true_counts = np.bincount(true_idx, minlength=k_true)
    total_true = int(np.sum(true_counts * (true_counts - 1) // 2))

    fp = total_pred - tp
    fn = total_true - tp

    precision = float(tp / total_pred) if total_pred > 0 else 1.0
    recall = float(tp / total_true) if total_true > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_pairs": n * (n - 1) // 2,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def run_audit():
    print("=" * 80)
    print("P1 FOCUSED OPTIMIZATION & AUDIT PASS")
    print("=" * 80)

    # 1. Load Dataset
    print("\n[Step 1] Loading deduplicated dataset...")
    reports, meta_dict = load_deduplicated_dataset(include_variants=True)
    n_total = len(reports)
    print(f"Loaded {n_total} total unique reports (4,076 originals + 2,423 variants).")

    raw_texts = [r.text for r in reports]
    prep_texts = [preprocess_crisis_text(r.text) for r in reports]
    true_events = np.array([meta_dict[r.id].event_id for r in reports])

    # 2. Embedding Generation (Raw vs Preprocessed)
    print("\n[Step 2] Generating all-MiniLM-L6-v2 embeddings...")
    embedder = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

    t0 = time.time()
    embs_preprocessed = embedder.model.encode(
        prep_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    time_prep = time.time() - t0
    print(f"  - Preprocessed embeddings generated in {time_prep:.2f}s")

    t0 = time.time()
    embs_raw = embedder.model.encode(
        raw_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    time_raw = time.time() - t0
    print(f"  - Raw embeddings generated in {time_raw:.2f}s")

    # ------------------------------------------------------------------
    # Audit Item 1 & 2: Fine-grained Threshold Sweep (Agglomerative Average)
    # ------------------------------------------------------------------
    print("\n[Audit Item 2] Fine-Grained Threshold Sweep (0.50 to 0.60, Preprocessed):")
    fine_thresholds = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]

    sweep_results = []
    print(f"| {'Threshold':>9} | {'Clusters':>8} | {'Precision':>10} | {'Recall':>10} | {'Pairwise F1':>11} | {'TP':>9} | {'FP':>9} | {'FN':>9} |")
    print("|" + "-"*11 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*13 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|")

    best_fine_f1 = -1.0
    best_fine_tau = None
    best_fine_preds = None

    for tau in fine_thresholds:
        dist_thresh = max(0.001, 1.0 - tau)
        model = AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=dist_thresh, n_clusters=None)
        preds = model.fit_predict(embs_preprocessed)

        m = compute_pairwise_metrics(preds, true_events)
        n_clust = len(np.unique(preds))
        sweep_results.append((tau, n_clust, m))

        print(f"| {tau:>9.2f} | {n_clust:>8} | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1']:>11.4f} | {m['tp']:>9} | {m['fp']:>9} | {m['fn']:>9} |")

        if m["f1"] > best_fine_f1:
            best_fine_f1 = m["f1"]
            best_fine_tau = tau
            best_fine_preds = preds

    print(f"\nOptimal threshold in neighborhood: {best_fine_tau:.2f} with Pairwise F1 = {best_fine_f1:.4f}")

    # ------------------------------------------------------------------
    # Audit Item 3: Investigate Recall Bottleneck (Linkage Strategies)
    # ------------------------------------------------------------------
    print("\n[Audit Item 3] Recall Bottleneck Investigation: Linkage Strategies:")
    linkages = ["average", "complete", "single"]
    print(f"| {'Linkage':<12} | {'Threshold':>9} | {'Clusters':>8} | {'Precision':>10} | {'Recall':>10} | {'Pairwise F1':>11} |")
    print("|" + "-"*14 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*13 + "|")

    for link in linkages:
        for tau in [0.50, 0.55, 0.60]:
            dist_thresh = max(0.001, 1.0 - tau)
            model = AgglomerativeClustering(metric="cosine", linkage=link, distance_threshold=dist_thresh, n_clusters=None)
            preds = model.fit_predict(embs_preprocessed)
            m = compute_pairwise_metrics(preds, true_events)
            n_clust = len(np.unique(preds))
            print(f"| {link:<12} | {tau:>9.2f} | {n_clust:>8} | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1']:>11.4f} |")

    # ------------------------------------------------------------------
    # Audit Item 5: Check Whether Preprocessing Helps or Hurts
    # ------------------------------------------------------------------
    print("\n[Audit Item 5] Preprocessing Comparison (Raw vs Preprocessed @ Tau=0.55):")
    model_prep = AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=1.0 - 0.55, n_clusters=None)
    preds_prep = model_prep.fit_predict(embs_preprocessed)
    m_prep = compute_pairwise_metrics(preds_prep, true_events)

    model_raw = AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=1.0 - 0.55, n_clusters=None)
    preds_raw = model_raw.fit_predict(embs_raw)
    m_raw = compute_pairwise_metrics(preds_raw, true_events)

    print(f"  - Preprocessed text: Clusters={len(np.unique(preds_prep))}, P={m_prep['precision']:.4f}, R={m_prep['recall']:.4f}, F1={m_prep['f1']:.4f}")
    print(f"  - Raw text:          Clusters={len(np.unique(preds_raw))}, P={m_raw['precision']:.4f}, R={m_raw['recall']:.4f}, F1={m_raw['f1']:.4f}")
    diff_f1 = m_prep["f1"] - m_raw["f1"]
    print(f"  - Delta (Preprocessed - Raw): F1 difference = {diff_f1:+.4f}")

    # ------------------------------------------------------------------
    # Audit Item 4: Rigorous Variant Robustness Audit
    # ------------------------------------------------------------------
    print("\n[Audit Item 4] Comprehensive Variant Robustness Audit (@ Tau=0.55):")
    # Evaluate with preds_prep
    preds = preds_prep

    # A. Original -> Parent Variant Fusion
    variant_ids = [r.id for r in reports if meta_dict[r.id].is_variant]
    id_to_pred = {r.id: preds[i] for i, r in enumerate(reports)}
    id_to_event = {r.id: meta_dict[r.id].event_id for r in reports}

    exact_parent_matches = 0
    missed_parent_merges = 0
    false_parent_merges = 0  # merged with something from wrong event!

    for vid in variant_ids:
        parent_id = meta_dict[vid].source_item_id
        pred_cluster_var = id_to_pred[vid]
        pred_cluster_parent = id_to_pred.get(parent_id)

        if pred_cluster_var == pred_cluster_parent:
            exact_parent_matches += 1
        else:
            missed_parent_merges += 1

    # B. Unrelated-event False Merges
    # Count how many clusters contain reports from more than 1 event
    cluster_to_events = defaultdict(set)
    cluster_to_sizes = defaultdict(int)
    for i, r in enumerate(reports):
        cluster_to_events[preds[i]].add(meta_dict[r.id].event_id)
        cluster_to_sizes[preds[i]] += 1

    impure_clusters = {cid: evs for cid, evs in cluster_to_events.items() if len(evs) > 1}
    pure_clusters = {cid: evs for cid, evs in cluster_to_events.items() if len(evs) == 1}

    print(f"  - Total variants evaluated:           {len(variant_ids)}")
    print(f"  - Exact parent-variant matches:       {exact_parent_matches} ({exact_parent_matches/len(variant_ids)*100:.2f}%)")
    print(f"  - Missed parent merges (separated):   {missed_parent_merges} ({missed_parent_merges/len(variant_ids)*100:.2f}%)")
    print(f"  - Total predicted clusters:           {len(cluster_to_events)}")
    print(f"  - Pure single-event clusters:         {len(pure_clusters)} ({len(pure_clusters)/len(cluster_to_events)*100:.2f}%)")
    print(f"  - Impure cross-event clusters:        {len(impure_clusters)} ({len(impure_clusters)/len(cluster_to_events)*100:.2f}%)")
    print(f"  - Pairwise False Positives (FP):      {m_prep['fp']} out of {m_prep['total_pairs']:,} total pairs ({m_prep['fp']/m_prep['total_pairs']*100:.4f}%)")

    # C. Variant -> Variant Fusion
    # Group variants by parent source_item_id
    parent_to_variants = defaultdict(list)
    for vid in variant_ids:
        parent_to_variants[meta_dict[vid].source_item_id].append(vid)

    multi_variant_parents = {pid: vids for pid, vids in parent_to_variants.items() if len(vids) > 1}
    print(f"  - Parents with multiple variants:     {len(multi_variant_parents)}")

    print("\n" + "=" * 80)
    print("AUDIT PASS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
