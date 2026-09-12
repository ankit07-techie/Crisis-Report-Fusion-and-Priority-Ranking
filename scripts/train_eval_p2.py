"""P2 Official Training, Evaluation, Audit, and Serialization Script for AI-03.
Primary Evaluation: Authoritative REAL TREC-IS 2020-A Crisis Collection (Events 35-49)
  - 4,076 unique original reports
  - 2,423 unique deterministic variants
  - 6,499 total reports.
Diagnostic Evaluation: Synthetic Benchmark Corpus (6,490 items: 1,298 original + 5,192 variants).
"""

import gzip
import json
import math
import os
import random
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# Official 11 TREC-IS Task-2 Reduced Information Categories
CATEGORIES = [
    "Location",
    "EmergingThreats",
    "MultimediaShare",
    "MovePeople",
    "NewSubEvent",
    "FirstPartyObservation",
    "InformationWanted",
    "ServiceAvailable",
    "SearchAndRescue",
    "Volunteer",
    "GoodsServices"
]
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}

TASK2_ACTIONABLE_PRIORITY = [
    "SearchAndRescue", "MovePeople", "EmergingThreats", "ServiceAvailable",
    "GoodsServices", "InformationWanted", "Volunteer", "NewSubEvent",
    "FirstPartyObservation", "Location", "MultimediaShare"
]
TASK2_SET = set(TASK2_ACTIONABLE_PRIORITY)

PRIORITY_SCORE_MAP = {
    "Critical": 5.0, "High": 4.0, "Medium": 3.0, "Low": 1.5
}

def compute_sha256_bytes(msg_id: str):
    digest = hashlib.sha256(f"20260911:{msg_id}".encode("utf-8")).digest()
    return digest[0], digest[1]

def generate_variant(msg_id: str, text: str):
    h0, h1 = compute_sha256_bytes(msg_id)
    if (h0 % 10) not in {0, 1, 2, 3, 4, 5}:
        return None
    t_type = h1 % 4
    if t_type == 0:
        v_text = text.lower().translate(str.maketrans("", "", string.punctuation))
        t_name = "lower_punct"
    elif t_type == 1:
        v_text = f"Update: {text.strip()} Please verify."
        t_name = "wrapper"
    elif t_type == 2:
        tokens = text.split()
        n = len(tokens)
        if n <= 1:
            v_text = text
        else:
            k = min(math.ceil(0.15 * n), n - 1)
            v_text = " ".join(tokens[:-k])
        t_name = "truncation"
    elif t_type == 3:
        chars = list(text)
        pos = 40
        while pos < len(chars):
            idx = pos - 1
            if idx + 1 < len(chars) and chars[idx].isalnum() and chars[idx+1].isalnum():
                chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            pos += 40
        v_text = "".join(chars)
        t_name = "char_swap"
    return {
        "variant_id": f"{msg_id}_var_{t_type}",
        "variant_text": v_text,
        "transform_type": t_type,
        "transform_name": t_name
    }

import hashlib

def load_real_trecis_2020a_dataset():
    """Load authoritative REAL TREC-IS 2020-A dataset (Events 35-49)."""
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    gz_path = raw_dir / "TRECIS-2018-2020B.json.gz"
    
    if not gz_path.exists():
        import urllib.request
        raw_dir.mkdir(parents=True, exist_ok=True)
        url = "https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2021/TRECIS-2018-2020B.json.gz"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(gz_path, "wb") as f:
            f.write(resp.read())

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    unique_originals = {}
    for ev in data.get("events", []):
        trecis_id = ev.get("trecisid", "")
        parts = trecis_id.split("-")
        try:
            num = int(parts[-1])
        except (ValueError, IndexError):
            continue
        if not (35 <= num <= 49):
            continue

        for tw in ev.get("tweets", []):
            post_id = str(tw.get("postID", ""))
            text = str(tw.get("postText", "")).strip()
            priority = tw.get("postPriority")
            all_cats = tw.get("postCategories", [])
            if not priority or priority == "None":
                continue
            task2_cats = [c.split("-")[-1] for c in all_cats if c.split("-")[-1] in TASK2_SET]
            if not task2_cats:
                continue
            primary_cat = next((c for c in TASK2_ACTIONABLE_PRIORITY if c in task2_cats), task2_cats[0])
            if post_id not in unique_originals:
                unique_originals[post_id] = {
                    "item_id": post_id,
                    "source_item_id": post_id,
                    "event_id": trecis_id,
                    "text": text,
                    "category": primary_cat,
                    "priority": priority,
                    "priority_score": PRIORITY_SCORE_MAP.get(priority, 3.0),
                    "is_variant": False,
                    "transform_type": "original"
                }

    orig_list = list(unique_originals.values())
    variants_by_source = {}
    all_real_variants = []
    for sid, r in unique_originals.items():
        var = generate_variant(sid, r["text"])
        if var:
            v_rec = dict(r)
            v_rec["item_id"] = var["variant_id"]
            v_rec["text"] = var["variant_text"]
            v_rec["is_variant"] = True
            v_rec["transform_type"] = var["transform_name"]
            variants_by_source[sid] = v_rec
            all_real_variants.append(v_rec)

    return orig_list, variants_by_source, all_real_variants

token_pattern = re.compile(r"\b[a-z0-9]+\b")
def get_doc_ngrams(text: str):
    toks = token_pattern.findall(text.lower())
    res = list(toks)
    for i in range(len(toks) - 1):
        res.append(toks[i] + "_" + toks[i+1])
    return res

def compute_all_category_metrics(y_true: list[int], y_pred: list[int], categories: list[str]):
    total = len(y_true)
    total_tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    micro_f1 = total_tp / total if total > 0 else 0.0

    f1s = []
    weighted_f1s = []
    class_metrics = {}
    
    n_cats = len(categories)
    cm = np.zeros((n_cats, n_cats), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    for c_idx, cat in enumerate(categories):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c_idx and p == c_idx)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c_idx and p == c_idx)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c_idx and p != c_idx)
        support = sum(1 for t in y_true if t == c_idx)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        f1s.append(f1)
        weighted_f1s.append(f1 * support)
        class_metrics[cat] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = sum(weighted_f1s) / total if total > 0 else 0.0

    return macro_f1, micro_f1, weighted_f1, class_metrics, cm

def dcg_at_k(r, k=None):
    r = np.asarray(r, dtype=float)
    if k is not None:
        r = r[:k]
    if not r.size:
        return 0.0
    return float(np.sum((2.0**r - 1.0) / np.log2(np.arange(2, r.size + 2))))

def ndcg_at_k(r, k=None):
    dcg = dcg_at_k(r, k)
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.0
    return dcg / idcg

def compute_group_ndcg(test_items: list[dict], y_true_pri: np.ndarray, y_pred_pri: np.ndarray):
    event_ndcgs = []
    test_events = Counter(x.get("event_id", "default_event") for x in test_items)
    for ev_id in test_events:
        ev_indices = [i for i, x in enumerate(test_items) if x.get("event_id", "default_event") == ev_id]
        if len(ev_indices) > 1:
            ev_true = y_true_pri[ev_indices]
            ev_pred = y_pred_pri[ev_indices]
            sorted_order = np.argsort(-ev_pred)
            sorted_true = ev_true[sorted_order]
            ndcg_val = ndcg_at_k(sorted_true, len(sorted_true))
            event_ndcgs.append(ndcg_val)
    return float(np.mean(event_ndcgs)) if event_ndcgs else 0.0

def run_training_and_evaluation():
    print("==========================================================================")
    print("STARTING P2 MODEL TRAINING AND DUAL EVALUATION (REAL TREC-IS vs SYNTHETIC)")
    print("==========================================================================\n")

    # 1. Load REAL TREC-IS 2020-A Dataset
    orig_list, variants_by_source, all_real_variants = load_real_trecis_2020a_dataset()
    
    # 2. Strict Split: REAL ORIGINAL reports first 80/20, then attach variants to same split
    random.seed(42)
    orig_shuffled = list(orig_list)
    random.shuffle(orig_shuffled)
    n_train_orig = int(len(orig_shuffled) * 0.8)
    train_origs = orig_shuffled[:n_train_orig]
    test_origs = orig_shuffled[n_train_orig:]

    train_items = []
    for r in train_origs:
        train_items.append(r)
        if r["source_item_id"] in variants_by_source:
            train_items.append(variants_by_source[r["source_item_id"]])

    test_items = []
    for r in test_origs:
        test_items.append(r)
        if r["source_item_id"] in variants_by_source:
            test_items.append(variants_by_source[r["source_item_id"]])

    # Leakage Check and Strict Filtering
    train_sids = set(x["source_item_id"] for x in train_items)
    test_sids = set(x["source_item_id"] for x in test_items)
    id_leakage = len(train_sids & test_sids)
    
    train_texts_set = set(x["text"] for x in train_items)
    # Remove any test item whose text identically matches any training text
    test_items = [x for x in test_items if x["text"] not in train_texts_set]
    test_origs = [x for x in test_origs if x["text"] not in train_texts_set]

    test_texts_set = set(x["text"] for x in test_items)
    text_leakage = len(train_texts_set & test_texts_set)

    print("--------------------------------------------------------------------------")
    print("DATASET SPLIT & INTEGRITY VERIFICATION (REAL TREC-IS 2020-A):")
    print(f"  - Total Real Originals:        {len(orig_list)}")
    print(f"  - Total Real Variants:         {len(all_real_variants)}")
    print(f"  - Total Real Combined Items:   {len(orig_list) + len(all_real_variants)}")
    print(f"  - Train Originals:             {len(train_origs)} (80%)")
    print(f"  - Train Total Items:           {len(train_items)} ({len(train_origs)} originals + {len(train_items)-len(train_origs)} variants)")
    print(f"  - Test Originals:              {len(test_origs)} (20%)")
    print(f"  - Test Combined Items:         {len(test_items)} ({len(test_origs)} originals + {len(test_items)-len(test_origs)} variants)")
    print(f"  - ID Leakage Check:            {id_leakage} overlapping IDs")
    print(f"  - Text Leakage Check:          {text_leakage} overlapping texts")
    print("--------------------------------------------------------------------------\n")

    # 3. Tokenize & Vocabulary
    train_ngram_lists = [get_doc_ngrams(item["text"]) for item in train_items]
    test_ngram_lists = [get_doc_ngrams(item["text"]) for item in test_items]

    df_counter = Counter()
    N_train = len(train_items)
    for ngrams in train_ngram_lists:
        for t in set(ngrams):
            df_counter[t] += 1

    vocab = [t for t, count in df_counter.most_common(2000) if count >= 3][:1500]
    vocab.sort()
    vocab_map = {t: i for i, t in enumerate(vocab)}
    vocab_size = len(vocab)
    idfs_arr = np.array([math.log((N_train + 1.0) / (df_counter[t] + 1.0)) + 1.0 for t in vocab], dtype=np.float32)
    idfs_dict = {t: float(idfs_arr[i]) for i, t in enumerate(vocab)}

    def build_dense_matrix(ngram_lists):
        M = np.zeros((len(ngram_lists), vocab_size), dtype=np.float32)
        for i, ngrams in enumerate(ngram_lists):
            tc = Counter(ngrams)
            for t, cnt in tc.items():
                if t in vocab_map:
                    M[i, vocab_map[t]] = (1.0 + math.log(cnt)) * idfs_arr[vocab_map[t]]
            norm = np.linalg.norm(M[i])
            if norm > 0:
                M[i] /= norm
        return M

    X_train = build_dense_matrix(train_ngram_lists)
    X_test = build_dense_matrix(test_ngram_lists)

    y_train_cat = np.array([CAT_TO_IDX[item["category"]] for item in train_items], dtype=int)
    y_train_pri = np.array([item["priority_score"] for item in train_items], dtype=np.float32)
    y_test_cat = np.array([CAT_TO_IDX[item["category"]] for item in test_items], dtype=int)
    y_test_pri = np.array([item["priority_score"] for item in test_items], dtype=np.float32)

    # 4. Train Models
    num_classes = len(CATEGORIES)
    cat_class_counts = np.bincount(y_train_cat, minlength=num_classes)
    cat_priors = np.log((cat_class_counts + 1.0) / (N_train + num_classes))

    cat_feat_sums = np.zeros((num_classes, vocab_size), dtype=np.float32)
    for c in range(num_classes):
        cat_feat_sums[c] = np.sum(X_train[y_train_cat == c], axis=0)

    cat_weights = np.log((cat_feat_sums + 0.1) / (np.sum(cat_feat_sums, axis=1, keepdims=True) + 0.1 * vocab_size))

    # Priority Model (Ridge)
    alpha = 10.0
    A = np.dot(X_train.T, X_train) + alpha * np.eye(vocab_size, dtype=np.float32)
    b = np.dot(X_train.T, y_train_pri)
    pri_weights = np.linalg.solve(A, b)
    pri_bias = float(np.mean(y_train_pri) - np.mean(np.dot(X_train, pri_weights)))

    # 5. Evaluate on REAL Test Split
    scores_test = np.dot(X_test, cat_weights.T) + cat_priors
    preds_test_cat = np.argmax(scores_test, axis=1)

    preds_test_pri = np.dot(X_test, pri_weights) + pri_bias
    preds_test_pri = np.clip(preds_test_pri, 1.0, 5.0)

    real_macro_f1, real_micro_f1, real_weighted_f1, real_per_class, cm_real = compute_all_category_metrics(y_test_cat, preds_test_cat, CATEGORIES)
    real_pri_mae = float(np.mean(np.abs(y_test_pri - preds_test_pri)))
    real_pri_ndcg = compute_group_ndcg(test_items, y_test_pri, preds_test_pri)

    print("==========================================================================")
    print("REAL TREC-IS EVALUATION (OFFICIAL PRIMARY BENCHMARK)")
    print("==========================================================================")
    print(f"Evaluated Items:        {len(test_items)} items ({len(test_origs)} originals + {len(test_items)-len(test_origs)} variants)")
    print(f"Category Macro-F1:      {real_macro_f1:.4f} (Official Weight: 25%)")
    print(f"Category Micro-F1:      {real_micro_f1:.4f}")
    print(f"Category Weighted-F1:   {real_weighted_f1:.4f}")
    print(f"Priority MAE:           {real_pri_mae:.4f}")
    print(f"Priority NDCG:          {real_pri_ndcg:.4f} (Official Weight: 20%)")
    print("--------------------------------------------------------------------------")
    print("Per-Class Information Category Results (REAL Test Set):")
    print(f"{'Category':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for cat, m in real_per_class.items():
        print(f"{cat:<25} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['support']:<8}")

    print("\n--------------------------------------------------------------------------")
    print("REAL Deterministic Variants Robustness Breakdown (All 2,423 Real Variants):")
    all_var_ngram_lists = [get_doc_ngrams(item["text"]) for item in all_real_variants]
    X_all_var = build_dense_matrix(all_var_ngram_lists)
    y_all_var_cat = np.array([CAT_TO_IDX[item["category"]] for item in all_real_variants], dtype=int)
    y_all_var_pri = np.array([item["priority_score"] for item in all_real_variants], dtype=np.float32)

    p_all_var_cat = np.argmax(np.dot(X_all_var, cat_weights.T) + cat_priors, axis=1)
    p_all_var_pri = np.clip(np.dot(X_all_var, pri_weights) + pri_bias, 1.0, 5.0)

    print(f"{'Transformation Family':<20} | {'Macro-F1':<10} | {'Micro-F1':<10} | {'Priority MAE':<12} | {'Count':<8}")
    print("-" * 70)
    for tf_name in ["lower_punct", "wrapper", "truncation", "char_swap"]:
        tf_indices = [i for i, x in enumerate(all_real_variants) if x["transform_type"] == tf_name]
        if tf_indices:
            sub_y_cat = y_all_var_cat[tf_indices]
            sub_p_cat = p_all_var_cat[tf_indices]
            sub_f1, sub_mic, _, _, _ = compute_all_category_metrics(sub_y_cat, sub_p_cat, CATEGORIES)
            sub_y_pri = y_all_var_pri[tf_indices]
            sub_p_pri = p_all_var_pri[tf_indices]
            sub_mae = float(np.mean(np.abs(sub_y_pri - sub_p_pri)))
            print(f"{tf_name:<20} | {sub_f1:<10.4f} | {sub_mic:<10.4f} | {sub_mae:<12.4f} | {len(tf_indices):<8}")
    print("==========================================================================\n")

    # 6. Evaluate on SYNTHETIC Diagnostic Benchmark
    syn_file = Path(__file__).resolve().parent.parent / "data" / "trec_is_prepared.json"
    if syn_file.exists():
        with open(syn_file, "r", encoding="utf-8") as f:
            syn_dataset = json.load(f)
        
        syn_ngram_lists = [get_doc_ngrams(item["text"]) for item in syn_dataset]
        X_syn = build_dense_matrix(syn_ngram_lists)
        y_syn_cat = np.array([CAT_TO_IDX[item["category"]] for item in syn_dataset], dtype=int)
        y_syn_pri = np.array([item["priority"] for item in syn_dataset], dtype=np.float32)

        preds_syn_cat = np.argmax(np.dot(X_syn, cat_weights.T) + cat_priors, axis=1)
        preds_syn_pri = np.clip(np.dot(X_syn, pri_weights) + pri_bias, 1.0, 5.0)

        syn_macro_f1, syn_micro_f1, _, _, _ = compute_all_category_metrics(y_syn_cat, preds_syn_cat, CATEGORIES)
        syn_pri_mae = float(np.mean(np.abs(y_syn_pri - preds_syn_pri)))
        syn_pri_ndcg = compute_group_ndcg(syn_dataset, y_syn_pri, preds_syn_pri)

        print("==========================================================================")
        print("SYNTHETIC DIAGNOSTIC EVALUATION (UNIT / DIAGNOSTIC BENCHMARK)")
        print("==========================================================================")
        print(f"Evaluated Corpus:       {len(syn_dataset)} synthetic items (1,298 original + 5,192 variants)")
        print(f"Category Macro-F1:      {syn_macro_f1:.4f}")
        print(f"Category Micro-F1:      {syn_micro_f1:.4f}")
        print(f"Priority MAE:           {syn_pri_mae:.4f}")
        print(f"Priority NDCG:          {syn_pri_ndcg:.4f}")
        print("Note: Synthetic results are for code path verification only.")
        print("==========================================================================\n")

    # 7. Serialize Trained Model Weights
    artifacts_dir = Path(__file__).resolve().parent.parent / "data" / "p2_models"
    artifacts_dir.mkdir(exist_ok=True)

    centroids = {}
    for c in range(num_classes):
        cat_vecs = X_train[y_train_cat == c]
        if len(cat_vecs) > 0:
            c_mean = np.mean(cat_vecs, axis=0)
            norm = np.linalg.norm(c_mean)
            if norm > 0:
                c_mean /= norm
            centroids[CATEGORIES[c]] = c_mean.tolist()
        else:
            centroids[CATEGORIES[c]] = [0.0] * vocab_size

    weights_data = {
        "categories": CATEGORIES,
        "vocab": vocab_map,
        "idf": idfs_dict,
        "centroids": centroids,
        "cat_weights": cat_weights.tolist(),
        "cat_priors": cat_priors.tolist(),
        "priority_weights": pri_weights.tolist(),
        "priority_bias": pri_bias,
        "real_trec_is_metrics": {
            "num_real_originals": len(orig_list),
            "num_real_variants": len(all_real_variants),
            "train_originals": len(train_origs),
            "test_originals": len(test_origs),
            "combined_test_items": len(test_items),
            "category_macro_f1": round(real_macro_f1, 4),
            "category_micro_f1": round(real_micro_f1, 4),
            "priority_mae": round(real_pri_mae, 4),
            "priority_ndcg": round(real_pri_ndcg, 4),
            "per_class": real_per_class
        }
    }

    weights_file = artifacts_dir / "model_weights.json"
    with open(weights_file, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2)

    print(f"Successfully serialized P2 model weights to {weights_file}")

if __name__ == "__main__":
    run_training_and_evaluation()
