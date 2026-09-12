# P1: Semantic Embedding + Report Fusion / Clustering

This document specifies the architecture, final production configuration, empirical benchmark results, and interface guarantees for the **P1 Semantic Fusion & Incident Clustering** subsystem of AI-03 (Crisis Report Fusion and Priority Ranking).

---

## 1. Final Locked Production Configuration

Based on systematic benchmarking and threshold optimization sweeps across candidate architectures, the following production configuration is locked in `src/clustering.py`:

| Hyperparameter / Component | Production Specification | Description |
| :--- | :--- | :--- |
| **Embedding Model** | `all-MiniLM-L6-v2` | Dense sentence embedding (384-dim, 22.7M params), L2-normalized |
| **Clustering Algorithm** | `AgglomerativeClustering` | Hierarchical agglomerative clustering |
| **Distance Metric** | Cosine Distance | $d(u, v) = 1 - \cos(u, v) = 1 - u \cdot v$ (for L2-normalized vectors) |
| **Linkage Strategy** | `average` | Merges clusters based on average pairwise distance |
| **Similarity Cutoff ($\tau$)**| **`0.54`** | Distance threshold $d_{\text{cut}} = 1.0 - 0.54 = 0.46$ |
| **Fallback Embedding** | `TfidfEmbedding` | Sublinear TF-IDF baseline used automatically if transformers unavailable |

---

## 2. Measured Benchmark Performance (AI-03 Official Metric)

Evaluated on the full deduplicated development dataset ($N = 6,499$ unique reports, covering 4,076 originals and 2,423 transformed variants across 15 TREC-IS 2020-A events):

- **Pairwise Precision**: **`0.9721`** (97.21% of fused pairs genuinely belong to the same physical crisis incident)
- **Pairwise Recall**: **`0.5450`**
- **Pairwise Clustering F1**: **`0.6984`** (peak verified F1 across the neighborhood $[0.50, 0.60]$)
- **Variant-to-Parent Semantic Fusion Rate**: **`98.72%`** (2,392 of 2,423 variants fused directly with their original source message)
- **Cluster Purity**: **`92.55%`** of predicted clusters are pure single-event clusters
- **Global Pairwise False Positive Rate**: **`0.1189%`** across all $21,115,251$ evaluated report pairs
- **Inference Runtime**: $\sim 6.7$ seconds for 6,499 reports on standard CPU ($\sim 970$ reports/second)

---

## 3. Interface Contracts & System Guarantees

Implemented strictly in `src/clustering.py` according to the P3 contract in `docs/INTERFACES.md`:

```python
from typing import List, Optional
from src.schemas import Report, ClusterResult

def cluster_reports(
    reports: List[Report],
    similarity_threshold: Optional[float] = None
) -> List[ClusterResult]:
    """Batch incident clustering."""
    ...

def assign_cluster(
    report: Report,
    existing_reports: Optional[List[Report]] = None,
    similarity_threshold: Optional[float] = None
) -> ClusterResult:
    """Online dynamic report clustering."""
    ...
```

### Safety & Integrity Guarantees:
1. **Opaque Input Isolation**: Production clustering accesses strictly `Report.id` and `Report.text`. It has zero access to `event_id`, `source_item_id`, category labels, priority ratings, or evaluation cluster annotations.
2. **Authentic Evidence Preservation**: `evidence_ids` in each `ClusterResult` contains strictly the original `Report.id` values belonging to that cluster. No synthetic or fabricated IDs are ever created.
3. **P3 Dynamic Discovery**: `CrisisPipeline` automatically resolves `src.clustering` as its primary P1 engine.

---

## 4. Reproducibility & Validation Commands

All data preparation, benchmarking, and audits can be reproduced with:

```bash
# 1. Prepare and filter development dataset with SHA256 variants
python scripts/prepare_p1_dataset.py

# 2. Run comprehensive data validation
python scripts/validate_dataset.py

# 3. Run candidate embedding and clustering benchmark
python scripts/benchmark_p1.py

# 4. Run fine-grained threshold and robustness audit
python scripts/audit_p1_optimization.py

# 5. Run full test suite
pytest -v tests/
```
