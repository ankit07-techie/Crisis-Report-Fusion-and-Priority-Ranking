# Interface Contracts: Crisis Report Fusion and Priority Ranking (AI-03)

This document specifies the exact function signatures, data types, and interface contracts agreed upon for the AI-03 system. Teammates implementing P1 and P2 should conform to these contracts.

All components import types directly from `src/schemas.py`.

---

## 1. P1: Fusion and Clustering Interface Contract

### Target Module Locations:
- Preferred: `src/clustering.py` or `src/fusion.py`
- Fallback Stub: `src/stubs/clustering_stub.py`

### Signatures:

```python
from typing import List, Optional
from src.schemas import Report, ClusterResult

def cluster_reports(
    reports: List[Report],
    similarity_threshold: Optional[float] = None
) -> List[ClusterResult]:
    """
    Cluster a collection of raw crisis reports into discrete incident groups.

    Args:
        reports: List of Report objects (each containing 'id' and 'text').
        similarity_threshold: Optional cosine/semantic similarity cutoff.

    Returns:
        List of ClusterResult objects, where each ClusterResult contains:
        - cluster_id (str): Unique incident identifier (e.g., 'INCIDENT_001').
        - confidence (float): Confidence score between 0.0 and 1.0.
        - evidence_ids (List[str]): List of report IDs belonging to this cluster.
        - summary (Optional[str]): Short incident summary or exemplar text.
    """
    ...

def assign_cluster(
    report: Report,
    existing_reports: Optional[List[Report]] = None,
    similarity_threshold: Optional[float] = None
) -> ClusterResult:
    """
    Assign a single incoming report to either an existing cluster or spawn a new cluster.

    Args:
        report: Single incoming Report.
        existing_reports: Optional history of previously indexed reports.
        similarity_threshold: Optional similarity cutoff.

    Returns:
        ClusterResult with cluster assignment and evidence report IDs.
    """
    ...
```

---

## 2. P2: Category & Priority Prediction Interface Contract

### Target Module Locations:
- Preferred: `src/classification.py` and `src/priority.py` (or combined `src/prediction.py`)
- Fallback Stub: `src/stubs/prediction_stub.py`

### Signatures:

```python
from typing import List, Tuple
from src.schemas import Report

def predict_category(text: str) -> str:
    """
    Predict the humanitarian crisis category for a single report.

    Args:
        text: Raw crisis message string.

    Returns:
        Category name (e.g. 'Search & Rescue', 'Medical Assistance', 'Infrastructure Damage').
    """
    ...

def predict_priority(text: str) -> float:
    """
    Predict operational priority score for a single report.

    Args:
        text: Raw crisis message string.

    Returns:
        Float priority score bounded between 1.0 (lowest) and 5.0 (critical emergency).
    """
    ...

def predict_category_and_priority_batch(
    reports: List[Report]
) -> List[Tuple[str, float]]:
    """
    Batch classification and priority scoring for enhanced throughput.

    Args:
        reports: List of Report objects.

    Returns:
        List of tuples: [(category, priority_score), ...]
    """
    ...
```

---

## 3. P3: Integration Pipeline Contract

### Module Location: `src/pipeline.py`

### Signatures:

```python
from typing import List
from src.schemas import Report, Prediction

class CrisisPipeline:
    def __init__(self, config=None, use_stubs_if_missing: bool = True):
        """Initializes pipeline, dynamically binding P1 and P2 modules or fallbacks."""
        ...

    def process_report(self, report: Report) -> Prediction:
        """Process a single report through clustering, classification, priority, and evidence linking."""
        ...

    def process_batch(self, reports: List[Report]) -> List[Prediction]:
        """Process a collection of reports end-to-end with evidence tracking."""
        ...
```

---

## 4. P3: Official Evaluation Runner Contract

### Module Location: `evaluate.py`

### CLI Execution:
```bash
python evaluate.py --input <path_to_input> --output <path_to_output> [--format jsonl|csv]
```

### Input Requirements:
- Accepts only opaque identifiers (`id` or `item_id`) and report text (`text` or `report_text`).
- No ground truth labels, cluster annotations, or source transformations expected.

### Output Guarantee:
- Structured JSONL or CSV containing:
  - `item_id` (str)
  - `predicted_cluster_id` (str)
  - `category` (str)
  - `predicted_information_category` (str)
  - `priority_score` (float)
  - `evidence_ids` (List[str] or comma-separated string)

---

## 5. P1/P2 HANDOFF

### Instructions for P1 (Clustering / Incident Fusion Engineer):
- **Target File**: Implement in `src/clustering.py` (or `src/fusion.py`).
- **Required Function**:
  ```python
  def cluster_reports(reports: List[Report]) -> List[ClusterResult]:
      ...
  ```
  *(Optionally also `assign_cluster(report: Report, existing_reports: Optional[List[Report]]) -> ClusterResult`)*
- **Input**: `List[Report]`, where each `Report` provides `.id` (str) and `.text` (str).
- **Expected Output**: `List[ClusterResult]`, where each `ClusterResult` has:
  - `cluster_id` (str): Incident group ID (e.g., `"INCIDENT_001"`).
  - `evidence_ids` (List[str]): List of report IDs belonging to this cluster.
  - `confidence` (float, default 1.0): Clustering confidence.
  - `summary` (Optional[str]): Incident summary or exemplar snippet.
- **Note**: Do NOT worry about UI or evaluation I/O; P3 pipeline consumes your function directly.

---

### Instructions for P2 (Classification & Priority Engineer):
- **Target File**: Implement in `src/classification.py` and `src/priority.py` (or `src/prediction.py`).
- **Required Functions**:
  ```python
  def predict_category(text: str) -> str:
      ...

  def predict_priority(text: str) -> float:
      ...
  ```
- **Inputs**: Raw crisis report string `text: str`.
- **Expected Outputs**:
  - `predict_category(text)`: Returns category string (e.g., `"Search & Rescue"`, `"Medical Assistance"`, `"Infrastructure Damage"`, `"Hazardous Material / Fire"`, `"Food & Water Shortage"`, `"Shelter & Evacuation"`, `"General Information"`).
  - `predict_priority(text)`: Returns a numeric float between `1.0` (Lowest) and `5.0` (Critical Emergency).
- **Note**: P3 integration handles batch orchestration and evidence binding automatically.

