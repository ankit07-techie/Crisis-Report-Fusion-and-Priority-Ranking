"""Shared data schemas for Crisis Report Fusion and Priority Ranking (AI-03).
Independent of UI/Streamlit.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Report(BaseModel):
    """Input crisis report schema."""
    id: str = Field(..., description="Unique identifier for the report/item")
    text: str = Field(..., min_length=1, description="Raw crisis text content")
    timestamp: Optional[str] = Field(None, description="Optional ISO timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary optional metadata")

    model_config = ConfigDict(extra="ignore")


class ClusterResult(BaseModel):
    """Output from clustering/fusion module (P1)."""
    cluster_id: str = Field(..., description="Predicted incident/cluster identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Clustering confidence score")
    evidence_ids: List[str] = Field(default_factory=list, description="Associated report IDs belonging to this incident cluster")
    summary: Optional[str] = Field(None, description="Optional synthesized incident cluster summary")

    model_config = ConfigDict(extra="ignore")


class Prediction(BaseModel):
    """End-to-end prediction output for a crisis report."""
    item_id: str = Field(..., description="Original report/item identifier")
    predicted_cluster_id: str = Field(..., description="Assigned incident/cluster identifier")
    category: str = Field(..., description="Predicted crisis information category")
    predicted_information_category: Optional[str] = Field(None, description="Predicted information category (alias)")
    priority_score: float = Field(..., description="Operational priority score (1.0 to 5.0 scale)")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of actual reports associated as evidence")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context or diagnostics")

    model_config = ConfigDict(extra="ignore")

    def model_post_init(self, __context: Any) -> None:
        if not self.predicted_information_category:
            self.predicted_information_category = self.category


class EvaluationRecord(BaseModel):
    """Evaluation record format strictly matching the evaluation specification."""
    item_id: str = Field(..., description="Original opaque item identifier")
    predicted_cluster_id: str = Field(..., description="Predicted incident/cluster identifier")
    category: str = Field(..., description="Predicted information category")
    predicted_information_category: str = Field(..., description="Predicted information category (alias)")
    priority_score: float = Field(..., description="Operational priority score")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence item IDs")

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_prediction(cls, pred: Prediction) -> "EvaluationRecord":
        """Convert an internal Prediction object into an official EvaluationRecord."""
        return cls(
            item_id=pred.item_id,
            predicted_cluster_id=pred.predicted_cluster_id,
            category=pred.category,
            predicted_information_category=pred.category,
            priority_score=round(pred.priority_score, 2),
            evidence_ids=list(pred.evidence_ids)
        )
