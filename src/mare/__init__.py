"""MARE: Modality-Aware Retrieval Engine."""

from .engine import MAREngine
from .fusion import WeightedScoreFusion
from .router import HeuristicModalityRouter
from .types import Document, Modality, QueryPlan, RetrievalExplanation, RetrievalHit

__all__ = [
    "Document",
    "HeuristicModalityRouter",
    "MAREngine",
    "Modality",
    "QueryPlan",
    "RetrievalExplanation",
    "RetrievalHit",
    "WeightedScoreFusion",
]

