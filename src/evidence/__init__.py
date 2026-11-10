"""
Stage C Evidence Inspection package.
"""

from src.evidence.schemas import (
    ClaimContent,
    EvidenceBlock,
    DecisionBlock,
    ClaimDecisionRecord,
)
from src.evidence.crop_extractor import compute_padded_bbox, extract_bitemporal_crops
from src.evidence.visual_inspector import VisualEvidenceInspector
from src.evidence.pipeline import StageCEvidencePipeline

__all__ = [
    "ClaimContent",
    "EvidenceBlock",
    "DecisionBlock",
    "ClaimDecisionRecord",
    "compute_padded_bbox",
    "extract_bitemporal_crops",
    "VisualEvidenceInspector",
    "StageCEvidencePipeline",
]
