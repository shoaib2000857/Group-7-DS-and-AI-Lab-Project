"""
Stage C Evidence Inspection package.
"""

from POC.evidence.schemas import (
    ClaimContent,
    EvidenceBlock,
    DecisionBlock,
    ClaimDecisionRecord,
)
from POC.evidence.crop_extractor import compute_padded_bbox, extract_bitemporal_crops
from POC.evidence.visual_inspector import VisualEvidenceInspector
from POC.evidence.pipeline import StageCEvidencePipeline

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
