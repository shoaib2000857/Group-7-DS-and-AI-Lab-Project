"""
Typed schemas and contracts for Stage C (Visual Evidence Inspection) and Stage D (Decision Referee).
Strictly adheres to POC/benchmarks/templates/claim_decision.json.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class ClaimContent(BaseModel):
    object: Optional[str] = Field(None, description="e.g. building, road, vegetation, other")
    action: Optional[str] = Field(None, description="e.g. constructed, demolished, modified, unchanged")
    location: Optional[str] = Field(None, description="e.g. upper-left, northeast, center")
    quantity: Optional[int] = Field(None, description="Integer count if distinct objects are distinguishable")
    quantity_unit: Optional[str] = Field("count", description="Units for quantity, typically 'count'")


class EvidenceBlock(BaseModel):
    mask_run_id: Optional[str] = Field(None, description="Identifier of the detector run that produced candidate mask")
    mask_ontology: Optional[str] = Field("binary_or_ternary", description="Ontology of change mask (e.g. building_road)")
    witness_run_id: str = Field(..., description="Identifier of the visual evidence inspector module")
    visible_change: Optional[bool] = Field(None, description="True if physical change is visually confirmed in crop")
    confidence: Optional[float] = Field(None, description="Confidence score in [0,1] for the visible_change verdict")
    state_transition: Optional[str] = Field(None, description="e.g. appearance_of_structure, removal_of_structure, surface_alteration, no_physical_change")
    observations: str = Field(..., description="Natural language observation describing T1 vs T2 visual state")
    visibility_flags: List[str] = Field(default_factory=list, description="e.g. ['clear_t1', 'clear_t2', 'shadow_hazard']")
    quality_flags: List[str] = Field(default_factory=list, description="e.g. ['registered', 'subpixel_jitter', 'good_contrast']")
    crop_paths: List[str] = Field(default_factory=list, description="Relative paths to saved T1 and T2 crop images")
    structural_edge_change: Optional[float] = Field(None, description="Quantified edge density change (Sobel/Canny)")
    mean_luminance_diff: Optional[float] = Field(None, description="T2 - T1 mean luminance shift in crop")
    crop_mean_diff: Optional[float] = Field(None, description="Mean absolute pixel difference after radiometric normalization")
    transform_record: Optional[Dict[str, Any]] = Field(None, description="Geometric transform / padding applied")


class DecisionBlock(BaseModel):
    outcome: Optional[Literal["accept", "reject", "defer"]] = Field(None, description="Referee verdict")
    allowed_outcomes: List[str] = Field(default=["accept", "reject", "defer"])
    reason_codes: List[str] = Field(default_factory=list, description="Audit reason codes for the decision")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Confidence/probabilities if available")
    probability_semantics: Optional[str] = Field(None)
    raw_response_path: Optional[str] = Field(None)
    parse_status: Optional[str] = Field("success")


class ClaimDecisionRecord(BaseModel):
    record_type: str = Field("stage_c_evidence_record", description="Record type indicator")
    pair_id: str = Field(..., description="Unique temporal pair identifier (dataset/split/filename)")
    claim_id: str = Field(..., description="Unique claim identifier")
    source_span: str = Field(..., description="Exact substring from caption that generated this claim")
    claim: ClaimContent = Field(..., description="Structured claim attributes")
    candidate_region_ids: List[List[int]] = Field(
        default_factory=list, 
        description="Bounding box [x0, y0, x1, y1] in half-open convention"
    )
    coordinate_frame: str = Field("original_T1_pixels", description="Coordinate reference frame")
    bbox_convention: str = Field("x0_y0_x1_y1_half_open", description="Bounding box coordinate convention")
    evidence: EvidenceBlock = Field(..., description="Stage C visual evidence observations")
    decision: Optional[DecisionBlock] = Field(default=None, description="Stage D referee decision block")
    human_reference: str = Field(
        "stored_separately_and_never_passed_to_inference",
        description="Preserves zero-leakage protocol"
    )
