"""
Automated Unit Tests for Stage C Visual Evidence Pipeline.
Verifies crop boundaries, physical change detection, hazard flags,
radiometric robustness, and schema contract compliance.
"""

import json
import pytest
import numpy as np

from POC.evidence.schemas import ClaimDecisionRecord
from POC.evidence.crop_extractor import compute_padded_bbox, extract_bitemporal_crops
from POC.evidence.visual_inspector import VisualEvidenceInspector
from POC.evidence.pipeline import StageCEvidencePipeline


def test_crop_padding_and_clamping():
    """Verifies that bounding boxes are padded with context and strictly clamped."""
    image_shape = (256, 256, 3)
    
    # 1. Box near border
    box_near_corner = [0, 0, 30, 30]
    padded_box, meta = compute_padded_bbox(box_near_corner, image_shape, padding_fraction=0.2)
    assert padded_box[0] >= 0
    assert padded_box[1] >= 0
    assert padded_box[2] <= 256
    assert padded_box[3] <= 256
    assert padded_box[0] == 0
    assert padded_box[1] == 0
    assert padded_box[2] > 30

    # 2. Box centered
    center_box = [100, 100, 150, 150]
    padded_box, meta = compute_padded_bbox(center_box, image_shape, padding_fraction=0.2)
    assert padded_box[0] < 100
    assert padded_box[1] < 100
    assert padded_box[2] > 150
    assert padded_box[3] > 150


def test_building_appearance_detection():
    """Verifies that newly constructed building edges trigger positive visual change."""
    inspector = VisualEvidenceInspector()
    
    # T1: Smooth grassy surface
    np.random.seed(42)
    crop_t1 = np.full((64, 64, 3), 100, dtype=np.uint8) + np.random.randint(-5, 5, (64, 64, 3), dtype=np.int16).astype(np.uint8)
    
    # T2: Same background + sharp rectangular building structure
    crop_t2 = crop_t1.copy()
    crop_t2[16:48, 16:48] = [210, 215, 220]  # High-contrast light roof
    crop_t2[15, 16:48] = [40, 40, 40]        # Building edge shadow
    crop_t2[16:48, 48] = [40, 40, 40]
    
    res = inspector.inspect_crop_pair(crop_t1, crop_t2, claimed_object="building", claimed_action="constructed")
    assert res["visible_change"] is True
    assert res["state_transition"] == "appearance_of_structure"
    assert "Visible change confirmed" in res["observations"]
    assert res["structural_edge_change"] > 0.05


def test_unchanged_crops_flag_no_change():
    """Verifies that unchanged ground crops are flagged as no physical change."""
    inspector = VisualEvidenceInspector()
    
    crop_t1 = np.full((64, 64, 3), 120, dtype=np.uint8)
    crop_t2 = crop_t1.copy()  # Exactly identical
    
    res = inspector.inspect_crop_pair(crop_t1, crop_t2, claimed_object="building", claimed_action="constructed")
    assert res["visible_change"] is False
    assert res["state_transition"] == "no_physical_change"
    assert "No verifiable physical structural change" in res["observations"]


def test_shadow_hazard_flag():
    """Verifies that uniform illumination drop without structural change triggers shadow_hazard."""
    inspector = VisualEvidenceInspector()
    
    # T1: Bright ground
    crop_t1 = np.full((64, 64, 3), 180, dtype=np.uint8)
    # T2: Uniformly darkened ground (cloud or sun shadow) without edge change
    crop_t2 = np.full((64, 64, 3), 80, dtype=np.uint8)
    
    res = inspector.inspect_crop_pair(crop_t1, crop_t2)
    assert "shadow_hazard" in res["quality_flags"]
    assert res["visible_change"] is False


def test_seasonal_illumination_hallucination_rejected():
    """
    Verifies that a large radiometric illumination drop (e.g. Delta L = -45)
    without localized rectilinear structure correctly REJECTS a phantom building claim.
    """
    inspector = VisualEvidenceInspector()
    np.random.seed(42)
    # T1: Bright natural ground (mean ~130)
    crop_t1 = np.random.randint(110, 150, (80, 80, 3), dtype=np.uint8)
    # T2: Same natural ground, but darkened by 45 DN due to season/sun-angle
    crop_t2 = np.clip(crop_t1.astype(np.int16) - 45, 0, 255).astype(np.uint8)

    res = inspector.inspect_crop_pair(crop_t1, crop_t2, claimed_object="building", claimed_action="constructed")
    assert res["visible_change"] is False
    assert res["state_transition"] == "no_physical_change"
    assert "illumination_corrected" in " ".join(res["quality_flags"])
    assert "No verifiable physical structural change" in res["observations"]


def test_small_crop_indeterminate():
    """Verifies that crops smaller than min_crop_size (24 px) return indeterminate."""
    inspector = VisualEvidenceInspector(min_crop_size=24)
    crop_t1 = np.full((16, 16, 3), 120, dtype=np.uint8)
    crop_t2 = np.full((16, 16, 3), 200, dtype=np.uint8)

    res = inspector.inspect_crop_pair(crop_t1, crop_t2)
    assert res["visible_change"] is None
    assert res["state_transition"] == "indeterminate_small_crop"
    assert "crop_too_small" in res["quality_flags"]


def test_road_corridor_aspect_ratio():
    """Verifies that an elongated corridor with surface alteration confirms road claims."""
    inspector = VisualEvidenceInspector()
    # High aspect ratio crop: 100 x 40 (aspect = 2.5) with road stripe
    crop_t1 = np.full((100, 40, 3), 100, dtype=np.uint8)
    crop_t2 = crop_t1.copy()
    crop_t2[:, 12:28] = [200, 200, 200]  # New paved road stripe

    res = inspector.inspect_crop_pair(crop_t1, crop_t2, claimed_object="road", claimed_action="constructed")
    assert res["visible_change"] is True
    assert res["state_transition"] == "surface_alteration"
    assert "road" in res["observations"]


def test_building_demolition_detection():
    """Verifies that removal of building structural edges triggers removal_of_structure."""
    inspector = VisualEvidenceInspector()
    # T1: Has building with sharp boundary edges
    crop_t1 = np.full((64, 64, 3), 100, dtype=np.uint8)
    crop_t1[16:48, 16:48] = [220, 220, 220]
    crop_t1[15, 16:48] = [30, 30, 30]
    crop_t1[16:48, 48] = [30, 30, 30]

    # T2: Demolished and cleared ground
    crop_t2 = np.full((64, 64, 3), 100, dtype=np.uint8)

    res = inspector.inspect_crop_pair(crop_t1, crop_t2, claimed_object="building", claimed_action="demolished")
    assert res["visible_change"] is True
    assert res["state_transition"] == "removal_of_structure"
    assert "removed" in res["observations"]


def test_pipeline_end_to_end_contract(tmp_path):
    """Verifies the complete Stage C pipeline producing valid JSON contracts."""
    pipeline = StageCEvidencePipeline(output_dir=str(tmp_path / "crops"))
    
    t1 = np.full((256, 256, 3), 120, dtype=np.uint8)
    t2 = t1.copy()
    # Draw building in bottom-right
    t2[150:200, 150:200] = [230, 230, 230]
    
    claim_dict = {
        "object": "building",
        "action": "constructed",
        "location": "bottom-right",
        "quantity": 1,
        "quantity_unit": "count"
    }
    
    record = pipeline.process_claim(
        pair_id="LEVIR-MCI/train/train_000010",
        claim_id="claim_001",
        source_span="a new building was built in the bottom right",
        claim_data=claim_dict,
        candidate_region=[145, 145, 205, 205],
        image_t1=t1,
        image_t2=t2,
        save_crops=True
    )
    
    assert isinstance(record, ClaimDecisionRecord)
    assert record.pair_id == "LEVIR-MCI/train/train_000010"
    assert record.evidence.visible_change is True
    assert record.evidence.state_transition == "appearance_of_structure"
    assert len(record.evidence.crop_paths) == 2
    
    # Verify contract serializes cleanly to JSON
    json_str = record.model_dump_json(indent=2)
    data = json.loads(json_str)
    assert "evidence" in data
    assert data["evidence"]["witness_run_id"] == "W01_structural_witness_v1"
    assert data["evidence"]["state_transition"] == "appearance_of_structure"
    assert data["coordinate_frame"] == "original_T1_pixels"
