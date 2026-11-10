"""
Stage C Evidence Extraction Pipeline.
Orchestrates bounding box crop extraction, visual evidence inspection,
and emission of the standard claim_decision contract.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from src.evidence.schemas import (
    ClaimContent,
    EvidenceBlock,
    ClaimDecisionRecord,
    DecisionBlock
)
from src.evidence.crop_extractor import extract_bitemporal_crops
from src.evidence.visual_inspector import VisualEvidenceInspector


class StageCEvidencePipeline:
    """
    Stage C Pipeline: Visual Evidence Inspection & Witness Module.
    Bridges Stage A/B detections & claims with Stage D referee decisions.
    """

    def __init__(
        self,
        output_dir: str = "outputs/pipeline_runs/crops",
        witness_run_id: str = "W01_structural_witness_v1"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.witness_run_id = witness_run_id
        self.inspector = VisualEvidenceInspector()

    def process_claim(
        self,
        pair_id: str,
        claim_id: str,
        source_span: str,
        claim_data: Dict[str, Any],
        candidate_region: List[int],
        image_t1: np.ndarray,
        image_t2: np.ndarray,
        save_crops: bool = True
    ) -> ClaimDecisionRecord:
        """
        Executes Stage C visual evidence verification for a single claim.
        
        Args:
            pair_id: Identifier of the temporal pair (e.g. 'train_000010')
            claim_id: Unique claim id (e.g. 'claim_001')
            source_span: The raw caption phrase (e.g. 'a new building appeared')
            claim_data: Dict containing keys object, action, location, quantity
            candidate_region: [x0, y0, x1, y1] bounding box in T1 pixels
            image_t1: RGB array (H, W, 3) at T1
            image_t2: RGB array (H, W, 3) at T2
            save_crops: Whether to persist cropped PNG artifacts to disk
            
        Returns:
            Populated ClaimDecisionRecord conforming to benchmark standards.
        """
        # 1. Extract context-padded bitemporal crops
        crop_t1, crop_t2, transform_meta = extract_bitemporal_crops(
            image_t1=image_t1,
            image_t2=image_t2,
            bbox=candidate_region,
            padding_fraction=0.15
        )

        # 2. Inspect crops with visual witness
        obj_type = claim_data.get("object", "building")
        act_type = claim_data.get("action", "constructed")
        evidence_dict = self.inspector.inspect_crop_pair(
            crop_t1=crop_t1,
            crop_t2=crop_t2,
            claimed_object=obj_type,
            claimed_action=act_type
        )

        # 3. Save crop artifacts if requested
        crop_paths = []
        if save_crops:
            safe_pair = re.sub(r"[^A-Za-z0-9_.-]", "_", pair_id)
            claim_id = re.sub(r"[^A-Za-z0-9_.-]", "_", claim_id)
            c1_path = self.output_dir / f"{safe_pair}_{claim_id}_crop_T1.png"
            c2_path = self.output_dir / f"{safe_pair}_{claim_id}_crop_T2.png"
            
            Image.fromarray(crop_t1).save(c1_path)
            Image.fromarray(crop_t2).save(c2_path)
            
            crop_paths = [str(c1_path), str(c2_path)]

        # 4. Construct typed EvidenceBlock
        evidence_block = EvidenceBlock(
            mask_run_id=claim_data.get("mask_run_id", "detector_open_cd_bit"),
            mask_ontology="building_road_binary",
            witness_run_id=self.witness_run_id,
            visible_change=evidence_dict["visible_change"],
            confidence=evidence_dict.get("confidence"),
            state_transition=evidence_dict.get("state_transition"),
            observations=evidence_dict["observations"],
            visibility_flags=evidence_dict["visibility_flags"],
            quality_flags=evidence_dict["quality_flags"],
            crop_paths=crop_paths,
            structural_edge_change=evidence_dict["structural_edge_change"],
            mean_luminance_diff=evidence_dict["mean_luminance_diff"],
            crop_mean_diff=evidence_dict.get("crop_mean_diff"),
            transform_record=transform_meta
        )

        # 5. Populate and return ClaimDecisionRecord
        record = ClaimDecisionRecord(
            record_type="stage_c_evidence_record",
            pair_id=pair_id,
            claim_id=claim_id,
            source_span=source_span,
            claim=ClaimContent(**claim_data),
            candidate_region_ids=[candidate_region],
            coordinate_frame="original_T1_pixels",
            bbox_convention="x0_y0_x1_y1_half_open",
            evidence=evidence_block,
            decision=None  # Left unpopulated for Stage D referee
        )

        return record
