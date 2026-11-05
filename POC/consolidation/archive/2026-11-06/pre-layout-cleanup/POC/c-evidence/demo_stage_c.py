"""
Executable Demonstration of Stage C Visual Evidence Verification.
Processes a representative satellite change pair, extracts candidate region crops,
performs structural verification, and outputs the populated claim_decision contract.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from src.evidence.pipeline import StageCEvidencePipeline


def run_demo():
    output_dir = Path("outputs/evidence_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TerraWatch -- Stage C: Visual Evidence Inspection Demonstration")
    print("=" * 70)

    # 1. Generate realistic bitemporal satellite synthetic scenario (256x256)
    # Scenario: Bare land in T1 -> New building constructed in lower-right in T2
    np.random.seed(101)
    base_ground = np.full((256, 256, 3), 115, dtype=np.uint8)
    noise = np.random.randint(-8, 8, (256, 256, 3), dtype=np.int16).astype(np.uint8)
    t1_image = np.clip(base_ground + noise, 0, 255).astype(np.uint8)
    
    # T2: Same background + building constructed at [140, 140, 210, 210]
    t2_image = t1_image.copy()
    t2_image[140:210, 140:210] = [215, 220, 225]  # Bright concrete roof
    t2_image[139, 140:210] = [30, 30, 30]         # Cast shadow north edge
    t2_image[140:210, 210] = [30, 30, 30]         # Cast shadow east edge

    # Save demo full scene images
    Image.fromarray(t1_image).save(output_dir / "demo_pair_T1.png")
    Image.fromarray(t2_image).save(output_dir / "demo_pair_T2.png")
    print(f"[*] Saved full-scene demo images in {output_dir}/")

    # 2. Simulate upstream inputs (Stage A detector box + Stage B caption claim)
    pair_id = "LEVIR-MCI/val/val_000042"
    claim_id = "claim_001"
    source_span = "a new building was constructed in the bottom right corner"
    candidate_box = [135, 135, 215, 215]  # BBox in T1 pixel frame
    claim_payload = {
        "object": "building",
        "action": "constructed",
        "location": "bottom-right",
        "quantity": 1,
        "quantity_unit": "count"
    }

    print(f"[*] Incoming Pair:   {pair_id}")
    print(f"[*] Stage B Caption: \"{source_span}\"")
    print(f"[*] Candidate BBox:  {candidate_box}")

    # 3. Initialize and execute Stage C Pipeline
    pipeline = StageCEvidencePipeline(output_dir=str(output_dir / "crops"))
    record = pipeline.process_claim(
        pair_id=pair_id,
        claim_id=claim_id,
        source_span=source_span,
        claim_data=claim_payload,
        candidate_region=candidate_box,
        image_t1=t1_image,
        image_t2=t2_image,
        save_crops=True
    )

    # 4. Display findings
    ev = record.evidence
    print("\nStage C Verification Output:")
    print(f"  - Witness Run ID:            {ev.witness_run_id}")
    print(f"  - Physical Change Confirmed: {ev.visible_change}")
    print(f"  - Structural Edge Delta:     {ev.structural_edge_change:.4f}")
    print(f"  - Luminance Shift:           {ev.mean_luminance_diff:.2f}")
    print(f"  - Quality Flags:             {ev.quality_flags}")
    print(f"  - Observations:              {ev.observations}")
    print(f"  - Generated Crops:           {ev.crop_paths}")

    # 5. Export JSON contract
    json_path = output_dir / "stage_c_claim_decision_record.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(record.model_dump_json(indent=2))
        
    print(f"\n[+] Contract exported to: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
