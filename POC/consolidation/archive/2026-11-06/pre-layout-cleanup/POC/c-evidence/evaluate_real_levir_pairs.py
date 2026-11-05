"""
Evaluation of Stage C Visual Evidence Inspector on REAL LEVIR-MCI Satellite Imagery.
Fetches actual Texas aerial photo pairs from the official Hugging Face repository,
extracts real crops, and verifies real construction vs. caption hallucinations.
"""

import io
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
from PIL import Image

from POC.execution.scripts.remote_zip import open_remote_zip, fetch_members, LEVIR_MCI_URL
from src.evidence.pipeline import StageCEvidencePipeline


def run_real_data_evaluation():
    out_dir = Path("outputs/real_data_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TERRAWATCH: STAGE C EVALUATION ON REAL LEVIR-MCI SATELLITE IMAGERY")
    print("=" * 80)
    
    # 1. Fetch real satellite images from remote archive
    target_files = [
        # Pair 1: Changed scene (train_000010)
        "LEVIR-MCI-dataset/images/train/A/train_000010.png",
        "LEVIR-MCI-dataset/images/train/B/train_000010.png",
        "LEVIR-MCI-dataset/images/train/label/train_000010.png",
        # Pair 2: Unchanged scene (val_000005)
        "LEVIR-MCI-dataset/images/val/A/val_000005.png",
        "LEVIR-MCI-dataset/images/val/B/val_000005.png",
        "LEVIR-MCI-dataset/images/val/label/val_000005.png",
    ]
    
    print(f"[*] Connecting to Hugging Face ({LEVIR_MCI_URL[:55]}...)...")
    zf, raw = open_remote_zip(LEVIR_MCI_URL)
    print(f"[*] Fetching real satellite image bytes for 2 pairs...")
    byte_dict = fetch_members(zf, raw, target_files)
    print(f"[+] Successfully downloaded {len(byte_dict)} files ({sum(len(v) for v in byte_dict.values()) / 1024:.1f} KB)\n")

    # Helper to decode PNG bytes to RGB numpy array
    def decode_png(key):
        return np.array(Image.open(io.BytesIO(byte_dict[key])).convert("RGB"))

    t1_changed = decode_png("LEVIR-MCI-dataset/images/train/A/train_000010.png")
    t2_changed = decode_png("LEVIR-MCI-dataset/images/train/B/train_000010.png")
    mask_changed = np.array(Image.open(io.BytesIO(byte_dict["LEVIR-MCI-dataset/images/train/label/train_000010.png"])))

    t1_unchanged = decode_png("LEVIR-MCI-dataset/images/val/A/val_000005.png")
    t2_unchanged = decode_png("LEVIR-MCI-dataset/images/val/B/val_000005.png")
    mask_unchanged = np.array(Image.open(io.BytesIO(byte_dict["LEVIR-MCI-dataset/images/val/label/val_000005.png"])))

    # Save real full images locally for inspection
    Image.fromarray(t1_changed).save(out_dir / "real_train_000010_T1.png")
    Image.fromarray(t2_changed).save(out_dir / "real_train_000010_T2.png")
    Image.fromarray(mask_changed).save(out_dir / "real_train_000010_mask.png")

    Image.fromarray(t1_unchanged).save(out_dir / "real_val_000005_T1.png")
    Image.fromarray(t2_unchanged).save(out_dir / "real_val_000005_T2.png")
    Image.fromarray(mask_unchanged).save(out_dir / "real_val_000005_mask.png")

    pipeline = StageCEvidencePipeline(output_dir=str(out_dir / "real_crops"))

    # Test cases comparing Sujal's Stage B Captioner vs Stage C Visual Evidence:
    test_cases = [
        {
            "case_name": "Case 1: Real Construction on Changed Scene (train_000010)",
            "pair_id": "LEVIR-MCI/train/train_000010",
            "t1": t1_changed,
            "t2": t2_changed,
            "claim_span": "several houses are built at the bottom of the scene",
            "claim": {"object": "building", "action": "constructed", "location": "bottom", "quantity": 3},
            "candidate_bbox": [60, 175, 230, 250],  # Bottom region where buildings appear in mask
            "ground_truth": "GENUINE_CHANGE",
            "stage_b_behavior": "RSCaMa predicts buildings correctly"
        },
        {
            "case_name": "Case 2: VLM Hallucination on Unchanged Scene (val_000005)",
            "pair_id": "LEVIR-MCI/val/val_000005",
            "t1": t1_unchanged,
            "t2": t2_unchanged,
            "claim_span": "two new houses were built in the center field",
            "claim": {"object": "building", "action": "constructed", "location": "center", "quantity": 2},
            "candidate_bbox": [80, 80, 180, 180],  # Center region where VLM claims houses exist
            "ground_truth": "UNCHANGED_GROUND_TRUTH",
            "stage_b_behavior": "VLM (Qwen/SmolVLM) hallucinated houses when mask is 0"
        },
        {
            "case_name": "Case 3: Surface / Road Appearance on Changed Scene (train_000010)",
            "pair_id": "LEVIR-MCI/train/train_000010",
            "t1": t1_changed,
            "t2": t2_changed,
            "claim_span": "a new road appears on the left",
            "claim": {"object": "road", "action": "constructed", "location": "left", "quantity": 1},
            "candidate_bbox": [10, 40, 70, 220],  # Left road corridor
            "ground_truth": "GENUINE_CHANGE",
            "stage_b_behavior": "RSCaMa describes lane on left"
        }
    ]

    print("-" * 80)
    print(f"{'Test Case':<32} | {'Sujal Stage B Output':<22} | {'Stage C Verdict':<18}")
    print("-" * 80)

    records = []
    for tc in test_cases:
        rec = pipeline.process_claim(
            pair_id=tc["pair_id"],
            claim_id=f"real_eval_{tc['pair_id'].split('/')[-1]}",
            source_span=tc["claim_span"],
            claim_data=tc["claim"],
            candidate_region=tc["candidate_bbox"],
            image_t1=tc["t1"],
            image_t2=tc["t2"],
            save_crops=True
        )
        records.append(rec)
        ev = rec.evidence
        
        status_str = f"CONFIRMED (edge={ev.structural_edge_change:+.3f})" if ev.visible_change else f"REJECTED (edge={ev.structural_edge_change:+.3f})"
        print(f"{tc['case_name'][:32]:<32} | {tc['claim_span'][:22]:<22} | {status_str:<18}")

    print("-" * 80)
    print("\n[+] Detailed Stage C Evidence Observations on Real Imagery:")
    for tc, rec in zip(test_cases, records):
        ev = rec.evidence
        print(f"\n--- {tc['case_name']} ---")
        print(f"  • Real Pair ID:       {rec.pair_id}")
        print(f"  • Candidate BBox:     {rec.candidate_region_ids[0]}")
        print(f"  • Visible Change:     {ev.visible_change} (Expected Ground Truth: {tc['ground_truth']})")
        print(f"  • Structural Edge:    {ev.structural_edge_change:.4f}")
        print(f"  • Mean Lum Diff:      {ev.mean_luminance_diff:.2f}")
        print(f"  • Quality Flags:      {ev.quality_flags}")
        print(f"  • Crop Paths:         {ev.crop_paths}")
        print(f"  • Witness Summary:    \"{ev.observations}\"")

    # Export combined JSON
    json_path = out_dir / "real_levir_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, indent=2)
        
    print(f"\n[+] Real satellite evaluation exported to: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_real_data_evaluation()
