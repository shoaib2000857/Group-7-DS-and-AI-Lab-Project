"""
Batch Evaluation of Stage C Visual Evidence Inspector across a diverse benchmark cohort.
Simulates real-world satellite conditions: genuine construction, caption hallucinations,
seasonal shadow hazards, and sub-pixel registration jitter.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from src.evidence.pipeline import StageCEvidencePipeline


def generate_synthetic_cohort(output_dir: Path, n_samples: int = 50):
    """Generates a controlled, diverse 50-pair benchmark cohort."""
    cohort = []
    np.random.seed(42)

    # Scenarios distribution:
    # 0-19: True construction (building appeared in T2)
    # 20-31: Hallucinated claim on unchanged ground (T1 == T2)
    # 32-39: Shadow hazard (illumination shift without structural edge)
    # 40-45: Sub-pixel camera jitter (1.5 px shift)
    # 46-49: True demolition (building in T1 removed in T2)

    for i in range(n_samples):
        pair_id = f"LEVIR-MCI/eval/eval_{i:04d}"
        claim_id = f"claim_{i:04d}"
        
        # Base terrain (soil / grass)
        base = np.full((256, 256, 3), 110, dtype=np.uint8)
        noise = np.random.randint(-6, 6, (256, 256, 3), dtype=np.int16).astype(np.uint8)
        t1 = np.clip(base + noise, 0, 255).astype(np.uint8)
        t2 = t1.copy()
        
        # Random bbox in scene
        rx = np.random.randint(30, 170)
        ry = np.random.randint(30, 170)
        rw = np.random.randint(40, 60)
        rh = np.random.randint(40, 60)
        candidate_bbox = [rx, ry, rx + rw, ry + rh]

        if i < 20:  # 1. Genuine Construction
            scenario = "genuine_construction"
            # Add building to T2
            t2[ry:ry+rh, rx:rx+rw] = [220, 225, 230]
            t2[ry-1, rx:rx+rw] = [30, 30, 30]  # Shadow
            t2[ry:ry+rh, rx+rw] = [30, 30, 30]
            claim = {"object": "building", "action": "constructed", "quantity": 1}
            source_span = "a new building was built here"
            expected_change = True

        elif i < 32:  # 2. Caption Hallucination (Ground is unchanged!)
            scenario = "caption_hallucination"
            # T1 and T2 remain identical bare ground!
            claim = {"object": "building", "action": "constructed", "quantity": 1}
            source_span = "two new houses appeared in the field"
            expected_change = False

        elif i < 40:  # 3. Shadow Hazard (Sun angle / cloud shadow with natural penumbra)
            scenario = "shadow_hazard"
            # Natural shadows have smooth penumbra gradients without internal building edges
            # Extend shadow across the wider crop context
            pad = 20
            y_s = max(0, ry - pad)
            y_e = min(256, ry + rh + pad)
            x_s = max(0, rx - pad)
            x_e = min(256, rx + rw + pad)
            t2[y_s:y_e, x_s:x_e] = np.clip(t1[y_s:y_e, x_s:x_e].astype(int) - 50, 0, 255).astype(np.uint8)
            claim = {"object": "building", "action": "constructed", "quantity": 1}
            source_span = "a new dark building was erected"
            expected_change = False

        elif i < 46:  # 4. Sub-pixel Registration Jitter (1.5 px shift)
            scenario = "subpixel_jitter"
            # Draw an existing feature in both dates, but shift T2 by 1-2 pixels
            t1[ry:ry+rh, rx:rx+rw] = [180, 180, 180]
            t2[ry+1:ry+rh+1, rx+1:rx+rw+1] = [180, 180, 180]
            claim = {"object": "building", "action": "constructed", "quantity": 1}
            source_span = "a new structure appeared"
            expected_change = False

        else:  # 5. Genuine Demolition
            scenario = "genuine_demolition"
            # Building in T1, cleared ground in T2
            t1[ry:ry+rh, rx:rx+rw] = [220, 225, 230]
            t1[ry-1, rx:rx+rw] = [30, 30, 30]
            claim = {"object": "building", "action": "demolished", "quantity": 1}
            source_span = "a building was removed"
            expected_change = True

        cohort.append({
            "pair_id": pair_id,
            "claim_id": claim_id,
            "scenario": scenario,
            "t1": t1,
            "t2": t2,
            "candidate_bbox": candidate_bbox,
            "claim": claim,
            "source_span": source_span,
            "expected_change": expected_change
        })

    return cohort


def run_batch_evaluation():
    output_dir = Path("outputs/evidence_batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cohort = generate_synthetic_cohort(output_dir, n_samples=50)
    pipeline = StageCEvidencePipeline(output_dir=str(output_dir / "crops"))

    print("=" * 75)
    print("STAGE C: BATCH EVALUATION BENCHMARK (50 DIVERSE SATELLITE CLAIMS)")
    print("=" * 75)
    
    start_time = time.time()
    results = []
    
    scenario_counts = {}
    scenario_correct = {}
    total_time_ms = 0.0

    for sample in cohort:
        t0 = time.time()
        record = pipeline.process_claim(
            pair_id=sample["pair_id"],
            claim_id=sample["claim_id"],
            source_span=sample["source_span"],
            claim_data=sample["claim"],
            candidate_region=sample["candidate_bbox"],
            image_t1=sample["t1"],
            image_t2=sample["t2"],
            save_crops=False  # speed up benchmark
        )
        t_el = (time.time() - t0) * 1000.0
        total_time_ms += t_el

        sc = sample["scenario"]
        scenario_counts[sc] = scenario_counts.get(sc, 0) + 1
        
        # Check correctness against expected physical change ground truth
        observed_change = record.evidence.visible_change
        expected = sample["expected_change"]
        
        # In shadow or jitter hazard, we expect visible_change=False or hazard flag
        is_correct = (observed_change == expected)
        if sc == "shadow_hazard" and "shadow_hazard" in record.evidence.quality_flags:
            is_correct = True
        if sc == "subpixel_jitter" and any("subpixel_jitter" in q for q in record.evidence.quality_flags):
            is_correct = True
            
        scenario_correct[sc] = scenario_correct.get(sc, 0) + (1 if is_correct else 0)

        results.append({
            "pair_id": sample["pair_id"],
            "scenario": sc,
            "visible_change": observed_change,
            "expected": expected,
            "is_correct": is_correct,
            "structural_edge_change": record.evidence.structural_edge_change,
            "quality_flags": record.evidence.quality_flags,
            "latency_ms": round(t_el, 2)
        })

    total_duration = time.time() - start_time
    avg_latency = total_time_ms / len(cohort)

    # Print Summary Table
    print(f"\nCompleted {len(cohort)} claim inspections in {total_duration:.2f} s (Avg: {avg_latency:.2f} ms/claim)\n")
    print(f"{'Scenario':<26} | {'Total':<6} | {'Correct':<8} | {'Accuracy / Precision':<20}")
    print("-" * 68)
    
    overall_correct = 0
    overall_total = len(cohort)

    for sc, count in scenario_counts.items():
        corr = scenario_correct.get(sc, 0)
        overall_correct += corr
        acc = (corr / count) * 100.0
        print(f"{sc:<26} | {count:<6} | {corr:<8} | {acc:.1f}%")

    overall_acc = (overall_correct / overall_total) * 100.0
    print("-" * 68)
    print(f"{'OVERALL BENCHMARK':<26} | {overall_total:<6} | {overall_correct:<8} | {overall_acc:.1f}%")
    print("=" * 75)

    # Save detailed JSON summary
    summary_path = output_dir / "batch_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": overall_total,
            "overall_accuracy": overall_acc,
            "avg_latency_ms": avg_latency,
            "scenario_breakdown": {
                sc: {
                    "count": scenario_counts[sc],
                    "correct": scenario_correct.get(sc, 0),
                    "accuracy": (scenario_correct.get(sc, 0) / scenario_counts[sc]) * 100.0
                }
                for sc in scenario_counts
            },
            "detailed_records": results
        }, f, indent=2)

    print(f"\n[+] Full batch metrics saved to: {summary_path}")


if __name__ == "__main__":
    run_batch_evaluation()
