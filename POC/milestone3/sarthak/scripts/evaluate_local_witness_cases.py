"""Re-run the two-pair Stage C demo using the user's local LEVIR-MCI ZIP.

Claims and boxes are hand-authored fixtures copied from the earlier demo; no
caption model or detector is invoked. The dataset mask is consulted only after
inference to compute an explicitly limited crop-presence check.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from POC.evidence.pipeline import StageCEvidencePipeline
from benchmark_pixel_control import PREFIX


def decode(zf, member):
    with Image.open(io.BytesIO(zf.read(member))) as im:
        return np.asarray(im.convert("RGB"))


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "POC/milestone3/sarthak/runs" / f"witness-real-demo-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    cases = [
        {"id": "changed_train", "split": "train", "pair_id": "train_000010",
         "claim": {"object": "building", "action": "constructed", "location": "bottom", "quantity": 3},
         "span": "several houses are built at the bottom of the scene", "box": [60, 175, 230, 250]},
        {"id": "unchanged_val", "split": "val", "pair_id": "val_000005",
         "claim": {"object": "building", "action": "constructed", "location": "center", "quantity": 2},
         "span": "two new houses were built in the center field", "box": [80, 80, 180, 180]},
        {"id": "road_train", "split": "train", "pair_id": "train_000010",
         "claim": {"object": "road", "action": "constructed", "location": "left", "quantity": 1},
         "span": "a new road appears on the left", "box": [10, 40, 70, 220]},
    ]
    pipeline = StageCEvidencePipeline(output_dir=str(out / "crops"))
    records = []
    with zipfile.ZipFile(ROOT / "data/raw/LEVIR-MCI-dataset.zip") as zf:
        for case in cases:
            base = f"{PREFIX}/{case['split']}"
            a = decode(zf, f"{base}/A/{case['pair_id']}.png")
            b = decode(zf, f"{base}/B/{case['pair_id']}.png")
            mask = decode(zf, f"{base}/label/{case['pair_id']}.png")[..., 0]
            x1, y1, x2, y2 = case["box"]
            local_mask_pixels = int(np.count_nonzero(mask[y1:y2, x1:x2]))
            rec = pipeline.process_claim(
                pair_id=f"LEVIR-MCI/{case['split']}/{case['pair_id']}",
                claim_id=case["id"], source_span=case["span"], claim_data=case["claim"],
                candidate_region=case["box"], image_t1=a, image_t2=b, save_crops=False)
            evidence = rec.evidence
            records.append({
                "case_id": case["id"], "pair_id": case["pair_id"], "claim": case["claim"],
                "source_span_hand_authored": case["span"], "bbox_hand_authored": case["box"],
                "source_mask_change_pixels_in_bbox": local_mask_pixels,
                "mask_region_has_change": local_mask_pixels > 0,
                "witness_visible_change": evidence.visible_change,
                "witness_matches_mask_region_presence": (evidence.visible_change == (local_mask_pixels > 0)),
                "structural_edge_change": evidence.structural_edge_change,
                "mean_luminance_diff": evidence.mean_luminance_diff,
                "quality_flags": evidence.quality_flags,
                "observations": evidence.observations,
            })
    summary = {
        "benchmark": "Stage C local real-image diagnostic",
        "run_started_utc": stamp, "distinct_source_pairs": 2, "claims": len(records),
        "matched_mask_region_presence": sum(r["witness_matches_mask_region_presence"] for r in records),
        "records": records,
        "limitations": [
            "The two distinct image pairs were chosen for a hand-authored demonstration, not sampled representatively.",
            "Claim wording and candidate boxes are manual fixtures; no captioner, BIT detector or referee ran.",
            "Mask-region presence is only a coarse pixel-label comparison; it does not verify class, temporal direction or quantity.",
            "The two claims on train_000010 are correlated observations, not independent pairs.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "matched": f"{summary['matched_mask_region_presence']}/{len(records)}",
                      "records": [{k: r[k] for k in ("case_id", "source_mask_change_pixels_in_bbox",
                          "witness_visible_change", "witness_matches_mask_region_presence", "quality_flags")}
                                  for r in records]}, indent=2))


if __name__ == "__main__":
    main()
