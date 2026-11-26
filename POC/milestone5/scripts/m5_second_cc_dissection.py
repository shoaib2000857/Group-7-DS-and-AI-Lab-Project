"""Milestone 5 WS5: SECOND-CC Per-Class Semantic Domain-Gap Dissection (530 Val + 1,142 Test).

Proves quantitatively WHY zero-shot LEVIR-MCI (2-class: road/building) detectors show lower Any-Change F1
on SECOND-CC (6 semantic classes: building, non-vegetated ground, low vegetation, tree, water, playground):
computes exact recall of D03_bit_lr3e4 and A03_siam_r18_unet per SECOND-CC native semantic category.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(SCRIPT_DIR), str(SCRIPT_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detector_models import build_model
from train_detector import read_jsonl, sha256_file

# Official SECOND dataset RGB palette -> semantic class name
SECOND_CLASSES: List[Tuple[str, Tuple[int, int, int]]] = [
    ("building", (128, 0, 0)),
    ("non_vegetated_ground", (128, 128, 128)),
    ("low_vegetation", (0, 128, 0)),
    ("tree", (0, 255, 0)),
    ("water", (0, 0, 255)),
    ("playground", (255, 0, 0)),
]


@torch.inference_mode()
def eval_second_per_class(
    model: torch.nn.Module,
    rows: List[dict],
    ext_root: Path,
    device: str,
) -> dict:
    model.eval()
    has_native_bf16 = device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if has_native_bf16 else torch.float16

    any_tp = any_fp = any_fn = 0
    cls_total: Dict[str, int] = {name: 0 for name, _ in SECOND_CLASSES}
    cls_hit_any: Dict[str, int] = {name: 0 for name, _ in SECOND_CLASSES}
    cls_hit_bldg: Dict[str, int] = {name: 0 for name, _ in SECOND_CLASSES}
    cls_hit_road: Dict[str, int] = {name: 0 for name, _ in SECOND_CLASSES}

    for i in range(0, len(rows), 16):
        chunk = rows[i : i + 16]
        x1s, x2s, s1s, s2s = [], [], [], []
        for r in chunk:
            t1 = np.array(Image.open(ext_root / r["t1"]).convert("RGB"))
            t2 = np.array(Image.open(ext_root / r["t2"]).convert("RGB"))
            s1 = np.array(Image.open(ext_root / r["labels"][0]).convert("RGB"))
            s2 = np.array(Image.open(ext_root / r["labels"][1]).convert("RGB"))
            x1s.append(torch.from_numpy(t1.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0)
            x2s.append(torch.from_numpy(t2.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0)
            s1s.append(s1)
            s2s.append(s2)

        b1 = torch.stack(x1s).to(device)
        b2 = torch.stack(x2s).to(device)
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=amp_dtype):
            preds = model(b1, b2).argmax(dim=1).cpu().numpy().astype(np.uint8)

        for pm, s1, s2 in zip(preds, s1s, s2s):
            # Resize pred mask to semantic label shape if different (SECOND is 512x512 or 256x256)
            if pm.shape != s1.shape[:2]:
                pm = np.array(Image.fromarray(pm).resize((s1.shape[1], s1.shape[0]), Image.NEAREST))
            p_any = pm > 0
            p_road = pm == 1
            p_bldg = pm == 2

            gt_any = (~((s1[:, :, 0] == 255) & (s1[:, :, 1] == 255) & (s1[:, :, 2] == 255))) | (
                ~((s2[:, :, 0] == 255) & (s2[:, :, 1] == 255) & (s2[:, :, 2] == 255))
            )
            any_tp += int((gt_any & p_any).sum())
            any_fp += int((~gt_any & p_any).sum())
            any_fn += int((gt_any & ~p_any).sum())

            for cname, (cr, cg, cb) in SECOND_CLASSES:
                m1 = (s1[:, :, 0] == cr) & (s1[:, :, 1] == cg) & (s1[:, :, 2] == cb)
                m2 = (s2[:, :, 0] == cr) & (s2[:, :, 1] == cg) & (s2[:, :, 2] == cb)
                cmask = m1 | m2
                cnt = int(cmask.sum())
                if cnt > 0:
                    cls_total[cname] += cnt
                    cls_hit_any[cname] += int((cmask & p_any).sum())
                    cls_hit_bldg[cname] += int((cmask & p_bldg).sum())
                    cls_hit_road[cname] += int((cmask & p_road).sum())

    any_p = any_tp / max(any_tp + any_fp, 1)
    any_r = any_tp / max(any_tp + any_fn, 1)
    any_f1 = 2 * any_tp / max(2 * any_tp + any_fp + any_fn, 1)
    tot_cls_px = sum(cls_total.values())

    per_class = {}
    for cname, _ in SECOND_CLASSES:
        tot = cls_total[cname]
        per_class[cname] = {
            "gt_pixels": tot,
            "gt_share_of_changes_pct": round(100.0 * tot / max(tot_cls_px, 1), 2),
            "recall_by_any_pred_pct": round(100.0 * cls_hit_any[cname] / max(tot, 1), 2),
            "recall_by_pred_bldg_pct": round(100.0 * cls_hit_bldg[cname] / max(tot, 1), 2),
            "recall_by_pred_road_pct": round(100.0 * cls_hit_road[cname] / max(tot, 1), 2),
        }

    return {
        "num_pairs": len(rows),
        "binary_any_change": {
            "precision": round(any_p, 6),
            "recall": round(any_r, 6),
            "f1": round(any_f1, 6),
        },
        "per_semantic_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--extracted-root", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--bit-source-dir", required=True)
    ap.add_argument("--changeformer-source-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    man = Path(args.manifests_dir)
    ext = Path(args.extracted_root)
    runs = Path(args.runs_dir)

    splits = {
        "val": read_jsonl(man / "second_cc_val.jsonl"),
        "test": read_jsonl(man / "second_cc_test.jsonl"),
    }

    out = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    for run_id, arch in [("D03_bit_lr3e4", "bit_resnet18"), ("A03_siam_r18_unet", "siam_resnet18_unet")]:
        ckpt = runs / run_id / "best.pt"
        model, _, _, _ = build_model(arch, args.device, args.bit_source_dir, args.changeformer_source_dir)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
        out["models"][run_id] = {"arch": arch, "sha256": sha256_file(ckpt), "splits": {}}
        for sp_name, rows in splits.items():
            res = eval_second_per_class(model, rows, ext, args.device)
            out["models"][run_id]["splits"][sp_name] = res
            print(f"[{run_id}] SECOND-CC {sp_name} ({len(rows)} pairs) Any-F1={res['binary_any_change']['f1']*100:.2f}% | bldg_rec={res['per_semantic_class']['building']['recall_by_any_pred_pct']:.1f}% | veg_rec={res['per_semantic_class']['low_vegetation']['recall_by_any_pred_pct']:.1f}%", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"SECOND_CC_DISSECTION_COMPLETE in {time.time()-t0:.1f}s -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
