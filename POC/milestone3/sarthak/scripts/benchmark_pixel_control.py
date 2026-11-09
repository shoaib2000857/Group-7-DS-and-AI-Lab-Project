"""Run a bounded, reproducible LEVIR-MCI pixel-difference control benchmark.

The control is binary and is not a substitute for a trained BIT detector.
Images and labels are read directly from the downloaded source ZIP; no files are
extracted or modified. A connected-component minimum is selected on validation
only, then frozen for the held-out test sample.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE_DEFAULT = ROOT / "data/raw/LEVIR-MCI-dataset.zip"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "runs"
PREFIX = "LEVIR-MCI-dataset/images"
MIN_AREAS = (0, 16, 32, 50, 64)


def png(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(zf.read(member))) as image:
        return np.asarray(image.convert("RGB"))


def ids_for(zf: zipfile.ZipFile, split: str) -> list[str]:
    folder = f"{PREFIX}/{split}/A/"
    return sorted(Path(name).stem for name in zf.namelist()
                  if name.startswith(folder) and name.lower().endswith(".png"))


def sample_ids(zf: zipfile.ZipFile, split: str, n: int, seed: int) -> list[str]:
    ids = ids_for(zf, split)
    if n > len(ids):
        raise ValueError(f"requested {n} {split} pairs, only {len(ids)} available")
    return sorted(random.Random(seed).sample(ids, n))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_data(zf: zipfile.ZipFile, split: str, pair_id: str):
    base = f"{PREFIX}/{split}"
    a, b = png(zf, f"{base}/A/{pair_id}.png"), png(zf, f"{base}/B/{pair_id}.png")
    label = png(zf, f"{base}/label/{pair_id}.png")
    # Source masks encode road=128, building=255; both classes are binary change here.
    y = np.any(label != 0, axis=2)
    # Mean absolute RGB change is a stable 8-bit input to OpenCV Otsu thresholding.
    diff = np.rint(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)).astype(np.uint8)
    threshold, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return y, binary > 0, float(threshold)


def remove_small_components(pred: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or not pred.any():
        return pred
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.count_nonzero(y & pred)),
        "fp": int(np.count_nonzero(~y & pred)),
        "fn": int(np.count_nonzero(y & ~pred)),
        "tn": int(np.count_nonzero(~y & ~pred)),
    }


def score(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0}


def evaluate(zf, split: str, selected: list[str], min_area: int):
    total = dict(tp=0, fp=0, fn=0, tn=0)
    rows = []
    thresholds, latencies = [], []
    for pair_id in selected:
        start = time.perf_counter()
        y, pred, threshold = pair_data(zf, split, pair_id)
        pred = remove_small_components(pred, min_area)
        latency = (time.perf_counter() - start) * 1000
        c = counts(y, pred)
        for key in total:
            total[key] += c[key]
        thresholds.append(threshold)
        latencies.append(latency)
        rows.append({"split": split, "pair_id": pair_id, **c,
                     "truth_changed": bool(y.any()), "predicted_changed": bool(pred.any()),
                     "otsu_threshold": threshold, "latency_ms": latency})
    pair_precision = sum(r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0 for r in rows) / len(rows)
    pair_recall = sum(r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else 0 for r in rows) / len(rows)
    pair_f1s = []
    for r in rows:
        pr = r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0
        re = r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else 0
        pair_f1s.append(2 * pr * re / (pr + re) if pr + re else 0)
    clean = [r for r in rows if not r["truth_changed"]]
    result = {
        "pairs": len(rows), "changed_pairs": sum(r["truth_changed"] for r in rows),
        "unchanged_pairs": len(clean),
        "unchanged_pair_false_alarm_rate": (sum(r["predicted_changed"] for r in clean) / len(clean) if clean else None),
        "pixel_counts": total, "micro": score(total),
        "macro_pair_precision": pair_precision, "macro_pair_recall": pair_recall,
        "macro_pair_f1": float(np.mean(pair_f1s)),
        "mean_otsu_threshold": float(np.mean(thresholds)),
        "median_pair_latency_ms": float(np.median(latencies)),
        "mean_pair_latency_ms": float(np.mean(latencies)),
        "records": rows,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ARCHIVE_DEFAULT)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--validation-pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or OUT_DEFAULT / stamp
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.archive) as zf:
        val_ids = sample_ids(zf, "val", args.validation_pairs, args.seed)
        test_ids = sample_ids(zf, "test", args.pairs, args.seed + 1)
        val_candidates = {str(m): evaluate(zf, "val", val_ids, m) for m in MIN_AREAS}
        selected_area = max(MIN_AREAS, key=lambda m: (val_candidates[str(m)]["macro_pair_f1"], -m))
        test_result = evaluate(zf, "test", test_ids, selected_area)
        metadata = {
            "benchmark": "LEVIR-MCI binary abs-RGB-difference + per-pair Otsu control",
            "run_started_utc": stamp, "archive": str(args.archive.resolve()),
            "archive_size_bytes": args.archive.stat().st_size,
            "archive_sha256": sha256_file(args.archive),
            "seed": args.seed, "validation_pairs": args.validation_pairs, "test_pairs": args.pairs,
            "unique_pair_count_test_split": len(ids_for(zf, "test")),
            "test_fraction": args.pairs / len(ids_for(zf, "test")),
            "label_projection": "any nonzero source label pixel (road=128 or building=255)",
            "algorithm": "mean absolute RGB difference -> per-pair Otsu -> 8-connected component size filter",
            "minimum_component_candidates_px": list(MIN_AREAS),
            "selected_minimum_component_px": selected_area,
            "tuning_rule": "maximum validation macro pair F1; ties prefer smaller component threshold",
            "validation_candidates": {m: {k: v for k, v in result.items() if k != "records"}
                                      for m, result in val_candidates.items()},
            "test": {k: v for k, v in test_result.items() if k != "records"},
            "runtime": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
                        "pillow": Image.__version__ if hasattr(Image, "__version__") else "unknown",
                        "device": "CPU"},
            "limitations": [
                "This is a binary hand-crafted control, not BIT or a learned detector.",
                "Per-pair Otsu is adaptive and is not a fixed global threshold.",
                "Labels mark road/building change but the control cannot identify the class.",
                "Only a seeded 100-pair test subset is measured; scores are not a full official test result.",
            ],
        }
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "pair_id", "tp", "fp", "fn", "tn",
            "truth_changed", "predicted_changed", "otsu_threshold", "latency_ms"])
        writer.writeheader()
        writer.writerows(val_candidates[str(selected_area)]["records"])
        writer.writerows(test_result["records"])
    print(json.dumps({"output": str(out.resolve()), "selected_minimum_component_px": selected_area,
                      "validation_macro_pair_f1": val_candidates[str(selected_area)]["macro_pair_f1"],
                      "test": metadata["test"]}, indent=2))


if __name__ == "__main__":
    main()
