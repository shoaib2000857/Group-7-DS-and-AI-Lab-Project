"""Prepare the exact prior 100-pair LEVIR validation screen from the local ZIP.

Only selected validation images/masks are copied. The full published caption
JSON is copied separately for the vocabulary reconstruction used by historical
B0. That reconstruction reads caption text across official splits, including
screen references; it is preserved for parity and disclosed as lexical exposure.
Both destinations are under ignored POC/execution/data/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "data/raw/LEVIR-MCI-dataset.zip"
PREFIX = "LEVIR-MCI-dataset/images"
from benchmark_pixel_control import sha256_file


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=ROOT / "POC/execution/data/sarthak_rscama_screen")
    args = ap.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / "POC/execution" / args.output
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as zf:
        captions_bytes = zf.read("LEVIR-MCI-dataset/LevirCCcaptions.json")
        captions = json.loads(captions_bytes)["images"]
        pool = sorted((r for r in captions if r["filepath"] == "val"), key=lambda r: r["filename"])
        selected = random.Random("42-val").sample(pool, 100)
        rows, refs, stats = [], {}, {}
        for order, rec in enumerate(selected):
            filename = rec["filename"]
            pair_id = f"val/{filename[:-4]}"
            row = {
                "dataset": "LEVIR-MCI", "dataset_version": "hf:lcybuaa/LEVIR-MCI@dff09c7",
                "pair_id": pair_id, "parent_scene_id": "", "official_split": "val",
                "poc_role": "screening", "selection_order": order, "selection_seed": 42,
                "selection_rule": "uniform random.Random('42-val').sample(sorted validation pool, 100)",
                "t1_path": "", "t2_path": "", "label_path": "",
                "t1_sha256": "", "t2_sha256": "", "label_sha256": "",
                "label_ontology": "0=background,128=road,255=building", "gsd_m": 0.5,
                "crs": "", "acquisition_t1": "", "acquisition_t2": "",
                "first_use_date": "2026-10-27", "exclusion_reason": "",
            }
            for sub, key in (("A", "t1"), ("B", "t2"), ("label", "label")):
                member = f"{PREFIX}/val/{sub}/{filename}"
                data = zf.read(member)
                relative = Path("pairs") / "val" / sub / filename
                target = out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                row[f"{key}_path"] = relative.as_posix()
                row[f"{key}_sha256"] = digest(data)
                if sub == "label":
                    mask = Image.open(io.BytesIO(data))
                    arr = np.array(mask)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    stats[pair_id] = {
                        "label_values": sorted(map(int, np.unique(arr))),
                        "building_px": int((arr == 255).sum()), "road_px": int((arr == 128).sum()),
                        "changed_frac": float(((arr == 255) | (arr == 128)).mean()),
                        "shape": list(arr.shape), "changeflag": int(rec["changeflag"]),
                    }
            rows.append(row)
            refs[pair_id] = {"changeflag": rec["changeflag"],
                             "captions": [s["raw"].strip() for s in rec["sentences"]]}
        caption_dir = ROOT / "POC/execution/data/levir_mci_meta"
        caption_dir.mkdir(parents=True, exist_ok=True)
        (caption_dir / "LevirCCcaptions.json").write_bytes(captions_bytes)
    fields = list(rows[0])
    import csv
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (out / "references").mkdir(exist_ok=True)
    (out / "references/references.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")
    (out / "mask_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    manifest_hash = digest((out / "manifest.csv").read_bytes())
    meta = {
        "dataset_version": "hf:lcybuaa/LEVIR-MCI@dff09c7", "in_scope_classes": {"building": 255, "road": 128},
        "archive_sha256": sha256_file(ARCHIVE), "manifest_sha256": manifest_hash,
        "pairs": len(rows), "official_split": "val", "selection": "seed 42, exact historical B0 screen",
        "source_archive_test_pair_count": 1929,
        "changeflag_mask_mismatches": [p for p, s in stats.items()
              if bool(s["building_px"] or s["road_px"]) != bool(s["changeflag"])],
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "pairs": len(rows), "manifest_sha256": manifest_hash,
                      "archive_sha256": meta["archive_sha256"], "mask_caption_conflicts": len(meta["changeflag_mask_mismatches"]),
                      "selection": meta["selection"]}, indent=2))


if __name__ == "__main__":
    main()
