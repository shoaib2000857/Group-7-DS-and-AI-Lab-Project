"""Freeze the Track M subset and extract only the selected pairs.

Selection: uniform seeded sampling (seed 42) within each official split.
  train -> 50 development pairs (first 10 in selection order = smoke fixture)
  val   -> 100 screening pairs
  test  -> 100 sealed pairs: IDs recorded, images NOT downloaded or inspected
Parent-scene IDs are not recoverable from LEVIR-CC patch names, so grouping
is by pair only (documented limitation).

Outputs (all under data/, git-ignored):
  data/pairs/<split>/{A,B,label}/<file>.png   model inputs (+ label for eval)
  data/references/references.json             captions, eval only
  data/manifest.csv / manifest.json           frozen manifest with hashes
  data/mask_stats.json                         per-pair label statistics
"""
import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import LEVIR_MCI_URL, fetch_members, open_remote_zip  # noqa: E402

ROOT = "LEVIR-MCI-dataset"
COUNTS = {"train": 50, "val": 100, "test": 100}
ROLES = {"train": "development", "val": "screening", "test": "sealed_evaluation"}
CAP = 0.20


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def mask_stats(label_bytes):
    m = np.array(Image.open(io.BytesIO(label_bytes)))
    if m.ndim == 3:
        m = m[..., 0]
    values = sorted(int(v) for v in np.unique(m))
    building = int((m == 255).sum())
    road = int((m == 128).sum())
    total = int(m.size)
    return {
        "label_values": values,
        "building_px": building,
        "road_px": road,
        "changed_frac": round((building + road) / total, 5),
        "shape": list(m.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--captions", default="data/levir_mci_meta/LevirCCcaptions.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)

    caps = json.load(open(args.captions))["images"]
    by_split = {}
    for rec in caps:
        by_split.setdefault(rec["filepath"], []).append(rec)

    z, raw = open_remote_zip()
    names = set(z.namelist())

    # Join check: every caption record has A, B and label in the archive.
    join_problems = []
    for rec in caps:
        for sub in ("A", "B", "label"):
            p = f"{ROOT}/images/{rec['filepath']}/{sub}/{rec['filename']}"
            if p not in names:
                join_problems.append(p)
    dup = [k for k, v in Counter((r["filepath"], r["filename"]) for r in caps).items() if v > 1]

    n_total = len(caps)
    selected = {}
    for split, n in COUNTS.items():
        pool = sorted(by_split[split], key=lambda r: r["filename"])
        cap_n = int(CAP * len(pool))
        assert n <= cap_n, (split, n, cap_n)
        rng = random.Random(f"{args.seed}-{split}")
        selected[split] = rng.sample(pool, n)
    assert sum(COUNTS.values()) <= int(CAP * n_total)

    wanted = [
        f"{ROOT}/images/{split}/{sub}/{rec['filename']}"
        for split, recs in selected.items() if split != "test"
        for rec in recs for sub in ("A", "B", "label")
    ]
    fetched = fetch_members(z, raw, wanted)

    rows, refs, stats = [], {}, {}
    for split, recs in selected.items():
        for order, rec in enumerate(recs):
            pid = f"{split}/{rec['filename'][:-4]}"
            role = ROLES[split]
            if split == "train" and order < 10:
                role = "development_smoke"
            row = {
                "dataset": "LEVIR-MCI",
                "dataset_version": "hf:lcybuaa/LEVIR-MCI@dff09c7",
                "pair_id": pid,
                "parent_scene_id": "",
                "official_split": split,
                "poc_role": role,
                "selection_order": order,
                "selection_seed": args.seed,
                "selection_rule": f"uniform random.Random('{args.seed}-{split}').sample(sorted pool, {COUNTS[split]})",
                "t1_path": "", "t2_path": "", "label_path": "",
                "caption_ids": " ".join(str(s) for s in rec["sentids"]),
                "t1_sha256": "", "t2_sha256": "", "label_sha256": "",
                "label_ontology": "0=background,128=road,255=building",
                "gsd_m": 0.5, "crs": "", "acquisition_t1": "", "acquisition_t2": "",
                "first_use_date": "" if split == "test" else date.today().isoformat(),
                "exclusion_reason": "",
            }
            if split != "test":  # sealed split stays undownloaded
                for sub, key in (("A", "t1"), ("B", "t2"), ("label", "label")):
                    member = f"{ROOT}/images/{split}/{sub}/{rec['filename']}"
                    b = fetched[member]
                    dest = out / "pairs" / split / sub / rec["filename"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b)
                    row[f"{key}_path"] = str(dest.relative_to(out))
                    row[f"{key}_sha256"] = sha256(b)
                    if sub == "label":
                        stats[pid] = mask_stats(b)
                        stats[pid]["changeflag"] = rec["changeflag"]
                    else:
                        im = Image.open(io.BytesIO(b))
                        assert im.size == (256, 256), (member, im.size)
                refs[pid] = {
                    "changeflag": rec["changeflag"],
                    "captions": [s["raw"].strip() for s in rec["sentences"]],
                }
            rows.append(row)

    (out / "references").mkdir(parents=True, exist_ok=True)
    json.dump(refs, open(out / "references" / "references.json", "w"), indent=1)
    json.dump(stats, open(out / "mask_stats.json", "w"), indent=1)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest_hash = sha256((out / "manifest.csv").read_bytes())

    # Consistency: changeflag versus mask content.
    flag_mismatch = [
        p for p, s in stats.items()
        if (s["changed_frac"] > 0) != bool(s["changeflag"])
    ]
    summary = {
        "source_url": LEVIR_MCI_URL,
        "archive_bytes": raw.size,
        "bytes_fetched": raw.bytes_fetched,
        "pairs_in_source": n_total,
        "split_sizes": {k: len(v) for k, v in by_split.items()},
        "selected": {k: len(v) for k, v in selected.items()},
        "cap_fraction": CAP,
        "cap_limits": {k: int(CAP * len(v)) for k, v in by_split.items()},
        "global_cap": int(CAP * n_total),
        "selected_changed": {
            k: sum(r["changeflag"] for r in v) for k, v in selected.items() if k != "test"
        },
        "join_problems": join_problems[:20],
        "join_problem_count": len(join_problems),
        "duplicate_records": dup,
        "changeflag_mask_mismatch": flag_mismatch,
        "parent_scene_recoverable": False,
        "manifest_sha256": manifest_hash,
        "frozen_on": date.today().isoformat(),
    }
    json.dump(summary, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
