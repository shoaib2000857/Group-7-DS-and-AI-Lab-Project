"""Out-of-domain screening sets in the same layout as data/ (LEVIR-MCI).

  python scripts/build_ood.py whu_cdc|second_cc

  data_ood/<name>/pairs/val/{A,B,label}/<file>.png   label re-encoded as LEVIR (255 = building change)
  data_ood/<name>/references/references.json, mask_stats.json, manifest.csv, manifest.json

Run any B script on it with B_DATA=data_ood/<name> (runs go to runs/ood/<name>/).
100 pairs from the official val split, seeded; test splits are not touched.
Neither dataset marks roads, so only building claims are mask-checked (in_scope_classes).
"""
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
from remote_zip import fetch_members, open_remote_zip  # noqa: E402

EXEC_ROOT = Path(__file__).resolve().parents[1]
N, SEED, CAP = 100, 42, 0.20
SECOND_BUILDING = (128, 0, 0)  # SECOND palette; white (255,255,255) = unchanged

SOURCES = {
    "whu_cdc": {
        "dataset": "WHU-CDC",
        "url": "https://huggingface.co/datasets/hygge10111/RS-CDC/resolve/main/whu_CDC_dataset.zip",
        "dataset_version": "hf:hygge10111/RS-CDC whu_CDC_dataset.zip (third-party upload, no licence stated)",
        "captions": "whu_CDC_dataset/whuCCcaptions.json",
        "members": lambda f: {"A": f"whu_CDC_dataset/images/val/A/{f}", "B": f"whu_CDC_dataset/images/val/B/{f}",
                              "label": f"whu_CDC_dataset/images/val/label/{f}"},
        "keep": lambda f: True,
        "gsd_m": None,  # Christchurch aerial; tile resampling not documented in this release
        "licence": "WHU-CD: academic use; captions from WHU-CDC (Shen et al.)",
    },
    "second_cc": {
        "dataset": "SECOND-CC",
        "url": "https://zenodo.org/records/16937571/files/SECOND-CC-AUG.zip?download=1",
        "dataset_version": "zenodo:10.5281/zenodo.16937571 SECOND-CC-AUG.zip",
        "captions": "SECOND-CC-AUG/SECOND-CC-AUG.json",
        "members": lambda f: {"A": f"SECOND-CC-AUG/val/rgb/A/{f}", "B": f"SECOND-CC-AUG/val/rgb/B/{f}",
                              "semA": f"SECOND-CC-AUG/val/sem/A/{f}", "semB": f"SECOND-CC-AUG/val/sem/B/{f}"},
        "keep": lambda f: "_random_augment" not in f,  # originals only, not the augmented copies
        "gsd_m": None,  # SECOND mixes sources (0.5-3 m)
        "licence": "CC-BY-4.0",
        "workers": 2,  # Zenodo answers parallel range bursts with 429
    },
}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def label_png(src, files):
    """Building-change mask in LEVIR encoding (uint8, 255 = building change)."""
    if src == "whu_cdc":
        m = np.array(Image.open(io.BytesIO(files["label"])).convert("L")) > 0
    else:
        a, b = (np.array(Image.open(io.BytesIO(files[k])).convert("RGB")) for k in ("semA", "semB"))
        m = np.all(a == SECOND_BUILDING, -1) | np.all(b == SECOND_BUILDING, -1)
    buf = io.BytesIO()
    Image.fromarray((m * 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue(), int(m.sum()), m.size


def main(name):
    S = SOURCES[name]
    out = EXEC_ROOT / "data_ood" / name
    z, raw = open_remote_zip(S["url"])
    caps = json.loads(fetch_members(z, raw, [S["captions"]])[S["captions"]])["images"]
    split_sizes = Counter(r["filepath"] for r in caps if S["keep"](r["filename"]))
    pool = sorted((r for r in caps if r["filepath"] == "val" and S["keep"](r["filename"])), key=lambda r: r["filename"])
    assert N <= int(CAP * len(pool)) + 1 and N <= CAP * sum(split_sizes.values()), (N, len(pool))
    rule = f"random.Random('{SEED}-ood-{name}-val').sample(sorted val originals, {N})"
    sel = random.Random(f"{SEED}-ood-{name}-val").sample(pool, N)
    fetched = fetch_members(z, raw, [m for r in sel for m in S["members"](r["filename"]).values()],
                            workers=S.get("workers", 16))

    rows, refs, stats = [], {}, {}
    for order, rec in enumerate(sel):
        f = rec["filename"]
        pid = f"val/{f[:-4]}"
        files = {k: fetched[m] for k, m in S["members"](f).items()}
        lab, bpx, tot = label_png(name, files)
        row = {"dataset": S["dataset"], "dataset_version": S["dataset_version"], "pair_id": pid, "parent_scene_id": "",
               "official_split": "val", "poc_role": "screening", "selection_order": order, "selection_seed": SEED,
               "selection_rule": rule, "caption_ids": " ".join(str(s) for s in rec["sentids"]),
               "label_ontology": "0=background,255=building change", "gsd_m": S["gsd_m"] or "",
               "first_use_date": date.today().isoformat(), "exclusion_reason": ""}
        for sub, key, b in (("A", "t1", files["A"]), ("B", "t2", files["B"]), ("label", "label", lab)):
            dest = out / "pairs" / "val" / sub / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b)
            row[f"{key}_path"] = str(dest.relative_to(out))
            row[f"{key}_sha256"] = sha256(b)
        rows.append(row)
        refs[pid] = {"changeflag": rec["changeflag"], "captions": [s["raw"].strip() for s in rec["sentences"]]}
        stats[pid] = {"building_px": bpx, "road_px": 0, "changed_frac": round(bpx / tot, 5),
                      "changeflag": rec["changeflag"], "shape": list(Image.open(io.BytesIO(files["A"])).size)}

    (out / "references").mkdir(parents=True, exist_ok=True)
    json.dump(refs, open(out / "references" / "references.json", "w"), indent=1)
    json.dump(stats, open(out / "mask_stats.json", "w"), indent=1)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "dataset": S["dataset"], "dataset_version": S["dataset_version"], "source_url": S["url"], "licence": S["licence"],
        "archive_bytes": raw.size, "bytes_fetched": raw.bytes_fetched,
        "split_sizes_originals": dict(split_sizes), "selected": {"val": N}, "selection_rule": rule,
        "in_scope_classes": {"building": 255},
        "selected_changeflag": sum(r["changeflag"] for r in sel),
        "selected_building_change": sum(s["building_px"] >= 50 for s in stats.values()),
        "changeflag_without_building_change": sum(bool(s["changeflag"]) and s["building_px"] < 50 for s in stats.values()),
        "manifest_sha256": sha256((out / "manifest.csv").read_bytes()), "frozen_on": date.today().isoformat(),
    }
    json.dump(summary, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
