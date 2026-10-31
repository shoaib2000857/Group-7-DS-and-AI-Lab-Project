"""Cross-check the evidence tables and reproduce the original 150-pair cohort."""
from collections import Counter
import csv
import json
from pathlib import Path
import random
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader((ROOT / "evidence/full_audit/pair_profile.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 10077
    assert all(all(r[k] == "True" for k in ["valid_files", "valid_shapes", "mask_channels_identical", "valid_mask_values", "palette_consistent_within_pair"]) for r in rows)
    assert all(r["caption_count"] == "5" and r["empty_captions"] == "0" for r in rows)
    profiles = {r["pair_id"]: r for r in rows}
    kept = []
    for split in ["train", "val", "test"]:
        kept += [json.loads(s) for s in (ROOT / f"data/prepared/{split}.jsonl").read_text().splitlines()]
    assert len(kept) == 9544
    signatures = set()
    frame_splits = {}
    for r in kept:
        p = profiles[r["pair_id"]]
        key = tuple(sorted([p["t1_pixel_sha256"], p["t2_pixel_sha256"]]))
        assert key not in signatures
        signatures.add(key)
        assert p["mask_caption_flag_conflict"] == "False"
        for k in ["t1", "t2"]:
            frame_splits.setdefault(p[k+"_pixel_sha256"], set()).add(r["official_split"])
    assert all(len(s) == 1 for s in frame_splits.values())
    with zipfile.ZipFile(ROOT / "data/raw/LEVIR-MCI-dataset.zip") as z:
        caps = json.loads(z.read("LEVIR-MCI-dataset/LevirCCcaptions.json"))["images"]
    original = []
    for split, n in [("train", 50), ("val", 100)]:
        pool = sorted((r for r in caps if r["filepath"] == split), key=lambda r: r["filename"])
        original += [profiles[f"LEVIR-MCI/{split}/{Path(r['filename']).stem}"] for r in random.Random(f"42-{split}").sample(pool,n)]
    composition = dict(Counter(r["mask_class"] for r in original))
    conflicts = sum(r["mask_caption_flag_conflict"] == "True" for r in original)
    assert composition == {"building_only": 28, "unchanged": 65, "road_only": 11, "both": 46}
    assert conflicts == 8
    checks = {"all_source_records_valid": len(rows), "retained_unique_pairs": len(kept), "retained_flag_conflicts": 0,
              "shared_exact_frames_across_retained_splits": 0, "original_150_reproduction": composition,
              "original_150_conflicts": conflicts}
    (ROOT / "evidence/full_audit/independent_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
