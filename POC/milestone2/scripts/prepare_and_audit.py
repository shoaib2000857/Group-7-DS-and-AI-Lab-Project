"""Audit the pinned LEVIR-MCI release and freeze a reproducible training handover.

No model predictions or test-set model selection are performed. All release pixels
are inspected for quality. Raw bytes remain in the local archive; public outputs
contain identifiers, hashes and aggregate statistics, never redistributed imagery.
Run from any directory: python POC/milestone2/scripts/prepare_and_audit.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import re
import sys
import threading
import time
import zipfile

import numpy as np
from PIL import Image, __version__ as pillow_version
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REVISION = "dff09c709999b5b1f1f1806024811c65a6b14dfe"
SOURCE = f"https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/resolve/{REVISION}/LEVIR-MCI-dataset.zip"
PREFIX = "LEVIR-MCI-dataset"
LOCAL = threading.local()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def csv_out(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def no_change(text):
    text = re.sub(r"[^a-z ]", "", text.lower())
    return bool(re.search(r"no (?:change|difference)|nothing (?:has )?changed|same as before|seem identical|seem the same|almost nothing", text))


def dhash(arr):
    a = np.asarray(Image.fromarray(arr).convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), "big")


def profile_one(archive, rec):
    row = {"pair_id": f"LEVIR-MCI/{rec['filepath']}/{Path(rec['filename']).stem}",
           "dataset": "LEVIR-MCI", "source_revision": REVISION,
           "official_split": rec["filepath"], "filename": rec["filename"],
           "caption_changeflag": int(rec["changeflag"])}
    if not hasattr(LOCAL, "archive"):
        LOCAL.archive = zipfile.ZipFile(archive)
    if True:
        z = LOCAL.archive
        images = {}
        for sub, key in [("A", "t1"), ("B", "t2"), ("label", "mask"), ("label_rgb", "mask_rgb")]:
            member = f"{PREFIX}/images/{rec['filepath']}/{sub}/{rec['filename']}"
            row[f"{key}_member"] = member
            try:
                data = z.read(member)
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    a = np.array(im)
                    row[f"{key}_mode"] = im.mode
                row[f"{key}_sha256"] = digest(data)
                row[f"{key}_pixel_sha256"] = digest(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes())
                row[f"{key}_shape"] = "x".join(map(str, a.shape))
                images[key] = a
            except Exception as exc:
                row[f"{key}_error"] = str(exc)
        row["valid_files"] = len(images) == 4
        if len(images) < 4:
            return row
        a, b, mask, rgb = (images[k] for k in ["t1", "t2", "mask", "mask_rgb"])
        # The published label PNG is RGB storage of three equal grayscale
        # channels. Verify equality before lossless reduction to class values.
        row["mask_channels_identical"] = mask.ndim == 2 or (mask.ndim == 3 and mask.shape[-1] == 3 and bool(np.all(mask == mask[..., :1])))
        if mask.ndim == 3 and row["mask_channels_identical"]:
            mask = mask[..., 0]
        row["valid_shapes"] = a.shape == b.shape == (256, 256, 3) and mask.shape == (256, 256) and rgb.shape == a.shape
        if not row["valid_shapes"]:
            return row
        row["mask_values"] = ";".join(str(int(v)) for v in np.unique(mask))
        row["valid_mask_values"] = set(np.unique(mask)).issubset({0, 128, 255})
        px = {v: int(np.count_nonzero(mask == v)) for v in [0, 128, 255]}
        for value, key in [(0, "background"), (128, "road"), (255, "building")]:
            row[f"{key}_px"] = px[value]
            row[f"{key}_fraction"] = px[value] / mask.size
        row["changed_fraction"] = (px[128] + px[255]) / mask.size
        row["mask_class"] = "both" if px[128] and px[255] else "building_only" if px[255] else "road_only" if px[128] else "unchanged"
        row["mask_caption_flag_conflict"] = bool(px[128] + px[255]) != bool(rec["changeflag"])
        row["identical_dates"] = row["t1_pixel_sha256"] == row["t2_pixel_sha256"]
        row["identical_dates_with_changed_mask"] = row["identical_dates"] and bool(px[128] + px[255])
        # Check categorical equivalence, not visual intensity preference.
        palette = {}
        palette_ok = True
        for val in np.unique(mask):
            packed = rgb[mask == val].astype(np.uint32)
            codes = np.unique((packed[:, 0] << 16) | (packed[:, 1] << 8) | packed[:, 2])
            colors = np.array([[(int(c) >> 16) & 255, (int(c) >> 8) & 255, int(c) & 255] for c in codes])
            palette[str(int(val))] = colors.tolist()
            palette_ok &= len(colors) == 1
        if palette_ok:
            palette_ok = len({tuple(c[0]) for c in palette.values()}) == len(palette)
        row["palette_consistent_within_pair"] = bool(palette_ok)
        row["palette"] = json.dumps(palette, separators=(",", ":"))
        for key, arr in [("t1", a), ("t2", b)]:
            gray = arr @ np.array([.2126, .7152, .0722])
            row[f"{key}_brightness"] = float(gray.mean())
            row[f"{key}_contrast"] = float(gray.std())
            row[f"{key}_dhash"] = str(dhash(arr))
            for j, channel in enumerate("rgb"):
                row[f"{key}_{channel}_mean"] = float(arr[:, :, j].mean())
        row["brightness_delta"] = row["t2_brightness"] - row["t1_brightness"]
        row["pair_mae"] = float(np.abs(a.astype(float) - b).mean())
        # Components and centroid are geometry diagnostics, never instance counts.
        for value, key in [(128, "road"), (255, "building")]:
            labels, n = ndimage.label(mask == value, structure=np.ones((3, 3)))
            row[f"{key}_components_8"] = int(n)
            if n:
                sizes = np.bincount(labels.ravel())[1:]
                row[f"{key}_max_component_px"] = int(sizes.max())
            else:
                row[f"{key}_max_component_px"] = 0
        yy, xx = np.where(mask > 0)
        row["change_centroid_x"] = float(xx.mean() / 255) if len(xx) else None
        row["change_centroid_y"] = float(yy.mean() / 255) if len(yy) else None
        grid = np.zeros((3, 3), dtype=np.int64)
        for yi, ys in enumerate(np.array_split(np.arange(256), 3)):
            for xi, xs in enumerate(np.array_split(np.arange(256), 3)):
                grid[yi, xi] = int((mask[np.ix_(ys, xs)] > 0).sum())
        row["grid_changed_px"] = json.dumps(grid.tolist())
        captions = [s["raw"].strip() for s in rec["sentences"]]
        lengths = [len(s.split()) for s in captions]
        row["caption_count"] = len(captions)
        row["empty_captions"] = sum(not s for s in captions)
        row["caption_words_mean"] = float(np.mean(lengths))
        row["caption_words_max"] = max(lengths, default=0)
        nc = sum(no_change(c) for c in captions)
        row["no_change_reference_count"] = nc
        row["mixed_change_reference_flags"] = 0 < nc < len(captions)
        row["caption_flag_text_conflict"] = (nc == len(captions) and bool(rec["changeflag"])) or (nc == 0 and not bool(rec["changeflag"]))
        row["direction_reference_count"] = sum(bool(re.search(r"\b(left|right|top|bottom|upper|lower|center|centre|middle|north|south|east|west)\b", c.lower())) for c in captions)
        row["count_reference_count"] = sum(bool(re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", c.lower())) for c in captions)
    return row


def near_duplicate_candidates(rows):
    """Exact enumeration of pairs within Hamming distance <=4 at BOTH dates.

    Five disjoint bands guarantee that at least one band is equal when the
    first-date distance is <=4. Hash similarity does not establish duplication.
    """
    eligible = [r for r in rows if "t1_dhash" in r]
    bands = defaultdict(list)
    results = []
    for idx, row in enumerate(eligible):
        h = int(row["t1_dhash"])
        cand = set()
        for band, (offset, width) in enumerate([(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]):
            key = (band, (h >> offset) & ((1 << width) - 1))
            cand.update(bands[key])
            bands[key].append(idx)
        for other_idx in cand:
            other = eligible[other_idx]
            if row["official_split"] == other["official_split"]:
                continue
            d1 = (h ^ int(other["t1_dhash"])).bit_count()
            d2 = (int(row["t2_dhash"]) ^ int(other["t2_dhash"])).bit_count()
            if d1 <= 4 and d2 <= 4:
                exact = {row["t1_pixel_sha256"], row["t2_pixel_sha256"]} == {other["t1_pixel_sha256"], other["t2_pixel_sha256"]}
                results.append({"pair_a": other["pair_id"], "pair_b": row["pair_id"], "t1_hamming": d1, "t2_hamming": d2, "exact_pixels": exact,
                                "disposition": "exact_group_policy" if exact else "conservative_development_exclusion"})
    return results


def aggregate(rows):
    good = [r for r in rows if r.get("valid_shapes") and r.get("valid_mask_values")]
    pixels = {k: sum(r[f"{k}_px"] for r in good) for k in ["background", "building", "road"]}
    denom = sum(pixels.values())
    changes = [r["changed_fraction"] for r in good if r["changed_fraction"] > 0]
    def quant(key):
        a = [r[key] for r in good]
        return dict(zip(["min", "q1", "median", "q3", "max"], [float(v) for v in np.quantile(a, [0, .25, .5, .75, 1])])) if a else {}
    return {"pairs": len(rows), "valid_pairs": len(good), "mask_classes": dict(Counter(r["mask_class"] for r in good)),
            "caption_changeflags": dict(Counter(r["caption_changeflag"] for r in good)), "pixels": pixels,
            "pixel_percent": {k: 100 * v / denom for k, v in pixels.items()} if denom else {},
            "mask_caption_conflicts": sum(r["mask_caption_flag_conflict"] for r in good),
            "mixed_reference_change_flags": sum(r["mixed_change_reference_flags"] for r in good),
            "caption_text_flag_conflicts": sum(r["caption_flag_text_conflict"] for r in good),
            "identical_dates": sum(r["identical_dates"] for r in good),
            "identical_dates_with_changed_mask": sum(r["identical_dates_with_changed_mask"] for r in good),
            "small_changed_pairs_under_1_percent": sum(0 < r["changed_fraction"] < .01 for r in good),
            "changed_fraction_positive_quantiles": dict(zip(["min", "q1", "median", "q3", "max"], map(float, np.quantile(changes, [0, .25, .5, .75, 1])))) if changes else {},
            "brightness_delta_quantiles": quant("brightness_delta"), "pair_mae_quantiles": quant("pair_mae"),
            "caption_words_mean": float(np.mean([r["caption_words_mean"] for r in good])) if good else None,
            "caption_words_max": max((r["caption_words_max"] for r in good), default=0),
            "threshold_sensitivity": {str(t): {k: sum(r[f"{k}_px"] >= t for r in good) for k in ["building", "road"]} for t in [1, 10, 50, 100]},
            "direction_reference_pairs": sum(r["direction_reference_count"] > 0 for r in good),
            "numeric_reference_pairs": sum(r["count_reference_count"] > 0 for r in good)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ROOT / "data/raw/LEVIR-MCI-dataset.zip")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = ROOT / "evidence/full_audit"
    out.mkdir(parents=True, exist_ok=True)
    prepared = ROOT / "data/prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with zipfile.ZipFile(args.archive) as z:
        cap_name = next(n for n in z.namelist() if n.endswith("LevirCCcaptions.json"))
        caps_bytes = z.read(cap_name)
        caps = json.loads(caps_bytes)["images"]
        names = set(z.namelist())
        print("Caption records", len(caps), "Archive members", len(names), flush=True)
        records = sorted(caps, key=lambda r: (r["filepath"], r["filename"]))
        missing = [f"{PREFIX}/images/{r['filepath']}/{sub}/{r['filename']}" for r in records for sub in ["A", "B", "label", "label_rgb"] if f"{PREFIX}/images/{r['filepath']}/{sub}/{r['filename']}" not in names]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(lambda r: profile_one(args.archive, r), records), 1):
            rows.append(row)
            if i % 500 == 0:
                print("PROFILE", i, "/", len(records), "seconds", round(time.time() - start), flush=True)
    csv_out(out / "pair_profile.csv", rows)
    print("Profiles saved; checking exact and near duplicates", flush=True)
    groups = defaultdict(list)
    for row in rows:
        if row.get("valid_shapes"):
            key = tuple(sorted([row["t1_pixel_sha256"], row["t2_pixel_sha256"]]))
            groups[key].append(row)
    duplicates = []
    rank = {"test": 0, "val": 1, "train": 2}
    exclusions = defaultdict(set)
    for key, group in groups.items():
        if len(group) > 1:
            group.sort(key=lambda r: (rank[r["official_split"]], r["pair_id"]))
            for row in group[1:]:
                exclusions[row["pair_id"]].add("exact_repeated_pair")
            target_keys = {r["mask_pixel_sha256"] for r in group}
            if len(target_keys) > 1:
                for row in group:
                    exclusions[row["pair_id"]].add("inconsistent_duplicate_targets")
            duplicates.append({"kept_by_priority": group[0]["pair_id"], "members": [r["pair_id"] for r in group], "same_mask": len(target_keys) == 1, "splits": sorted({r["official_split"] for r in group})})
    near = near_duplicate_candidates(rows)
    by_id = {r["pair_id"]: r for r in rows}
    # Conservative removal of lower-priority split records eliminates all detected
    # cross-split near-pair matches without claiming they are true duplicates.
    for c in near:
        if not c["exact_pixels"]:
            a, b = by_id[c["pair_a"]], by_id[c["pair_b"]]
            loser = max([a, b], key=lambda r: rank[r["official_split"]])
            exclusions[loser["pair_id"]].add("perceptual_cross_split_candidate")
    for row in rows:
        pid = row["pair_id"]
        if not row.get("valid_files") or not row.get("valid_shapes") or not row.get("valid_mask_values"):
            exclusions[pid].add("invalid_input")
            continue
        if not row["palette_consistent_within_pair"]:
            exclusions[pid].add("rgb_palette_conflict")
        if row["mask_caption_flag_conflict"]:
            exclusions[pid].add("mask_caption_changeflag_conflict")
        if row["mixed_change_reference_flags"] or row["caption_flag_text_conflict"]:
            exclusions[pid].add("caption_reference_flag_conflict")
        if row["identical_dates_with_changed_mask"]:
            exclusions[pid].add("identical_dates_with_change_target")
        if row["caption_count"] != 5 or row["empty_captions"]:
            exclusions[pid].add("invalid_captions")
    dump(out / "exact_duplicate_groups.json", duplicates)
    csv_out(out / "near_duplicate_candidates.csv", near)
    csv_out(out / "exclusion_ledger.csv", [{"pair_id": k, "reasons": ";".join(sorted(v))} for k, v in sorted(exclusions.items())])
    accepted = [r for r in rows if not exclusions[r["pair_id"]]]
    cap_by_id = {f"LEVIR-MCI/{r['filepath']}/{Path(r['filename']).stem}": r for r in records}
    references = {r["pair_id"]: {"captions": [s["raw"].strip() for s in cap_by_id[r["pair_id"]]["sentences"]], "source_changeflag": r["caption_changeflag"], "mask_member": r["mask_member"]} for r in accepted}
    # Image-only inference records cannot expose evaluation references by accident.
    full = []
    for r in accepted:
        full.append({"pair_id": r["pair_id"], "dataset": "LEVIR-MCI", "source_revision": REVISION,
                     "official_split": r["official_split"], "archive": "../raw/LEVIR-MCI-dataset.zip",
                     "t1_member": r["t1_member"], "t2_member": r["t2_member"], "target_member": r["mask_member"],
                     "t1_sha256": r["t1_sha256"], "t2_sha256": r["t2_sha256"], "target_sha256": r["mask_sha256"],
                     "shape": [256, 256, 3], "mask_encoding": {"0": "background", "128": "road_change", "255": "building_change"},
                     "mask_class": r["mask_class"], "acquisition_t1": None, "acquisition_t2": None, "crs": None,
                     "per_tile_gsd_m": None, "nominal_source_gsd_m": .5})
    for split in ["train", "val", "test"]:
        manifest = [r for r in full if r["official_split"] == split]
        text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in manifest)
        (prepared / f"{split}.jsonl").write_text(text, encoding="utf-8")
        # Public metadata handover is byte-identical, except archive path is interpreted
        # relative to data/prepared in the documented local layout.
        (out / f"{split}_manifest.jsonl").write_text(text, encoding="utf-8")
        inference = [{k: v for k, v in r.items() if k in ["pair_id", "archive", "t1_member", "t2_member", "t1_sha256", "t2_sha256"]} for r in manifest]
        (prepared / f"{split}_inference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inference), encoding="utf-8")
    dump(prepared / "references.json", references)
    # Keep image-text POC training below the original 20% cap; complete-release EDA
    # does not train a model. Selection is stratified by mask category.
    train = [r for r in full if r["official_split"] == "train"]
    desired = min(1363, len(train))
    strata = defaultdict(list)
    for r in train:
        strata[r["mask_class"]].append(r)
    selected = []
    for k, pool in sorted(strata.items()):
        random.Random(f"42-m2-clean-{k}").shuffle(pool)
        n = int(desired * len(pool) / len(train))
        selected.extend(pool[:n])
    selected_ids = {r["pair_id"] for r in selected}
    rest = [r for r in train if r["pair_id"] not in selected_ids]
    random.Random("42-m2-clean-remainder").shuffle(rest)
    selected += rest[:desired - len(selected)]
    (prepared / "poc_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected), encoding="utf-8")
    (out / "poc_train_manifest.jsonl").write_text((prepared / "poc_train.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    archive_hash = hashlib.sha256()
    with args.archive.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            archive_hash.update(chunk)
    unchanged_refs = Counter(tuple(s["raw"].strip() for s in r["sentences"]) for r in records if not r["changeflag"])
    palette_values = defaultdict(set)
    for r in rows:
        for value, colors in json.loads(r.get("palette", "{}")).items():
            palette_values[value].update(tuple(c) for c in colors)
    numeric = ["changed_fraction", "brightness_delta", "pair_mae", "caption_words_mean"]
    matrix = np.array([[r[k] for k in numeric] for r in rows if r.get("valid_shapes") and r.get("valid_mask_values")])
    if matrix.size == 0:
        raise ValueError("No valid pairs: inspect recorded input modes, dimensions and class values")
    outliers = {}
    for j, key in enumerate(numeric):
        q1, q3 = np.quantile(matrix[:, j], [.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outliers[key] = {"lower_fence": float(lo), "upper_fence": float(hi),
                         "flagged_pairs": int(((matrix[:, j] < lo) | (matrix[:, j] > hi)).sum()),
                         "disposition": "retained unless a separate integrity/consistency exclusion applies"}
    summary = {"source_url": SOURCE, "revision": REVISION, "archive_bytes": args.archive.stat().st_size,
               "archive_sha256": archive_hash.hexdigest(), "caption_json_sha256": digest(caps_bytes),
               "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pillow": pillow_version,
               "audit_scope": "All source pairs and all four image/label representations; quality-only inspection includes test pixels.",
               "source": aggregate(rows), "prepared": aggregate(accepted),
               "source_by_split": {s: aggregate([r for r in rows if r["official_split"] == s]) for s in ["train", "val", "test"]},
               "prepared_by_split": {s: aggregate([r for r in accepted if r["official_split"] == s]) for s in ["train", "val", "test"]},
               "missing_archive_members": missing,
               "duplicate_caption_keys": len(records) - len(cap_by_id),
               "exact_duplicate_groups": len(duplicates), "redundant_pair_records": sum(len(g["members"]) - 1 for g in duplicates),
               "cross_split_exact_groups": sum(len(g["splits"]) > 1 for g in duplicates),
               "cross_split_near_candidates": len(near),
               "cross_split_near_candidates_nonexact": sum(not c["exact_pixels"] for c in near),
               "exclusions_unique_pairs": len([v for v in exclusions.values() if v]),
               "exclusion_reason_counts_nonexclusive": dict(Counter(reason for v in exclusions.values() for reason in v)),
               "rgb_palette": {k: sorted(v) for k, v in palette_values.items()},
               "unchanged_reference_template_sets": len(unchanged_refs),
               "most_common_unchanged_template_pairs": unchanged_refs.most_common(1)[0][1] if unchanged_refs else 0,
               "poc_training_pairs": len(selected),
               "correlation_features": numeric, "pearson_correlation": np.corrcoef(matrix.T).tolist(),
               "iqr_outliers": outliers,
               "manifest_sha256": {p.name: digest(p.read_bytes()) for p in prepared.glob('*.jsonl')},
               "elapsed_seconds": round(time.time() - start, 2)}
    # Assertions certify the prepared table, not latent factual correctness.
    kept = {r["pair_id"] for r in accepted}
    assert all(not ({c["pair_a"], c["pair_b"]} <= kept) for c in near)
    assert sum(summary["prepared"]["mask_classes"].values()) == len(accepted)
    assert summary["source"]["pairs"] == len(records)
    assert len(rows) == len(cap_by_id)
    dump(out / "summary.json", summary)
    dump(prepared / "provenance.json", summary)
    sample = next(r for r in full if r["official_split"] == "train" and r["mask_class"] == "both")
    dump(out / "sample_record.json", {"input_record": sample, "evaluation_only_reference": references[sample["pair_id"]]})
    print(json.dumps({k: summary[k] for k in ["source", "prepared", "exclusions_unique_pairs", "exclusion_reason_counts_nonexclusive", "elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
