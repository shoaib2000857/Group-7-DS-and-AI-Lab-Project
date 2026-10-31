"""Prepare an author-release SECOND-CC evaluation probe without a multi-GB download.

Requires the captured official ZIP index and caption JSON. Downloads only the
seeded 100 original validation pairs; never mixes them into primary training.
Archive CRC checks validate range extraction; SHA-256 records acquired files.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
import hashlib
import io
import json
import random
import struct
import time
import zlib

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/16937571/files/SECOND-CC-AUG.zip?download=1"


def get_member(item):
    dest = ROOT / "data/transfer/SECOND-CC" / item["name"].removeprefix("SECOND-CC-AUG/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        b = dest.read_bytes()
        if zlib.crc32(b) == item["crc32"] and len(b) == item["size"]:
            return item["name"], b
        raise ValueError(f"Existing cached source failed CRC: {dest}")
    start = item["offset"]
    end = min(2539187782 - 1, start + 30 + len(item["name"].encode()) + 1024 + item["compressed"])
    for attempt in range(4):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 35))
            if r.status_code == 429 and attempt < 3:
                wait = min(60, max(15, int(r.headers.get("Retry-After", "60"))))
                print("Zenodo rate limit; cached files retained; wait", wait, "seconds", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            if r.status_code != 206 or len(r.content) != end - start + 1:
                raise ValueError("Range response not honoured")
            buf = r.content
            if buf[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP member header")
            method = struct.unpack_from("<H", buf, 8)[0]
            n, extra = struct.unpack_from("<HH", buf, 26)
            comp = buf[30+n+extra:30+n+extra+item["compressed"]]
            b = zlib.decompress(comp, -15) if method == 8 else comp
            if method not in (0, 8) or len(b) != item["size"] or zlib.crc32(b) != item["crc32"]:
                raise ValueError("CRC or length mismatch")
            dest.write_bytes(b)
            return item["name"], b
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    index = json.loads((ROOT / "evidence/second_index.json").read_text())
    idx = {r["name"]: r for r in index}
    caps = json.loads((ROOT / "data/raw/second_captions.json").read_text())["images"]
    originals = [r for r in caps if "_random_augment" not in r["filename"]]
    pool = sorted((r for r in originals if r["filepath"] == "val"), key=lambda r: r["filename"])
    selection = random.Random("42-ood-second_cc-val").sample(pool, 100)
    names = [f"SECOND-CC-AUG/val/{kind}/{date}/{r['filename']}" for r in selection for kind in ["rgb", "sem"] for date in ["A", "B"]]
    files = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(get_member, idx[n]) for n in names]
        for i, f in enumerate(as_completed(futures), 1):
            n, data = f.result()
            files[n] = data
            if i % 25 == 0:
                print("TRANSFER FILES", i, "/", len(names), flush=True)
    out = ROOT / "data/transfer/SECOND-CC"
    rows, refs = [], {}
    for rec in selection:
        name = rec["filename"]
        pid = "SECOND-CC/val/" + Path(name).stem
        arrays = {}
        row = {"pair_id": pid, "dataset": "SECOND-CC", "official_split": "val", "role": "transfer_evaluation", "source_record": "zenodo:16937571", "source_changeflag": rec["changeflag"]}
        for kind in ["rgb", "sem"]:
            for date in ["A", "B"]:
                key = f"{kind}_{date}"
                rel = f"val/{kind}/{date}/{name}"
                b = files[f"SECOND-CC-AUG/{rel}"]
                arrays[key] = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
                row[key+"_path"] = rel
                row[key+"_sha256"] = hashlib.sha256(b).hexdigest()
        assert len({a.shape for a in arrays.values()}) == 1
        row["shape"] = list(arrays["rgb_A"].shape)
        mask = np.all(arrays["sem_A"] == (128, 0, 0), -1) | np.all(arrays["sem_B"] == (128, 0, 0), -1)
        row["building_change_pixels"] = int(mask.sum())
        row["road_supervision"] = False
        row["source_has_change_without_building"] = bool(rec["changeflag"]) and not bool(mask.any())
        target = out / "val/building_change" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(target)
        row["building_mask_path"] = target.relative_to(out).as_posix()
        row["building_mask_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        refs[pid] = [s["raw"].strip() for s in rec["sentences"]]
        rows.append(row)
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "references.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")
    # CRC/size matches identify archive candidates, not confirmed decoded-pixel identity.
    groups = {}
    for r in originals:
        split, name = r["filepath"], r["filename"]
        members = [idx[f"SECOND-CC-AUG/{split}/rgb/{date}/{name}"] for date in ["A", "B"]]
        signature = tuple(sorted((m["crc32"], m["size"]) for m in members))
        groups.setdefault(signature, []).append((split, name))
    cross = [g for g in groups.values() if len({p[0] for p in g}) > 1]
    summary = {"source": URL, "licence": "CC-BY-4.0, Zenodo record 16937571", "original_split_sizes": dict(Counter(r["filepath"] for r in originals)),
               "augmented_and_original_records": len(caps), "selected_original_val_pairs": len(rows),
               "selection_rule": "random.Random('42-ood-second_cc-val').sample(sorted original val,100)",
               "selected_shapes": dict(Counter(str(r["shape"]) for r in rows)),
               "building_positive_at_one_pixel": sum(r["building_change_pixels"] > 0 for r in rows),
               "building_positive_at_50_pixels": sum(r["building_change_pixels"] >= 50 for r in rows),
               "source_changeflag_without_building": sum(r["source_has_change_without_building"] for r in rows),
               "cross_split_crc_candidate_groups_originals": len(cross),
               "source_index_sha256": hashlib.sha256((ROOT/"evidence/second_index.json").read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256((out/"manifest.jsonl").read_bytes()).hexdigest(),
               "interpretation": "Evaluation only; original validation pairs. Building masks have narrower scope than source change captions. No new captioner or human annotation run."}
    (ROOT / "evidence/second_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
