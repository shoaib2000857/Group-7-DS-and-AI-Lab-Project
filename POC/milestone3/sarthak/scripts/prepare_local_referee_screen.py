"""Build the local 30-pair, 150-claim referee screen from the frozen MCI archive."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "data/raw/LEVIR-MCI-dataset.zip"
OUT = ROOT / "POC/milestone3/sarthak/runs/referee_local_screen"
RUNTIME = ROOT / "POC/benchmarks/colab/benchmark_runtime.py"
SEED = 42
N_PAIRS = 30


def load_runtime():
    spec = importlib.util.spec_from_file_location("tw_benchmark_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tw = load_runtime()
    image_root = OUT / "inputs/images"
    image_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        meta_path = next(name for name in archive.namelist() if name.endswith("/LevirCCcaptions.json"))
        root = meta_path.rsplit("/", 1)[0]
        index = json.loads(archive.read(meta_path))["images"]
        validation = [row for row in index if row["filepath"] == "val"]
        selected = sorted(validation, key=lambda row: tw.digest([SEED, "val", row["filename"]]))[:N_PAIRS]
        pairs, annotations, shared = [], [], []
        for item in selected:
            pair_id = "mci_" + Path(item["filename"]).stem
            paths, hashes = [], []
            for band in ("A", "B"):
                member = f"{root}/images/val/{band}/{item['filename']}"
                raw = archive.read(member)
                path = image_root / pair_id / f"{band}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                paths.append(str(path.resolve()))
                hashes.append(hashlib.sha256(raw).hexdigest())
            pairs.append({"pair_id": pair_id, "scene_id": pair_id, "official_split": "val", "phase": "val",
                          "filename": item["filename"], "image_paths": paths,
                          "image_metadata": [{"sha256": h} for h in hashes],
                          "coordinate_frame": "original_T1_pixels", "transform": "none"})
            annotations.append({"pair_id": pair_id, "changeflag": item.get("changeflag"),
                                "captions": [sentence["raw"].strip() for sentence in item["sentences"]],
                                "label_source": "dataset_annotation_proxy_not_human_adjudication"})
            shared.append({"pair_id": pair_id, "filename": item["filename"], "phase": "val",
                           "official_split": "val", "image_sha256": hashes})

    witnesses = {}
    for pair in pairs:
        pair["mask"] = tw.mask_observation(pair, OUT)
    cases, refs = tw.make_real_cases(pairs, annotations, witnesses, OUT)
    tw.write_json(OUT / "inputs/pairs.json", pairs)
    tw.write_json(OUT / "references/dataset_annotations.json", annotations)
    tw.write_json(OUT / "shared_manifest.json", {"seed": SEED, "selection": "rank by SHA256 of [42, 'val', filename], first 30",
                                                   "pairs": shared, "sha256": tw.digest(shared),
                                                   "source_archive_sha256": "8f7d52298fa3ca32aec983e84addef80f61ad7d981c0ff97a92c5880c4b83e9d"})
    tw.write_json(OUT / "references/targets.json", refs)
    print(json.dumps({"run_dir": str(OUT), "pairs": len(pairs), "proxy_claims": len(cases),
                      "reference_targets": len(refs), "sample_pair_ids": [p["pair_id"] for p in pairs[:5]],
                      "note": "Cases are five annotation-proxy claims per pair; these targets are not human adjudications. Qwen witness observations will be added before inference."}, indent=2))


if __name__ == "__main__":
    main()
