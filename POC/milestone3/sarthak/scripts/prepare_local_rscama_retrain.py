"""Materialise the frozen RSCaMa 20% train/dev/screen subset from local archive."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "data/raw/LEVIR-MCI-dataset.zip"
OUT = ROOT / "POC/execution/data/sarthak_rscama_retrain"
MODULE = ROOT / "POC/b-captioning/colab/rscama_colab.py"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not ARCHIVE.is_file():
        raise SystemExit(f"Local source archive missing: {ARCHIVE}")
    spec = importlib.util.spec_from_file_location("rscama_colab", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    root_in_zip = "LEVIR-MCI-dataset/"
    with zipfile.ZipFile(ARCHIVE) as archive:
        captions_bytes = archive.read(root_in_zip + "LevirCCcaptions.json")
        captions = json.loads(captions_bytes)
        selected = module.select(captions["images"])
        files = [(role, rec, split)
                 for role, split in (("development", "train"), ("screening", "val"), ("finetune", "train"))
                 for rec in selected[role]]
        extracted = []
        for role, rec, split in files:
            for band in ("A", "B"):
                rel = f"images/{split}/{band}/{rec['filename']}"
                data = archive.read(root_in_zip + rel)
                dest = OUT / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                extracted.append({"role": role, "path": rel, "sha256": sha256(data)})
        (OUT / "LevirCCcaptions.json").write_bytes(captions_bytes)
    ids = {role: module.ids_sha(selected[role]) for role in selected}
    record = {
        "source_archive": str(ARCHIVE.relative_to(ROOT)),
        "source_archive_sha256": sha256(ARCHIVE.read_bytes()),
        "caption_json_sha256": sha256(captions_bytes),
        "selected_pair_counts": {key: len(val) for key, val in selected.items()},
        "selection_ids_sha256": ids,
        "expected_selection_ids_sha256": module.EXPECTED,
        "local_images": extracted,
        "note": "Selection matches frozen RSCaMa train/dev/screen lists; validation/screening images are materialised for scoring but not passed to training.",
    }
    (OUT / "local_prepare_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "selected_pair_counts": record["selected_pair_counts"],
                      "selection_ids_sha256": ids, "images_written": len(extracted)}, indent=2))


if __name__ == "__main__":
    main()
