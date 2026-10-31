"""Training-ready, framework-neutral loading of the frozen pair records.

Images remain uint8 RGB for VLM processors; detector batches use CHW float32
in [0,1] and class-index targets. There is no learned preprocessing and no
automatic radiometric matching between dates. Model adapters may apply their
checkpoint's mean/std after loading. Evaluation references are opt-in.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class PreparedPairs:
    def __init__(self, manifest, include_references=False, verify_hashes=True):
        self.manifest = Path(manifest).resolve()
        self.rows = [json.loads(s) for s in self.manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
        self.verify_hashes = verify_hashes
        self.references = json.loads((self.manifest.parent / "references.json").read_text(encoding="utf-8")) if include_references else None
        self.archives = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.manifest.parent / row["archive"]).resolve()
        if path not in self.archives:
            self.archives[path] = zipfile.ZipFile(path)
        z = self.archives[path]
        result = {"pair_id": row["pair_id"]}
        for key in ["t1", "t2", "target"]:
            if f"{key}_member" not in row:
                continue
            data = z.read(row[f"{key}_member"])
            if self.verify_hashes and hashlib.sha256(data).hexdigest() != row[f"{key}_sha256"]:
                raise ValueError(f"hash mismatch {row['pair_id']} {key}")
            with Image.open(io.BytesIO(data)) as im:
                arr = np.array(im)
            if key == "target":
                if arr.ndim == 3:
                    if arr.shape[-1] != 3 or not np.all(arr == arr[..., :1]):
                        raise ValueError("Categorical mask RGB channels disagree")
                    arr = arr[..., 0]
                if not set(np.unique(arr)).issubset({0, 128, 255}):
                    raise ValueError("Unexpected categorical mask value")
                target = np.zeros_like(arr, dtype=np.int64)
                target[arr == 128] = 1
                target[arr == 255] = 2
                result["target"] = target
            else:
                result[key] = arr
        if self.references is not None:
            result["references"] = self.references[row["pair_id"]]["captions"]
        return result

    def detector_batch(self, indices):
        rows = [self[i] for i in indices]
        x = np.stack([np.stack([r["t1"], r["t2"]]) for r in rows])
        return {"pair_ids": [r["pair_id"] for r in rows],
                "images": np.ascontiguousarray(x.transpose(0, 1, 4, 2, 3), dtype=np.float32) / 255.,
                "targets": np.stack([r["target"] for r in rows])}

    def close(self):
        for z in self.archives.values():
            z.close()
        self.archives.clear()


def verify_handover():
    result = {}
    prepared = ROOT / "data/prepared"
    seen = set()
    for split in ["train", "val", "test"]:
        ds = PreparedPairs(prepared / f"{split}.jsonl", include_references=True)
        ids = {r["pair_id"] for r in ds.rows}
        assert len(ids) == len(ds) and not seen.intersection(ids)
        seen |= ids
        # Visit every retained sample, verify hashes and exercise label conversion.
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["t1"].shape == sample["t2"].shape == (256, 256, 3)
            assert sample["target"].shape == (256, 256)
            assert len(sample["references"]) == 5
            assert set(np.unique(sample["target"])).issubset({0, 1, 2})
        batch = ds.detector_batch(range(min(4, len(ds))))
        assert batch["images"].dtype == np.float32
        assert 0 <= batch["images"].min() <= batch["images"].max() <= 1
        inference = PreparedPairs(prepared / f"{split}_inference.jsonl")
        assert set(inference[0]) == {"pair_id", "t1", "t2"}
        result[split] = {"records_verified": len(ds), "batch_images_shape": list(batch["images"].shape),
                         "batch_target_shape": list(batch["targets"].shape), "hash_and_decode_check": "passed",
                         "inference_reference_separation": "passed"}
        ds.close()
        inference.close()
        print(split, "verified", len(ids), flush=True)
    path = ROOT / "evidence/full_audit/handover_validation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    verify_handover()
