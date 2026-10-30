"""Complete run_record.json fields that inference could not fill (run once after scoring).

- checkpoint_sha256: RSCaMa.pth hashed; HF LFS blobs are named by their sha256
- limitations as a list (template type); record_type
- RSCaMa exposure: upstream keeps the checkpoint with the best val BLEU-4
- parsed outputs per split; code-state note
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import EXEC_ROOT, RUNS  # noqa: E402

HF = Path.home() / ".cache" / "huggingface" / "hub"
HF_REPOS = {"C04": "models--Qwen--Qwen3-VL-4B-Instruct", "C07": "models--HuggingFaceTB--SmolVLM2-2.2B-Instruct"}
RSCAMA_EXPOSURE = (
    "Trained on the LEVIR-CC train split (author paper); development pairs here come from that split. "
    "Upstream train_CC.py keeps the checkpoint with the best val BLEU-4, so the screening (val) pairs "
    "were used for checkpoint selection: screening is not unseen data for RSCaMa. Only the sealed test "
    "split is untouched by it."
)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_shards(repo_dir, revision):
    snap = HF / repo_dir / "snapshots" / revision
    return {p.name: Path(os.readlink(p)).name for p in sorted(snap.glob("*.safetensors"))}


def main(code_commit):
    rscama_sha = None
    for f in sorted(RUNS.glob("B0_*/run_record.json")):
        rr = json.load(open(f))
        rid = rr["role_model_id"]
        rr["record_type"] = "run_record"
        if isinstance(rr.get("limitations"), str):
            rr["limitations"] = [s.strip() for s in rr["limitations"].split(";") if s.strip()]
        if rid == "C02":
            rscama_sha = rscama_sha or sha256_file(EXEC_ROOT / "weights" / "rscama" / "RSCaMa.pth")
            rr["checkpoint_sha256"] = rscama_sha
            rr["training_data_exposure"] = RSCAMA_EXPOSURE
        elif rid in HF_REPOS:
            rr["checkpoint_sha256"] = hf_shards(HF_REPOS[rid], rr["checkpoint_revision"])
        rr["parsed_output_path"] = {
            split: str(p.relative_to(EXEC_ROOT)) for split in ("screening", "development")
            for p in [f.parent / f"claims_{split}.jsonl"] if p.exists()
        }
        rr["code_commit_at_run"] = rr.get("code_commit")
        rr["code_commit"] = code_commit
        rr["code_commit_note"] = ("Inference ran from the uncommitted execution/ tree that was first "
                                  "committed as code_commit; later commits change parsing/scoring only.")
        json.dump(rr, open(f, "w"), indent=1)
        print(f.parent.name, "ok")


if __name__ == "__main__":
    main(sys.argv[1])
