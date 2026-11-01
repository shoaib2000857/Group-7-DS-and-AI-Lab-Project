"""Shared helpers: manifest loading, run records, timing."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

EXEC_ROOT = Path(__file__).resolve().parents[1]
# B_DATA=data_ood/<name> points every script at an out-of-domain set (scripts/build_ood.py)
DATA = EXEC_ROOT / os.environ.get("B_DATA", "data")
RUNS = EXEC_ROOT / "runs" / ("ood/" + DATA.name if "B_DATA" in os.environ else "")

PROMPT_V1 = (
    "Image T1 is earlier and T2 is later. Describe visible building or road changes "
    "in at most two sentences. Use image-relative locations. If the evidence is "
    "insufficient, state that. Do not infer legality, ownership or cause. Do not give "
    "an exact count unless individual objects are distinguishable. If no relevant "
    "change is visible, say so."
)
PROMPT_REVISION = "B-prompt-v1 (B_captioning.md, unrevised)"


def load_manifest(roles=None):
    with open(DATA / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    if roles:
        rows = [r for r in rows if r["poc_role"] in roles]
    if any(r["official_split"] == "test" for r in rows):
        raise SystemExit("Sealed test pairs are not available to runs yet.")
    return rows


def pair_paths(row):
    return DATA / row["t1_path"], DATA / row["t2_path"]


def dataset_meta():
    p = DATA / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def manifest_sha256():
    return hashlib.sha256((DATA / "manifest.csv").read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=EXEC_ROOT, text=True
        ).strip()
    except Exception:
        return None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment():
    import torch

    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }


def new_run_record(**kw):
    template = json.loads(
        (EXEC_ROOT.parent / "benchmarks" / "templates" / "run_record.json").read_text()
    )
    rec = {k: None for k in template}
    rec.update(
        record_type="run_record",
        owner="Sujal Raj Pradhan",
        code_commit=git_commit(),
        track="M",
        dataset_version=dataset_meta().get("dataset_version", "hf:lcybuaa/LEVIR-MCI@dff09c7"),
        manifest_sha256=manifest_sha256(),
        environment_record=environment(),
        status="running",
        start_utc=now(),
        errors=[],
    )
    rec.update(kw)
    return rec


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
