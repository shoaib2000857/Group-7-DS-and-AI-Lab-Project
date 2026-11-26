"""Aggregate Milestone 4 training cost from run.json receipts into JSON and Markdown tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE = ROOT / "docs/milestone4/evidence/full_data_m4"
DEFAULT_OUT = ROOT / "POC/milestone5/evidence/compute_profile"


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def detector_row(run_json: Path) -> dict:
    d = json.loads(run_json.read_text())
    hist = read_history(run_json.parent / "history.jsonl")
    trainable = max((h.get("trainable_params", 0) for h in hist), default=None)
    return {
        "run_id": d["run_id"],
        "family": d["config"]["arch"],
        "kind": "detector",
        "params_total": d["model_params"]["total"],
        "params_trainable_max": trainable,
        "gpu": d["environment"]["gpu_name"],
        "torch": d["environment"]["torch"],
        "amp": d.get("amp_actual"),
        "seed": d["config"]["seed"],
        "epochs_cap": d["config"]["epochs"],
        "epochs_run": d["epochs_completed"],
        "best_epoch": d["best_epoch"],
        "status": d["status"],
        "wall_sec": d["total_wall_sec"],
        "peak_vram_mb": max((h.get("peak_vram_mb", 0) for h in hist), default=None),
    }


def captioner_row(run_json: Path) -> dict:
    d = json.loads(run_json.read_text())
    hist = read_history(run_json.parent / "history.jsonl")
    tr = d["trainability"]
    return {
        "run_id": d["run_id"],
        "family": "rscama",
        "kind": "captioner",
        "params_total": sum(v["total"] for v in tr.values()),
        "params_trainable_max": sum(v["trainable"] for v in tr.values()),
        "gpu": d["environment"]["gpu_name"],
        "torch": d["environment"]["torch"],
        "amp": d["config"]["amp"],
        "seed": d["config"]["seed"],
        "epochs_cap": d["config"]["epochs"],
        "epochs_run": d["epochs_completed"],
        "best_epoch": d["best_epoch"],
        "status": d["status"],
        "wall_sec": d["total_wall_sec"],
        "peak_vram_mb": max((h.get("peak_vram_mb", 0) for h in hist), default=None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = [detector_row(p) for p in sorted((args.evidence_dir / "detector_runs").glob("*/run.json"))]
    rows += [captioner_row(p) for p in sorted((args.evidence_dir / "captioner_runs").glob("*/run.json"))]
    for r in rows:
        r["wall_min"] = round(r["wall_sec"] / 60, 2)
        r["sec_per_epoch"] = round(r["wall_sec"] / r["epochs_run"], 1)

    per_gpu = defaultdict(lambda: {"runs": 0, "wall_sec": 0.0, "epochs": 0})
    for r in rows:
        g = per_gpu[r["gpu"]]
        g["runs"] += 1
        g["wall_sec"] += r["wall_sec"]
        g["epochs"] += r["epochs_run"]
    per_gpu_out = {
        k: {"runs": v["runs"], "epochs": v["epochs"], "wall_min": round(v["wall_sec"] / 60, 1),
            "gpu_hours": round(v["wall_sec"] / 3600, 3)}
        for k, v in sorted(per_gpu.items())
    }
    per_kind = defaultdict(float)
    for r in rows:
        per_kind[r["kind"]] += r["wall_sec"]
    total_sec = sum(r["wall_sec"] for r in rows)
    out = {
        "source": str(args.evidence_dir.relative_to(ROOT)) if args.evidence_dir.is_relative_to(ROOT) else str(args.evidence_dir),
        "num_runs": len(rows),
        "per_gpu": per_gpu_out,
        "per_kind_gpu_hours": {k: round(v / 3600, 3) for k, v in per_kind.items()},
        "grand_total": {"wall_min": round(total_sec / 60, 1), "gpu_hours": round(total_sec / 3600, 3),
                        "epochs": sum(r["epochs_run"] for r in rows)},
        "runs": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "training_cost.json").write_text(json.dumps(out, indent=2))

    def m(x):
        return "n/a" if x is None else f"{x / 1e6:.2f}"

    lines = ["| Run | Family | Params total (M) | Trainable (M) | GPU | Epochs run / cap | Best epoch | Wall (min) | Sec/epoch | Peak train VRAM (MB) |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        status = "" if r["status"] == "completed" else f" ({r['status']})"
        vram = "n/a" if r["peak_vram_mb"] is None else f"{r['peak_vram_mb']:,.0f}"
        lines.append(f"| `{r['run_id']}`{status} | {r['family']} | {m(r['params_total'])} | {m(r['params_trainable_max'])} | {r['gpu']} | "
                     f"{r['epochs_run']} / {r['epochs_cap']} | {r['best_epoch']} | {r['wall_min']:.1f} | {r['sec_per_epoch']:.1f} | {vram} |")
    lines += ["", "| GPU | Runs | Epochs | Wall (min) | GPU-hours |", "|---|---:|---:|---:|---:|"]
    for k, v in per_gpu_out.items():
        lines.append(f"| {k} | {v['runs']} | {v['epochs']} | {v['wall_min']:.1f} | {v['gpu_hours']:.2f} |")
    g = out["grand_total"]
    lines.append(f"| **Total** | **{len(rows)}** | **{g['epochs']}** | **{g['wall_min']:.1f}** | **{g['gpu_hours']:.2f}** |")
    (args.out / "training_cost.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
