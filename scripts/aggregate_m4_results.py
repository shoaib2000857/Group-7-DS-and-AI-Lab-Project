"""Aggregate all completed Milestone 4 run receipts into structured JSON & Markdown tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np


def load_runs(root: Path) -> Dict[str, dict]:
    runs = {}
    if not root.exists():
        return runs
    for sub in sorted(root.iterdir()):
        rj = sub / "run.json"
        if rj.is_file():
            runs[sub.name] = json.loads(rj.read_text(encoding="utf-8"))
    return runs


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Aggregate Milestone 4 detector, captioner, robustness, and verification receipts.")
    ap.add_argument(
        "--evidence-dir",
        type=Path,
        default=repo / "docs/milestone4/evidence/full_data_m4",
        help="Directory containing detector_runs/, captioner_runs/, robustness JSON, and verification/",
    )
    args = ap.parse_args()

    ev = args.evidence_dir
    out = ev / "aggregated_summary.json"
    md_out = ev / "EXPERIMENT_TABLES.md"

    det_runs = load_runs(ev / "detector_runs")
    cap_runs = load_runs(ev / "captioner_runs")
    rob_path = ev / "robustness_and_transfer_bit.json"
    if not rob_path.is_file():
        rob_path = ev / "robustness_and_transfer.json"
    rob_siam_path = ev / "robustness_and_transfer_siam.json"
    ver_path = ev / "verification/verification_comparison_summary.json"
    rel_cap_val = ev / "captioner_runs/C00_rscama_released/eval_val.json"
    rel_cap_test = ev / "captioner_runs/C00_rscama_released/eval_test.json"

    det_table = []
    for rid, r in det_runs.items():
        cfg = r["config"]
        m = r["reloaded_best_val_metrics"]
        det_table.append({
            "run_id": rid,
            "arch": cfg["arch"],
            "seed": cfg["seed"],
            "head_lr": cfg["head_lr"],
            "backbone_lr": cfg.get("backbone_lr"),
            "wd": cfg["weight_decay"],
            "eff_batch": r["effective_batch_size"],
            "scheduler": cfg["scheduler"],
            "loss": cfg["loss"],
            "freeze_epochs": cfg["freeze_epochs"],
            "aug": cfg["aug"],
            "epochs_run": r["epochs_completed"],
            "best_epoch": r["best_epoch"],
            "val_change_miou": round(r["best_val_change_miou"] * 100, 2),
            "val_macro_iou": round(m["macro_iou"] * 100, 2),
            "val_road_iou": round(m["road_iou"] * 100, 2),
            "val_building_iou": round(m["building_iou"] * 100, 2),
            "val_binary_f1": round(m["binary_f1"] * 100, 2),
            "val_small_f1": round(m["small_change_f1"] * 100, 2),
            "fa_1px_pct": round(m["unchanged_fa_rates"]["1"] * 100, 2),
            "fa_50px_pct": round(m["unchanged_fa_rates"]["50"] * 100, 2),
            "wall_sec": r["total_wall_sec"],
            "ckpt_sha256": r["best_checkpoint_sha256"],
        })
    det_table.sort(key=lambda x: -x["val_change_miou"])

    cap_table = []
    for rid, r in cap_runs.items():
        cfg = r["config"]
        hist = [
            json.loads(line)
            for line in (ev / "captioner_runs" / rid / "history.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        best_row = max(hist, key=lambda h: h["val_bleu4"])
        cap_table.append({
            "run_id": rid,
            "seed": cfg["seed"],
            "lr": cfg["lr"],
            "optimizer": cfg["optimizer"],
            "wd": cfg["weight_decay"],
            "scheduler": cfg["scheduler"],
            "label_smoothing": cfg["label_smoothing"],
            "freeze_encoder": cfg["freeze_encoder"],
            "batch_size": cfg["batch_size"],
            "vocab_size": r["vocab_size"],
            "epochs_run": r["epochs_completed"],
            "best_epoch": r["best_epoch"],
            "val_bleu4": round(best_row["val_bleu4"], 4),
            "val_rouge_l": round(best_row["val_rouge_l"], 4),
            "val_changed_bleu4": round(best_row["val_changed_bleu4"], 4),
            "val_ce_loss": round(best_row["val_ce_loss"], 4),
            "nc_fa_rate_pct": round(best_row["val_unchanged_fa_rate"] * 100, 2),
            "wall_sec": r["total_wall_sec"],
            "ckpt_sha256": r["best_checkpoint_sha256"],
        })
    cap_table.sort(key=lambda x: -x["val_bleu4"])

    def group_multiseed(table, primary_key, secondary_key):
        by_id = {r["run_id"]: r for r in table}
        groups = defaultdict(list)
        for row in table:
            rid = row["run_id"]
            matched_base = None
            for suf in ("_s0", "_s42", "_s2026"):
                if rid.endswith(suf):
                    matched_base = rid[: -len(suf)]
                    break
            if matched_base is not None:
                groups[matched_base].append(row)
        out_ms = {}
        for base, rows in groups.items():
            if base in by_id and all(r["run_id"] != base for r in rows):
                rows = [by_id[base]] + rows
            if len(rows) >= 2:
                vals = [x[primary_key] for x in rows]
                secs = [x[secondary_key] for x in rows]
                out_ms[base] = {
                    "n_seeds": len(rows),
                    "seeds": [x["seed"] for x in rows],
                    "run_ids": [x["run_id"] for x in rows],
                    f"{primary_key}_mean": round(float(np.mean(vals)), 4 if primary_key == "val_bleu4" else 2),
                    f"{primary_key}_std": round(float(np.std(vals, ddof=1)), 4 if primary_key == "val_bleu4" else 2),
                    f"{secondary_key}_mean": round(float(np.mean(secs)), 4 if secondary_key == "val_rouge_l" else 2),
                    f"{secondary_key}_std": round(float(np.std(secs, ddof=1)), 4 if secondary_key == "val_rouge_l" else 2),
                }
        return out_ms

    multiseed_det = group_multiseed(det_table, "val_change_miou", "val_binary_f1")
    multiseed_cap = group_multiseed(cap_table, "val_bleu4", "val_rouge_l")

    agg = {
        "detector_runs_count": len(det_table),
        "detector_ranking": det_table,
        "detector_multiseed": multiseed_det,
        "captioner_runs_count": len(cap_table),
        "captioner_ranking": cap_table,
        "captioner_multiseed": multiseed_cap,
        "released_rscama_val": json.loads(rel_cap_val.read_text(encoding="utf-8")) if rel_cap_val.is_file() else None,
        "released_rscama_test": json.loads(rel_cap_test.read_text(encoding="utf-8")) if rel_cap_test.is_file() else None,
        "robustness_and_transfer": json.loads(rob_path.read_text(encoding="utf-8")) if rob_path.is_file() else None,
        "robustness_and_transfer_siam": json.loads(rob_siam_path.read_text(encoding="utf-8")) if rob_siam_path.is_file() else None,
        "verification": json.loads(ver_path.read_text(encoding="utf-8")) if ver_path.is_file() else None,
    }
    out.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    lines = ["# TerraWatch Milestone 4 — Measured Experiment Tables\n"]
    lines.append("## 1. Full-Data Detector Experiments (6,444 Train / 1,242 Val LEVIR-MCI)\n")
    lines.append("| Run ID | Arch | Head LR | WD | Eff Batch | Sched | Loss | Freeze | Aug | Best Ep | Chg mIoU (%) | Road IoU (%) | Bldg IoU (%) | Bin F1 (%) | FA@50px (%) | SHA-256 (12) |")
    lines.append("|---|---|---:|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for r in det_table:
        lines.append(
            f"| `{r['run_id']}` | {r['arch']} | {r['head_lr']} | {r['wd']} | {r['eff_batch']} | {r['scheduler']} | {r['loss']} | {r['freeze_epochs']} | {r['aug']} | {r['best_epoch']} | **{r['val_change_miou']:.2f}** | {r['val_road_iou']:.2f} | {r['val_building_iou']:.2f} | {r['val_binary_f1']:.2f} | {r['fa_50px_pct']:.2f} | `{r['ckpt_sha256'][:12]}` |"
        )

    if cap_table:
        lines.append("\n## 2. Full-Data Captioner Experiments (6,444 Train Pairs / 32,220 Captions, Train-Only Vocab)\n")
        lines.append("| Run ID | LR | Opt | Sched | LS | Freeze CLIP | Batch | Vocab | Best Ep | Val BLEU-4 | Val ROUGE-L | Chg BLEU-4 | Val CE | NC FA (%) | SHA-256 (12) |")
        lines.append("|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in cap_table:
            lines.append(
                f"| `{r['run_id']}` | {r['lr']} | {r['optimizer']} | {r['scheduler']} | {r['label_smoothing']} | {r['freeze_encoder']} | {r['batch_size']} | {r['vocab_size']} | {r['best_epoch']} | **{r['val_bleu4']:.4f}** | {r['val_rouge_l']:.4f} | {r['val_changed_bleu4']:.4f} | {r['val_ce_loss']:.4f} | {r['nc_fa_rate_pct']:.2f} | `{r['ckpt_sha256'][:12]}` |"
            )

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AGGREGATED: {len(det_table)} detector runs, {len(cap_table)} captioner runs -> {out}")


if __name__ == "__main__":
    main()
