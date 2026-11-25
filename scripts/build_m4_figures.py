"""Generate publication-quality Matplotlib figures for TerraWatch Milestone 4
from committed run receipts under docs/milestone4/evidence/full_data_m4/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

NAVY = "#102630"
TEAL = "#1f6f78"
LIME = "#7aa82a"
AMBER = "#d9822b"
CORAL = "#c94040"
SLATE = "#5c6f78"
BG = "#fafbf8"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": BG,
    "axes.edgecolor": "#c8d3cf",
    "axes.grid": True,
    "grid.color": "#e2e8e5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_runs(root: Path) -> Dict[str, dict]:
    runs = {}
    if not root.exists():
        return runs
    for sub in sorted(root.iterdir()):
        rj = sub / "run.json"
        hj = sub / "history.jsonl"
        if rj.is_file() and hj.is_file():
            meta = json.loads(rj.read_text(encoding="utf-8"))
            meta["_history"] = load_jsonl(hj)
            meta["_dir"] = str(sub)
            runs[sub.name] = meta
    return runs


def plot_detector_curves(det_runs: Dict[str, dict], out_dir: Path) -> None:
    if not det_runs:
        return
    key_arms = [
        ("D01_bit_ref", "BIT Ref (LR 1e-4, Cosine)", NAVY),
        ("D02_bit_lr3e5", "BIT LR 3e-5", SLATE),
        ("D03_bit_lr3e4", "BIT LR 3e-4", TEAL),
        ("D12_bit_unfreeze0", "BIT Unfreeze Ep 0", LIME),
        ("D13_bit_frozen", "BIT Frozen Backbone", CORAL),
        ("D15_bit_photo", "BIT + Photometric Aug", AMBER),
    ]
    present = [(k, lbl, col) for k, lbl, col in key_arms if k in det_runs]
    if not present:
        present = [(k, k, NAVY) for k in list(det_runs.keys())[:6]]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=180)
    for k, lbl, col in present:
        hist = det_runs[k]["_history"]
        eps = [h["epoch"] + 1 for h in hist]
        axes[0].plot(eps, [h["train_loss"] for h in hist], label=lbl, color=col, lw=1.8)
        axes[1].plot(eps, [h["val_loss"] for h in hist], label=lbl, color=col, lw=1.8)
        axes[2].plot(eps, [h["val_change_miou"] * 100 for h in hist], label=lbl, color=col, lw=2.0)

    axes[0].set_title("Training Loss (Full 6,444 MCI Pairs)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal + Soft Dice Loss")
    axes[1].set_title("Validation Loss (Full 1,242 MCI Pairs)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
    axes[2].set_title("Validation Change-Class mIoU (Selection Metric)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Road/Building Mean IoU (%)")
    axes[2].legend(fontsize=7.8, loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "m4_detector_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    items = sorted(
        [(k, r["best_val_change_miou"] * 100, r["reloaded_best_val_metrics"]["binary_f1"] * 100, r["config"]["arch"])
         for k, r in det_runs.items()],
        key=lambda t: t[1],
    )
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.36 * len(items))), dpi=180)
    names = [t[0] for t in items]
    mious = [t[1] for t in items]
    f1s = [t[2] for t in items]
    bar_colors = [TEAL if t[3] == "bit_resnet18" else AMBER for t in items]
    y = np.arange(len(items))
    ax.barh(y, mious, color=bar_colors, height=0.68, alpha=0.92, label="Val Change-Class mIoU (%)")
    ax.scatter(f1s, y, color=NAVY, zorder=5, s=28, label="Val Binary Change F1 (%)")
    for i, (m, f) in enumerate(zip(mious, f1s)):
        ax.text(m + 0.6, i, f"{m:.2f}% (F1 {f:.1f}%)", va="center", fontsize=8, color=NAVY)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Validation Score (%) — Full 1,242 LEVIR-MCI Validation Partition")
    ax.set_title("Detector Controlled Experiments & Architecture Comparisons (Teal=BIT, Amber=Baselines)")
    ax.set_xlim(0, min(100, max(max(f1s), max(mious)) + 16))
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "m4_detector_comparison_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_captioner_curves(cap_runs: Dict[str, dict], out_dir: Path) -> None:
    if not cap_runs:
        return
    palette = [NAVY, TEAL, LIME, AMBER, CORAL, SLATE, "#7b4ea3", "#2b8cbe"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=180)
    for idx, (k, r) in enumerate(sorted(cap_runs.items())):
        col = palette[idx % len(palette)]
        hist = r["_history"]
        eps = [h["epoch"] + 1 for h in hist]
        axes[0].plot(eps, [h["train_ce_loss"] for h in hist], label=k, color=col, lw=1.7)
        axes[1].plot(eps, [h["val_ce_loss"] for h in hist], label=k, color=col, lw=1.7)
        axes[2].plot(eps, [h["val_bleu4"] * 100 for h in hist], label=k, color=col, lw=1.9)
    axes[0].set_title("RSCaMa Train Token CE Loss (6,444 Pairs / 32,220 Refs)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("RSCaMa Validation Token CE Loss (1,242 Pairs)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation CE Loss")
    axes[2].set_title("Validation Corpus BLEU-4 (Selection Metric)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("BLEU-4 (%)")
    axes[2].legend(fontsize=7.5, loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "m4_captioner_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_and_fa(rob_path: Path, out_dir: Path) -> None:
    if not rob_path.is_file():
        return
    data = json.loads(rob_path.read_text(encoding="utf-8"))
    corr = data.get("levir_mci_val_corruptions", {})
    if not corr:
        return
    siam_path = rob_path.parent / "robustness_and_transfer_siam.json"
    siam_corr = json.loads(siam_path.read_text(encoding="utf-8")).get("levir_mci_val_corruptions", {}) if siam_path.is_file() else {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.5), dpi=180)
    names = list(corr.keys())
    mious_bit = [corr[n]["change_miou"] * 100 for n in names]
    x = np.arange(len(names))
    w = 0.38
    if siam_corr:
        mious_siam = [siam_corr[n]["change_miou"] * 100 for n in names]
        ax1.bar(x - w / 2, mious_bit, w, label="BIT-ResNet18 (D03) Change mIoU (%)", color=TEAL)
        ax1.bar(x + w / 2, mious_siam, w, label="Siam-ResNet18-UNet (A03) Change mIoU (%)", color=AMBER)
        ax1.set_title("Corruption Robustness: BIT vs Siam-UNet (1,242 Val Pairs)")
    else:
        f1s = [corr[n]["binary_f1"] * 100 for n in names]
        ax1.bar(x - w / 2, mious_bit, w, label="Change-Class mIoU (%)", color=TEAL)
        ax1.bar(x + w / 2, f1s, w, label="Binary Change F1 (%)", color=NAVY)
        ax1.set_title("Corruption Robustness Ladder (1,242 Val Pairs)")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=32, ha="right", fontsize=8)
    ax1.set_ylabel("Score (%)")
    ax1.legend(fontsize=8.0, loc="lower left")

    clean_fa = corr["clean"]["unchanged_fa_rates"]
    test_fa = (data.get("levir_mci_test") or {}).get("unchanged_fa_rates", {})
    ths = sorted([int(k) for k in clean_fa.keys()])
    ax2.plot(ths, [clean_fa[str(t)] * 100 for t in ths], "o-", color=TEAL, lw=2.0, label="Val Unchanged FA Rate")
    if test_fa:
        ax2.plot(ths, [test_fa[str(t)] * 100 for t in ths], "s--", color=CORAL, lw=2.0, label="Test Unchanged FA Rate")
    ax2.set_xlabel("Minimum Changed-Pixel Operating Threshold")
    ax2.set_ylabel("Unchanged-Pair False Alarm Rate (%)")
    ax2.set_title("Unchanged-Scene False Alarm Operating Curve")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "m4_robustness_and_false_alarms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_verification_comparison(ver_path: Path, out_dir: Path) -> None:
    if not ver_path.is_file():
        return
    data = json.loads(ver_path.read_text(encoding="utf-8"))
    comp = data["settings_comparison"]
    sweep = data["threshold_sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.4), dpi=180)
    labels = [
        ("1_unverified_captioner", "Unverified\nCaptioner"),
        ("2_mask_only_checker", "Mask-Only\nChecker"),
        ("4_direct_qwen3vl_gated", "Direct\nQwen3-VL"),
        ("5_reinspection_all_gated", "4-View\nReinspection"),
        ("6_disagreement_triggered_reinspection", "Triggered\nReinspection"),
    ]
    present = [(k, l) for k, l in labels if k in comp]
    x = np.arange(len(present))
    prec = [comp[k]["proxy_supported_precision_strict"] * 100 for k, _ in present]
    rec = [comp[k]["proxy_supported_recall"] * 100 for k, _ in present]
    acc = [comp[k]["accept_rate"] * 100 for k, _ in present]
    w = 0.26
    ax1.bar(x - w, prec, w, label="Supported Precision (%)", color=NAVY)
    ax1.bar(x, rec, w, label="Supported Recall (%)", color=TEAL)
    ax1.bar(x + w, acc, w, label="Accept Coverage (%)", color=LIME)
    ax1.set_xticks(x); ax1.set_xticklabels([l for _, l in present], fontsize=8.3)
    ax1.set_ylabel("Percentage (%)"); ax1.set_ylim(0, 108)
    ax1.set_title("Checking Mechanism Comparison (Fixed Masks & Claims)")
    ax1.legend(fontsize=8, loc="lower right")

    ths = sorted(sweep.keys(), key=float)
    for key, lbl, col, mk in [
        ("direct_qwen_gated", "Direct Qwen3-VL", SLATE, "o-"),
        ("reinsp_all_gated", "4-View Reinspection (All)", TEAL, "s-"),
        ("triggered_reinsp", "Disagreement-Triggered", NAVY, "^-"),
    ]:
        cov = [sweep[t][key]["accept_rate"] * 100 for t in ths]
        risk = [sweep[t][key]["proxy_selective_risk"] * 100 for t in ths]
        ax2.plot(cov, risk, mk, label=lbl, color=col, lw=2.0)
        for t, cx, ry in zip(ths, cov, risk):
            ax2.annotate(t, (cx, ry), textcoords="offset points", xytext=(3, 4), fontsize=7, color=col)
    ax2.set_xlabel("Acceptance Coverage (%)")
    ax2.set_ylabel("Selective Risk / Error Among Accepted (%)")
    ax2.set_title("Selective Risk vs Coverage Across Gate Thresholds")
    ax2.legend(fontsize=8.2)
    fig.tight_layout()
    fig.savefig(out_dir / "m4_verification_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_e2e_trace_sheet(traces_dir: Path, out_dir: Path) -> None:
    idx_path = traces_dir / "trace_index.json"
    if not idx_path.is_file():
        return
    all_idx = json.loads(idx_path.read_text(encoding="utf-8"))
    preferred = ["accepted_building_change", "accepted_road_or_mixed", "rejected_claim", "deferred_exact_count"]
    by_cat = {x.get("category"): x for x in all_idx}
    idx = [by_cat[c] for c in preferred if c in by_cat]
    if len(idx) < 4:
        idx = all_idx[:4]
    if not idx:
        return
    fig, axes = plt.subplots(len(idx), 5, figsize=(13.5, 2.85 * len(idx)), dpi=170)
    if len(idx) == 1:
        axes = [axes]
    col_titles = ["T1 (Earlier)", "T2 (Later)", "BIT Mask + Box", "Zoomed Crop T1", "Zoomed Crop T2"]
    for r_i, item in enumerate(idx):
        folder = traces_dir / item["folder"]
        rep = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        c0 = rep["claims"][0] if rep["claims"] else None
        imgs = [
            Image.open(folder / "T1.png"),
            Image.open(folder / "T2.png"),
            Image.open(folder / "mask_overlay.png") if (folder / "mask_overlay.png").exists() else Image.open(folder / "detector_response.mask.png"),
            Image.open(folder / c0["evidence"]["crop_paths"][0]) if c0 and c0["evidence"]["crop_paths"] else Image.open(folder / "T1.png"),
            Image.open(folder / c0["evidence"]["crop_paths"][1]) if c0 and len(c0["evidence"]["crop_paths"]) > 1 else Image.open(folder / "T2.png"),
        ]
        outcome = c0["decision"]["outcome"].upper() if c0 else "N/A"
        for c_i, im in enumerate(imgs):
            ax = axes[r_i][c_i]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=9.5)
            if c_i == 0:
                ax.set_ylabel(f"{item['pair_id'].split('/')[-1]}\n[{outcome}]", fontsize=8.5, fontweight="bold")
        cap = rep["generated_caption"]
        axes[r_i][2].set_xlabel(f"Caption: \"{cap[:64]}{'...' if len(cap) > 64 else ''}\"", fontsize=7.8)
    fig.suptitle("Authentic Full-Data End-to-End Pipeline Traces (Selected BIT + Selected RSCaMa + Qwen3-VL)", fontsize=11.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "m4_e2e_traces.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Build Milestone 4 Matplotlib figures from committed evidence receipts.")
    ap.add_argument("--evidence-dir", type=Path, default=repo / "docs/milestone4/evidence/full_data_m4")
    ap.add_argument("--output-dir", type=Path, default=repo / "reports/figures")
    args = ap.parse_args()

    ev = args.evidence_dir
    figs = args.output_dir
    figs.mkdir(parents=True, exist_ok=True)

    det_runs = load_runs(ev / "detector_runs")
    cap_runs = load_runs(ev / "captioner_runs")
    rob_path = ev / "robustness_and_transfer_bit.json"
    if not rob_path.is_file():
        rob_path = ev / "robustness_and_transfer.json"

    plot_detector_curves(det_runs, figs)
    plot_captioner_curves(cap_runs, figs)
    plot_robustness_and_fa(rob_path, figs)
    plot_verification_comparison(ev / "verification/verification_comparison_summary.json", figs)
    plot_e2e_trace_sheet(ev / "verification/e2e_traces", figs)
    print("GENERATED_FIGURES:", [p.name for p in sorted(figs.glob("m4_*.png"))])


if __name__ == "__main__":
    main()
