"""Generate all Milestone 5 publication-quality Matplotlib figures (180 DPI) and summary tables
from completed M5 evidence receipts under POC/milestone5/evidence/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

NAVY = "#102630"
TEAL = "#1F6F78"
LIME = "#7AA82A"
AMBER = "#D9822B"
CORAL = "#C94040"
SLATE = "#5C6F78"
PURPLE = "#6C4AB6"
BG = "#FAFBF8"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": BG,
    "axes.edgecolor": "#C8D3CF",
    "axes.grid": True,
    "grid.color": "#E2E8E5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.family": "DejaVu Sans",
    "font.size": 9.3,
    "axes.titlesize": 10.8,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


def plot_roc_pr_curves(det_full: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=180)
    styles = [
        ("D03_bit_lr3e4", "BIT-ResNet18 (D03, Selected)", TEAL, "-", 2.3),
        ("A03_siam_r18_unet", "Siam-ResNet18-UNet (A03)", AMBER, "--", 2.0),
        ("A01_ef_unet", "EF-UNet (A01)", NAVY, "-.", 1.7),
        ("A02_fc_siam_diff", "FC-Siam-Diff (A02)", LIME, ":", 1.9),
        ("A04_changeformer", "ChangeFormer-B2 (A04)", PURPLE, "--", 1.7),
    ]
    for key, lbl, col, ls, lw in styles:
        m = det_full["models"].get(key)
        if not m or "roc_pr" not in m:
            continue
        rp = m["roc_pr"]
        ax1.plot(rp["roc_fpr"], rp["roc_tpr"], color=col, ls=ls, lw=lw, label=f"{lbl} (AUC={rp['roc_auc']:.4f})")
        ax2.plot(rp["pr_recall"], rp["pr_precision"], color=col, ls=ls, lw=lw, label=f"{lbl} (AP={rp['pr_auc']:.4f})")

    otsu = det_full.get("non_learning_controls", {}).get("otsu_diff", {}).get("roc_pr")
    if otsu:
        ax1.plot(otsu["roc_fpr"], otsu["roc_tpr"], color=CORAL, ls=":", lw=1.6, label=f"Otsu Pixel Diff (AUC={otsu['roc_auc']:.4f})")
        ax2.plot(otsu["pr_recall"], otsu["pr_precision"], color=CORAL, ls=":", lw=1.6, label=f"Otsu Pixel Diff (AP={otsu['pr_auc']:.4f})")

    ax1.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.0, ls="--")
    ax1.set_title("Pixel-Level ROC Curve (1,858 Held-Out Test Pairs)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=7.8, loc="lower right", framealpha=0.94)

    ax2.set_title("Pixel-Level Precision-Recall Curve (121.7M Test Pixels)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=7.8, loc="lower left", framealpha=0.94)

    fig.tight_layout()
    fig.savefig(out_dir / "m5_roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stratified_area(det_full: dict, out_dir: Path) -> None:
    buckets = ["tiny_1_250", "small_251_1000", "medium_1001_4000", "large_gt_4000"]
    blabels = ["Tiny\n(1–250 px)", "Small\n(251–1k px)", "Medium\n(1k–4k px)", "Large\n(>4k px)"]
    models = [
        ("D03_bit_lr3e4", "BIT-ResNet18 (D03)", TEAL),
        ("A03_siam_r18_unet", "Siam-R18-UNet (A03)", AMBER),
        ("A01_ef_unet", "EF-UNet (A01)", NAVY),
        ("A02_fc_siam_diff", "FC-Siam-Diff (A02)", LIME),
        ("A04_changeformer", "ChangeFormer (A04)", PURPLE),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=180)
    x = np.arange(len(buckets))
    w = 0.15
    for idx, (k, lbl, col) in enumerate(models):
        st = det_full["models"][k]["area_stratified"]
        vals = [st[b]["binary_f1"] * 100 for b in buckets]
        ax1.bar(x + (idx - 2) * w, vals, w, label=lbl, color=col, alpha=0.92)
    ax1.set_xticks(x)
    ax1.set_xticklabels(blabels)
    ax1.set_ylabel("Binary Change F1 (%)")
    ax1.set_ylim(0, 102)
    ax1.set_title("Held-Out Test Detector F1 Stratified by Change Area")
    ax1.legend(fontsize=7.8, loc="lower right")

    # Right panel: Unchanged-Scene False Alarm Operating Curve across all 5 models
    ths = [1, 16, 50, 100, 200, 500]
    for k, lbl, col in models:
        far = det_full["models"][k]["unchanged_fa_rates"]
        ax2.plot(ths, [far[str(t)] * 100 for t in ths], "o-", color=col, lw=1.9, markersize=4.5, label=lbl)
    ax2.set_xlabel("Minimum Changed-Pixel Alarm Threshold")
    ax2.set_ylabel("Unchanged-Pair False Alarm Rate (%)")
    ax2.set_title("Unchanged-Scene False Alarm Operating Curve (Test)")
    ax2.legend(fontsize=7.8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "m5_stratified_area_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(det_full: dict, ver_summary: dict | None, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=180)
    cm_norm = np.array(det_full["models"]["D03_bit_lr3e4"]["confusion_matrix_row_norm"]) * 100.0
    cm_raw = np.array(det_full["models"]["D03_bit_lr3e4"]["confusion_matrix"])
    classes = ["Background", "Road", "Building"]

    im = axes[0].imshow(cm_norm, cmap="YlGnBu", vmin=0, vmax=100)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(classes)
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels(classes)
    axes[0].set_xlabel("Predicted Class"); axes[0].set_ylabel("Ground Truth Class")
    axes[0].set_title("D03 Pixel Confusion Matrix (1,858 Test Pairs)")
    for i in range(3):
        for j in range(3):
            col = "white" if cm_norm[i, j] > 55 else NAVY
            axes[0].text(j, i, f"{cm_norm[i, j]:.2f}%\n({cm_raw[i, j]/1e6:.2f}M)", ha="center", va="center", color=col, fontsize=8.5, fontweight="bold")
    axes[0].grid(False)

    # Right panel: Hybrid Verifier outcome breakdown vs Proxy GT on Held-Out Test
    if ver_summary:
        bd = ver_summary["test_settings_comparison"]["7_m5_calibrated_mask_first_hybrid"]["breakdown"]
        tot_sup = ver_summary["audit_scope"]["proxy_gt_counts"].get("supported", 1)
        tot_con = ver_summary["audit_scope"]["proxy_gt_counts"].get("contradicted", 1)
        tot_unv = ver_summary["audit_scope"]["proxy_gt_counts"].get("unverifiable", 1)
        # Rows: Proxy GT (Supported, Contradicted, Unverifiable); Cols: Predicted (Accept, Reject, Defer)
        mat = np.zeros((3, 3), dtype=np.int64)
        mat[0, 0] = bd["accepted_supported"]
        mat[0, 1] = bd["rejected_supported"]
        mat[0, 2] = max(0, tot_sup - bd["accepted_supported"] - bd["rejected_supported"])
        mat[1, 0] = bd["accepted_contradicted"]
        mat[1, 1] = bd["rejected_contradicted"]
        mat[1, 2] = max(0, tot_con - bd["accepted_contradicted"] - bd["rejected_contradicted"])
        mat[2, 0] = bd["accepted_unverifiable"]
        mat[2, 1] = 0
        mat[2, 2] = max(0, tot_unv - bd["accepted_unverifiable"])

        mat_pct = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1) * 100.0
        axes[1].imshow(mat_pct, cmap="YlGnBu", vmin=0, vmax=100)
        axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["Accept", "Reject", "Defer"])
        axes[1].set_yticks(range(3)); axes[1].set_yticklabels(["Supported", "Contradicted", "Unverifiable"])
        axes[1].set_xlabel("Mask-First Hybrid Decision"); axes[1].set_ylabel("Reference Proxy Ground Truth")
        axes[1].set_title("M5 Mask-First Hybrid Verifier Matrix (150 Test Pairs)")
        for i in range(3):
            for j in range(3):
                col = "white" if mat_pct[i, j] > 55 else NAVY
                axes[1].text(j, i, f"{mat_pct[i, j]:.1f}%\n(n={mat[i, j]})", ha="center", va="center", color=col, fontsize=8.8, fontweight="bold")
        axes[1].grid(False)

    fig.tight_layout()
    fig.savefig(out_dir / "m5_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_captioner_semantics(cap_full: dict, out_dir: Path) -> None:
    order = [
        ("C06_rscama_frozen_clip", "C06 Frozen CLIP\n(Selected)"),
        ("C00_rscama_released", "C00 Released\n(Benchmark)"),
        ("C04_rscama_adamw_cos", "C04 AdamW\n(Unfrozen)"),
        ("C01_rscama_ref", "C01 Adam\n(Ref)"),
        ("bit_mask_template", "BIT Mask\nTemplate"),
        ("always_no_change", "Always\nNo-Change"),
    ]
    present = [(k, lbl) for k, lbl in order if k in cap_full["models"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.6), dpi=180)
    x = np.arange(len(present))
    w = 0.20

    b4 = [cap_full["models"][k]["bleu4"] for k, _ in present]
    rl = [cap_full["models"][k]["rouge_l"] for k, _ in present]
    met = [cap_full["models"][k]["meteor"] for k, _ in present]
    bf1 = [cap_full["models"][k]["bertscore"]["f1"] for k, _ in present]

    ax1.bar(x - 1.5 * w, b4, w, label="BLEU-4", color=NAVY)
    ax1.bar(x - 0.5 * w, rl, w, label="ROUGE-L", color=TEAL)
    ax1.bar(x + 0.5 * w, met, w, label="METEOR", color=LIME)
    ax1.bar(x + 1.5 * w, bf1, w, label="BERTScore-F1", color=AMBER)
    ax1.set_xticks(x); ax1.set_xticklabels([l for _, l in present], fontsize=8.2)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Score (0–1)")
    ax1.set_title("Held-Out Test Caption Quality Metrics (1,858 Pairs)")
    ax1.legend(fontsize=8.0, loc="lower left", ncol=2)

    nc_fa = [cap_full["models"][k]["unchanged_fa_rate"] * 100 for k, _ in present]
    miss = [cap_full["models"][k]["changed_miss_rate"] * 100 for k, _ in present]
    obj_h = [cap_full["models"][k]["object_hallucination_rate"] * 100 for k, _ in present]
    w2 = 0.25
    ax2.bar(x - w2, nc_fa, w2, label="Unchanged Hallucination (NC FA%)", color=CORAL)
    ax2.bar(x, miss, w2, label="Missed Change Rate (%)", color=SLATE)
    ax2.bar(x + w2, obj_h, w2, label="Object-Class Hallucination (%)", color=AMBER)
    ax2.set_xticks(x); ax2.set_xticklabels([l for _, l in present], fontsize=8.2)
    ax2.set_ylabel("Error Rate (%) — Lower Is Better")
    ax2.set_title("Hallucination & Miss Rates on Held-Out Test (1,858 Pairs)")
    ax2.legend(fontsize=8.0, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "m5_captioner_semantic_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_verifier(ver_summary: dict, out_dir: Path) -> None:
    comp = ver_summary["test_settings_comparison"]
    sweep = ver_summary["test_threshold_sweep"]
    cal = ver_summary["test_calibration"]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6), dpi=180)
    labels = [
        ("1_unverified_captioner", "Unverified"),
        ("3_witness_only_checker", "Witness\nOnly"),
        ("4_direct_qwen3vl_gated", "Direct\nQwen3-VL"),
        ("5_reinspection_all_gated", "4-View\nAll"),
        ("2_mask_only_checker", "BIT Mask\nOnly"),
        ("7_m5_calibrated_mask_first_hybrid", "M5 Mask-First\nHybrid"),
    ]
    present = [(k, l) for k, l in labels if k in comp]
    x = np.arange(len(present))
    w = 0.26
    axes[0].bar(x - w, [comp[k]["proxy_supported_precision_strict"] * 100 for k, _ in present], w, label="Precision (%)", color=NAVY)
    axes[0].bar(x, [comp[k]["proxy_supported_recall"] * 100 for k, _ in present], w, label="Recall (%)", color=TEAL)
    axes[0].bar(x + w, [comp[k]["proxy_contradicted_catch_rate"] * 100 for k, _ in present], w, label="Contra Catch (%)", color=LIME)
    axes[0].set_xticks(x); axes[0].set_xticklabels([l for _, l in present], fontsize=7.6)
    axes[0].set_ylim(0, 108); axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Held-Out Test Verifier Regimes (150 Pairs)")
    axes[0].legend(fontsize=7.4, loc="lower left")

    ths = sorted(sweep.keys(), key=float)
    for key, lbl, col, mk in [
        ("direct_qwen_gated", f"Direct Qwen (AURC={cal['direct_qwen_aurc']:.3f})", SLATE, "o-"),
        ("reinsp_all_gated", f"4-View All (AURC={cal['reinsp_all_aurc']:.3f})", AMBER, "s-"),
        ("m5_mask_first_hybrid", f"M5 Hybrid (AURC={cal['m5_hybrid_aurc']:.3f})", TEAL, "^-"),
    ]:
        cov = [sweep[t][key]["accept_rate"] * 100 for t in ths]
        risk = [sweep[t][key]["proxy_selective_risk"] * 100 for t in ths]
        axes[1].plot(cov, risk, mk, label=lbl, color=col, lw=2.0)
    axes[1].set_xlabel("Acceptance Coverage (%)")
    axes[1].set_ylabel("Selective Risk Among Accepted (%)")
    axes[1].set_title("Risk-Coverage Sweep Across Thresholds (Test)")
    axes[1].legend(fontsize=7.6, loc="upper left")

    # Reliability diagram
    axes[2].plot([0, 1], [0, 1], "--", color="#999999", lw=1.1, label="Perfect Calibration")
    for key, lbl, col, mk in [
        ("m5_hybrid", f"M5 Hybrid (ECE={cal['m5_hybrid']['ece']:.3f}, Brier={cal['m5_hybrid']['brier_score']:.3f})", TEAL, "o-"),
        ("direct_qwen", f"Direct Qwen (ECE={cal['direct_qwen']['ece']:.3f}, Brier={cal['direct_qwen']['brier_score']:.3f})", CORAL, "s--"),
    ]:
        bins = cal[key]["reliability_bins"]
        axes[2].plot([b["mean_conf"] for b in bins], [b["empirical_acc"] for b in bins], mk, color=col, lw=2.0, label=lbl)
    axes[2].set_xlabel("Mean Predicted Support Probability")
    axes[2].set_ylabel("Empirical Support Frequency")
    axes[2].set_title("Reliability / Calibration Diagram (Test)")
    axes[2].legend(fontsize=7.3, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "m5_hybrid_verifier_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_extended_robustness(stress_data: dict, cap_full: dict, out_dir: Path) -> None:
    conds = stress_data["conditions"]
    bit_r = stress_data["results"]["D03_bit_lr3e4"]
    siam_r = stress_data["results"]["A03_siam_r18_unet"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 4.9), dpi=180, gridspec_kw={"width_ratios": [2.2, 1.0]})
    x = np.arange(len(conds))
    w = 0.39
    b_vals = [bit_r[c]["change_miou"] * 100 for c in conds]
    s_vals = [siam_r[c]["change_miou"] * 100 for c in conds]
    ax1.bar(x - w / 2, b_vals, w, label="BIT-ResNet18 (D03, Selected)", color=TEAL)
    ax1.bar(x + w / 2, s_vals, w, label="Siam-ResNet18-UNet (A03)", color=AMBER)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds, rotation=38, ha="right", fontsize=7.8)
    ax1.set_ylabel("Held-Out Test Change-Class mIoU (%)")
    ax1.set_ylim(0, 88)
    ax1.set_title("17-Condition Held-Out Test Stress Suite (1,858 Pairs): BIT vs Siam-UNet")
    ax1.legend(fontsize=8.2, loc="upper right")

    # Right panel: Captioner 200-pair corruption probe
    probe = cap_full.get("robustness_probe_200", {})
    pnames = list(probe.keys())
    px = np.arange(len(pnames))
    ax2.bar(px - 0.18, [probe[k]["bleu4"] * 100 for k in pnames], 0.36, label="BLEU-4 (%)", color=NAVY)
    ax2.bar(px + 0.18, [probe[k]["rouge_l"] * 100 for k in pnames], 0.36, label="ROUGE-L (%)", color=LIME)
    ax2.set_xticks(px)
    ax2.set_xticklabels([n.replace("_", "\n") for n in pnames], fontsize=8.0)
    ax2.set_ylabel("Score (%)")
    ax2.set_ylim(0, 90)
    ax2.set_title("C06 Captioner Stress Probe (200 Test Pairs)")
    ax2.legend(fontsize=8.0, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_dir / "m5_extended_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_gallery(err_data: dict, cases_dir: Path, out_dir: Path) -> None:
    exported = err_data.get("exported_visual_cases", [])
    # Pick 6 diverse failure cases (2 FP, 2 FN, 1 road/bldg swap, 1 tiny miss)
    picks = []
    for cat, limit in [("false_positive_unchanged", 2), ("false_negative_missed", 2), ("road_building_swap", 1), ("tiny_structure_miss", 1)]:
        picks.extend([x for x in exported if x["category"] == cat][:limit])
    if not picks:
        return

    fig, axes = plt.subplots(len(picks), 5, figsize=(13.2, 2.35 * len(picks)), dpi=170)
    col_titles = ["T1 (Before)", "T2 (After)", "Ground Truth\n(Gold=Road, Red=Bldg)", "D03 Prediction", "Error Map\n(G=TP, R=FP, B=FN, Y=Swap)"]
    for ri, item in enumerate(picks):
        cdir = cases_dir / item["folder"]
        imgs = [Image.open(cdir / fn) for fn in ("T1.png", "T2.png", "gt_color.png", "pred_color.png", "error_map.png")]
        for ci, im in enumerate(imgs):
            ax = axes[ri][ci]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=8.8)
            if ci == 0:
                short_cat = item["category"].replace("_", " ").title()
                ax.set_ylabel(f"{item['pair_id'].split('/')[-1]}\n{short_cat}\nF1={item['binary_f1']*100:.1f}%", fontsize=7.6, fontweight="bold")
    fig.suptitle("Held-Out Test Detector Failure Case Gallery (D03_bit_lr3e4 — Color-Coded TP/FP/FN Error Maps)", fontsize=11.0, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "m5_error_taxonomy_gallery.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_test_e2e_traces(traces_dir: Path, out_dir: Path) -> None:
    idx_p = traces_dir / "trace_index.json"
    if not idx_p.is_file():
        return
    all_idx = json.loads(idx_p.read_text(encoding="utf-8"))
    preferred = ["test_accepted_building", "test_accepted_road", "test_rejected_hallucination", "test_deferred_out_of_ontology", "test_vlm_escalated_case"]
    by_cat = {x["category"]: x for x in all_idx}
    picks = [by_cat[c] for c in preferred if c in by_cat][:5]
    if not picks:
        picks = all_idx[:4]

    fig, axes = plt.subplots(len(picks), 5, figsize=(13.5, 2.55 * len(picks)), dpi=170)
    if len(picks) == 1:
        axes = [axes]
    titles = ["T1 (Before)", "T2 (After)", "BIT Mask + Box", "Zoom Crop T1", "Zoom Crop T2"]
    for ri, item in enumerate(picks):
        f = traces_dir / item["folder"]
        rep = json.loads((f / "report.json").read_text(encoding="utf-8"))
        c0 = rep["claims"][0]
        imgs = [
            Image.open(f / "T1.png"),
            Image.open(f / "T2.png"),
            Image.open(f / "mask_overlay.png"),
            Image.open(f / c0["evidence"]["crop_paths"][0]),
            Image.open(f / c0["evidence"]["crop_paths"][1]),
        ]
        out = c0["decision"]["outcome"].upper()
        vlm_tag = "VLM" if c0["decision"].get("vlm_triggered") else "MASK"
        for ci, im in enumerate(imgs):
            ax = axes[ri][ci]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(titles[ci], fontsize=9.2)
            if ci == 0:
                ax.set_ylabel(f"{item['pair_id'].split('/')[-1]}\n[{out} | {vlm_tag}]", fontsize=8.0, fontweight="bold")
        axes[ri][2].set_xlabel(f"\"{rep['generated_caption'][:64]}\"", fontsize=7.6)
    fig.suptitle("Authentic Held-Out Test End-to-End Traces (D03 BIT + C06 RSCaMa + M5 Mask-First Hybrid Verifier)", fontsize=11.0, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "m5_test_e2e_traces.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_second_cc_6class(sec_data: dict, out_dir: Path) -> None:
    d03 = sec_data["models"]["D03_bit_lr3e4"]["splits"]["test"]["per_semantic_class"]
    a03 = sec_data["models"]["A03_siam_r18_unet"]["splits"]["test"]["per_semantic_class"]
    cls_order = ["building", "non_vegetated_ground", "low_vegetation", "tree", "water", "playground"]
    cls_labels = ["Building\n(In-Ontology)", "Non-Veg\nGround", "Low\nVegetation", "Tree", "Water", "Playground"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=180)
    shares = [d03[c]["gt_share_of_changes_pct"] for c in cls_order]
    colors = [TEAL, AMBER, LIME, "#3E8E41", "#3B82C4", PURPLE]
    bars = ax1.barh(cls_labels[::-1], shares[::-1], color=colors[::-1], alpha=0.92)
    for b, v in zip(bars, shares[::-1]):
        ax1.text(v + 0.7, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.4, fontweight="bold", color=NAVY)
    ax1.set_xlim(0, 40)
    ax1.set_xlabel("Share of Ground-Truth Changed Pixels (%)")
    ax1.set_title("SECOND-CC Test (1,142 Pairs): 70.5% Non-Building Ontology")

    x = np.arange(len(cls_order))
    w = 0.36
    d_rec = [d03[c]["recall_by_any_pred_pct"] for c in cls_order]
    a_rec = [a03[c]["recall_by_any_pred_pct"] for c in cls_order]
    ax2.bar(x - w / 2, d_rec, w, label="BIT-ResNet18 (D03, F1=23.2%)", color=TEAL)
    ax2.bar(x + w / 2, a_rec, w, label="Siam-R18-UNet (A03, F1=17.7%)", color=AMBER)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cls_labels, fontsize=8.0)
    ax2.set_ylabel("Zero-Shot Pixel Recall (%)")
    ax2.set_ylim(0, 29)
    ax2.set_title("Zero-Shot Recall per Semantic Class (LEVIR-MCI -> SECOND-CC)")
    ax2.legend(fontsize=8.2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "m5_second_cc_6class.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compute_profile(comp_dir: Path, out_dir: Path) -> None:
    profiles = {}
    for label, fname in [
        ("RTX 4060 (CUDA)", "compute_profile_rtx4060.json"),
        ("Colab T4 (CUDA)", "compute_profile_colab_t4.json"),
        ("Apple M4 (MPS)", "compute_profile_mac_m4_mps.json"),
        ("Apple M4 (CPU)", "compute_profile_mac_m4_cpu.json"),
    ]:
        p = comp_dir / fname
        if p.is_file():
            profiles[label] = json.loads(p.read_text(encoding="utf-8"))
    if not profiles:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.7), dpi=180)
    # Left panel: Detector P50 latency across hardware platforms (log scale)
    det_keys = ["A01_ef_unet", "A02_fc_siam_diff", "A03_siam_r18_unet", "D03_bit_lr3e4", "A04_changeformer"]
    det_names = ["A01\nEF-UNet", "A02\nFC-Siam", "A03\nSiam-R18", "D03 BIT\n(Selected)", "A04\nChgFormer"]
    plat_colors = {"RTX 4060 (CUDA)": TEAL, "Colab T4 (CUDA)": NAVY, "Apple M4 (MPS)": AMBER, "Apple M4 (CPU)": CORAL}
    x = np.arange(len(det_keys))
    n_p = len(profiles)
    w = 0.76 / max(n_p, 1)
    for pi, (plbl, pdata) in enumerate(profiles.items()):
        vals = []
        for dk in det_keys:
            st = pdata.get("stages", {}).get(dk, {})
            b1 = st.get("batch1", {})
            vals.append(b1.get("p50_ms", st.get("latency_ms", np.nan)))
        ax1.bar(x + (pi - (n_p - 1) / 2) * w, vals, w, label=plbl, color=plat_colors.get(plbl, SLATE), alpha=0.92)
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(det_names, fontsize=8.3)
    ax1.set_ylabel("Batch-1 Median Latency (ms, log scale)")
    ax1.set_title("Cross-Platform Detector Latency (5 Architectures)")
    ax1.legend(fontsize=7.8, loc="upper left")

    # Right panel: End-to-End Stage Latency Breakdown on CUDA GPUs (D03 + C06 Ref vs Triton + Witness + Hybrid VLM)
    gpu_labels = []
    d03_ms, cap_ref_ms, cap_tri_ms, wit_ms, ref_ms = [], [], [], [], []
    for plbl, fname, ref_fname in [
        ("RTX 4060\n(8 GB)", "compute_profile_rtx4060.json", None),
        ("Colab T4\n(15 GB)", "compute_profile_colab_t4.json", "compute_profile_colab_t4_referee.json"),
    ]:
        p = comp_dir / fname
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        st = d.get("stages", {})
        gpu_labels.append(plbl)
        d03_ms.append(st.get("D03_bit_lr3e4", {}).get("batch1", {}).get("p50_ms", 0.0))
        c06 = st.get("C06_rscama_frozen_clip", {})
        cap_ref_ms.append(c06.get("ref_scan", {}).get("p50_ms", 0.0))
        cap_tri_ms.append(c06.get("triton_scan", {}).get("p50_ms", 0.0))
        wit_ms.append(st.get("witness_opencv", {}).get("p50_ms", 0.0))
        r_val = 0.0
        if ref_fname and (comp_dir / ref_fname).is_file():
            rd = json.loads((comp_dir / ref_fname).read_text(encoding="utf-8"))
            ref_st = rd.get("stages", {}).get("Qwen3-VL-4B-Instruct") or rd.get("stages", {}).get("qwen3_vl_4b_referee") or {}
            r_val = ref_st.get("full_pair_2img", {}).get("p50_ms", 0.0)
        ref_ms.append(r_val)

    if gpu_labels:
        gx = np.arange(len(gpu_labels))
        gw = 0.16
        ax2.bar(gx - 2 * gw, d03_ms, gw, label="D03 BIT Detector", color=TEAL)
        ax2.bar(gx - gw, wit_ms, gw, label="OpenCV Witness", color=LIME)
        ax2.bar(gx, cap_tri_ms, gw, label="C06 Captioner (Triton)", color=NAVY)
        ax2.bar(gx + gw, cap_ref_ms, gw, label="C06 Captioner (Ref Scan)", color=AMBER)
        if any(v > 0 for v in ref_ms):
            ax2.bar(gx + 2 * gw, ref_ms, gw, label="Qwen3-VL-4B Referee (1 call)", color=CORAL)
        ax2.set_xticks(gx); ax2.set_xticklabels(gpu_labels, fontsize=8.8)
        ax2.set_ylabel("Median Per-Call Latency (ms)")
        ax2.set_title("Pipeline Stage Latency & Triton Scan Acceleration")
        ax2.legend(fontsize=7.6, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "m5_compute_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    m5 = Path(__file__).resolve().parents[1]
    ev_det = m5 / "evidence/test_detector_and_stress"
    ev_cap = m5 / "evidence/test_captioner_and_verifier"
    ev_str = m5 / "evidence/stress_tests"
    ev_cmp = m5 / "evidence/compute_profile"
    figs = m5 / "submission/figures"
    figs.mkdir(parents=True, exist_ok=True)

    det_full = json.loads((ev_det / "detector_test_full.json").read_text(encoding="utf-8")) if (ev_det / "detector_test_full.json").is_file() else None
    stress = json.loads((ev_det / "stress_test_levir_mci_test.json").read_text(encoding="utf-8")) if (ev_det / "stress_test_levir_mci_test.json").is_file() else None
    err = json.loads((ev_det / "detector_error_mining.json").read_text(encoding="utf-8")) if (ev_det / "detector_error_mining.json").is_file() else None
    cap_full = json.loads((ev_cap / "captioner_test_full.json").read_text(encoding="utf-8")) if (ev_cap / "captioner_test_full.json").is_file() else None
    ver_sum = json.loads((ev_cap / "test_verification_summary.json").read_text(encoding="utf-8")) if (ev_cap / "test_verification_summary.json").is_file() else None
    sec_6 = json.loads((ev_str / "second_cc_6class_dissection.json").read_text(encoding="utf-8")) if (ev_str / "second_cc_6class_dissection.json").is_file() else None

    if det_full:
        plot_roc_pr_curves(det_full, figs)
        plot_stratified_area(det_full, figs)
        plot_confusion_matrices(det_full, ver_sum, figs)
    if cap_full:
        plot_captioner_semantics(cap_full, figs)
    if ver_sum:
        plot_hybrid_verifier(ver_sum, figs)
    if stress and cap_full:
        plot_extended_robustness(stress, cap_full, figs)
    if err:
        plot_error_gallery(err, ev_det / "detector_visual_cases", figs)
    if (ev_cap / "test_e2e_traces").exists():
        plot_test_e2e_traces(ev_cap / "test_e2e_traces", figs)
    if sec_6:
        plot_second_cc_6class(sec_6, figs)
    if ev_cmp.exists():
        plot_compute_profile(ev_cmp, figs)

    print("GENERATED_M5_FIGURES:", [p.name for p in sorted(figs.glob("m5_*.png"))])


if __name__ == "__main__":
    main()
