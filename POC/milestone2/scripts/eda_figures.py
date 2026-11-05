"""Render report figures and detailed EDA from the executed full-release audit."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BLUE, NAVY, GRAY = "#1F5FAE", "#12202B", "#9DA7AF"


def main():
    out = ROOT / "evidence/full_audit"
    summary = json.loads((out / "summary.json").read_text())
    rows = list(csv.DictReader((out / "pair_profile.csv").open(newline="", encoding="utf-8")))
    assets = ROOT / "assets"
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold"})
    s = summary["source"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.8), layout="constrained")
    keys = ["unchanged", "building_only", "road_only", "both"]
    values = [s["mask_classes"].get(k, 0) for k in keys]
    bars = axes[0].barh(["No labelled change", "Building only", "Road only", "Both"], values, color=[GRAY, BLUE, "#526A86", NAVY])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(values) * 1.24)
    axes[0].set_xlabel("Image pairs")
    axes[0].set_title(f"All {s['pairs']:,} source pairs")
    for b, v in zip(bars, values):
        axes[0].text(v + max(values) * .025, b.get_y() + b.get_height()/2, f"{v:,}", va="center")
    percentages = [s["pixel_percent"][k] for k in ["background", "building", "road"]]
    bars = axes[1].barh(["Background", "Building change", "Road change"], percentages, color=[GRAY, BLUE, "#526A86"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Share of source pixels (%)")
    axes[1].set_title("Pixel-class distribution")
    for b, v in zip(bars, percentages):
        axes[1].text(v + 1, b.get_y() + b.get_height()/2, f"{v:.2f}%", va="center")
    fig.savefig(assets / "eda_classes.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.7), layout="constrained")
    positive = np.array([float(r["changed_fraction"]) * 100 for r in rows if float(r["changed_fraction"]) > 0])
    axes[0].hist(positive, bins=np.arange(0, 101, 5), color=BLUE, edgecolor="white", linewidth=.45)
    axes[0].set_xlabel("Changed pixels per changed pair (%)")
    axes[0].set_ylabel("Image pairs")
    axes[0].set_title("Change-area distribution")
    grid = sum(np.array(json.loads(r["grid_changed_px"])) for r in rows)
    # Grid cells differ by one pixel; normalise each by its exact area first.
    dims = np.array([len(x) for x in np.array_split(np.arange(256), 3)])
    areas = dims[:, None] * dims[None, :]
    density = grid / (len(rows) * areas) * 100
    axes[1].imshow(density, cmap="Blues", vmin=0, vmax=max(15, density.max()))
    axes[1].set_xticks([0, 1, 2], ["Left", "Centre", "Right"])
    axes[1].set_yticks([0, 1, 2], ["Top", "Middle", "Bottom"])
    axes[1].set_title("Changed-pixel density by image region")
    for y in range(3):
        for x in range(3):
            axes[1].text(x, y, f"{density[y,x]:.1f}%", ha="center", va="center", color="white" if density[y,x] > 9 else NAVY)
    fig.savefig(assets / "eda_geometry.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), layout="constrained")
    axes[0,0].hist([float(r["brightness_delta"]) for r in rows], bins=40, color=BLUE, edgecolor="white")
    axes[0,0].set(title="Brightness differences", xlabel="T2 minus T1 mean luminance (0-255)", ylabel="Image pairs")
    axes[0,1].scatter([float(r["changed_fraction"]) * 100 for r in rows], [float(r["pair_mae"]) for r in rows], s=3, alpha=.15, c=BLUE, rasterized=True)
    axes[0,1].set(title="Pixel difference and annotated change", xlabel="Changed-pixel share (%)", ylabel="Mean absolute RGB difference (0-255)")
    axes[1,0].hist([float(r["caption_words_mean"]) for r in rows], bins=25, color=BLUE, edgecolor="white")
    axes[1,0].set(title="Caption length per pair", xlabel="Mean whitespace tokens across five references", ylabel="Image pairs")
    corr = np.array(summary["pearson_correlation"])
    axes[1,1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["Change area", "Brightness delta", "RGB difference", "Caption length"]
    axes[1,1].set_xticks(range(4), labels, rotation=35, ha="right", fontsize=8)
    axes[1,1].set_yticks(range(4), labels, fontsize=8)
    axes[1,1].set_title("Pearson correlations (descriptive)")
    for y in range(4):
        for x in range(4):
            axes[1,1].text(x,y,f"{corr[y,x]:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(corr[y,x])>.65 else NAVY)
    fig.savefig(assets / "eda_diagnostics.png", dpi=220)
    plt.close(fig)
    (assets / "figure_data.json").write_text(json.dumps({"source": "Executed full-release audit", "source_pairs": s["pairs"], "mask_classes": s["mask_classes"], "pixel_percent": s["pixel_percent"], "grid_density_percent": density.tolist()}, indent=2), encoding="utf-8")
    lines = ["# Milestone 2 - Detailed EDA", "", "Executed over every LEVIR-MCI source pair. This document records dataset preparation, not model performance or new human factuality annotations.", "", "## Coverage and source integrity", "",
             f"- Source records: {s['pairs']:,}; usable input representations: {s['valid_pairs']:,}.",
             f"- Missing archive members: {len(summary['missing_archive_members'])}; duplicate caption keys: {summary['duplicate_caption_keys']}.",
             f"- Archive SHA-256: `{summary['archive_sha256']}`.",
             f"- Release revision: `{summary['revision']}`.",
             "- All test pixels were examined for dataset quality, not model selection. The new preparation does not describe them as unopened.", "",
             "## Source and retained populations", "", "| Population | Train | Validation | Test | Total |", "|---|---:|---:|---:|---:|"]
    for key, title in [("source", "Source release"), ("prepared", "Prepared joint-task records")]:
        by = summary[f"{key}_by_split"]
        lines.append(f"| {title} | {by['train']['pairs']:,} | {by['val']['pairs']:,} | {by['test']['pairs']:,} | {summary[key]['pairs']:,} |")
    lines += ["", "## Class balance", "", "![Source class balance](assets/eda_classes.png)", "", "| Population | Unchanged | Building only | Road only | Both |", "|---|---:|---:|---:|---:|"]
    for key in ["source", "prepared"]:
        c = summary[key]["mask_classes"]
        lines.append(f"| {key.title()} | " + " | ".join(str(c.get(k,0)) for k in keys) + " |")
    lines += ["", "## Conflicts, duplicates and exclusions", "",
              f"- Exact repeated unordered image-pair groups: {summary['exact_duplicate_groups']}; redundant records: {summary['redundant_pair_records']}; groups spanning official splits: {summary['cross_split_exact_groups']}.",
              f"- Cross-split dHash candidates: {summary['cross_split_near_candidates']}; nonexact candidates: {summary['cross_split_near_candidates_nonexact']}. These are similarity candidates, not confirmed scene duplicates.",
              f"- Unique records excluded from the strict joint-task package: {summary['exclusions_unique_pairs']:,}. Reason counts below overlap.", ""]
    for k,v in summary["exclusion_reason_counts_nonexclusive"].items():
        lines.append(f"- {k}: {v:,}.")
    lines += ["", "Exact duplicates retain test before validation before train, then lexical pair ID. Nonexact cross-split perceptual matches exclude the lower-priority partition record conservatively. Reference/mask conflicts are quarantined without rewriting source captions or masks. This is a strict training cohort, not a claim that every excluded annotation is objectively wrong. All raw records remain recoverable. Scene-level geographic independence cannot be certified without parent-scene coordinates.", "",
              "## Geometry, small changes and directions", "", "![Change area and spatial distribution](assets/eda_geometry.png)", "",
              f"- Source changed pairs below 1% changed pixels: {s['small_changed_pairs_under_1_percent']:,}.",
              f"- Positive changed-area quantiles: `{json.dumps(s['changed_fraction_positive_quantiles'])}` (fractions).",
              f"- Pairs with at least one location/direction term in a reference: {s['direction_reference_pairs']:,}.",
              f"- Pairs with at least one numeric/count term: {s['numeric_reference_pairs']:,}. These lexical flags do not establish verified instance counts.",
              "- Grid labels refer to image coordinates, not geographic north/east. Counts of connected mask components are shape diagnostics, not building-instance labels.", "",
              "## Captions and reference agreement", "",
              f"- Mixed lexical change/no-change reference sets: {s['mixed_reference_change_flags']:,}.",
              f"- Source changeflag versus nonzero mask conflicts: {s['mask_caption_conflicts']:,}.",
              f"- Unchanged-reference template sets: {summary['unchanged_reference_template_sets']:,}; most common set occurs on {summary['most_common_unchanged_template_pairs']:,} pairs.",
              "- Retaining five references preserves alternative wording. A lexical consistency check does not certify semantic agreement on action, location or count. No new independent human adjudication is claimed.", "",
              "## Image variation, outliers and correlations", "", "![Photometric and text diagnostics](assets/eda_diagnostics.png)", "",
              "| Feature | IQR lower fence | IQR upper fence | Flagged pairs |", "|---|---:|---:|---:|"]
    for k,v in summary["iqr_outliers"].items():
        lines.append(f"| {k} | {v['lower_fence']:.4f} | {v['upper_fence']:.4f} | {v['flagged_pairs']:,} |")
    lines += ["", "Outliers are retained unless another integrity rule excludes the record: a large true change can be a valid tail observation. Brightness differences and pixel MAE are descriptive, not proof of misregistration or physical change. Correlations are associations and are not causal or performance claims.", "",
              "## Threshold sensitivity", "", "The 50-pixel threshold belongs to the earlier automatic caption evaluator; it is not a dataset label rule and is not used to erase small regions in this preparation.", "", "| Minimum class pixels | Building-present pairs | Road-present pairs |", "|---|---:|---:|"]
    for t,v in s["threshold_sensitivity"].items():
        lines.append(f"| {t} | {v['building']:,} | {v['road']:,} |")
    lines += ["", "## Reproduce and inspect", "", "Run `scripts/fetch_primary.py`, `scripts/prepare_and_audit.py`, `scripts/prepared_dataset.py`, then `scripts/eda_figures.py`. Exact inputs, hashes and settings are in `evidence/full_audit/summary.json`; per-pair values, near-pair candidates, exact groups and exclusion reasons are alongside it. Training data stays in `data/` and is not redistributed through Git."]
    probe_path = ROOT / "evidence/second_probe_summary.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        counts = probe["original_split_sizes"]
        lines += ["", "## Separate SECOND-CC transfer preparation", "",
                  f"The author-linked source index contains {probe['augmented_and_original_records']:,} original/augmented records. Excluding filenames marked `_random_augment` leaves {sum(counts.values()):,} originals: {counts['train']:,} train, {counts['val']:,} validation and {counts['test']:,} test.",
                  f"The original-pair archive signatures contain {probe['cross_split_crc_candidate_groups_originals']} cross-split CRC/size candidate groups. These are not independently confirmed decoded-pixel duplicates. This source is used only for transfer evaluation, without project training on its partitions.", "",
                  f"The deterministic {probe['selected_original_val_pairs']}-pair original-validation probe contains compatible 256 x 256 RGB images, semantic maps and five captions per pair. Its building-change projection has {probe['building_positive_at_one_pixel']} positive pairs at one pixel and {probe['building_positive_at_50_pixels']} at 50 pixels. {probe['source_changeflag_without_building']} source change-positive pairs have no building-change pixel; source captions can describe other classes, so this is not automatically an annotation error. No road-negative supervision is inferred from building-only masks.", "",
                  "Acquired source members pass ZIP CRC and length checks and receive SHA-256 hashes. Generated masks and the probe manifest also have hashes; references are separate. This is dataset preparation, not a new model experiment or human audit. Raw files stay local under `data/transfer/SECOND-CC/`; `evidence/second_probe_summary.json` records settings, selection and counts."]
    (ROOT / "EDA_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Generated three figures and EDA_Report.md")


if __name__ == "__main__":
    main()
