"""Create static evidence plots from evaluation_proof.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PROOF = ROOT / "POC/milestone3/sarthak/evaluation_proof.json"
OUT = ROOT / "POC/milestone3/sarthak/figures"
OUT.mkdir(parents=True, exist_ok=True)
proof = json.loads(PROOF.read_text(encoding="utf-8"))
rows = proof["comparisons"]
names = ["Released RSCaMa", "20% retrain", "SmolVLM2", "Qwen3-VL", "Pixel template", "No-change constant"]
colors = list(plt.get_cmap("tab10").colors)
metrics = ["strict_precision", "in_scope_precision", "BLEU-4"]
labels = ["Strict precision", "In-scope precision", "BLEU-4"]

# Plot 1: delta against displayed M3 table, with precision converted to percentage points.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
for ax, metric, label in zip(axes, metrics, labels):
    matched = {"RSCaMa (specialist)", "Evidence template (pixel difference)", 'Constant "there is no difference"'}
    rounding_rows = [r for r in rows if r["configuration"] in matched]
    rounding_names = [names[i] for i, r in enumerate(rows) if r["configuration"] in matched]
    vals = [r["metrics"][metric]["local_minus_displayed_report"] for r in rounding_rows]
    if metric != "BLEU-4":
        ax.barh(rounding_names, vals, color=[colors[i] for i, r in enumerate(rows) if r["configuration"] in matched], edgecolor="#233044", linewidth=.5)
        ax.set_xlim(-.018, .008)
        ax.set_xticks([-.01, 0], ["−0.01", "0"])
        unit = "percentage points"
    else:
        ax.barh(rounding_names, vals, color=[colors[i] for i, r in enumerate(rows) if r["configuration"] in matched], edgecolor="#233044", linewidth=.5)
        ax.set_xlim(-.001, .001)
        ax.set_xticks([-.001, 0, .001], ["−0.001", "0", "+0.001"])
        unit = "BLEU score"
    ax.axvline(0, color="#293849", linewidth=1)
    ax.set_title(label)
    ax.set_xlabel(f"Δ {unit}")
    ax.tick_params(axis="y", labelsize=8)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.2f}" if metric != "BLEU-4" else f"{v:+.4f}",
                    (v, i), xytext=(-5 if v < 0 else 5, 0), textcoords="offset points",
                    ha="right" if v < 0 else "left", va="center", fontsize=8)
    ax.grid(axis="x", alpha=.2)
fig.suptitle("Display rounding for the three exactly reproduced configurations (zoomed around zero)")
fig.savefig(OUT / "m3-displayed-rounding-delta.png", dpi=180)
plt.close(fig)

# Plot 2: exact raw historical aggregate vs local re-run; overlapping values are annotated.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
for ax, metric, label in zip(axes, metrics, labels):
    scale = 100 if metric != "BLEU-4" else 1
    values = [row["metrics"][metric][key] * scale for row in rows
              for key in ("historical_b0_json", "local_rerun")]
    margin = max((max(values) - min(values)) * .08, .02 if metric == "BLEU-4" else 1.0)
    bounds = (min(values) - margin, max(values) + margin)
    ax.plot(bounds, bounds, color="#7c8793", linestyle="--", linewidth=1, zorder=1)
    for i, row in enumerate(rows):
        old = row["metrics"][metric]["historical_b0_json"] * scale
        new = row["metrics"][metric]["local_rerun"] * scale
        ax.scatter(old, new, marker="o", s=65, color=colors[i], label=names[i], zorder=3)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_title(label)
    ax.grid(alpha=.2)
    ax.set_xlabel("Historical B0 JSON")
    ax.set_ylabel("Local re-run")
    ax.legend(frameon=False, fontsize=7, loc="best")
fig.suptitle("Local results lie on the 1:1 parity line (100 pairs per configuration)")
fig.savefig(OUT / "historical-local-exact-parity.png", dpi=180)
plt.close(fig)

# Plot 3: new local supplementary RSCaMa metrics not shown in the compact M3 table.
rscama = rows[0]["supplementary_local_metrics"]
tm = rscama["text_metrics"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
bleu_names = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]
bleu_vals = [tm[k] for k in bleu_names]
axes[0].bar(bleu_names, bleu_vals, color=["#3b82a0"] * 4 + ["#df9b35"])
axes[0].set_ylim(0, 1)
axes[0].set_title("Caption overlap metrics")
axes[0].set_ylabel("Score")
axes[0].grid(axis="y", alpha=.2)
for i, v in enumerate(bleu_vals): axes[0].text(i, v + .02, f"{v:.4f}", ha="center", fontsize=8)

precisions = np.array([.7869, .8889]) * 100
cilo = np.array([rscama["strict_precision_ci95"][0], rscama["in_scope_precision_ci95"][0]]) * 100
cihi = np.array([rscama["strict_precision_ci95"][1], rscama["in_scope_precision_ci95"][1]]) * 100
axes[1].errorbar(["Strict precision", "In-scope precision"], precisions,
                 yerr=[precisions-cilo, cihi-precisions], fmt="o", capsize=5,
                 color="#2c7fb8", ecolor="#6c8ebf")
axes[1].set_ylim(50, 100)
axes[1].set_title("Pair-bootstrap 95% intervals")
axes[1].set_ylabel("Precision (%)")
axes[1].grid(axis="y", alpha=.2)
for i, (v, lo, hi) in enumerate(zip(precisions, cilo, cihi)):
    axes[1].text(i, hi + 1, f"{v:.2f}% [{lo:.2f}, {hi:.2f}]", ha="center", fontsize=8)
fig.suptitle("Additional local RSCaMa evidence (not in the compact M3 table)")
fig.savefig(OUT / "rscama-supplementary-evidence.png", dpi=180)
plt.close(fig)

print("Created:", *(str(p) for p in sorted(OUT.glob("*.png"))), sep="\n")
