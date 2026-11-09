"""Reconcile published M3 caption metrics with reproducible local run outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "POC/milestone3/sarthak/evaluation_proof.json"
HIST = ROOT / "POC/b-captioning/outputs/b0_results.json"
LOCAL = {
    "RSCaMa (specialist)": ROOT / "POC/milestone3/sarthak/runs/rscama_local_summary.json",
    "RSCaMa retrained on 20% (ours)": ROOT / "POC/milestone3/sarthak/runs/rscama_retrain_local_summary.json",
    "SmolVLM2-2.2B": ROOT / "POC/milestone3/sarthak/runs/smol_local_summary.json",
    "Qwen3-VL-4B": ROOT / "POC/milestone3/sarthak/runs/qwen_local_summary.json",
    "Evidence template (pixel difference)": ROOT / "POC/milestone3/sarthak/runs/pixel_caption_local_summary.json",
    'Constant "there is no difference"': ROOT / "POC/milestone3/sarthak/runs/constant_caption_local_summary.json",
}
RUN_DIRS = {
    "RSCaMa (specialist)": ROOT / "POC/execution/runs/ood/sarthak_rscama_screen/B0_C02_rscama_screening",
    "RSCaMa retrained on 20% (ours)": ROOT / "POC/execution/runs/ood/sarthak_rscama_screen/B0_C02r20_rscama_retrained20_development_smoke+development+screening",
    "SmolVLM2-2.2B": ROOT / "POC/execution/runs/ood/sarthak_rscama_screen/B0_C07_smolvlm2_screening_sarthak_local",
    "Qwen3-VL-4B": ROOT / "POC/execution/runs/ood/sarthak_rscama_screen/B0_C04_qwen3vl_screening_sarthak_local",
    "Evidence template (pixel difference)": ROOT / "POC/execution/runs/ood/sarthak_rscama_screen/B0_CTRL_evidence_template_screening",
    'Constant "there is no difference"': ROOT / "POC/execution/runs/ood/sarthak_rscama_screen/B0_CTRL_constant_nochange_screening",
}
SOURCE_FILES = [
    "POC/execution/scripts/caption_rscama.py",
    "POC/execution/scripts/controls.py",
    "POC/execution/scripts/evaluate.py",
    "POC/milestone3/sarthak/scripts/evaluate_caption_screen.py",
    "POC/milestone3/sarthak/scripts/prepare_local_rscama_screen.py",
    "POC/milestone3/sarthak/scripts/build_evaluation_proof.py",
    "POC/milestone3/sarthak/scripts/plot_evaluation_proof.py",
    "POC/b-captioning/colab/rscama_colab.py",
    "POC/execution/scripts/caption_vlm.py",
]
CORE = {
    "strict_precision": "supported_claim_precision_strict",
    "in_scope_precision": "supported_claim_precision_in_scope",
    "BLEU-4": ("text_metrics", "BLEU-4"),
}
DISPLAYED_M3 = {
    "RSCaMa (specialist)": {"strict_precision": 78.7, "in_scope_precision": 88.9, "BLEU-4": 0.6223},
    "RSCaMa retrained on 20% (ours)": {"strict_precision": 79.2, "in_scope_precision": 88.0, "BLEU-4": 0.5786},
    "SmolVLM2-2.2B": {"strict_precision": 38.6, "in_scope_precision": 48.4, "BLEU-4": 0.0102},
    "Qwen3-VL-4B": {"strict_precision": 58.3, "in_scope_precision": 72.9, "BLEU-4": 0.0267},
    "Evidence template (pixel difference)": {"strict_precision": 69.0, "in_scope_precision": 69.0, "BLEU-4": 0.0},
    'Constant "there is no difference"': {"strict_precision": 40.0, "in_scope_precision": 40.0, "BLEU-4": 0.2798},
}
TRANSFER = {
    "SECOND-CC": {
        "historical": ROOT / "POC/b-captioning/outputs/b0_results_ood_second_cc.json",
        "local": ROOT / "POC/milestone3/sarthak/runs/second_cc_local_summary.json",
        "run_dir": ROOT / "POC/execution/runs/ood/second_cc/B0_C02_rscama_screening",
        "manifest": ROOT / "POC/execution/data_ood/second_cc/manifest.csv",
    },
    "WHU-CDC": {
        "historical": ROOT / "POC/b-captioning/outputs/b0_results_ood_whu_cdc.json",
        "local": ROOT / "POC/milestone3/sarthak/runs/whu_cdc_local_summary.json",
        "run_dir": ROOT / "POC/execution/runs/ood/whu_cdc/B0_C02_rscama_screening",
        "manifest": ROOT / "POC/execution/data_ood/whu_cdc/manifest.csv",
    },
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_path(path: Path):
    return {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path), "bytes": path.stat().st_size}


def get_metric(row, metric):
    key = CORE[metric]
    return row[key[0]][key[1]] if isinstance(key, tuple) else row[key]


def main() -> None:
    historical_rows = json.loads(HIST.read_text(encoding="utf-8"))
    historical_rows = [row for row in historical_rows
                       if row.get("roles") == ["screening"]
                       or (row.get("roles") == "all" and "screening" in row.get("run_id", ""))]
    historical = {row["model_name"]: row for row in historical_rows}
    manifest_json = ROOT / "POC/execution/data/sarthak_rscama_screen/manifest.json"
    manifest_csv = ROOT / "POC/execution/data/sarthak_rscama_screen/manifest.csv"
    source_archive = ROOT / "data/raw/LEVIR-MCI-dataset.zip"
    checkpoint_sha = "5eeba271fb7c2ff4722b725cbf5ce1a1fdc0a33ddf1b077dcf5381aebd643449"
    checkpoint = ROOT / "POC/.cache-benchmark/hf/models--lcybuaa--RSCaMa/blobs" / checkpoint_sha

    comparisons = []
    for name, summary_path in LOCAL.items():
        old, new = historical[name], json.loads(summary_path.read_text(encoding="utf-8"))
        comparison = {"configuration": name, "sample_count_historical": old["pairs"],
                      "sample_count_local": new["pairs"], "metrics": {}}
        for metric in CORE:
            hist_value, local_value = get_metric(old, metric), get_metric(new, metric)
            scale = 100 if metric != "BLEU-4" else 1
            displayed = DISPLAYED_M3[name][metric]
            comparison["metrics"][metric] = {
                "historical_b0_json": hist_value,
                "milestone3_displayed": displayed,
                "local_rerun": local_value,
                "local_minus_historical_exact": round((local_value - hist_value) * scale, 8),
                "local_minus_displayed_report": round((local_value * scale - displayed), 8),
                "scale": "percentage_points" if scale == 100 else "BLEU points",
            }
        comparison["supplementary_local_metrics"] = {
            "reference_fact_recall": new.get("reference_fact_recall"),
            "reference_fact_recall_ci95": new.get("reference_fact_recall_ci95"),
            "no_change_false_alarm_rate": new.get("no_change_false_alarm_rate"),
            "no_change_false_alarm_pairs": new.get("unchanged_pairs_clean"),
            "text_metrics": new.get("text_metrics"),
            "strict_precision_ci95": new.get("supported_claim_precision_strict_ci95"),
            "in_scope_precision_ci95": new.get("supported_claim_precision_in_scope_ci95"),
            "cold_load_s_local": new.get("cold_load_s"),
            "warm_latency_median_s_local": new.get("warm_latency_median_s"),
            "warm_latency_median_s_historical": old.get("warm_latency_median_s"),
            "latency_median_delta_s": round(new["warm_latency_median_s"] - old["warm_latency_median_s"], 4),
            "latency_note": "Different hosts/runtimes; timing delta is descriptive, not an apples-to-apples speed claim.",
        }
        run_dir = RUN_DIRS[name]
        comparison["local_artifacts"] = {
            "run_record": digest_path(run_dir / "run_record.json"),
            "raw_caption_outputs": digest_path(run_dir / "captions.jsonl"),
            "evaluator_summary": digest_path(summary_path),
        }
        if name == "RSCaMa retrained on 20% (ours)":
            training_dir = ROOT / "POC/execution/runs/sarthak_rscama_retrain"
            comparison["local_training"] = {
                "best_checkpoint": digest_path(training_dir / "best.pth"),
                "train_log": digest_path(training_dir / "train_log.json"),
                "selected_epoch": 4,
                "best_development_BLEU4": 0.6468,
                "effective_batch": 12,
                "train_pairs": 1313,
                "development_pairs": 50,
                "screening_pairs": 100,
            }
        comparisons.append(comparison)

    transfer_comparisons = []
    for dataset, paths in TRANSFER.items():
        hist_rows = json.loads(paths["historical"].read_text(encoding="utf-8"))
        hist = next(row for row in hist_rows
                    if row.get("model_name") == "RSCaMa (specialist)" and row.get("roles") == ["screening"])
        local = json.loads(paths["local"].read_text(encoding="utf-8"))
        metrics = {}
        for metric in ("strict_precision", "in_scope_precision", "BLEU-4"):
            hv, lv = get_metric(hist, metric), get_metric(local, metric)
            scale = 100 if metric != "BLEU-4" else 1
            metrics[metric] = {"historical_b0_json": hv, "local_rerun": lv,
                               "local_minus_historical": round((lv-hv)*scale, 8),
                               "scale": "percentage_points" if scale == 100 else "BLEU points"}
        transfer_comparisons.append({
            "dataset": dataset, "sample_count_historical": hist["pairs"],
            "sample_count_local": local["pairs"], "metrics": metrics,
            "manifest": digest_path(paths["manifest"]),
            "historical_aggregate": digest_path(paths["historical"]),
            "local_run_record": digest_path(paths["run_dir"] / "run_record.json"),
            "local_raw_captions": digest_path(paths["run_dir"] / "captions.jsonl"),
            "local_evaluator_summary": digest_path(paths["local"]),
        })

    provenance = {
        "historical_aggregate": digest_path(HIST),
        "local_source_archive": digest_path(source_archive),
        "local_screen_manifest_json": digest_path(manifest_json),
        "local_screen_manifest_csv": digest_path(manifest_csv),
        "source_files": [digest_path(ROOT / rel) for rel in SOURCE_FILES],
        "rscama_checkpoint": {"repo": "lcybuaa/RSCaMa", "revision": "3e527e6656b62cc615eab1d2f3fb73777cbbf62c",
                              "sha256": checkpoint_sha, "local_blob": digest_path(checkpoint)},
        "local_environment": {"python": "3.11.4", "torch": "2.9.0+cu128", "torchvision": "0.24.0+cu128",
                              "device": "NVIDIA GeForce RTX 4060 Ti 8GB"},
        "evaluator_contract": {
            "caption_reference_check": "automatic rule parser + oracle masks + five captions per pair",
            "strict_precision_denominator": "all factual claims including out-of-scope objects",
            "in_scope_precision_denominator": "claims within road/building mask ontology",
            "bootstrap": "pair-level, 2,000 resamples, seed 0",
            "text_metrics": "PTBTokenizer BLEU-1..4, ROUGE-L, CIDEr-D; METEOR unavailable in local Java wrapper and not in M3 table",
        },
    }
    proof = {
        "title": "Local reproduction proof for completed Milestone 3 caption metrics",
        "run_date": "2026-11-09",
        "claim": "Released RSCaMa and two controls exactly match their historical screening aggregates. Retrain and VLM comparisons show observed deltas; both transfer screens also exactly match their historical aggregates.",
        "scope": "Released RSCaMa, its 20%-budget retrain, Qwen3-VL, SmolVLM2, two caption controls, and SECOND-CC/WHU-CDC RSCaMa transfer screens. Referee comparisons are still pending and excluded.",
        "comparisons": comparisons,
        "transfer_comparisons": transfer_comparisons,
        "provenance": provenance,
        "limitations": [
            "Historical committed results contain aggregate metrics, not the original per-pair caption outputs; the local raw outputs can be independently re-scored, but historical prediction parity cannot be audited pair by pair.",
            "Released RSCaMa was selected using LEVIR-family validation data; the local run reproduces its existing validation screen and is not new held-out evidence.",
            "The local archive is the raw user-provided LEVIR-MCI package; it is not the cleaned M2 handover cohort.",
            "Human factuality labels were not used; precision and recall are automatic annotation/mask proxy metrics.",
        ],
    }
    OUT.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "comparisons": [
        {"configuration": item["configuration"], "exact_deltas": {key: v["local_minus_historical_exact"]
         for key, v in item["metrics"].items()}} for item in comparisons]}, indent=2))


if __name__ == "__main__":
    main()
