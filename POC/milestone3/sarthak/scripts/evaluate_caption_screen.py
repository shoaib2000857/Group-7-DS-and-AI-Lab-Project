"""Score the local RSCaMa rerun with the canonical claim and BLEU evaluator.

The repo's evaluator also requests METEOR, whose bundled Java wrapper does not
start correctly under this Windows runtime. This adapter preserves the same
PTB tokenizer, BLEU, ROUGE-L, CIDEr-D and claim scoring; METEOR is omitted because
it is not a Milestone 3 report metric.
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXEC_SCRIPTS = ROOT / "POC/execution/scripts"
sys.path.insert(0, str(EXEC_SCRIPTS))
import evaluate as evaluator
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def text_metrics(run_rows, refs):
    keys = [row["pair_id"] for row in run_rows]
    gts = {key: [{"caption": caption} for caption in refs[key]["captions"]] for key in keys}
    res = {row["pair_id"]: [{"caption": row["raw_text"] or ""}] for row in run_rows}
    tokenizer = PTBTokenizer()
    gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)
    out = {}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    out.update({f"BLEU-{i + 1}": round(value, 4) for i, value in enumerate(bleu)})
    out["ROUGE-L"] = round(Rouge().compute_score(gts, res)[0], 4)
    out["CIDEr-D"] = round(Cider().compute_score(gts, res)[0], 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "POC/execution/runs/ood/sarthak_rscama_screen/B0_C02_rscama_screening")
    parser.add_argument("--output", type=Path, default=ROOT / "POC/milestone3/sarthak/runs/rscama_local_summary.json")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    evaluator.text_metrics = text_metrics
    truth = evaluator.Truth()
    summary = evaluator.evaluate_run(run_dir, truth, {"screening"}, "_local_rerun")
    result_path = args.output if args.output.is_absolute() else ROOT / args.output
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = ("run_id", "pairs", "claims_total", "supported_claim_precision_strict",
              "supported_claim_precision_strict_ci95", "supported_claim_precision_in_scope",
              "supported_claim_precision_in_scope_ci95", "reference_fact_recall", "no_change_false_alarm_rate",
              "text_metrics", "completion_rate", "cold_load_s", "warm_latency_median_s", "warm_latency_p95_s")
    print(json.dumps({field: summary.get(field) for field in fields}, indent=2))


if __name__ == "__main__":
    main()
