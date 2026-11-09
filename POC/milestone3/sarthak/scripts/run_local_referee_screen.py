"""Run Qwen3 text/VL referees on the frozen local 30-pair, 150-claim screen."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "POC/milestone3/sarthak/runs/referee_local_screen"
RUNTIME = ROOT / "POC/benchmarks/colab/benchmark_runtime.py"
REVISIONS = {"J05": "cdbee75f17c01a7cc42f958dc650907174af0554",
             "J06": "ebb281ec70b05090aa6165b016eac8ec08e71b17"}


def load_runtime():
    spec = importlib.util.spec_from_file_location("tw_benchmark_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / "POC/.cache-benchmark/hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tw = load_runtime()
    import numpy as np
    import torch

    config = tw.default_config()
    config.update(PROFILE="screen", N_DEV=10, N_VAL=30, MAX_IMAGE_SIDE=512, MAX_NEW_TOKENS=180,
                  MODEL_DEVICE_MAP="auto", MAX_GPU_MEMORY="6GiB", MAX_CPU_MEMORY="22GiB")
    pairs = json.loads((RUN / "inputs/pairs.json").read_text(encoding="utf-8"))
    annotations = json.loads((RUN / "references/dataset_annotations.json").read_text(encoding="utf-8"))
    if len(pairs) != 30:
        raise RuntimeError(f"Expected 30 frozen pairs; found {len(pairs)}")
    (RUN / "predictions").mkdir(parents=True, exist_ok=True)

    vl = tw.HFJudge("J06", REVISIONS["J06"], config)
    witnesses, witness_rows = {}, []
    for i, pair in enumerate(pairs, 1):
        started = time.perf_counter()
        torch.cuda.synchronize()
        try:
            observation, raw, tokens = vl.observe(pair["image_paths"])
            torch.cuda.synchronize()
            row = {"pair_id": pair["pair_id"], "model_id": "J06", "status": "ok",
                   "observation": observation, "raw_text": raw, "input_tokens": tokens,
                   "latency_s": round(time.perf_counter() - started, 4)}
            witnesses[pair["pair_id"]] = observation
        except Exception as exc:
            row = {"pair_id": pair["pair_id"], "model_id": "J06", "status": "error",
                   "error": repr(exc), "latency_s": round(time.perf_counter() - started, 4)}
        witness_rows.append(row)
        print(f"witness {i}/30 {pair['pair_id']} {row['status']} {row['latency_s']}s", flush=True)
    write_rows(RUN / "predictions/J06_witness.jsonl", witness_rows)
    for pair in pairs:
        pair["witness"] = witnesses.get(pair["pair_id"], {
            "source": "unavailable", "summary": None, "visible_change": None,
            "quality_flags": ["witness_failed_no_oracle_fallback"]})
    cases, refs = tw.make_real_cases(pairs, annotations, witnesses, RUN)
    proxy_cases = [case for case in cases if case["benchmark_group"] == "satellite_annotation_proxy"]
    proxy_refs = {ref["case_id"]: ref for ref in refs if ref["label_source"] == "annotation_proxy_not_human_factuality"}
    if len(proxy_cases) != 150 or len(proxy_refs) != 150:
        raise RuntimeError(f"Expected 150 proxy claims/targets; got {len(proxy_cases)}/{len(proxy_refs)}")

    all_predictions = {}
    for model_id, judge in (("J06", vl),):
        for track in ("common_evidence", "image_backed"):
            rows = []
            for i, case in enumerate(proxy_cases, 1):
                paths = case["image_paths"] if track == "image_backed" else []
                started = time.perf_counter()
                torch.cuda.synchronize()
                try:
                    outcome, probabilities, raw, semantics, usage = judge.decide(
                        tw.inference_state(case), paths, reverse=False)
                    torch.cuda.synchronize()
                    row = {"case_id": case["case_id"], "pair_id": case["pair_id"], "model_id": model_id,
                           "track": track, "status": "ok", "outcome": outcome,
                           "probabilities": probabilities, "probability_semantics": semantics,
                           "raw_output": raw, "usage": usage,
                           "latency_s": round(time.perf_counter() - started, 4)}
                except Exception as exc:
                    row = {"case_id": case["case_id"], "pair_id": case["pair_id"], "model_id": model_id,
                           "track": track, "status": "error", "error": repr(exc),
                           "latency_s": round(time.perf_counter() - started, 4)}
                rows.append(row)
                if i % 10 == 0 or i == len(proxy_cases):
                    print(f"{model_id} {track} {i}/150 {row['status']} {row['latency_s']}s", flush=True)
            write_rows(RUN / f"predictions/{model_id}_{track}.jsonl", rows)
            all_predictions[(model_id, track)] = rows

    del judge, vl
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    text = tw.HFJudge("J05", REVISIONS["J05"], config)
    rows = []
    for i, case in enumerate(proxy_cases, 1):
        started = time.perf_counter()
        torch.cuda.synchronize()
        try:
            outcome, probabilities, raw, semantics, usage = text.decide(
                tw.inference_state(case), [], reverse=False)
            torch.cuda.synchronize()
            row = {"case_id": case["case_id"], "pair_id": case["pair_id"], "model_id": "J05",
                   "track": "common_evidence", "status": "ok", "outcome": outcome,
                   "probabilities": probabilities, "probability_semantics": semantics,
                   "raw_output": raw, "usage": usage,
                   "latency_s": round(time.perf_counter() - started, 4)}
        except Exception as exc:
            row = {"case_id": case["case_id"], "pair_id": case["pair_id"], "model_id": "J05",
                   "track": "common_evidence", "status": "error", "error": repr(exc),
                   "latency_s": round(time.perf_counter() - started, 4)}
        rows.append(row)
        if i % 10 == 0 or i == len(proxy_cases):
            print(f"J05 common_evidence {i}/150 {row['status']} {row['latency_s']}s", flush=True)
    write_rows(RUN / "predictions/J05_common_evidence.jsonl", rows)
    all_predictions[("J05", "common_evidence")] = rows
    del text
    torch.cuda.empty_cache()

    target_by_id = {case["case_id"]: proxy_refs[case["case_id"]]["target"] for case in proxy_cases}
    summaries = {}
    rng = np.random.default_rng(42)
    for (model_id, track), prediction_rows in all_predictions.items():
        completed = [row for row in prediction_rows if row["status"] == "ok"]
        correct = [row for row in completed if row["outcome"] == target_by_id[row["case_id"]]]
        by_pair = {}
        for row in completed:
            by_pair.setdefault(row["pair_id"], []).append(row["outcome"] == target_by_id[row["case_id"]])
        pair_ids = sorted({case["pair_id"] for case in proxy_cases})
        rates = []
        for _ in range(2000):
            sample = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            vals = [value for pair_id in sample for value in by_pair.get(pair_id, [])]
            if vals:
                rates.append(float(np.mean(vals)))
        summaries[f"{model_id}:{track}"] = {
            "planned_proxy_claims": 150, "completed_proxy_claims": len(completed),
            "raw_decision_agreement": round(len(correct) / len(completed), 4) if completed else None,
            "agreement_ci95_pair_bootstrap": [round(float(np.quantile(rates, .025)), 4),
                                               round(float(np.quantile(rates, .975)), 4)] if rates else None,
            "median_latency_s": round(float(np.median([r["latency_s"] for r in completed])), 4) if completed else None,
            "p95_latency_s": round(float(np.quantile([r["latency_s"] for r in completed], .95)), 4) if completed else None,
            "pair_groups": len(pair_ids), "label_source": "dataset_annotation_proxy_not_human_adjudication"}

    record = {"run_date": "2026-11-09", "pairs": 30, "proxy_claims": 150,
              "synthetic_logic_fixture_cases_not_scored_here": len(cases) - len(proxy_cases),
              "selection_seed": 42, "selection_rule": "current D screen: lowest SHA256([42, 'val', filename]), 30 official val pairs",
              "revisions": REVISIONS, "prompt": tw.DECISION_INSTRUCTIONS, "witness_prompt": tw.WITNESS_PROMPT,
              "input_tracks": {"common_evidence": "same Otsu diagnostic and fallible Qwen3-VL witness; no original pixels sent to judge",
                               "image_backed": "the common evidence plus original ordered T1/T2 images"},
              "device_map": config["MODEL_DEVICE_MAP"], "max_gpu_memory": config["MAX_GPU_MEMORY"],
              "dtype": "bfloat16", "quantisation": None,
              "correlated_witness_judge": "Qwen3-VL both creates witness and performs J06 referee runs",
              "metrics": summaries,
              "limitations": ["Targets are dataset annotation proxies, not independent human adjudications.",
                             "The historical 150-claim export bundle was unavailable; this is a newly reconstructed local screen.",
                             "The released models may have seen LEVIR-family data during pretraining.",
                             "The 4060 Ti 8 GB uses CPU offload; latency is not directly comparable to the historical Apple M4/L4 figures."]}
    (RUN / "referee_local_summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
