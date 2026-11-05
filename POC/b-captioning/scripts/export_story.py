"""Build the data-story datasets from screening results (runs/story/, git-ignored).

  results.json   one row per system: screening metrics + cost
  examples.json  one row per (example pair, system): caption + claim verdicts
  pairs.json     one row per example pair: T1/T2 as small JPEG data URLs,
                 one reference caption, what the mask marks
  data.json      dataset/budget facts (counts only)
  ood.json       one row per (dataset, system): LEVIR plus WHU-CDC and SECOND-CC

Examples are chosen by a fixed rule, not by looking at outputs: the first
changed, first unchanged and first reference-conflict pair in the seeded
screening order.
"""
import base64
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, RUNS  # noqa: E402

RUNS_BY_SYSTEM = [
    ("RSCaMa", "specialist", "B0_C02_rscama_development_smoke+development+screening"),
    ("RSCaMa retrained", "specialist", "B0_C02r20_rscama_retrained20_development_smoke+development+screening"),
    ("Qwen3-VL-4B", "general VLM", "B0_C04_qwen3vl_development_smoke+development+screening"),
    ("SmolVLM2-2.2B", "general VLM", "B0_C07_smolvlm2_development_smoke+development+screening"),
    ("Pixel-difference template", "control", "B0_CTRL_evidence_template_screening"),
    ("Always \"no difference\"", "control", "B0_CTRL_constant_nochange_screening"),
]
# out-of-domain sets (scripts/build_ood.py), scored like LEVIR; runs/ood/<name>/<run>
OOD = [("WHU-CDC", "whu_cdc"), ("SECOND-CC", "second_cc")]
VERDICT = {"supported": "checks out", "contradicted": "wrong", "direction_conflict": "wrong",
           "reference_conflict": "answer key disagrees", "unverifiable": "can't check",
           "abstention": "abstains"}


def jsonl(p):
    return [json.loads(l) for l in open(p)]


def data_url(path):
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    out = RUNS / "story"
    out.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader(open(DATA / "manifest.csv")))
    refs = json.load(open(DATA / "references" / "references.json"))
    stats = json.load(open(DATA / "mask_stats.json"))

    results = []
    for name, kind, run in RUNS_BY_SYSTEM:
        s = json.load(open(RUNS / run / "eval_summary_screening.json"))
        rr = json.load(open(RUNS / run / "run_record.json"))
        tm = s["text_metrics"] or {}
        lc = s["label_counts"]
        results.append({
            "system": name, "kind": kind, "pairs": s["pairs"],
            "claims": s["claims_total"], "claims_per_pair": s["claims_per_pair"],
            "precision": s["supported_claim_precision_strict"],
            "precision_lo": (s["supported_claim_precision_strict_ci95"] or [None, None])[0],
            "precision_hi": (s["supported_claim_precision_strict_ci95"] or [None, None])[1],
            "precision_in_scope": s["supported_claim_precision_in_scope"],
            "precision_located": s["supported_claim_precision_located"],
            "recall": s["reference_fact_recall"],
            "recall_lo": (s["reference_fact_recall_ci95"] or [None, None])[0],
            "recall_hi": (s["reference_fact_recall_ci95"] or [None, None])[1],
            "false_alarm_rate": s["no_change_false_alarm_rate"],
            "false_alarms": round(s["no_change_false_alarm_rate"] * s["unchanged_pairs_clean"]),
            "unchanged_pairs": s["unchanged_pairs_clean"],
            "missed_rate": s["missed_change_rate"],
            "missed": round(s["missed_change_rate"] * s["changed_pairs_clean"]),
            "changed_pairs": s["changed_pairs_clean"],
            "claims_wrong": lc.get("contradicted", 0) + lc.get("direction_conflict", 0),
            "claims_supported": lc.get("supported", 0),
            "claims_unverifiable": lc.get("unverifiable", 0) + lc.get("reference_conflict", 0),
            "bleu4": tm.get("BLEU-4"), "cider": tm.get("CIDEr-D"), "meteor": tm.get("METEOR"),
            "rouge_l": tm.get("ROUGE-L"),
            # latency only for runs on the Apple M4 (the retrain was captioned on a Kaggle T4)
            "seconds_per_pair": None if rr["role_model_id"] == "C02-r20" else rr.get("warm_latency_median_s"),
            "peak_memory_gb": None if rr.get("peak_gpu_memory_mb") is None else round(rr["peak_gpu_memory_mb"] / 1024, 1),
        })
    json.dump(results, open(out / "results.json", "w"), indent=1)

    # building(-and-road) claim precision on LEVIR and each out-of-domain set; roads are only marked in LEVIR
    by_id = {}
    for name, _, run in RUNS_BY_SYSTEM:
        by_id[json.load(open(RUNS / run / "run_record.json"))["role_model_id"]] = name
    ood = []
    sets = [("LEVIR-MCI", [RUNS / run for _, _, run in RUNS_BY_SYSTEM])] + \
           [(ds, sorted((RUNS / "ood" / d).glob("B0_*"))) for ds, d in OOD]
    for ds, dirs in sets:
        for d in dirs:
            s = json.load(open(d / "eval_summary_screening.json"))
            ci = s["supported_claim_precision_in_scope_ci95"] or [None, None]
            ood.append({
                "dataset": ds, "system": by_id[s["role_model_id"]], "pairs": s["pairs"],
                "precision_in_scope": s["supported_claim_precision_in_scope"],
                "precision_in_scope_lo": ci[0], "precision_in_scope_hi": ci[1],
                "precision": s["supported_claim_precision_strict"], "recall": s["reference_fact_recall"],
                "false_alarms": round(s["no_change_false_alarm_rate"] * s["unchanged_pairs_clean"]),
                "unchanged_pairs": s["unchanged_pairs_clean"],
                "missed": round(s["missed_change_rate"] * s["changed_pairs_clean"]),
                "changed_pairs": s["changed_pairs_clean"],
            })
    json.dump(ood, open(out / "ood.json", "w"), indent=1)

    # fixed-rule example picks
    order = sorted((r for r in manifest if r["poc_role"] == "screening"), key=lambda r: int(r["selection_order"]))
    pp = {p["pair_id"]: p for p in jsonl(RUNS / RUNS_BY_SYSTEM[0][2] / "per_pair_screening.jsonl")}
    picks = {}
    for r in order:
        p = pp[r["pair_id"]]
        k = "conflict" if p["conflict"] else ("changed" if p["truth_change"] else "unchanged")
        picks.setdefault(k, r)
    pairs, examples = [], []
    for k in ("changed", "unchanged", "conflict"):
        r = picks[k]
        pid = r["pair_id"]
        st = stats[pid]
        marks = [c for c, px in (("buildings", st["building_px"]), ("roads", st["road_px"])) if px >= 50]
        pairs.append({
            "pair_id": pid, "story_role": k,
            "t1": data_url(DATA / r["t1_path"]), "t2": data_url(DATA / r["t2_path"]),
            "reference_caption": refs[pid]["captions"][0].rstrip(" ."),
            "mask_marks": " and ".join(marks) if marks else "nothing",
        })
        for name, _, run in RUNS_BY_SYSTEM:
            cap = next(c for c in jsonl(RUNS / run / "captions.jsonl") if c["pair_id"] == pid)
            cl = [c for c in jsonl(RUNS / run / "claims_screening.jsonl") if c["pair_id"] == pid]
            v = Counter(VERDICT[c["b_reference_check"]["label"]] for c in cl)
            examples.append({
                "pair_id": pid, "story_role": k, "system": name, "caption": cap["raw_text"],
                "claims": len(cl), "checks_out": v.get("checks out", 0), "wrong": v.get("wrong", 0),
                "cannot_check": v.get("can't check", 0) + v.get("answer key disagrees", 0),
                "verdicts": "; ".join(f"{c['claim']['object']} {c['claim']['action']}: "
                                      f"{VERDICT[c['b_reference_check']['label']]}" for c in cl),
            })
    json.dump(pairs, open(out / "pairs.json", "w"))
    json.dump(examples, open(out / "examples.json", "w"), indent=1)

    m = json.load(open(DATA / "manifest.json"))
    conflicts = sum(1 for p in pp.values() if p["conflict"])
    all_conf = len(m["changeflag_mask_mismatch"])
    facts = [
        {"fact": "pairs_in_dataset", "value": m["pairs_in_source"]},
        {"fact": "pairs_used", "value": sum(m["selected"].values())},
        {"fact": "pairs_inspected", "value": m["selected"]["train"] + m["selected"]["val"]},
        {"fact": "development_pairs", "value": m["selected"]["train"]},
        {"fact": "screening_pairs", "value": m["selected"]["val"]},
        {"fact": "audit_pairs", "value": 10},
        {"fact": "sealed_pairs", "value": m["selected"]["test"]},
        {"fact": "cap_pairs", "value": m["global_cap"]},
        {"fact": "screening_reference_conflicts", "value": conflicts},
        {"fact": "inspected_reference_conflicts", "value": all_conf},
        # independent AI-assisted visual spot check of 36 screening claims (not a human audit)
        {"fact": "spot_check_claims", "value": 36},
        {"fact": "spot_check_extraction_faithful", "value": 34},
        {"fact": "spot_check_judgeable", "value": 29},
        {"fact": "spot_check_label_agrees", "value": 27},
        {"fact": "megabytes_downloaded_of_archive", "value": round(m["bytes_fetched"] / 1e6)},
        {"fact": "archive_megabytes", "value": round(m["archive_bytes"] / 1e6)},
        {"fact": "train_split_pairs", "value": m["split_sizes"]["train"]},
        {"fact": "val_split_pairs", "value": m["split_sizes"]["val"]},
        {"fact": "test_split_pairs", "value": m["split_sizes"]["test"]},
        # colab/rscama_colab.py select(): 20% of train minus the development pairs
        {"fact": "retrain_pairs", "value": int(0.20 * m["split_sizes"]["train"]) - m["selected"]["train"]},
    ]
    json.dump(facts, open(out / "data.json", "w"), indent=1)
    print({k: v["pair_id"] for k, v in picks.items()}, "files:", [p.name for p in out.iterdir()],
          {p.name: p.stat().st_size for p in out.iterdir()})


if __name__ == "__main__":
    main()
