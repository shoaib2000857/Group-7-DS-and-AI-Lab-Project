"""Score B0 caption runs: text metrics, claims, automatic reference check.

The reference check is an evaluation instrument that uses ORACLE masks and
the five reference captions. It is not the human factual audit and not
TerraWatch's checker (which may only use predicted evidence). Its labels:

  supported           mask shows change of that class and references do not
                      contradict the direction (added/removed/modified)
  direction_conflict  mask shows the class changed but references describe
                      only the opposite direction  -> counted as not supported
  contradicted        mask shows no change of that class and no reference
                      mentions it (false change), or a no-change claim on a
                      pair whose mask shows building/road change
  reference_conflict  mask and references disagree with each other -> unverifiable
  unverifiable        object outside the mask ontology (vegetation, water, ...)

  python scripts/evaluate.py runs/<run_id> [runs/<run_id> ...]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from claims import extract_claims, to_contract  # noqa: E402
from common import DATA, dataset_meta  # noqa: E402

# classes the oracle mask marks; out-of-domain sets mark buildings only (road claims -> unverifiable)
IN_SCOPE = dataset_meta().get("in_scope_classes", {"building": 255, "road": 128})
MIN_CLASS_PX = 50          # ignore single-blob slivers when deciding "class changed"
CELL_MIN_FRAC = 0.02       # class pixels needed in a 3x3 cell to support a location
CHANGE_ACTIONS = {"added", "removed", "modified", "changed"}
COMPATIBLE = {"changed": {"added", "removed", "modified", "changed"},"added": {"added", "modified"}, "modified": {"added", "modified", "removed"},
              "removed": {"removed", "modified"}}


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


class Truth:
    def __init__(self):
        self.refs = json.load(open(DATA / "references" / "references.json"))
        self.stats = json.load(open(DATA / "mask_stats.json"))
        self._masks = {}
        self.ref_claims = {
            pid: [c for cap in v["captions"] for c in extract_claims(cap)]
            for pid, v in self.refs.items()
        }

    def mask(self, pid):
        if pid not in self._masks:
            split, name = pid.split("/")
            m = np.array(Image.open(DATA / "pairs" / split / "label" / f"{name}.png"))
            self._masks[pid] = m[..., 0] if m.ndim == 3 else m
        return self._masks[pid]

    def class_changed(self, pid, cls):
        return int((self.mask(pid) == IN_SCOPE[cls]).sum()) >= MIN_CLASS_PX

    def any_change(self, pid):
        return any(self.class_changed(pid, c) for c in IN_SCOPE)

    def conflict(self, pid):
        """Captions and mask disagree on whether anything changed."""
        return self.any_change(pid) != bool(self.refs[pid]["changeflag"])

    def cells(self, pid, cls):
        m = self.mask(pid) == IN_SCOPE[cls]
        h, w = m.shape
        out = set()
        for r in range(3):
            for c in range(3):
                cell = m[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
                if cell.mean() >= CELL_MIN_FRAC:
                    out.add((r, c))
        return out

    def components(self, pid, cls):
        m = self.mask(pid) == IN_SCOPE[cls]
        lab, n = ndimage.label(m)
        sizes = ndimage.sum(m, lab, range(1, n + 1)) if n else []
        return int(sum(1 for s in sizes if s >= MIN_CLASS_PX))


def check_claim(c, pid, T):
    obj, act = c["claim"]["object"], c["claim"]["action"]
    loc = c["claim"]["location"]
    res = {"label": None, "location_check": None, "quantity_check": None}
    ref = T.ref_claims[pid]
    if act == "insufficient_evidence":
        res["label"] = "abstention"
        return res
    if act == "no_change":
        if obj == "scene":
            truth_change = T.any_change(pid)
        elif obj in IN_SCOPE:
            truth_change = T.class_changed(pid, obj)
        else:
            res["label"] = "unverifiable"
            return res
        if not truth_change:
            res["label"] = "reference_conflict" if T.refs[pid]["changeflag"] and obj == "scene" else "supported"
        else:
            res["label"] = "reference_conflict" if not T.refs[pid]["changeflag"] else "contradicted"
        return res
    if obj == "region":  # generic "changed region": any building/road change, location over both
        res["label"] = "supported" if T.any_change(pid) else (
            "reference_conflict" if T.refs[pid]["changeflag"] else "contradicted")
        if res["label"] == "supported" and loc and not loc.get("relational"):
            claimed = {tuple(x) for x in loc["cells"]} if loc.get("cells") else {(r, cc) for r in loc["rows"] for cc in loc["cols"]}
            res["location_check"] = bool(claimed & set().union(*(T.cells(pid, k) for k in IN_SCOPE)))
        return res
    if obj not in IN_SCOPE or act not in CHANGE_ACTIONS:
        res["label"] = "unverifiable"
        res["reference_mentions"] = any(r["claim"]["object"] == obj for r in ref)
        return res
    mask_ok = T.class_changed(pid, obj)
    ref_obj = [r for r in ref if r["claim"]["object"] == obj and r["claim"]["action"] in CHANGE_ACTIONS]
    ref_dir = [r for r in ref_obj if r["claim"]["action"] in COMPATIBLE[act]]
    if mask_ok:
        res["label"] = "supported" if (ref_dir or not ref_obj) else "direction_conflict"
    else:
        res["label"] = "reference_conflict" if ref_obj else "contradicted"
    if res["label"] == "supported" and loc and not loc.get("relational"):
        claimed = {tuple(x) for x in loc["cells"]} if loc.get("cells") else {(r, cc) for r in loc["rows"] for cc in loc["cols"]}
        res["location_check"] = bool(claimed & T.cells(pid, obj))
    q = c["claim"]["quantity"]
    if res["label"] == "supported" and isinstance(q, int) and c["parser"]["quantity_kind"] == "exact":
        n = T.components(pid, obj)
        res["quantity_check"] = {"claimed": q, "mask_components": n,
                                 "within_tolerance": abs(q - n) <= max(1, round(0.3 * n))}
    return res


def text_metrics(run_rows, refs):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    gts = {r["pair_id"]: [{"caption": c} for c in refs[r["pair_id"]]["captions"]] for r in run_rows}
    res = {r["pair_id"]: [{"caption": r["raw_text"] or ""}] for r in run_rows}
    tok = PTBTokenizer()
    gts, res = tok.tokenize(gts), tok.tokenize(res)
    out = {}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0)
    out.update({f"BLEU-{i + 1}": round(v, 4) for i, v in enumerate(b)})
    out["METEOR"] = round(Meteor().compute_score(gts, res)[0], 4)
    out["ROUGE-L"] = round(Rouge().compute_score(gts, res)[0], 4)
    out["CIDEr-D"] = round(Cider().compute_score(gts, res)[0], 4)  # coco-caption CIDEr-D implementation
    return out


def bootstrap(pairs, fn, n=2000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]


def summarise(per_pair):
    def precision(ps):
        sup = sum(p["n_supported"] for p in ps)
        tot = sum(p["n_factual"] for p in ps)
        return sup / tot if tot else None

    def located_precision(ps):
        # stricter: a supported change claim whose stated image location misses the mask counts as wrong
        sup = sum(p["n_supported"] - p["loc_checked"] + p["loc_ok"] for p in ps)
        tot = sum(p["n_factual"] for p in ps)
        return sup / tot if tot else None

    def inscope_precision(ps):
        sup = sum(p["n_supported"] for p in ps)
        tot = sum(p["n_inscope"] for p in ps)
        return sup / tot if tot else None

    def recall(ps):
        cov = sum(p["facts_covered"] for p in ps)
        tot = sum(p["n_facts"] for p in ps)
        return cov / tot if tot else None

    clean_unchanged = [p for p in per_pair if not p["truth_change"] and not p["conflict"]]
    changed = [p for p in per_pair if p["truth_change"] and not p["conflict"]]
    s = {
        "pairs": len(per_pair),
        "claims_total": sum(p["n_claims"] for p in per_pair),
        "claims_per_pair": round(sum(p["n_claims"] for p in per_pair) / len(per_pair), 3),
        "pairs_with_zero_claims": sum(p["n_claims"] == 0 for p in per_pair),
        "label_counts": dict(sum((p["labels"] for p in per_pair), start=__import__("collections").Counter())),
        "supported_claim_precision_strict": precision(per_pair),
        "supported_claim_precision_strict_ci95": bootstrap(per_pair, precision),
        "supported_claim_precision_in_scope": inscope_precision(per_pair),
        "supported_claim_precision_in_scope_ci95": bootstrap(per_pair, inscope_precision),
        "supported_claim_precision_located": located_precision(per_pair),
        "supported_claim_precision_located_ci95": bootstrap(per_pair, located_precision),
        "reference_fact_recall": recall(per_pair),
        "reference_fact_recall_ci95": bootstrap(per_pair, recall),
        "unchanged_pairs_clean": len(clean_unchanged),
        "no_change_false_alarm_rate": (sum(p["asserts_change"] for p in clean_unchanged) / len(clean_unchanged)) if clean_unchanged else None,
        "changed_pairs_clean": len(changed),
        # a changed pair is missed when the caption describes no building/road change at all
        # (explicit "no change" or silent), per protocol "omission"
        "missed_change_rate": (sum(not p["asserts_change"] for p in changed) / len(changed)) if changed else None,
        "missed_change_explicit_rate": (sum(p["says_no_change"] and not p["asserts_change"] for p in changed) / len(changed)) if changed else None,
        "pair_change_accuracy": (sum(p["asserts_change"] == p["truth_change"] for p in clean_unchanged + changed) / len(clean_unchanged + changed)) if (clean_unchanged or changed) else None,
        "location_checked": sum(p["loc_checked"] for p in per_pair),
        "location_supported": sum(p["loc_ok"] for p in per_pair),
        "quantity_checked": sum(p["q_checked"] for p in per_pair),
        "quantity_within_tolerance": sum(p["q_ok"] for p in per_pair),
        "reference_conflict_pairs": sum(p["conflict"] for p in per_pair),
    }
    for k, v in list(s.items()):
        if isinstance(v, float):
            s[k] = round(v, 4)
    return s


def evaluate_run(run_dir, T, roles=None, tag=""):
    run_dir = Path(run_dir)
    rows = [r for r in load_jsonl(run_dir / "captions.jsonl") if not roles or r["poc_role"] in roles]
    ok = [r for r in rows if r["status"] == "ok"]
    claims_out, per_pair = [], []
    from collections import Counter
    for r in rows:
        pid = r["pair_id"]
        cs = extract_claims(r.get("raw_text") or "", pid, r["run_id"])
        labels = Counter()
        n_inscope = asserts = says_nc = loc_c = loc_ok = q_c = q_ok = 0
        covered = set()
        for c in cs:
            chk = check_claim(c, pid, T)
            labels[chk["label"]] += 1
            obj, act = c["claim"]["object"], c["claim"]["action"]
            if act in CHANGE_ACTIONS and (obj in IN_SCOPE or obj == "region") and not c["parser"]["negated"]:
                asserts = 1
            if act == "no_change" and obj in ("scene", "building", "road"):
                says_nc = 1
            if obj in IN_SCOPE or obj in ("scene", "region"):
                n_inscope += chk["label"] != "abstention"
            if chk["label"] == "supported":
                if act in CHANGE_ACTIONS and obj == "region":
                    covered |= {c2 for c2 in IN_SCOPE if T.class_changed(pid, c2)}
                elif act in CHANGE_ACTIONS:
                    covered.add(obj)
                elif act == "no_change":
                    covered.add("no_change")
            if chk["location_check"] is not None:
                loc_c += 1
                loc_ok += chk["location_check"]
            if chk["quantity_check"]:
                q_c += 1
                q_ok += chk["quantity_check"]["within_tolerance"]
            rec = to_contract(c)
            rec["b_reference_check"] = chk
            claims_out.append(rec)
        truth_change = T.any_change(pid)
        facts = {c for c in IN_SCOPE if T.class_changed(pid, c)} or {"no_change"}
        conflict = T.conflict(pid)
        n_factual = sum(v for k, v in labels.items() if k != "abstention")
        per_pair.append({
            "pair_id": pid, "n_claims": len(cs), "labels": labels,
            "n_supported": labels["supported"], "n_factual": n_factual, "n_inscope": n_inscope,
            "n_facts": 0 if conflict else len(facts),
            "facts_covered": 0 if conflict else len(facts & covered),
            "truth_change": truth_change, "conflict": conflict,
            "asserts_change": asserts, "says_no_change": says_nc,
            "loc_checked": loc_c, "loc_ok": loc_ok, "q_checked": q_c, "q_ok": q_ok,
        })
    with open(run_dir / f"claims{tag}.jsonl", "w") as f:
        for c in claims_out:
            f.write(json.dumps(c) + "\n")
    with open(run_dir / f"per_pair{tag}.jsonl", "w") as f:
        for p in per_pair:
            f.write(json.dumps({**p, "labels": dict(p["labels"])}) + "\n")
    summary = summarise(per_pair)
    summary["text_metrics"] = text_metrics(ok, T.refs) if ok else None
    summary["completion_rate"] = round(len(ok) / len(rows), 4)
    rr = json.load(open(run_dir / "run_record.json"))
    for k in ("cold_load_s", "warm_latency_median_s", "warm_latency_p95_s", "peak_gpu_memory_mb"):
        summary[k] = rr.get(k)
    summary["run_id"] = rr["run_id"]
    summary["role_model_id"] = rr["role_model_id"]
    summary["evaluator"] = {"reference_check": "oracle mask + 5 reference captions, automatic",
                            "MIN_CLASS_PX": MIN_CLASS_PX, "CELL_MIN_FRAC": CELL_MIN_FRAC,
                            "parser": "rules-v3", "bootstrap": "pair-level, 2000 resamples, seed 0"}
    summary["roles"] = sorted(roles) if roles else "all"
    json.dump(summary, open(run_dir / f"eval_summary{tag}.json", "w"), indent=1)
    rr.setdefault("metrics", {})
    if not isinstance(rr["metrics"], dict):
        rr["metrics"] = {}
    rr["metrics"][tag.strip("_") or "all"] = {k: summary[k] for k in (
        "pairs", "supported_claim_precision_strict", "reference_fact_recall",
        "no_change_false_alarm_rate", "text_metrics")}
    rr["parsed_output_path"] = str((run_dir / f"claims{tag}.jsonl").relative_to(run_dir.parents[1]))
    json.dump(rr, open(run_dir / "run_record.json", "w"), indent=1)
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--roles", default="", help="e.g. screening, or development_smoke,development")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    roles = set(args.roles.split(",")) if args.roles else None
    T = Truth()
    for d in args.runs:
        s = evaluate_run(d, T, roles, args.tag)
        print(json.dumps({k: s[k] for k in ("run_id", "pairs", "claims_per_pair", "supported_claim_precision_strict",
                                            "reference_fact_recall", "no_change_false_alarm_rate",
                                            "missed_change_rate", "text_metrics")}))


if __name__ == "__main__":
    main()
