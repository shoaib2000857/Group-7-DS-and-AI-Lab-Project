"""Automatic annotation-proxy checks, distinct from human factuality."""
import random
from src.models.claims import extract_claims
IN_SCOPE={"road":1,"building":2}
MIN_CLASS_PX=50
CELL_MIN_FRAC=.02
CHANGE_ACTIONS={"added","removed","modified","changed"}
COMPATIBLE={"changed":CHANGE_ACTIONS,"added":{"added","modified"},"modified":{"added","modified","removed"},"removed":{"removed","modified"}}

def check_claim(c, pid, T):
    IN_SCOPE=T.in_scope
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
