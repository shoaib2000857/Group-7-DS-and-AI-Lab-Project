import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

LABELS = ["supported", "contradicted", "unverifiable"]
CATS = ["none", "hallucinated_object", "missed_change", "wrong_class_road_vs_bldg",
        "wrong_count", "wrong_location", "shadow_or_seasonal", "tiny_or_occluded"]
SLICES = {
    "001-030": {"range": (1, 30), "raters": ("SSSM", "PA"), "adjudicator": "SRP"},
    "031-060": {"range": (31, 60), "raters": ("PA", "SS"), "adjudicator": "SSSM"},
    "061-090": {"range": (61, 90), "raters": ("SS", "SRP"), "adjudicator": "PA"},
    "091-120": {"range": (91, 120), "raters": ("SRP", "SSSM"), "adjudicator": "SS"},
}
ACCEPT = {"accept", "accepted"}
LEAK_COLS = {"proxy_label", "hybrid_verifier_prob", "vlm_triggered"}
DEFAULT_POLICIES = {
    "unverified": "unverified_decision", "mask_only": "mask_only_decision",
    "direct_vlm": "direct_vlm_decision", "reinsp_all": "reinsp_all_decision",
    "hybrid": "hybrid_verifier_decision"}
KNOWN_DECISIONS = ACCEPT | {"reject", "rejected", "defer", "deferred"}
NAVY, TEAL, LIME, AMBER, CORAL, SLATE, BG = (
    "#102630", "#1f6f78", "#7aa82a", "#d9822b", "#c94040", "#5c6f78", "#fafbf8")


class AuditError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_id(raw, where):
    m = re.search(r"(\d+)\s*$", str(raw))
    if not m:
        raise AuditError(f"{where}: cannot parse id {raw!r}")
    return int(m.group(1))


def slice_of(cid):
    for name, s in SLICES.items():
        if s["range"][0] <= cid <= s["range"][1]:
            return name
    return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in r]
        return r.fieldnames or [], rows


def load_rater(path, rater):
    fields, rows = read_csv(path)
    fields = [x.strip() for x in fields]
    leaks = [c for c in fields if c.endswith("_decision") or c in LEAK_COLS]
    if leaks:
        raise AuditError(f"{path.name}: blindness violation, model columns present: {leaks}")
    for col in ("audit_id", "human_label", "error_category"):
        if col not in fields:
            raise AuditError(f"{path.name}: missing required column {col!r}")
    out, warnings = {}, []
    blank = 0
    for i, row in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        cid = parse_id(row["audit_id"], where)
        if not row["human_label"]:
            blank += 1
            continue
        label = row["human_label"].lower()
        if label not in LABELS:
            raise AuditError(f"{where}: unknown human_label {row['human_label']!r}; allowed {LABELS}")
        cat = row["error_category"].lower() or "none"
        if cat not in CATS:
            raise AuditError(f"{where}: unknown error_category {row['error_category']!r}; allowed {CATS}")
        if cid in out:
            raise AuditError(f"{where}: duplicate claim_id {cid} in rater {rater}")
        sl = slice_of(cid)
        if sl is None:
            raise AuditError(f"{where}: claim_id {cid} outside 1-120")
        if rater not in SLICES[sl]["raters"]:
            raise AuditError(f"{where}: rater {rater} is not assigned to slice {sl} (claim {cid})")
        if label == "supported" and cat != "none":
            warnings.append(f"{where}: supported claim {cid} has error_category {cat}")
        out[cid] = {"label": label, "cat": cat}
    return out, warnings, blank


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None, False
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in LABELS)
    if math.isclose(pe, 1.0):
        return (1.0 if math.isclose(po, 1.0) else 0.0), True
    return (po - pe) / (1 - pe), False


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def pair_stats(pairs):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    n = len(pairs)
    kappa, degenerate = cohen_kappa(a, b)
    cm = [[sum(1 for x, y in pairs if x == r and y == c) for c in LABELS] for r in LABELS]
    return {
        "n": n,
        "percent_agreement": (sum(x == y for x, y in pairs) / n) if n else None,
        "kappa": kappa,
        "kappa_degenerate": degenerate,
        "confusion_matrix": cm,
        "confusion_labels": LABELS,
    }


def category_agreement(items):
    sel = [(x, y) for (lx, ly, x, y) in items if lx != "supported" and ly != "supported"]
    return {"n": len(sel), "percent_agreement": (sum(x == y for x, y in sel) / len(sel)) if sel else None}


def load_adjudication(path):
    fields, rows = read_csv(path)
    for col in ("claim_id", "adjudicated_label"):
        if col not in fields:
            raise AuditError(f"{path.name}: missing required column {col!r}")
    out = {}
    for i, row in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        cid = parse_id(row["claim_id"], where)
        lab = row["adjudicated_label"].lower()
        if lab not in LABELS:
            raise AuditError(f"{where}: unknown adjudicated_label {row['adjudicated_label']!r}")
        if cid in out:
            raise AuditError(f"{where}: duplicate claim_id {cid}")
        sl = slice_of(cid)
        adj = row.get("adjudicator", "")
        if sl and adj and adj != SLICES[sl]["adjudicator"]:
            raise AuditError(f"{where}: adjudicator {adj} is not assigned to slice {sl}")
        out[cid] = lab
    return out


def compute_raters(raters):
    per_slice, disagreements, gold_rows, coverage = {}, [], {}, {}
    all_pairs, all_items = [], []
    for name, s in SLICES.items():
        ra, rb = s["raters"]
        lo, hi = s["range"]
        da, db = raters.get(ra, {}), raters.get(rb, {})
        ids = range(lo, hi + 1)
        both = [c for c in ids if c in da and c in db]
        coverage[name] = {
            "total": hi - lo + 1,
            f"labelled_{ra}": sum(c in da for c in ids),
            f"labelled_{rb}": sum(c in db for c in ids),
            "labelled_both": len(both),
            "missing_either": hi - lo + 1 - len(both),
        }
        pairs = [(da[c]["label"], db[c]["label"]) for c in both]
        items = [(da[c]["label"], db[c]["label"], da[c]["cat"], db[c]["cat"]) for c in both]
        st = pair_stats(pairs)
        st["error_category_agreement"] = category_agreement(items)
        st.update({"rater_a": ra, "rater_b": rb, "adjudicator": s["adjudicator"]})
        per_slice[name] = st
        all_pairs += pairs
        all_items += items
        for c in both:
            if da[c]["label"] == db[c]["label"]:
                gold_rows[c] = (da[c]["label"], "agreed")
            else:
                disagreements.append({
                    "claim_id": c, "slice": name, "rater_a": ra, "label_a": da[c]["label"],
                    "rater_b": rb, "label_b": db[c]["label"], "adjudicator": s["adjudicator"]})
    pooled = pair_stats(all_pairs)
    pooled["error_category_agreement"] = category_agreement(all_items)
    return per_slice, pooled, disagreements, gold_rows, coverage


def build_gold(gold_rows, disagreements, adjudication):
    gold = dict(gold_rows)
    unresolved = []
    for d in disagreements:
        c = d["claim_id"]
        if c in adjudication:
            gold[c] = (adjudication[c], "adjudicated")
        else:
            unresolved.append(c)
    return gold, sorted(unresolved)


def load_packet(path, id_map):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise AuditError(f"{path.name} line {i}: invalid JSON ({e})")
    out, unmapped = [], 0
    for r in rows:
        if r.get("audit_id") not in (None, ""):
            aid = parse_id(r["audit_id"], "packet audit_id")
        elif id_map is not None and str(r.get("claim_id")) in id_map:
            aid = id_map[str(r["claim_id"])]
        else:
            unmapped += 1
            continue
        out.append((aid, r))
    ids = [a for a, _ in out]
    dup = sorted({a for a in ids if ids.count(a) > 1})
    if dup:
        raise AuditError(f"packet: duplicate audit ids {dup[:10]}")
    return out, unmapped, len(rows)


def load_master_key(path):
    fields, rows = read_csv(path)
    if "audit_id" not in fields:
        raise AuditError(f"{path.name}: missing column 'audit_id'")
    out = []
    for i, r in enumerate(rows, start=2):
        out.append((parse_id(r["audit_id"], f"{path.name} line {i}"), r))
    ids = [a for a, _ in out]
    dup = sorted({a for a in ids if ids.count(a) > 1})
    if dup:
        raise AuditError(f"master key: duplicate audit ids {dup[:10]}")
    return out, 0, len(rows)


def load_id_map(path):
    fields, rows = read_csv(path)
    for col in ("audit_id", "claim_id"):
        if col not in fields:
            raise AuditError(f"{path.name}: missing column {col!r}")
    return {r["claim_id"]: parse_id(r["audit_id"], path.name) for r in rows}


def policy_stats(packet, policies, gold):
    res = {}
    for name, col in policies.items():
        acc = total = 0
        sup = con = unv = excl = 0
        for aid, r in packet:
            if col not in r:
                raise AuditError(f"policy {name}: column {col!r} not in packet row {aid}")
            dec = str(r[col]).strip().lower()
            if dec not in KNOWN_DECISIONS:
                raise AuditError(f"policy {name}: unknown decision {r[col]!r} (row {aid}); known {sorted(KNOWN_DECISIONS)}")
            total += 1
            if dec not in ACCEPT:
                continue
            acc += 1
            g = gold.get(aid)
            if g is None:
                excl += 1
            elif g[0] == "supported":
                sup += 1
            elif g[0] == "contradicted":
                con += 1
            else:
                unv += 1
        ng = sup + con + unv
        trig = [str(r["vlm_triggered"]).strip().lower() in ("true", "1", "yes") for _, r in packet if "vlm_triggered" in r]
        res[name] = {
            "column": col, "n_packet": total, "n_accepted": acc,
            "coverage": acc / total if total else None,
            "n_accepted_with_gold": ng, "n_accepted_without_gold": excl,
            "n_supported": sup, "n_contradicted": con, "n_unverifiable": unv,
            "groundedness": sup / ng if ng else None, "groundedness_ci95": wilson(sup, ng),
            "hallucination_rate": con / ng if ng else None, "hallucination_ci95": wilson(con, ng),
        }
        if name == "hybrid" and trig:
            res[name]["vlm_call_rate"] = sum(trig) / len(trig)
    return res


def proxy_vs_gold(packet, gold):
    pairs = []
    for aid, r in packet:
        p = str(r.get("proxy_label") or r.get("proxy_gt") or "").strip().lower()
        if p in LABELS and aid in gold:
            pairs.append((p, gold[aid][0]))
    if not pairs:
        return None
    st = pair_stats(pairs)
    return {k: st[k] for k in ("n", "percent_agreement", "kappa", "kappa_degenerate", "confusion_matrix")}


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_figure(path, per_slice, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": BG, "axes.edgecolor": "#c8d3cf",
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelweight": "bold"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    cm = pooled["confusion_matrix"]
    cmap = LinearSegmentedColormap.from_list("m5", ["#ffffff", TEAL])
    mx = max(max(r) for r in cm) or 1
    a1.imshow(cm, cmap=cmap, vmin=0, vmax=mx)
    a1.grid(False)
    for i in range(3):
        for j in range(3):
            a1.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > mx * 0.55 else NAVY, fontweight="bold")
    a1.set_xticks(range(3), LABELS, rotation=20)
    a1.set_yticks(range(3), LABELS)
    a1.set_xlabel("Rater B")
    a1.set_ylabel("Rater A")
    pk = pooled["kappa"]
    a1.set_title(f"Pooled confusion (n={pooled['n']}, kappa={'n/a' if pk is None else f'{pk:.2f}'})")
    names = list(per_slice) + ["pooled"]
    vals = [per_slice[k]["kappa"] for k in per_slice] + [pk]
    cols = [TEAL] * len(per_slice) + [NAVY]
    ys = [0 if v is None else v for v in vals]
    a2.bar(names, ys, color=cols)
    for i, v in enumerate(vals):
        a2.text(i, ys[i] + 0.02, "n/a" if v is None else f"{v:.2f}", ha="center", color=NAVY)
    a2.axhline(0.6, color=AMBER, ls="--", lw=1)
    a2.text(len(names) - 0.5, 0.61, "0.60", color=AMBER, ha="right", va="bottom")
    a2.set_ylim(min(0, min(ys) - 0.1), 1.1)
    a2.set_ylabel("Cohen's kappa")
    a2.set_title("Inter-rater kappa by slice")
    a2.grid(axis="y", color="#e2e8e5", ls="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(raters_dir, out, adjudication=None, packet=None, id_map=None, policies=None, master_key=None):
    raters_dir, out = Path(raters_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raters_dir.glob("rater_*.csv"))
    if not files:
        raise AuditError(f"no rater_*.csv files in {raters_dir}")
    raters, warnings, hashes, unrated = {}, [], {}, {}
    for p in files:
        name = re.sub(r"_\d+$", "", p.stem[len("rater_"):])
        if name not in {r for s in SLICES.values() for r in s["raters"]}:
            raise AuditError(f"{p.name}: unknown rater {name!r}")
        raters[name], w, nb = load_rater(p, name)
        warnings += w
        unrated[name] = nb
        hashes[str(p.name)] = sha256(p)
    per_slice, pooled, dis, gold_rows, coverage = compute_raters(raters)
    adj = {}
    if adjudication:
        adj = load_adjudication(Path(adjudication))
        hashes[Path(adjudication).name] = sha256(adjudication)
        extra = sorted(set(adj) - {d["claim_id"] for d in dis})
        if extra:
            warnings.append(f"adjudication rows for claims without a disagreement ignored: {extra}")
    gold, unresolved = build_gold(gold_rows, dis, adj)
    write_csv(out / "disagreements.csv",
              ["claim_id", "slice", "rater_a", "label_a", "rater_b", "label_b", "adjudicator"], dis)
    write_csv(out / "gold_labels.csv", ["claim_id", "gold_label", "source"],
              [{"claim_id": c, "gold_label": g[0], "source": g[1]} for c, g in sorted(gold.items())])
    summary = {
        "input_sha256": hashes,
        "warnings": warnings,
        "coverage": coverage,
        "unrated_blank_rows": unrated,
        "per_slice": per_slice,
        "pooled": pooled,
        "counts": {
            "claims_total": 120, "disagreements": len(dis), "gold_labels": len(gold),
            "gold_from_agreement": sum(1 for g in gold.values() if g[1] == "agreed"),
            "gold_from_adjudication": sum(1 for g in gold.values() if g[1] == "adjudicated"),
            "unresolved_excluded": len(unresolved), "unresolved_claim_ids": unresolved,
        },
        "kappa_note": "kappa is 1.0 when raters agree perfectly with a single category (chance agreement 1), 0.0 if chance agreement is 1 but they disagree; flagged by kappa_degenerate",
    }
    if packet or master_key:
        if master_key:
            rows, unmapped, n_rows = load_master_key(Path(master_key))
            hashes[Path(master_key).name] = sha256(master_key)
            policies = policies or DEFAULT_POLICIES
        else:
            idm = load_id_map(Path(id_map)) if id_map else None
            if id_map:
                hashes[Path(id_map).name] = sha256(id_map)
            rows, unmapped, n_rows = load_packet(Path(packet), idm)
            hashes[Path(packet).name] = sha256(packet)
        summary["packet"] = {"rows": n_rows, "mapped": len(rows), "unmapped": unmapped}
        if policies:
            summary["policies"] = policy_stats(rows, policies, gold)
        summary["proxy_gt_vs_gold"] = proxy_vs_gold(rows, gold)
    (out / "human_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(out / "m5_human_audit_agreement.png", per_slice, pooled)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compute M5 human audit agreement, gold labels and policy groundedness.")
    ap.add_argument("--raters-dir", required=True)
    ap.add_argument("--adjudication")
    ap.add_argument("--packet")
    ap.add_argument("--master-key")
    ap.add_argument("--id-map")
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=COL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pol = {}
    for p in a.policy:
        if "=" not in p:
            ap.error(f"--policy expects name=col, got {p!r}")
        k, v = p.split("=", 1)
        pol[k] = v
    try:
        s = run(a.raters_dir, a.out, a.adjudication, a.packet, a.id_map, pol, a.master_key)
    except AuditError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"pooled_kappa": s["pooled"]["kappa"], "pooled_n": s["pooled"]["n"],
                      "counts": s["counts"], "warnings": len(s["warnings"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
