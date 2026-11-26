import csv
import json
import random
import sys
from pathlib import Path

import pytest
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m5_compute_human_audit as m  # noqa: E402

L = m.LABELS


def write_rater(d, name, labels, cats=None):
    with open(Path(d) / f"rater_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "claim_id", "human_label", "error_category", "notes"])
        for cid, lab in labels.items():
            cat = (cats or {}).get(cid, "none" if lab == "supported" else "missed_change")
            w.writerow([f"M5T_{cid:03d}", f"c{cid}", lab, cat, ""])


def slice1(fn_a, fn_b=None):
    ids = range(1, 31)
    return {i: fn_a(i) for i in ids}, {i: (fn_b or fn_a)(i) for i in ids}


def test_identical_labels(tmp_path):
    a, b = slice1(lambda i: L[i % 3])
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)
    s = m.run(tmp_path, tmp_path / "o")
    st = s["per_slice"]["001-030"]
    assert st["kappa"] == 1.0 and st["percent_agreement"] == 1.0
    assert s["counts"]["disagreements"] == 0 and s["counts"]["gold_labels"] == 30
    assert s["coverage"]["031-060"]["labelled_both"] == 0


def test_textbook_kappa_and_sklearn():
    pairs = ([("supported", "supported")] * 20 + [("supported", "contradicted")] * 5
             + [("contradicted", "supported")] * 10 + [("contradicted", "contradicted")] * 15)
    st = m.pair_stats(pairs)
    assert round(st["kappa"], 4) == 0.4
    assert st["percent_agreement"] == 0.7
    assert st["confusion_matrix"][0][:2] == [20, 5] and st["confusion_matrix"][1][:2] == [10, 15]
    assert st["kappa"] == pytest.approx(cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs]))


def test_three_class_matches_sklearn():
    rng = random.Random(3)
    pairs = [(rng.choice(L), rng.choice(L)) for _ in range(200)]
    pairs = [(x, x if rng.random() < 0.6 else y) for x, y in pairs]
    k, _ = m.cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs])
    assert k == pytest.approx(cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs], labels=L))


def test_independent_random_kappa_small():
    rng = random.Random(0)
    a = [rng.choice(L) for _ in range(20000)]
    b = [rng.choice(L) for _ in range(20000)]
    k, _ = m.cohen_kappa(a, b)
    assert abs(k) < 0.1


def test_degenerate_all_supported(tmp_path):
    a, b = slice1(lambda i: "supported")
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)
    s = m.run(tmp_path, tmp_path / "o")
    st = s["per_slice"]["001-030"]
    assert st["kappa"] == 1.0 and st["kappa_degenerate"] is True
    saved = json.loads((tmp_path / "o" / "human_audit_summary.json").read_text())
    assert saved["per_slice"]["001-030"]["kappa_degenerate"] is True


def test_bad_label(tmp_path):
    a, b = slice1(lambda i: "supported")
    a[5] = "maybe"
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)
    with pytest.raises(m.AuditError, match="unknown human_label"):
        m.run(tmp_path, tmp_path / "o")


def test_wrong_slice(tmp_path):
    a, b = slice1(lambda i: "supported")
    a[45] = "supported"
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)
    with pytest.raises(m.AuditError, match="not assigned to slice 031-060"):
        m.run(tmp_path, tmp_path / "o")


def test_duplicate_claim(tmp_path):
    a, b = slice1(lambda i: "supported")
    write_rater(tmp_path, "PA", b)
    write_rater(tmp_path, "SSSM", a)
    p = tmp_path / "rater_SSSM.csv"
    p.write_text(p.read_text() + "M5T_005,c5,supported,none,\n")
    with pytest.raises(m.AuditError, match="duplicate claim_id 5"):
        m.run(tmp_path, tmp_path / "o")


def disagreeing(tmp_path):
    a, b = slice1(lambda i: "supported")
    b[1] = "contradicted"
    b[2] = "unverifiable"
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)


def test_adjudication_and_unresolved(tmp_path):
    disagreeing(tmp_path)
    adj = tmp_path / "adj.csv"
    adj.write_text("claim_id,adjudicated_label,adjudicator,notes\nM5T_001,contradicted,SRP,x\n")
    s = m.run(tmp_path, tmp_path / "o", adjudication=adj)
    assert s["counts"]["disagreements"] == 2
    assert s["counts"]["gold_from_adjudication"] == 1
    assert s["counts"]["unresolved_claim_ids"] == [2]
    gold = {r["claim_id"]: r for r in csv.DictReader(open(tmp_path / "o" / "gold_labels.csv"))}
    assert gold["1"]["gold_label"] == "contradicted" and "2" not in gold and len(gold) == 29
    dis = list(csv.DictReader(open(tmp_path / "o" / "disagreements.csv")))
    assert [d["claim_id"] for d in dis] == ["1", "2"] and dis[0]["adjudicator"] == "SRP"


def test_wrong_adjudicator(tmp_path):
    disagreeing(tmp_path)
    adj = tmp_path / "adj.csv"
    adj.write_text("claim_id,adjudicated_label,adjudicator,notes\nM5T_001,contradicted,PA,x\n")
    with pytest.raises(m.AuditError, match="not assigned"):
        m.run(tmp_path, tmp_path / "o", adjudication=adj)


def test_packet_join_policy(tmp_path):
    labs = {1: "supported", 2: "supported", 3: "contradicted", 4: "unverifiable", 5: "supported", 6: "contradicted"}
    a, _ = slice1(lambda i: labs.get(i, "supported"))
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", a)
    dec = ["accept", "accept", "accept", "reject", "defer", "accept"]
    proxy = ["supported", "contradicted", "contradicted", "supported", "supported", "contradicted"]
    trig = ["true", "false", "true", "false", "false", "false"]
    mk = tmp_path / "test_audit_master_key.csv"
    with open(mk, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "claim_id", "proxy_label", "unverified_decision", "mask_only_decision", "direct_vlm_decision", "reinsp_all_decision", "hybrid_verifier_decision", "vlm_triggered"])
        for i in range(6):
            w.writerow([f"M5T_{i+1:03d}", f"c{i+1}", proxy[i], "accept", "reject", "defer", "accept", dec[i], trig[i]])
    s = m.run(tmp_path, tmp_path / "o", master_key=mk, policies={"hybrid": "hybrid_verifier_decision"})
    p = s["policies"]["hybrid"]
    assert (p["n_accepted"], p["n_supported"], p["n_contradicted"]) == (4, 2, 2)
    assert p["coverage"] == pytest.approx(4 / 6)
    assert p["groundedness"] == 0.5 and p["hallucination_rate"] == 0.5
    assert p["vlm_call_rate"] == pytest.approx(2 / 6)
    lo, hi = p["groundedness_ci95"]
    assert lo == pytest.approx(0.15, abs=0.01) and hi == pytest.approx(0.85, abs=0.01)
    g = s["proxy_gt_vs_gold"]
    assert g["n"] == 6 and g["percent_agreement"] == pytest.approx(4 / 6)
    s2 = m.run(tmp_path, tmp_path / "o2", master_key=mk)
    assert set(s2["policies"]) == set(m.DEFAULT_POLICIES) and s2["policies"]["unverified"]["coverage"] == 1.0


def test_unknown_decision(tmp_path):
    a, b = slice1(lambda i: "supported")
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)
    mk = tmp_path / "k.csv"
    mk.write_text("audit_id,p\nM5T_001,maybe\n")
    with pytest.raises(m.AuditError, match="unknown decision"):
        m.run(tmp_path, tmp_path / "o", master_key=mk, policies={"pol": "p"})


@pytest.mark.parametrize("col", ["hybrid_verifier_decision", "proxy_label", "hybrid_verifier_prob", "vlm_triggered"])
def test_blindness_violation(tmp_path, col):
    a, b = slice1(lambda i: "supported")
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)
    p = tmp_path / "rater_SSSM.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0] + "," + col] + [x + ",x" for x in lines[1:]]) + "\n")
    with pytest.raises(m.AuditError, match="blindness"):
        m.run(tmp_path, tmp_path / "o")


def test_blank_rows_unrated(tmp_path):
    a, b = slice1(lambda i: "supported")
    write_rater(tmp_path, "SSSM", a)
    write_rater(tmp_path, "PA", b)
    p = tmp_path / "rater_SSSM.csv"
    lines = p.read_text().splitlines()
    lines[1] = "M5T_001,c1,,,"
    p.write_text("\n".join(lines) + "\n")
    s = m.run(tmp_path, tmp_path / "o")
    assert s["unrated_blank_rows"]["SSSM"] == 1
    assert s["coverage"]["001-030"]["labelled_both"] == 29


def test_wilson_known():
    lo, hi = m.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
