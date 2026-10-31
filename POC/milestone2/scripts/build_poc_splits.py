"""Freeze small experiment cohorts inside the cleaned official partitions."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data/prepared"


def main():
    cohorts = {}
    selected = [json.loads(s) for s in (ROOT / "evidence/full_audit/poc_train_manifest.jsonl").read_text().splitlines()]
    random.Random("42-clean-poc-development").shuffle(selected)
    cohorts["poc_development"] = selected[:50]
    cohorts["poc_training"] = selected[50:]
    for split, role in [("val", "poc_screening"), ("test", "poc_evaluation")]:
        pool = [json.loads(s) for s in (PREP / f"{split}.jsonl").read_text().splitlines()]
        cohorts[role] = random.Random(f"42-clean-{role}").sample(sorted(pool, key=lambda r: r["pair_id"]), 100)
    ledger = {}
    for role, rows in cohorts.items():
        path = PREP / f"{role}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        ledger[role] = {"pairs": len(rows), "source_partition": rows[0]["official_split"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "ids": [r["pair_id"] for r in rows]}
    sets = [{r["pair_id"] for r in rows} for rows in cohorts.values()]
    assert sum(len(s) for s in sets) == len(set.union(*sets))
    (ROOT / "evidence/full_audit/poc_cohort_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print({k: len(v) for k,v in cohorts.items()})


if __name__ == "__main__":
    main()
