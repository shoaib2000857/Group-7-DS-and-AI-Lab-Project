"""Cached B0 captions + contract-shaped claims for C (visual evidence) and D (decisions).

Baseline RSCaMa, challenger Qwen3-VL; development and screening, unchanged
pairs included. References and automatic-check labels are kept in a separate
file so they are never passed to inference.

  runs/handoff/<system>/captions.jsonl
  runs/handoff/<system>/claims.jsonl          evidence/decision fields empty
  runs/handoff/evaluation_only/reference_check.jsonl
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, RUNS  # noqa: E402

SYSTEMS = {"baseline_rscama": "B0_C02_rscama_development_smoke+development+screening",
           "challenger_qwen3vl": "B0_C04_qwen3vl_development_smoke+development+screening"}


def main():
    out = RUNS / "handoff"
    if out.exists():
        shutil.rmtree(out)
    (out / "evaluation_only").mkdir(parents=True)
    checks = []
    for name, run in SYSTEMS.items():
        d = out / name
        d.mkdir()
        shutil.copy(RUNS / run / "captions.jsonl", d / "captions.jsonl")
        shutil.copy(RUNS / run / "run_record.json", d / "run_record.json")
        with open(d / "claims.jsonl", "w") as f:
            for split in ("development", "screening"):
                for line in open(RUNS / run / f"claims_{split}.jsonl"):
                    c = json.loads(line)
                    chk = c.pop("b_reference_check")
                    checks.append({"system": name, "claim_id": c["claim_id"], "pair_id": c["pair_id"], **chk})
                    f.write(json.dumps(c) + "\n")
    with open(out / "evaluation_only" / "reference_check.jsonl", "w") as f:
        for c in checks:
            f.write(json.dumps(c) + "\n")
    shutil.copy(DATA / "manifest.csv", out / "manifest.csv")
    (out / "README.md").write_text(
        "B0 handoff for C and D. captions.jsonl: raw model output per pair; claims.jsonl: one claim per line "
        "in templates/claim_decision.json shape (b_source holds offsets, parser flags and location grid). "
        "evaluation_only/ holds automatic oracle-mask labels: never feed it to a checker. "
        "Pairs: 50 development + 100 screening; sealed test excluded. See POC/b-captioning/README.md.\n")
    print({p.name: sum(1 for _ in open(p / "claims.jsonl")) for p in out.iterdir() if (p / "claims.jsonl").exists()},
          len(checks))


if __name__ == "__main__":
    main()
