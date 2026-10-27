"""Blinded human-audit packet for the pilot (10 common screening pairs).

Writes, under runs/audit_pilot/ (git-ignored, contains imagery):
  sheet.csv     one row per (anonymous output, claim) in the team's
                claim_annotations.csv format; judgment columns left blank
  viewer.html   T1 / T2 side by side with anonymised outputs, for raters
  key.json      anonymous id -> run id (keep away from raters)
Raters establish reference facts from the images first, then judge each
claim: supported / unsupported / contradicted / unverifiable.

  python scripts/make_audit_packet.py runs/<run_a> runs/<run_b> ...
"""
import base64
import csv
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claims import extract_claims  # noqa: E402
from common import DATA, RUNS, load_manifest, pair_paths  # noqa: E402

FIELDS = ("pair_id,anonymous_output_id,claim_id,rater_id,source_span,object_type,action,location,quantity,"
          "quantity_unit,judgment,object_correct,action_correct,location_correct,count_correct,evidence_reference,"
          "ambiguity_reason,omitted_reference_fact_ids,annotation_round,notes").split(",")


def main():
    run_dirs = [Path(p) for p in sys.argv[1:]]
    pairs = sorted(load_manifest(["screening"]), key=lambda r: int(r["selection_order"]))[:10]
    out = RUNS / "audit_pilot"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    outputs = {}
    for d in run_dirs:
        for l in open(d / "captions.jsonl"):
            r = json.loads(l)
            outputs.setdefault(r["pair_id"], []).append((d.name, r.get("raw_text") or ""))
    key, rows, blocks = {}, [], []
    for p in pairs:
        pid = p["pair_id"]
        outs = outputs.get(pid, [])
        rng.shuffle(outs)
        cards = []
        for j, (run, text) in enumerate(outs):
            anon = f"{pid.split('/')[1]}-{chr(65 + j)}"
            key[anon] = run
            cs = extract_claims(text, pid, anon)
            for c in cs or [None]:
                rows.append({
                    "pair_id": pid, "anonymous_output_id": anon,
                    "claim_id": c["claim_id"] if c else f"{anon}:no-claims",
                    "source_span": c["source_span"] if c else text,
                    "object_type": c["claim"]["object"] if c else "",
                    "action": c["claim"]["action"] if c else "",
                    "location": (c["claim"]["location"] or {}).get("text", "") if c else "",
                    "quantity": c["claim"]["quantity"] if c and c["claim"]["quantity"] is not None else "",
                    "quantity_unit": c["claim"]["quantity_unit"] or "" if c else "",
                    "annotation_round": "pilot-1",
                })
            cards.append(f"<div class=out><b>Output {chr(65 + j)}</b><p>{html.escape(text)}</p></div>")
        imgs = "".join(
            f"<figure><img src='data:image/png;base64,{base64.b64encode(Path(x).read_bytes()).decode()}'>"
            f"<figcaption>{lab}</figcaption></figure>"
            for x, lab in zip(pair_paths(p), ("T1 (earlier)", "T2 (later)")))
        blocks.append(f"<section><h2>{pid}</h2><div class=imgs>{imgs}</div>"
                      f"<p class=hint>Write the reference facts for this pair before reading the outputs.</p>"
                      f"<details><summary>Show anonymised outputs</summary>{''.join(cards)}</details></section>")
    with open(out / "sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    json.dump(key, open(out / "key.json", "w"), indent=1)
    (out / "viewer.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Audit pilot</title><style>"
        "body{font:15px system-ui;max-width:900px;margin:24px auto;padding:0 16px}"
        "img{width:300px;image-rendering:pixelated}.imgs{display:flex;gap:16px;flex-wrap:wrap}"
        "figure{margin:0}.out{border:1px solid #ccc;border-radius:6px;padding:8px 12px;margin:8px 0}"
        ".hint{color:#666}</style><h1>TerraWatch B audit pilot</h1>"
        "<p>Judge each claim in sheet.csv: supported / unsupported / contradicted / unverifiable. "
        "Do not open key.json.</p>" + "".join(blocks))
    print(f"{len(pairs)} pairs, {len(key)} outputs, {len(rows)} claim rows -> {out}")


if __name__ == "__main__":
    main()
