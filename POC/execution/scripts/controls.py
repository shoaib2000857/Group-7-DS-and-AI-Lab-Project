"""Non-learned B0 controls (no reference data at inference).

evidence_template: absolute-difference candidate regions (A's listed classical
  control), reported only as candidate changed regions at image-relative
  positions. It never names an object type or a count of objects.
constant_nochange: always "there is no difference ." -- the majority-style
  caption, to show what n-gram metrics reward on a 50%-unchanged dataset.

  python scripts/controls.py --roles development_smoke,development,screening
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from common import RUNS, Timer, load_manifest, new_run_record, now, pair_paths, percentile  # noqa: E402

GRID_NAMES = [["upper-left", "top", "upper-right"],
              ["left", "centre", "right"],
              ["lower-left", "bottom", "lower-right"]]
# Frozen on the 50 development pairs (pair-level accuracy 0.84 there); not tuned on screening.
DIFF_SIGMA = 1.75       # threshold on the standardised absolute difference
MIN_CHANGED_FRAC = 0.05 # pair-level gate: below this, report no candidate change
MIN_AREA_FRAC = 0.004   # ~262 px at 256x256; smallest reported region


def candidate_regions(t1, t2):
    a = np.asarray(t1.convert("RGB"), dtype=np.float32) / 255
    b = np.asarray(t2.convert("RGB"), dtype=np.float32) / 255
    # per-image standardisation reduces global illumination shifts
    a = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6)
    b = (b - b.mean((0, 1))) / (b.std((0, 1)) + 1e-6)
    d = ndimage.gaussian_filter(np.abs(b - a).mean(2), 2)
    mask = ndimage.binary_opening(d > DIFF_SIGMA, iterations=2)
    if mask.mean() <= MIN_CHANGED_FRAC:
        return [], np.zeros_like(mask)
    lab, n = ndimage.label(mask)
    regions = []
    h, w = mask.shape
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        area = int((lab[sl] == i).sum())
        if area < MIN_AREA_FRAC * h * w:
            continue
        ys, xs = np.nonzero(lab == i)
        cy, cx = ys.mean(), xs.mean()
        regions.append({
            "region_id": f"r{len(regions)}",
            "bbox_x0y0x1y1": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
            "area_px": area,
            "grid": [min(2, int(3 * cy / h)), min(2, int(3 * cx / w))],
        })
    return regions, mask


def template_caption(regions):
    if not regions:
        return "No candidate changed region was detected."
    names = []
    for r in sorted(regions, key=lambda r: -r["area_px"]):
        n = GRID_NAMES[r["grid"][0]][r["grid"][1]]
        if n not in names:
            names.append(n)
    names = names[:3]
    where = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    noun = "region" if len(regions) == 1 else "regions"
    return f"Candidate changed {noun} at the {where} of the image; object type not determined."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="development_smoke")
    args = ap.parse_args()
    roles = args.roles.split(",")
    rows = load_manifest(roles)
    for control in ("evidence_template", "constant_nochange"):
        run_id = f"B0_CTRL_{control}_{'+'.join(roles)}"
        out = RUNS / run_id
        out.mkdir(parents=True, exist_ok=True)
        rec = new_run_record(
            run_id=run_id, role_model_id=f"CTRL_{control}",
            model_identity={"type": "non_learned_control", "name": control,
                            "params": {"DIFF_SIGMA": DIFF_SIGMA, "MIN_CHANGED_FRAC": MIN_CHANGED_FRAC, "MIN_AREA_FRAC": MIN_AREA_FRAC}
                            if control == "evidence_template" else {}},
            training_data_exposure="none", split="+".join(sorted({r["official_split"] for r in rows})),
            sample_count=len(rows), condition="B0", prompt_revision=None, seed=None,
            precision_and_quantisation="float32 numpy, CPU",
            limitations="Evidence template reports pixel differences, not semantic change; shadows, "
                        "vegetation and illumination also trigger regions." if control == "evidence_template"
            else "Ignores the images entirely.",
        )
        times = []
        with open(out / "captions.jsonl", "w") as f:
            for row in rows:
                p1, p2 = pair_paths(row)
                with Timer() as t:
                    if control == "evidence_template":
                        regions, _ = candidate_regions(Image.open(p1), Image.open(p2))
                        text = template_caption(regions)
                    else:
                        regions, text = None, "there is no difference ."
                times.append(t.s)
                f.write(json.dumps({"pair_id": row["pair_id"], "run_id": run_id, "model": f"CTRL_{control}",
                                    "condition": "B0", "poc_role": row["poc_role"], "raw_text": text,
                                    "status": "ok", "gen_s": round(t.s, 4), "regions": regions}) + "\n")
        rec.update(status="completed", end_utc=now(), completion_rate=1.0, cold_load_s=0.0,
                   warm_latency_median_s=percentile(times, 0.5), warm_latency_p95_s=percentile(times, 0.95),
                   raw_output_path=str((out / "captions.jsonl").relative_to(RUNS.parent)))
        json.dump(rec, open(out / "run_record.json", "w"), indent=1)
        print(run_id, "done", len(rows))


if __name__ == "__main__":
    main()
