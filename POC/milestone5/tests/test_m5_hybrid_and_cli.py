"""Automated pytest suite for Milestone 5 Calibrated Mask-First Hybrid Verifier,
model registry SHA-256 verification, and single-pair UI bundle generation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
M5_SCRIPTS = REPO_ROOT / "POC/milestone5/scripts"
for p in (str(REPO_ROOT), str(M5_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_terrawatch_inference import (
    route_mask_first_hybrid,
    run_single_pair_bundle,
    verify_registry_checksums,
)


def test_mask_first_hybrid_routing_rules():
    # 1. Out-of-ontology -> deterministic defer (0 VLM)
    out = route_mask_first_hybrid(
        {"claim": {"object": "vegetation", "action": "added"}},
        "defer", {"class_change_pixels": 500, "any_change_pixels": 500},
        ["T_out_of_mask_ontology"], [],
    )
    assert out["outcome"] == "defer" and not out["vlm_triggered"]

    # 2. Strong BIT reject -> trust reject (0 VLM)
    out = route_mask_first_hybrid(
        {"claim": {"object": "building", "action": "added"}},
        "reject", {"class_change_pixels": 0, "any_change_pixels": 0},
        [], [],
    )
    assert out["outcome"] == "reject" and not out["vlm_triggered"]

    # 3. High-confidence BIT accept -> accept directly (0 VLM)
    out = route_mask_first_hybrid(
        {"claim": {"object": "building", "action": "added"}},
        "accept", {"class_change_pixels": 1200, "any_change_pixels": 1200},
        [], [],
    )
    assert out["outcome"] == "accept" and not out["vlm_triggered"]

    # 4. Direction-sensitive 'removed' -> escalates to VLM and rejects on high pR
    out = route_mask_first_hybrid(
        {"claim": {"object": "building", "action": "removed"}},
        "accept", {"class_change_pixels": 1200, "any_change_pixels": 1200},
        [], [], vlm_probs={"accept": 0.05, "reject": 0.90, "defer": 0.05},
    )
    assert out["outcome"] == "reject" and out["vlm_triggered"]

    # 5. Residual no_change without VLM -> safe defer
    out = route_mask_first_hybrid(
        {"claim": {"object": "scene", "action": "no_change"}},
        "accept", {"class_change_pixels": 25, "any_change_pixels": 25},
        [], [], vlm_probs=None,
    )
    assert out["outcome"] == "defer" and out["vlm_triggered"]


def test_val_147_claims_replay_matches_locked_calibration():
    val_path = REPO_ROOT / "POC/milestone4/rework/evidence/verification/verification_claims.jsonl"
    if not val_path.is_file():
        return
    claims = [json.loads(l) for l in val_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(claims) == 147

    acc_sup = acc_tot = rej_con = tot_con = vlm_calls = 0
    for c in claims:
        if c["proxy_gt"] == "contradicted":
            tot_con += 1
        res = route_mask_first_hybrid(
            c, c["dec_mask_only"], c["mask_summary"], c["triggers"], c["forced_reasons"],
            vlm_probs=c["raw_direct"]["probabilities"],
            tau_accept=0.60, tau_reject=0.75,
        )
        vlm_calls += int(res["vlm_triggered"])
        if res["outcome"] == "accept":
            acc_tot += 1
            if c["proxy_gt"] == "supported":
                acc_sup += 1
        elif res["outcome"] == "reject" and c["proxy_gt"] == "contradicted":
            rej_con += 1

    prec = acc_sup / acc_tot
    catch = rej_con / tot_con
    assert prec >= 0.94, f"Expected >=94% val precision, got {prec:.4f}"
    assert catch >= 0.70, f"Expected >=70% contradicted catch, got {catch:.4f}"
    assert vlm_calls <= 20, f"Expected <=20 VLM calls (<=13.6%), got {vlm_calls}"


def test_model_registry_verifies_real_local_checkpoints():
    reg = REPO_ROOT / "POC/milestone5/configs/model_registry.json"
    runs = REPO_ROOT / "POC/milestone4/rework/runs"
    res = verify_registry_checksums(reg, runs)
    assert res["all_present_verified"] is True
    assert res["checkpoints"]["D03_bit_lr3e4"]["status"] == "ok"
    assert res["checkpoints"]["A03_siam_r18_unet"]["status"] == "ok"


def test_single_pair_bundle_generation_cpu_offline(tmp_path: Path):
    t1 = tmp_path / "T1.png"
    t2 = tmp_path / "T2.png"
    img1 = np.full((256, 256, 3), 120, dtype=np.uint8)
    img2 = img1.copy()
    img2[80:140, 80:140] = (210, 180, 160)
    Image.fromarray(img1).save(t1)
    Image.fromarray(img2).save(t2)

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[80:140, 80:140] = 2  # building

    bundle = run_single_pair_bundle(
        t1, t2, tmp_path / "out", "smoke_pair_01", mask,
        "a new building has been built in the center",
        detector_meta={"arch": "bit_resnet18"}, captioner_meta={"mode": "test"}, referee=None,
    )
    rep = json.loads((bundle / "report.json").read_text(encoding="utf-8"))
    assert (bundle / "mask_overlay.png").is_file()
    assert (bundle / "review_inputs.json").is_file()
    assert rep["claims"][0]["decision"]["outcome"] == "accept"
    assert rep["claims"][0]["decision"]["vlm_triggered"] is False
