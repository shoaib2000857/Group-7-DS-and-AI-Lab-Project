"""TerraWatch Production Single-Command Inference & Checkpoint Integrity CLI (Milestone 5 WS7).

Usage:
  1. Verify local checkpoints against configs/model_registry.json:
     python POC/milestone5/scripts/run_terrawatch_inference.py --verify-checksums POC/milestone4/rework/runs

  2. Run full end-to-end change detection, captioning, and Calibrated Mask-First Hybrid verification
     on any bitemporal satellite pair (T1, T2) and write an Evidence Review UI-ready bundle:
     python POC/milestone5/scripts/run_terrawatch_inference.py \
       --t1 path/to/T1.png --t2 path/to/T2.png --out outputs/demo_pair \
       --detector-ckpt path/to/D03_bit_lr3e4/best.pt \
       --captioner-ckpt path/to/C06_rscama_frozen_clip/best.pth \
       --bit-source-dir path/to/BIT_CD --rscama-source-dir path/to/RSCaMa
     python -m src.app.server outputs/demo_pair --port 8765
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_registry_checksums(registry_path: Path, checkpoints_root: Path) -> Dict[str, Any]:
    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    results = {}
    all_ok = True
    for name, meta in reg.get("checkpoints", {}).items():
        p = checkpoints_root / meta["rel_path"]
        if not p.is_file():
            results[name] = {"status": "missing", "expected_path": str(p)}
            continue
        actual_bytes = p.stat().st_size
        size_ok = actual_bytes == meta.get("bytes", actual_bytes)
        expected_sha = meta.get("sha256")
        if expected_sha:
            actual_sha = sha256_file(p)
            sha_ok = actual_sha == expected_sha
        else:
            actual_sha = None
            sha_ok = True
        ok = size_ok and sha_ok
        if not ok:
            all_ok = False
        results[name] = {
            "status": "ok" if ok else "mismatch",
            "bytes": actual_bytes,
            "expected_bytes": meta.get("bytes"),
            "sha256_match": sha_ok,
            "sha256": actual_sha[:16] if actual_sha else None,
        }
    return {"all_present_verified": all_ok, "checked_root": str(checkpoints_root), "checkpoints": results}


def route_mask_first_hybrid(
    claim_dict: dict,
    mask_only_dec: str,
    mask_summary: dict,
    triggers: List[str],
    forced_reasons: List[str],
    vlm_probs: Optional[Dict[str, float]] = None,
    tau_accept: float = 0.60,
    tau_reject: float = 0.75,
) -> Dict[str, Any]:
    """Val-locked Calibrated Mask-First Hybrid Verifier routing decision."""
    obj = claim_dict["claim"]["object"]
    act = claim_dict["claim"]["action"]
    trigs = set(triggers)
    forced = set(forced_reasons)
    any_px = int(mask_summary.get("any_change_pixels", 0))
    cls_px = int(mask_summary.get("class_change_pixels", 0))

    if (
        "T_out_of_mask_ontology" in trigs
        or "T_exact_numeric_count" in trigs
        or "exact_count_without_independent_instance_evidence" in forced
        or "exact_numeric_count_unverified" in forced
    ):
        return {
            "outcome": "defer",
            "probabilities": {"accept": 0.50, "reject": 0.0, "defer": 0.50},
            "vlm_triggered": False,
            "route_reason": "deterministic_defer_ontology_or_exact_count",
            "reason_codes": sorted(forced | (trigs & {"T_out_of_mask_ontology", "T_exact_numeric_count"})),
        }

    if mask_only_dec == "reject":
        return {
            "outcome": "reject",
            "probabilities": {"accept": 0.05, "reject": 0.95, "defer": 0.0},
            "vlm_triggered": False,
            "route_reason": "mask_first_reject",
            "reason_codes": ["mask_contradicts_claim"],
        }

    escalate_reason = None
    if act == "removed":
        escalate_reason = "vlm_escalate_direction_removed"
    elif act == "no_change" and (any_px >= 10 or "T_hedged_claim" in trigs):
        escalate_reason = "vlm_escalate_no_change_residual_or_hedged"
    elif act == "added" and obj in ("road", "building") and cls_px < 200:
        escalate_reason = "vlm_escalate_weak_mask_support"

    if not escalate_reason:
        conf = 0.94 if (act == "no_change" and any_px == 0) else min(0.96, 0.84 + 0.12 * min(cls_px / 2000.0, 1.0))
        return {
            "outcome": "accept",
            "probabilities": {"accept": round(conf, 4), "reject": 0.0, "defer": round(1.0 - conf, 4)},
            "vlm_triggered": False,
            "route_reason": "mask_first_accept",
            "reason_codes": [],
        }

    if vlm_probs is None:
        return {
            "outcome": "defer",
            "probabilities": {"accept": 0.50, "reject": 0.0, "defer": 0.50},
            "vlm_triggered": True,
            "route_reason": f"{escalate_reason}->vlm_unavailable_safe_defer",
            "reason_codes": [escalate_reason, "vlm_referee_disabled"],
        }

    pA = float(vlm_probs.get("accept", 0.0))
    pR = float(vlm_probs.get("reject", 0.0))
    pD = float(vlm_probs.get("defer", max(0.0, 1.0 - pA - pR)))
    if pR >= tau_reject:
        out = "reject"
    elif pA >= tau_accept:
        out = "accept"
    else:
        out = "defer"
    return {
        "outcome": out,
        "probabilities": {"accept": round(pA, 4), "reject": round(pR, 4), "defer": round(pD, 4)},
        "vlm_triggered": True,
        "route_reason": f"{escalate_reason}->vlm_{out}",
        "reason_codes": [escalate_reason],
    }


def run_single_pair_bundle(
    t1_path: Path,
    t2_path: Path,
    out_dir: Path,
    pair_id: str,
    pred_mask: np.ndarray,
    caption: str,
    detector_meta: Dict[str, Any],
    captioner_meta: Dict[str, Any],
    referee: Optional[Any] = None,
) -> Path:
    """Write a complete Evidence Review UI-ready bundle using the Calibrated Mask-First Hybrid Verifier."""
    from src.evidence.visual_inspector import VisualEvidenceInspector
    try:
        from src.inference.verification_pipeline import (
            candidate_box_and_components,
            compute_contextual_bbox,
            compute_disagreement_triggers,
            make_mask_overlay,
            mask_only_decision,
        )
    except ImportError:
        m4_scripts = str(REPO_ROOT / "POC/milestone4/rework/scripts")
        if m4_scripts not in sys.path:
            sys.path.insert(0, m4_scripts)
        from pipeline_recovery import (
            candidate_box_and_components,
            compute_contextual_bbox,
            compute_disagreement_triggers,
            make_mask_overlay,
            mask_only_decision,
        )
    from src.models.claims import extract_claims, to_contract

    pair_dir = out_dir / pair_id.replace("/", "_")
    crops_dir = pair_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    t1_dst = pair_dir / "T1.png"
    t2_dst = pair_dir / "T2.png"
    t1_rgb = np.array(Image.open(t1_path).convert("RGB"))
    t2_rgb = np.array(Image.open(t2_path).convert("RGB"))
    Image.fromarray(t1_rgb).save(t1_dst)
    Image.fromarray(t2_rgb).save(t2_dst)
    Image.fromarray(pred_mask.astype(np.uint8)).save(pair_dir / "detector_response.mask.png")

    parsed = [(c, to_contract(c)) for c in extract_claims(caption, pair_id=pair_id, run_id="terrawatch_cli")]
    if not parsed:
        fb = {
            "pair_id": pair_id,
            "claim_id": f"{pair_id}:c0",
            "source_span": caption,
            "claim": {"object": "scene", "action": "no_change", "location": None, "quantity": None, "quantity_unit": None},
            "parser": {"quantity_kind": None, "quantity_text": None, "hedged": False, "negated": True, "version": "fallback"},
        }
        parsed = [(fb, fb)]

    inspector = VisualEvidenceInspector(min_crop_size=24)
    report_claims = []
    first_box = None

    for ci, (c_raw, contract) in enumerate(parsed):
        claim_id = f"{pair_id}:c{ci}"
        bbox, localized, mask_summary = candidate_box_and_components(pred_mask, c_raw, min_component_px=16)
        padded_bbox, transform_meta = compute_contextual_bbox(bbox, t1_rgb.shape[:2], padding_fraction=0.25, min_size=32)
        if first_box is None:
            first_box = padded_bbox
        x0, y0, x1, y1 = padded_bbox
        c1 = t1_rgb[y0:y1, x0:x1].copy()
        c2 = t2_rgb[y0:y1, x0:x1].copy()
        c1_name, c2_name = f"claim_{ci}_crop_T1.png", f"claim_{ci}_crop_T2.png"
        Image.fromarray(c1).save(crops_dir / c1_name)
        Image.fromarray(c2).save(crops_dir / c2_name)

        wit_dict = inspector.inspect_crop_pair(
            c1, c2,
            claimed_object=c_raw["claim"]["object"] or "building",
            claimed_action=c_raw["claim"]["action"] or "added",
        )
        triggers = compute_disagreement_triggers(c_raw, localized, mask_summary, wit_dict, mask_alarm_px=50)
        forced_reasons = []
        if wit_dict.get("visible_change") is None:
            forced_reasons.append("insufficient_visual_evidence")
        if c_raw.get("parser", {}).get("quantity_kind") == "exact":
            forced_reasons.append("exact_count_without_independent_instance_evidence")
        dec_mask = mask_only_decision(c_raw, localized, mask_summary, alarm_px=50)

        # Peek whether hybrid routing needs VLM before invoking Qwen3-VL
        probe = route_mask_first_hybrid(c_raw, dec_mask, mask_summary, triggers, forced_reasons, vlm_probs=None)
        if probe["vlm_triggered"] and referee is not None:
            direct_state = {
                "claim": contract["source_span"],
                "structured_claim": contract["claim"],
                "temporal_order": "T1_earlier_T2_later",
            }
            raw_dir = referee.decide(direct_state, [str(t1_dst), str(t2_dst)], mode="direct_full_image")
            decision = route_mask_first_hybrid(c_raw, dec_mask, mask_summary, triggers, forced_reasons, vlm_probs=raw_dir["probabilities"])
        else:
            decision = probe

        report_claims.append({
            "pair_id": pair_id,
            "claim_id": claim_id,
            "source_span": contract["source_span"],
            "claim": contract["claim"],
            "candidate_bbox": padded_bbox,
            "parser_provenance": {"parser": c_raw.get("parser", {})},
            "evidence": {
                **wit_dict,
                "crop_paths": [f"crops/{c1_name}", f"crops/{c2_name}"],
                "transform_record": transform_meta,
                "mask_summary": mask_summary,
                "disagreement_triggers": triggers,
            },
            "decision": decision,
            "mask_only_decision": dec_mask,
        })

    Image.fromarray(make_mask_overlay(t2_rgb, pred_mask, first_box)).save(pair_dir / "mask_overlay.png")
    accepted_spans = list(dict.fromkeys(rc["source_span"] for rc in report_claims if rc["decision"]["outcome"] == "accept"))
    report = {
        "pair_id": pair_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "terrawatch_m5_calibrated_mask_first_hybrid",
        "input_sha256": [sha256_file(t1_dst), sha256_file(t2_dst)],
        "generated_caption": caption,
        "verified_description": "; ".join(accepted_spans) if accepted_spans else "The image pair requires review before a change description can be confirmed.",
        "review_required": (not accepted_spans) or any(rc["decision"]["outcome"] == "defer" for rc in report_claims),
        "claims": report_claims,
        "errors": [],
        "model_metadata": {
            "detector": detector_meta,
            "captioner": captioner_meta,
            "referee": getattr(referee, "metadata", {"mode": "disabled_safe_defer"}),
        },
    }
    (pair_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (pair_dir / "review_inputs.json").write_text(
        json.dumps({"images": ["T1.png", "T2.png"], "mask": "detector_response.mask.png", "overlay": "mask_overlay.png"}, indent=2),
        encoding="utf-8",
    )
    return pair_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="TerraWatch Production Inference & Checkpoint Integrity CLI")
    ap.add_argument("--registry", type=Path, default=Path(__file__).resolve().parents[1] / "configs/model_registry.json")
    ap.add_argument("--verify-checksums", type=Path, default=None, help="Verify all local checkpoints under this root against model_registry.json")
    ap.add_argument("--t1", type=Path, default=None)
    ap.add_argument("--t2", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/terrawatch_cli_bundle"))
    ap.add_argument("--pair-id", default="cli_pair_001")
    ap.add_argument("--detector-ckpt", type=Path, default=None)
    ap.add_argument("--captioner-ckpt", type=Path, default=None)
    ap.add_argument("--caption", type=str, default=None, help="Optional caption override (skips neural captioner)")
    ap.add_argument("--bit-source-dir", type=Path, default=None)
    ap.add_argument("--rscama-source-dir", type=Path, default=None)
    ap.add_argument("--no-vlm", action="store_true", help="Disable Qwen3-VL escalation (escalated claims safely defer)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.verify_checksums is not None:
        res = verify_registry_checksums(args.registry, args.verify_checksums)
        print(json.dumps(res, indent=2))
        sys.exit(0 if res["all_present_verified"] else 1)

    if not args.t1 or not args.t2:
        ap.error("Provide --verify-checksums DIR, or --t1 and --t2 for inference.")

    import torch
    from src.models.detector_architectures import build_model
    from scripts.train_captioner_full import build_rscama_components

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    t1_rgb = np.array(Image.open(args.t1).convert("RGB"))
    t2_rgb = np.array(Image.open(args.t2).convert("RGB"))

    assert args.detector_ckpt and args.detector_ckpt.is_file(), "Provide valid --detector-ckpt"
    det_model, _, _, _ = build_model("bit_resnet18", device, str(args.bit_source_dir) if args.bit_source_dir else None, None)
    det_model.load_state_dict(torch.load(args.detector_ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
    det_model.eval()
    with torch.inference_mode():
        x1 = torch.from_numpy(t1_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 127.5 - 1.0
        x2 = torch.from_numpy(t2_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 127.5 - 1.0
        pred_mask = det_model(x1, x2).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    det_meta = {"arch": "bit_resnet18", "checkpoint": args.detector_ckpt.name, "sha256": sha256_file(args.detector_ckpt)}

    if args.caption is not None:
        caption = args.caption
        cap_meta = {"mode": "cli_override"}
    else:
        assert args.captioner_ckpt and args.rscama_source_dir, "Provide --captioner-ckpt and --rscama-source-dir (or --caption)"
        c_state = torch.load(args.captioner_ckpt, map_location="cpu", weights_only=False)
        vocab = c_state["vocab"]
        models, processor, kernel_mode = build_rscama_components(args.rscama_source_dir, vocab, device)
        for m, k in zip(models, ["encoder_dict", "encoder_trans_dict", "decoder_dict"]):
            m.load_state_dict(c_state[k], strict=True)
            m.to(device).eval()
        inv = {int(v): w for w, v in vocab.items()}
        skip = {vocab[w] for w in ("<START>", "<END>", "<NULL>")}
        with torch.inference_mode():
            im1 = processor(Image.fromarray(t1_rgb)).unsqueeze(0).to(device)
            im2 = processor(Image.fromarray(t2_rgb)).unsqueeze(0).to(device)
            f1, f2 = models[0](im1, im2)
            toks = models[2].sample(models[1](f1, f2), k=1)
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            while toks and isinstance(toks[0], list):
                toks = toks[0]
            caption = " ".join(inv[int(t)] for t in toks if int(t) not in skip)
        cap_meta = {"model": "C06_rscama_frozen_clip", "checkpoint": args.captioner_ckpt.name, "sha256": sha256_file(args.captioner_ckpt), "kernel_mode": kernel_mode}

    referee = None
    if not args.no_vlm:
        from src.inference.verification_pipeline import Qwen3VLReferee
        referee = Qwen3VLReferee(device=device)

    out_bundle = run_single_pair_bundle(args.t1, args.t2, args.out, args.pair_id, pred_mask, caption, det_meta, cap_meta, referee=referee)
    print(f"WROTE_UI_BUNDLE: {out_bundle}")
    print(f"Launch UI: python -m src.app.server {args.out} --port 8765")


if __name__ == "__main__":
    main()
