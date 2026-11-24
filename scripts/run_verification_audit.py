"""Run frozen-checkpoint Verification Comparison Study (120 stratified val pairs)
and generate the Authentic End-to-End Trace Bundle compatible with the Evidence Review UI.

Uses ONLY real learned model weights (selected BIT checkpoint, selected RSCaMa checkpoint,
OpenCV structural visual witness, frozen Qwen3-VL-4B-Instruct) with zero GT leakage into inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.detector_architectures import build_model
from src.inference.verification_pipeline import (
    Qwen3VLReferee,
    apply_decision_gate,
    candidate_box_and_components,
    compute_contextual_bbox,
    compute_disagreement_triggers,
    make_mask_overlay,
    mask_only_decision,
    sha256_file,
)


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Import existing repo modules cleanly (works both in repo and standalone on Colab if copied)
from src.evidence.visual_inspector import VisualEvidenceInspector
from src.models.claims import extract_claims, to_contract


def select_stratified_pairs(
    val_rows: List[dict],
    ext_root: Path,
    refs: dict,
    seed: int = 42,
) -> List[Tuple[dict, str]]:
    """Select 120 stratified validation pairs across unchanged, building, road, mixed, and small-change."""
    buckets: Dict[str, List[dict]] = {
        "unchanged": [],
        "small_change": [],
        "building_only": [],
        "road_only": [],
        "mixed_both": [],
    }
    for r in val_rows:
        lbl = np.array(Image.open(ext_root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        m = np.zeros_like(lbl, dtype=np.uint8)
        m[lbl == 128] = 1
        m[lbl == 255] = 2
        road_px = int((m == 1).sum())
        bldg_px = int((m == 2).sum())
        tot = road_px + bldg_px
        if tot == 0:
            buckets["unchanged"].append(r)
        elif tot < 655:  # <1% area
            buckets["small_change"].append(r)
        elif bldg_px >= 50 and road_px >= 50:
            buckets["mixed_both"].append(r)
        elif bldg_px >= 50:
            buckets["building_only"].append(r)
        elif road_px >= 50:
            buckets["road_only"].append(r)
        else:
            buckets["small_change"].append(r)

    rng = np.random.default_rng(seed)
    quotas = [
        ("unchanged", 40),
        ("building_only", 30),
        ("road_only", 15),
        ("mixed_both", 20),
        ("small_change", 15),
    ]
    selected: List[Tuple[dict, str]] = []
    seen = set()
    for name, q in quotas:
        pool = buckets[name]
        idx = rng.choice(len(pool), size=min(q, len(pool)), replace=False)
        for i in sorted(idx.tolist()):
            r = pool[i]
            if r["pair_id"] not in seen:
                seen.add(r["pair_id"])
                selected.append((r, name))
    return selected


def proxy_ground_truth_label(claim_dict: dict, gt_mask: np.ndarray, ref_captions: List[str], changeflag: int) -> str:
    """Automatic annotation-proxy label (NOT a substitute for independent human annotation).
    Returns: 'supported' | 'contradicted' | 'unverifiable' | 'abstention'.
    """
    obj = claim_dict["claim"]["object"]
    act = claim_dict["claim"]["action"]
    loc = claim_dict["claim"].get("location")
    if act == "insufficient_evidence":
        return "abstention"
    road_chg = int((gt_mask == 1).sum()) >= 50
    bldg_chg = int((gt_mask == 2).sum()) >= 50
    any_chg = road_chg or bldg_chg

    if act == "no_change":
        if obj == "scene":
            return "supported" if not any_chg else "contradicted"
        if obj == "building":
            return "supported" if not bldg_chg else "contradicted"
        if obj == "road":
            return "supported" if not road_chg else "contradicted"
        return "unverifiable"

    if obj == "region":
        return "supported" if any_chg else "contradicted"
    if obj not in {"building", "road"}:
        return "unverifiable"

    cls_idx = 1 if obj == "road" else 2
    cls_mask = gt_mask == cls_idx
    if int(cls_mask.sum()) < 50:
        return "contradicted"

    # Check location compatibility if stated
    if loc and not loc.get("relational"):
        h, w = cls_mask.shape
        gt_cells = {
            (r, c)
            for r in range(3)
            for c in range(3)
            if cls_mask[r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3].mean() >= 0.015
        }
        claimed = (
            {tuple(x) for x in loc["cells"]}
            if loc.get("cells")
            else {(r, c) for r in loc["rows"] for c in loc["cols"]}
        )
        if gt_cells and not (claimed & gt_cells):
            return "contradicted"

    # Check action compatibility with reference captions if available
    ref_claims = [c for cap in ref_captions for c in extract_claims(cap)]
    ref_obj = [c for c in ref_claims if c["claim"]["object"] == obj]
    if ref_obj:
        compat = {
            "added": {"added", "modified", "changed"},
            "removed": {"removed", "modified", "changed"},
            "modified": {"added", "removed", "modified", "changed"},
            "changed": {"added", "removed", "modified", "changed"},
        }
        if not any(c["claim"]["action"] in compat.get(act, {act}) for c in ref_obj):
            return "contradicted"
    return "supported"


def evaluate_setting(records: List[dict], decision_key: str) -> Dict[str, Any]:
    """Compute supported-claim precision, selective risk, coverage, accept/reject/defer rates."""
    total = len(records)
    accepted = [r for r in records if r[decision_key] == "accept"]
    rejected = [r for r in records if r[decision_key] == "reject"]
    deferred = [r for r in records if r[decision_key] == "defer"]

    sup_acc = sum(1 for r in accepted if r["proxy_gt"] == "supported")
    con_acc = sum(1 for r in accepted if r["proxy_gt"] == "contradicted")
    unv_acc = sum(1 for r in accepted if r["proxy_gt"] == "unverifiable")
    con_rej = sum(1 for r in rejected if r["proxy_gt"] == "contradicted")
    sup_rej = sum(1 for r in rejected if r["proxy_gt"] == "supported")
    total_supported = sum(1 for r in records if r["proxy_gt"] == "supported")
    total_contradicted = sum(1 for r in records if r["proxy_gt"] == "contradicted")

    precision_strict = sup_acc / max(len(accepted), 1)
    selective_risk = (len(accepted) - sup_acc) / max(len(accepted), 1)
    false_accept_rate = con_acc / max(total_contradicted, 1)
    false_reject_rate = sup_rej / max(total_supported, 1)
    supported_recall = sup_acc / max(total_supported, 1)
    coverage = len(accepted) / max(total, 1)

    return {
        "total_claims": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "deferred": len(deferred),
        "accept_rate": round(coverage, 4),
        "reject_rate": round(len(rejected) / max(total, 1), 4),
        "defer_rate": round(len(deferred) / max(total, 1), 4),
        "proxy_supported_precision_strict": round(precision_strict, 4),
        "proxy_selective_risk": round(selective_risk, 4),
        "proxy_supported_recall": round(supported_recall, 4),
        "proxy_contradicted_catch_rate": round(con_rej / max(total_contradicted, 1), 4),
        "proxy_false_accept_rate": round(false_accept_rate, 4),
        "proxy_false_reject_rate": round(false_reject_rate, 4),
        "breakdown": {
            "accepted_supported": sup_acc,
            "accepted_contradicted": con_acc,
            "accepted_unverifiable": unv_acc,
            "rejected_contradicted": con_rej,
            "rejected_supported": sup_rej,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-arch", required=True)
    ap.add_argument("--detector-ckpt", required=True)
    ap.add_argument("--bit-source-dir", required=True)
    ap.add_argument("--changeformer-source-dir", default="")
    ap.add_argument("--captioner-ckpt", default=None)
    ap.add_argument("--captioner-source-dir", default=None)
    ap.add_argument("--captioner-vocab", default=None)
    ap.add_argument("--pred-captions-jsonl", default=None, help="Precomputed frozen-checkpoint greedy captions JSONL (e.g. val_best_captions.jsonl)")
    ap.add_argument("--captioner-run-json", default=None, help="Associated captioner run.json for verified checkpoint SHA-256 provenance")
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--extracted-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--accept-threshold", type=float, default=0.70)
    ap.add_argument("--mask-alarm-px", type=int, default=50)
    ap.add_argument("--num-audit-pairs", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = Path(args.manifests_dir)
    ext = Path(args.extracted_root)

    val_rows = read_jsonl(manifests / "mci_val.jsonl")
    refs = json.loads((manifests / "mci_references.json").read_text(encoding="utf-8"))
    selected = select_stratified_pairs(val_rows, ext, refs, seed=42)[: args.num_audit_pairs]

    det_ckpt = Path(args.detector_ckpt)
    det_sha = sha256_file(det_ckpt)

    if args.pred_captions_jsonl:
        cap_run_meta = json.loads(Path(args.captioner_run_json).read_text(encoding="utf-8")) if args.captioner_run_json else {}
        cap_ckpt_name = Path(cap_run_meta.get("best_checkpoint", args.captioner_ckpt or "best.pth")).name
        cap_ckpt = Path(cap_run_meta.get("best_checkpoint", args.captioner_ckpt or cap_ckpt_name))
        cap_sha = cap_run_meta.get("best_checkpoint_sha256") or (sha256_file(Path(args.captioner_ckpt)) if args.captioner_ckpt and Path(args.captioner_ckpt).is_file() else "unknown")
        kernel_mode = cap_run_meta.get("kernel_mode", "precomputed_from_frozen_best_pth")
        cap_run_id = cap_run_meta.get("run_id", "selected_captioner")
    else:
        assert args.captioner_ckpt and args.captioner_source_dir, "Provide either --pred-captions-jsonl or --captioner-ckpt + --captioner-source-dir"
        cap_ckpt = Path(args.captioner_ckpt)
        cap_sha = sha256_file(cap_ckpt)
        cap_run_id = cap_ckpt.parent.name

    print(f"=== Stage 1/3: Running frozen detector ({args.detector_arch}, sha={det_sha[:12]}) on {len(selected)} stratified val pairs ===", flush=True)
    det_model, _, _, _ = build_model(args.detector_arch, args.device, args.bit_source_dir, args.changeformer_source_dir)
    det_state = torch.load(det_ckpt, map_location="cpu", weights_only=False)
    det_model.load_state_dict(det_state["model"], strict=True)
    det_model.eval()

    pred_masks: Dict[str, np.ndarray] = {}
    amp_dtype = torch.bfloat16 if (args.device.startswith("cuda") and torch.cuda.is_bf16_supported()) else None
    with torch.inference_mode():
        for r, _ in selected:
            t1 = np.array(Image.open(ext / r["t1"]).convert("RGB"))
            t2 = np.array(Image.open(ext / r["t2"]).convert("RGB"))
            x1 = torch.from_numpy(t1.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(args.device) / 127.5 - 1.0
            x2 = torch.from_numpy(t2.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(args.device) / 127.5 - 1.0
            with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
                pred = det_model(x1, x2).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            pred_masks[r["pair_id"]] = pred
    del det_model
    torch.cuda.empty_cache()

    pred_captions: Dict[str, str] = {}
    if args.pred_captions_jsonl:
        print(f"=== Stage 2/3: Loading frozen captioner outputs ({cap_run_id}, sha={cap_sha[:12]}) for {len(selected)} stratified val pairs ===", flush=True)
        all_caps = {rec["pair_id"]: rec["caption"] for rec in read_jsonl(Path(args.pred_captions_jsonl))}
        for r, _ in selected:
            pid = r["pair_id"]
            if pid not in all_caps:
                raise KeyError(f"Missing pair_id {pid} in {args.pred_captions_jsonl}")
            pred_captions[pid] = all_caps[pid]
    else:
        from scripts.train_captioner_full import build_rscama_components
        print(f"=== Stage 2/3: Running frozen captioner (sha={cap_sha[:12]}) on {len(selected)} stratified val pairs ===", flush=True)
        c_state = torch.load(cap_ckpt, map_location="cpu", weights_only=False)
        vocab = c_state.get("vocab") or json.loads(Path(args.captioner_vocab).read_text(encoding="utf-8"))
        models, processor, kernel_mode = build_rscama_components(Path(args.captioner_source_dir), vocab, args.device)
        for m, k in zip(models, ["encoder_dict", "encoder_trans_dict", "decoder_dict"]):
            m.load_state_dict(c_state[k], strict=True)
            m.to(args.device).eval()
        inverse = {int(v): w for w, v in vocab.items()}
        skip_ids = {vocab[w] for w in ("<START>", "<END>", "<NULL>")}

        encoder, fusion, decoder = models
        with torch.inference_mode():
            for r, _ in selected:
                im1 = processor(Image.open(ext / r["t1"]).convert("RGB")).unsqueeze(0).to(args.device)
                im2 = processor(Image.open(ext / r["t2"]).convert("RGB")).unsqueeze(0).to(args.device)
                f1, f2 = encoder(im1, im2)
                toks = decoder.sample(fusion(f1, f2), k=1)
                if hasattr(toks, "tolist"):
                    toks = toks.tolist()
                while toks and isinstance(toks[0], list):
                    toks = toks[0]
                cap = " ".join(inverse[int(t)] for t in toks if int(t) not in skip_ids)
                pred_captions[r["pair_id"]] = cap
        del models, encoder, fusion, decoder
        torch.cuda.empty_cache()

    print("=== Stage 3/3: Witness extraction + Qwen3-VL-4B-Instruct verification comparison ===", flush=True)
    inspector = VisualEvidenceInspector(min_crop_size=24)
    referee = Qwen3VLReferee(device=args.device)

    work_crops = out_dir / "_audit_crops"
    work_crops.mkdir(parents=True, exist_ok=True)

    claim_rows: List[dict] = []
    pair_bundles: Dict[str, dict] = {}
    t_start = time.time()

    for idx, (r, stratum) in enumerate(selected):
        pid = r["pair_id"]
        safe_pid = pid.replace("/", "_")
        p1_path = ext / r["t1"]
        p2_path = ext / r["t2"]
        t1_rgb = np.array(Image.open(p1_path).convert("RGB"))
        t2_rgb = np.array(Image.open(p2_path).convert("RGB"))
        lbl = np.array(Image.open(ext / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        gt_mask = np.zeros_like(lbl, dtype=np.uint8)
        gt_mask[lbl == 128] = 1
        gt_mask[lbl == 255] = 2
        pred_mask = pred_masks[pid]
        caption = pred_captions[pid]
        parsed = [(c, to_contract(c)) for c in extract_claims(caption, pair_id=pid, run_id="m4_recovery")]

        pair_claims = []
        for c_idx, (c_raw, contract) in enumerate(parsed):
            bbox, localized, mask_summary = candidate_box_and_components(pred_mask, c_raw)
            padded_box, transform_meta = compute_contextual_bbox(bbox, t1_rgb.shape[:2], padding_fraction=0.25, min_size=32)
            px0, py0, px1, py1 = padded_box
            crop_t1 = t1_rgb[py0:py1, px0:px1].copy()
            crop_t2 = t2_rgb[py0:py1, px0:px1].copy()

            c1_path = work_crops / f"{safe_pid}_c{c_idx}_crop_T1.png"
            c2_path = work_crops / f"{safe_pid}_c{c_idx}_crop_T2.png"
            Image.fromarray(crop_t1).save(c1_path)
            Image.fromarray(crop_t2).save(c2_path)

            w_obs = inspector.inspect_crop_pair(
                crop_t1, crop_t2,
                claimed_object=c_raw["claim"]["object"] or "building",
                claimed_action=c_raw["claim"]["action"] or "added",
            )
            triggers = compute_disagreement_triggers(c_raw, localized, mask_summary, w_obs, mask_alarm_px=args.mask_alarm_px)
            m_only = mask_only_decision(c_raw, localized, mask_summary, alarm_px=args.mask_alarm_px)

            # Deterministic safety deferrals (FIXED policy: only explicit numeric counts defer, NOT 'a'/'an' articles)
            forced_reasons: List[str] = []
            if w_obs.get("visible_change") is None:
                forced_reasons.append("insufficient_visual_evidence")
            if c_raw["parser"].get("quantity_kind") == "exact":
                forced_reasons.append("exact_count_without_independent_instance_evidence")
            if re.search(r"\billegal\b|\bunauthori[sz]ed\b|\bownership\b", contract["source_span"], re.I):
                forced_reasons.append("nonvisual_legal_or_ownership_claim")

            # Setting 4: Direct full-image Qwen3-VL judge (T1, T2 only, no witness/crop guidance)
            direct_state = {
                "claim": contract["source_span"],
                "structured_claim": contract["claim"],
                "temporal_order": "T1_earlier_T2_later",
            }
            raw_direct = referee.decide(direct_state, [str(p1_path), str(p2_path)], mode="direct_full_image")
            gated_direct = apply_decision_gate(raw_direct, args.accept_threshold, forced_reasons)

            # Setting 5: 4-view Targeted Reinspection (full T1, full T2, contextual crop T1, contextual crop T2 + fallible mask/witness)
            reinsp_state = {
                "claim": contract["source_span"],
                "structured_claim": contract["claim"],
                "predicted_mask_summary": mask_summary,
                "region_localized": localized,
                "candidate_bbox": padded_box,
                "fallible_witness_observation": {
                    "visible_change": w_obs["visible_change"],
                    "state_transition": w_obs["state_transition"],
                    "observations": w_obs["observations"],
                    "quality_flags": w_obs["quality_flags"],
                    "visibility_flags": w_obs["visibility_flags"],
                    "ssim_score": round(w_obs["ssim_score"], 4),
                },
                "disagreement_triggers": triggers,
                "temporal_order": "T1_earlier_T2_later",
            }
            raw_reinsp = referee.decide(
                reinsp_state,
                [str(p1_path), str(p2_path), str(c1_path), str(c2_path)],
                mode="targeted_crop_context_reinspection",
            )
            gated_reinsp = apply_decision_gate(raw_reinsp, args.accept_threshold, forced_reasons)

            # Natural triggered policy: if no trigger fired, keep mask_only_decision; if any trigger fired, use gated_reinsp!
            # (Also record matched-budget all-claim gated_reinsp separately!)
            triggered_decision = gated_reinsp["outcome"] if triggers else m_only
            if forced_reasons:
                triggered_decision = "defer"

            proxy_gt = proxy_ground_truth_label(
                c_raw, gt_mask, refs[pid]["captions"], int(refs[pid].get("source_changeflag", 1))
            )

            row_rec = {
                "pair_id": pid,
                "stratum": stratum,
                "claim_id": contract["claim_id"],
                "caption": caption,
                "source_span": contract["source_span"],
                "claim": contract["claim"],
                "parser": c_raw["parser"],
                "localized": localized,
                "bbox": bbox,
                "padded_bbox": padded_box,
                "transform_meta": transform_meta,
                "crop_paths": [str(c1_path), str(c2_path)],
                "mask_summary": mask_summary,
                "witness": w_obs,
                "triggers": triggers,
                "forced_reasons": forced_reasons,
                "proxy_gt": proxy_gt,
                "dec_unverified": "accept",
                "dec_mask_only": m_only,
                "dec_direct_qwen_raw": raw_direct["outcome"],
                "dec_direct_qwen_gated": gated_direct["outcome"],
                "dec_reinsp_all_raw": raw_reinsp["outcome"],
                "dec_reinsp_all_gated": gated_reinsp["outcome"],
                "dec_triggered_reinsp": triggered_decision,
                "raw_direct": raw_direct,
                "gated_direct": gated_direct,
                "raw_reinsp": raw_reinsp,
                "gated_reinsp": gated_reinsp,
            }
            claim_rows.append(row_rec)
            pair_claims.append(row_rec)

        pair_bundles[pid] = {
            "row": r,
            "stratum": stratum,
            "caption": caption,
            "pred_mask": pred_mask,
            "claims": pair_claims,
        }
        if (idx + 1) % 20 == 0 or (idx + 1) == len(selected):
            print(f"  Verified {idx+1}/{len(selected)} pairs ({len(claim_rows)} claims, elapsed {time.time()-t_start:.1f}s)", flush=True)

    # Compute threshold sweep on validation claims for selective-risk vs coverage
    threshold_sweep = {}
    for th in [0.50, 0.60, 0.70, 0.80, 0.90]:
        for cr in claim_rows:
            g_d = apply_decision_gate(cr["raw_direct"], th, cr["forced_reasons"])
            g_r = apply_decision_gate(cr["raw_reinsp"], th, cr["forced_reasons"])
            cr[f"_tmp_dir_{th}"] = g_d["outcome"]
            cr[f"_tmp_rei_{th}"] = g_r["outcome"]
            cr[f"_tmp_trg_{th}"] = "defer" if cr["forced_reasons"] else (g_r["outcome"] if cr["triggers"] else cr["dec_mask_only"])
        threshold_sweep[f"{th:.2f}"] = {
            "direct_qwen_gated": evaluate_setting(claim_rows, f"_tmp_dir_{th}"),
            "reinsp_all_gated": evaluate_setting(claim_rows, f"_tmp_rei_{th}"),
            "triggered_reinsp": evaluate_setting(claim_rows, f"_tmp_trg_{th}"),
        }

    # Analyze untriggered claims (M1 requirement: measure errors missed by trigger)
    untriggered = [c for c in claim_rows if not c["triggers"]]
    triggered = [c for c in claim_rows if c["triggers"]]
    untriggered_errors = sum(1 for c in untriggered if c["dec_mask_only"] == "accept" and c["proxy_gt"] == "contradicted")

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": {
            "split": "levir_mci_val_stratified",
            "num_pairs": len(selected),
            "strata_counts": dict(Counter(s for _, s in selected)),
            "total_claims": len(claim_rows),
            "note": "Proxy labels derive from GT masks & reference captions for automated development comparison; they are NOT a substitute for independent dual-rater human annotation.",
        },
        "models": {
            "detector": {"arch": args.detector_arch, "checkpoint": str(det_ckpt), "sha256": det_sha},
            "captioner": {"checkpoint": str(cap_ckpt), "sha256": cap_sha, "kernel_mode": kernel_mode},
            "referee": referee.metadata,
        },
        "operating_threshold": args.accept_threshold,
        "trigger_statistics": {
            "triggered_claims": len(triggered),
            "untriggered_claims": len(untriggered),
            "trigger_rate": round(len(triggered) / max(len(claim_rows), 1), 4),
            "untriggered_accepted_contradicted_errors": untriggered_errors,
            "trigger_reason_counts": dict(Counter(t for c in claim_rows for t in c["triggers"])),
            "avg_vlm_calls_per_claim_direct": 1.0,
            "avg_vlm_calls_per_claim_triggered": round(len(triggered) / max(len(claim_rows), 1), 4),
        },
        "settings_comparison": {
            "1_unverified_captioner": evaluate_setting(claim_rows, "dec_unverified"),
            "2_mask_only_checker": evaluate_setting(claim_rows, "dec_mask_only"),
            "3_direct_qwen3vl_raw": evaluate_setting(claim_rows, "dec_direct_qwen_raw"),
            "4_direct_qwen3vl_gated": evaluate_setting(claim_rows, "dec_direct_qwen_gated"),
            "5_reinspection_all_gated": evaluate_setting(claim_rows, "dec_reinsp_all_gated"),
            "6_disagreement_triggered_reinspection": evaluate_setting(claim_rows, "dec_triggered_reinsp"),
        },
        "threshold_sweep": threshold_sweep,
    }
    (out_dir / "verification_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Strip temp keys and write claim-level audit log
    clean_rows = [{k: v for k, v in c.items() if not k.startswith("_tmp_")} for c in claim_rows]
    with (out_dir / "verification_claims.jsonl").open("w", encoding="utf-8") as f:
        for cr in clean_rows:
            f.write(json.dumps(cr) + "\n")

    # Export blind human-audit CSV template for the team's independent dual-rater evaluation
    import csv
    with (out_dir / "human_audit_template.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "stratum", "claim_id", "caption", "claim_span", "object", "action", "location", "quantity", "rater1_label", "rater2_label", "adjudicated_label", "adjudication_notes"])
        for cr in clean_rows:
            w.writerow([
                cr["pair_id"], cr["stratum"], cr["claim_id"], cr["caption"], cr["source_span"],
                cr["claim"]["object"], cr["claim"]["action"], cr["claim"]["location"], cr["claim"]["quantity"],
                "", "", "", ""
            ])

    # Export Authentic End-to-End Trace Bundle (select diverse genuine accepted, rejected, deferred, unchanged cases)
    traces_root = out_dir / "e2e_traces"
    if traces_root.exists():
        shutil.rmtree(traces_root)
    traces_root.mkdir(parents=True)

    def pick_trace(pred_fn, used: set) -> Optional[str]:
        for pid, b in pair_bundles.items():
            if pid in used or not b["claims"]:
                continue
            if pred_fn(b):
                used.add(pid)
                return pid
        return None

    used_pids: set = set()
    chosen_traces: List[Tuple[str, str]] = []
    categories = [
        ("accepted_building_change", lambda b: b["stratum"] in ("building_only", "mixed_both") and any(c["dec_reinsp_all_gated"] == "accept" and c["claim"]["object"] == "building" for c in b["claims"])),
        ("accepted_road_or_mixed", lambda b: b["stratum"] in ("road_only", "mixed_both") and any(c["dec_reinsp_all_gated"] == "accept" and c["claim"]["object"] in ("road", "building") for c in b["claims"])),
        ("accepted_unchanged_scene", lambda b: b["stratum"] == "unchanged" and any(c["dec_reinsp_all_gated"] == "accept" and c["claim"]["action"] == "no_change" for c in b["claims"])),
        ("rejected_claim", lambda b: any(c["dec_reinsp_all_gated"] == "reject" for c in b["claims"])),
        ("deferred_exact_count", lambda b: any("exact_count_without_independent_instance_evidence" in c["forced_reasons"] for c in b["claims"])),
        ("deferred_ambiguous_or_threshold", lambda b: any(c["dec_reinsp_all_gated"] == "defer" and "exact_count_without_independent_instance_evidence" not in c["forced_reasons"] for c in b["claims"])),
        ("small_change_case", lambda b: b["stratum"] == "small_change"),
        ("second_accepted_change", lambda b: b["stratum"] != "unchanged" and any(c["dec_reinsp_all_gated"] == "accept" for c in b["claims"])),
    ]
    for label, fn in categories:
        pid = pick_trace(fn, used_pids)
        if pid:
            chosen_traces.append((label, pid))

    trace_index = []
    for label, pid in chosen_traces:
        b = pair_bundles[pid]
        r = b["row"]
        folder = traces_root / f"{label}__{pid.replace('/', '_')}"
        crops_dir = folder / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        t1_dst = folder / "T1.png"
        t2_dst = folder / "T2.png"
        shutil.copy2(ext / r["t1"], t1_dst)
        shutil.copy2(ext / r["t2"], t2_dst)
        mask_dst = folder / "detector_response.mask.png"
        Image.fromarray(b["pred_mask"]).save(mask_dst)

        t2_rgb = np.array(Image.open(t2_dst).convert("RGB"))
        first_box = b["claims"][0]["padded_bbox"] if b["claims"] else None
        overlay = make_mask_overlay(t2_rgb, b["pred_mask"], first_box)
        Image.fromarray(overlay).save(folder / "mask_overlay.png")

        report_claims = []
        for idx_c, c in enumerate(b["claims"]):
            c1_name = f"claim_{idx_c}_crop_T1.png"
            c2_name = f"claim_{idx_c}_crop_T2.png"
            shutil.copy2(c["crop_paths"][0], crops_dir / c1_name)
            shutil.copy2(c["crop_paths"][1], crops_dir / c2_name)
            report_claims.append({
                "pair_id": pid,
                "claim_id": c["claim_id"],
                "source_span": c["source_span"],
                "claim": c["claim"],
                "candidate_bbox": c["padded_bbox"],
                "parser_provenance": {"parser": c["parser"]},
                "evidence": {
                    **c["witness"],
                    "crop_paths": [f"crops/{c1_name}", f"crops/{c2_name}"],
                    "transform_record": c["transform_meta"],
                    "mask_summary": c["mask_summary"],
                    "disagreement_triggers": c["triggers"],
                },
                "decision": c["gated_reinsp"],
                "direct_qwen_decision": c["gated_direct"],
                "mask_only_decision": c["dec_mask_only"],
            })

        accepted_spans = list(dict.fromkeys(rc["source_span"] for rc in report_claims if rc["decision"]["outcome"] == "accept"))
        report = {
            "pair_id": pid,
            "trace_category": label,
            "stratum": b["stratum"],
            "execution_mode": "model_workers_full_data_recovery",
            "input_sha256": [sha256_file(t1_dst), sha256_file(t2_dst)],
            "generated_caption": b["caption"],
            "verified_description": "; ".join(accepted_spans) if accepted_spans else "The image pair requires review before a change description can be confirmed.",
            "review_required": (not accepted_spans) or any(rc["decision"]["outcome"] == "defer" for rc in report_claims),
            "claims": report_claims,
            "errors": [],
            "model_metadata": {
                "detector": {"arch": args.detector_arch, "checkpoint": det_ckpt.name, "sha256": det_sha},
                "captioner": {"model": "RSCaMa-full-data", "checkpoint": cap_ckpt.name, "sha256": cap_sha},
                "referee": referee.metadata,
            },
        }
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (folder / "review_inputs.json").write_text(
            json.dumps({"images": ["T1.png", "T2.png"], "mask": "detector_response.mask.png", "overlay": "mask_overlay.png"}, indent=2),
            encoding="utf-8",
        )
        trace_index.append({
            "category": label,
            "pair_id": pid,
            "folder": folder.name,
            "caption": b["caption"],
            "outcomes": [rc["decision"]["outcome"] for rc in report_claims],
            "report_sha256": sha256_file(folder / "report.json"),
        })

    (traces_root / "trace_index.json").write_text(json.dumps(trace_index, indent=2), encoding="utf-8")
    print("=== VERIFICATION & E2E TRACES COMPLETE ===", flush=True)
    print(json.dumps(summary["settings_comparison"], indent=2), flush=True)
    print("Traces exported:", json.dumps(trace_index, indent=2), flush=True)


if __name__ == "__main__":
    main()
