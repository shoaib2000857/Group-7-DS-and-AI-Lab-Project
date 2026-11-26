"""Milestone 5 WS1-Cap, WS2 & WS4-Cap Suite (1,858 Held-Out LEVIR-MCI Test Pairs + 150-Pair Blind Verifier Audit).

Runs on Colab L4 (or any CUDA GPU) with frozen Milestone 4 checkpoints:
  1. Full 1,858-pair Test Captioning Benchmark:
     - C06_rscama_frozen_clip (selected), C00_rscama_released, C01_rscama_ref, C04_rscama_adamw_cos,
       Always-No-Change baseline, and BIT-Mask-Template baseline
     - Metrics: BLEU-1..4, ROUGE-L, METEOR, CIDEr-D, BERTScore-F1 (roberta-large),
       Unchanged Hallucination Rate (NC FA%), Changed Miss Rate (%), Object-Class Hallucination Rate (%)
     - 1,000-resample paired bootstrap 95% CIs + paired Wilcoxon p-values
     - 200-pair captioner corruption robustness probe (clean, blur_r1.0, brightness_0.75, cloud_haze_15pct)
     - WS4-Cap error mining (Top 20 hallucinations & Top 20 missed changes)
  2. Locked Val-Calibrated Mask-First Hybrid Verifier on 150 Stratified Held-Out Test Pairs:
     - Evaluates all 7 checking regimes on fixed D03 masks & C06 captions
     - Computes Precision, Recall, Selective Risk, Contradicted Catch, VLM Call Rate, ECE, Brier, AURC
     - Exports 10 authentic Held-Out Test E2E traces (Evidence Review UI compatible)
     - Exports 120-claim Blind Human-Audit Packet (M5T_001..M5T_120) matching locked Sujal schema
       + per-claim visual inspection strips (T1 | T2 | Crop T1 | Crop T2, zero verdict leakage)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import nltk
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(SCRIPT_DIR), str(SCRIPT_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detector_models import build_model
from pipeline_recovery import (
    Qwen3VLReferee,
    apply_decision_gate,
    candidate_box_and_components,
    compute_contextual_bbox,
    compute_disagreement_triggers,
    make_mask_overlay,
    mask_only_decision,
    sha256_file,
)
from src.evidence.visual_inspector import VisualEvidenceInspector
from src.models.claims import extract_claims, to_contract


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_cap(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


NO_CHANGE_PAT = re.compile(
    r"\b(no change|nothing has changed|nothing changed|same as before|remain(?:s|ed)? the same|seem(?:s)? identical|identical|no obvious change|unchanged|no significant change)\b",
    re.I,
)


def is_no_change_caption(cap: str) -> bool:
    return bool(NO_CHANGE_PAT.search(cap.strip()))


# --- Standard NLP Metrics: BLEU-1..4, ROUGE-L, METEOR, CIDEr-D ---
def corpus_bleu_n(hyps: List[List[str]], refs_list: List[List[List[str]]], max_n: int = 4) -> Dict[str, float]:
    clipped = [0] * max_n
    total = [0] * max_n
    hyp_len = 0
    ref_len = 0
    for hyp, refs in zip(hyps, refs_list):
        hl = len(hyp)
        hyp_len += hl
        best_rl = min((len(r) for r in refs), key=lambda rl: (abs(rl - hl), rl)) if refs else 0
        ref_len += best_rl
        for n in range(1, max_n + 1):
            h_ng = Counter(tuple(hyp[i : i + n]) for i in range(max(0, hl - n + 1)))
            max_ref_ng: Counter = Counter()
            for r in refs:
                r_ng = Counter(tuple(r[i : i + n]) for i in range(max(0, len(r) - n + 1)))
                for k, v in r_ng.items():
                    if v > max_ref_ng[k]:
                        max_ref_ng[k] = v
            clipped[n - 1] += sum(min(v, max_ref_ng[k]) for k, v in h_ng.items())
            total[n - 1] += sum(h_ng.values())

    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    out = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_n = (clipped[n - 1] + 1e-9) / max(total[n - 1], 1)
        log_sum += math.log(max(p_n, 1e-12))
        out[f"bleu{n}"] = round(bp * math.exp(log_sum / n), 6)
    return out


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l_pair(hyp: List[str], refs: List[List[str]]) -> float:
    best = 0.0
    for r in refs:
        if not hyp or not r:
            continue
        lcs = lcs_len(hyp, r)
        p = lcs / len(hyp)
        rc = lcs / len(r)
        if p + rc > 0:
            f = (1.2**2 + 1) * p * rc / (1.2**2 * p + rc)
            if f > best:
                best = f
    return best


def compute_cider_d(hyps: List[List[str]], refs_list: List[List[List[str]]], n_max: int = 4, sigma: float = 6.0) -> Tuple[float, List[float]]:
    """Standard CIDEr-D (Vedantam et al., 2015) across 1..4-grams with IDF computed over reference corpus."""
    num_docs = len(refs_list)
    df: Dict[Tuple[str, ...], int] = defaultdict(int)
    for refs in refs_list:
        doc_ngs = set()
        for r in refs:
            for n in range(1, n_max + 1):
                for i in range(max(0, len(r) - n + 1)):
                    doc_ngs.add(tuple(r[i : i + n]))
        for ng in doc_ngs:
            df[ng] += 1

    log_nd = math.log(max(num_docs, 1))

    def tf_idf_vec(toks: List[str], n: int) -> Tuple[Dict[Tuple[str, ...], float], float]:
        cnt = Counter(tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1)))
        vec = {}
        norm2 = 0.0
        for ng, c in cnt.items():
            idf = max(0.0, log_nd - math.log(max(df.get(ng, 1), 1)))
            val = float(c) * idf
            vec[ng] = val
            norm2 += val * val
        return vec, math.sqrt(norm2)

    per_pair_scores = []
    for hyp, refs in zip(hyps, refs_list):
        if not refs:
            per_pair_scores.append(0.0)
            continue
        score_n = 0.0
        for n in range(1, n_max + 1):
            hv, hn = tf_idf_vec(hyp, n)
            r_scores = []
            for r in refs:
                rv, rn = tf_idf_vec(r, n)
                if hn == 0.0 or rn == 0.0:
                    cos = 0.0
                else:
                    dot = sum(min(hv[k], rv[k]) * rv[k] for k in hv if k in rv)
                    cos = dot / (hn * rn)
                delta = len(hyp) - len(r)
                penalty = math.exp(-(delta * delta) / (2.0 * sigma * sigma))
                r_scores.append(cos * penalty)
            score_n += float(np.mean(r_scores))
        per_pair_scores.append(10.0 * score_n / n_max)
    return round(float(np.mean(per_pair_scores)), 6), per_pair_scores


def compute_bertscore_batch(hyps_str: List[str], refs_str_list: List[List[str]], device: str) -> Tuple[dict, List[float]]:
    """Compute max-over-5-references BERTScore (roberta-large) on GPU."""
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=False, device=device, batch_size=64)
    P, R, F1 = scorer.score(hyps_str, refs_str_list, verbose=False)
    del scorer
    torch.cuda.empty_cache()
    return (
        {
            "precision": round(float(P.mean().item()), 6),
            "recall": round(float(R.mean().item()), 6),
            "f1": round(float(F1.mean().item()), 6),
        },
        [float(x) for x in F1.tolist()],
    )


def evaluate_caption_predictions(
    name: str,
    preds_by_pid: Dict[str, str],
    test_rows: List[dict],
    refs: Dict[str, dict],
    gt_masks_meta: Dict[str, Tuple[int, int]],  # pid -> (road_px, bldg_px)
    device: str,
    run_bertscore: bool = True,
) -> Tuple[dict, List[dict]]:
    from nltk.translate.meteor_score import meteor_score

    hyps_tok = []
    refs_tok = []
    hyps_str = []
    refs_str = []
    per_pair = []

    unchanged_n = 0
    unchanged_fa = 0
    changed_n = 0
    changed_miss = 0
    obj_claims_total = 0
    obj_hallucinations = 0

    for r in test_rows:
        pid = r["pair_id"]
        hyp_s = preds_by_pid[pid].strip()
        ref_ss = refs[pid]["captions"]
        ht = tokenize_cap(hyp_s)
        rts = [tokenize_cap(x) for x in ref_ss]

        hyps_tok.append(ht)
        refs_tok.append(rts)
        hyps_str.append(hyp_s)
        refs_str.append(ref_ss)

        rl = rouge_l_pair(ht, rts)
        met = float(meteor_score(rts, ht)) if ht else 0.0
        # Pair-level smoothed BLEU-4 for Wilcoxon/bootstrap
        p_b4 = corpus_bleu_n([ht], [rts], max_n=4)["bleu4"]

        road_px, bldg_px = gt_masks_meta[pid]
        gt_changed = (road_px + bldg_px) > 0
        pred_nc = is_no_change_caption(hyp_s)

        if not gt_changed:
            unchanged_n += 1
            if not pred_nc:
                unchanged_fa += 1
        else:
            changed_n += 1
            if pred_nc:
                changed_miss += 1

        # Object-class hallucination check (road/building mentioned when GT has <50px of that class)
        cl_list = [to_contract(c) for c in extract_claims(hyp_s, pair_id=pid)]
        pair_hall = False
        for cd in cl_list:
            obj = cd["claim"]["object"]
            act = cd["claim"]["action"]
            if act in ("added", "removed") and obj in ("road", "building"):
                obj_claims_total += 1
                gt_cls_px = road_px if obj == "road" else bldg_px
                if gt_cls_px < 50:
                    obj_hallucinations += 1
                    pair_hall = True

        per_pair.append({
            "pair_id": pid,
            "caption": hyp_s,
            "gt_road_px": road_px,
            "gt_bldg_px": bldg_px,
            "gt_changed": gt_changed,
            "pred_no_change": pred_nc,
            "rouge_l": round(rl, 6),
            "meteor": round(met, 6),
            "pair_bleu4": round(p_b4, 6),
            "object_hallucination": pair_hall,
        })

    bleus = corpus_bleu_n(hyps_tok, refs_tok, max_n=4)
    cider_mean, cider_list = compute_cider_d(hyps_tok, refs_tok)
    for row, cid in zip(per_pair, cider_list):
        row["cider"] = round(cid, 6)

    # Changed-only BLEU-4
    chg_idx = [i for i, row in enumerate(per_pair) if row["gt_changed"]]
    chg_bleu4 = corpus_bleu_n([hyps_tok[i] for i in chg_idx], [refs_tok[i] for i in chg_idx], max_n=4)["bleu4"]

    bert_summary = None
    if run_bertscore:
        bert_summary, f1_list = compute_bertscore_batch(hyps_str, refs_str, device)
        for row, bf1 in zip(per_pair, f1_list):
            row["bertscore_f1"] = round(bf1, 6)

    summary = {
        "model_name": name,
        "num_pairs": len(test_rows),
        **bleus,
        "changed_bleu4": chg_bleu4,
        "rouge_l": round(float(np.mean([x["rouge_l"] for x in per_pair])), 6),
        "meteor": round(float(np.mean([x["meteor"] for x in per_pair])), 6),
        "cider_d": cider_mean,
        "bertscore": bert_summary,
        "unchanged_pairs": unchanged_n,
        "unchanged_fa_rate": round(unchanged_fa / max(unchanged_n, 1), 6),
        "changed_pairs": changed_n,
        "changed_miss_rate": round(changed_miss / max(changed_n, 1), 6),
        "object_claims_total": obj_claims_total,
        "object_hallucination_rate": round(obj_hallucinations / max(obj_claims_total, 1), 6),
    }
    return summary, per_pair


@torch.inference_mode()
def generate_rscama_captions(
    ckpt_path: Path,
    rscama_src: Path,
    rows: List[dict],
    ext_root: Path,
    device: str,
    corrupt_fn=None,
) -> Dict[str, str]:
    from train_captioner import build_rscama_components
    c_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = c_state["vocab"]
    models, processor, kernel_mode = build_rscama_components(rscama_src, vocab, device)
    for m, k in zip(models, ["encoder_dict", "encoder_trans_dict", "decoder_dict"]):
        m.load_state_dict(c_state[k], strict=True)
        m.to(device).eval()
    inverse = {int(v): w for w, v in vocab.items()}
    skip_ids = {vocab[w] for w in ("<START>", "<END>", "<NULL>")}
    encoder, fusion, decoder = models

    preds: Dict[str, str] = {}
    has_native_bf16 = device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if has_native_bf16 else torch.float16
    for idx, r in enumerate(rows):
        t1 = np.array(Image.open(ext_root / r["t1"]).convert("RGB"))
        t2 = np.array(Image.open(ext_root / r["t2"]).convert("RGB"))
        if corrupt_fn is not None:
            t1, t2 = corrupt_fn(t1, t2, 20000 + idx)
        im1 = processor(Image.fromarray(t1)).unsqueeze(0).to(device)
        im2 = processor(Image.fromarray(t2)).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=amp_dtype):
            f1, f2 = encoder(im1, im2)
            feat = fusion(f1, f2)
        toks = decoder.sample(feat.float(), k=1)
        if hasattr(toks, "tolist"):
            toks = toks.tolist()
        while toks and isinstance(toks[0], list):
            toks = toks[0]
        preds[r["pair_id"]] = " ".join(inverse[int(t)] for t in toks if int(t) not in skip_ids)
    del models, encoder, fusion, decoder
    torch.cuda.empty_cache()
    return preds


def build_bit_mask_template_captions(det_masks_meta: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
    """Deterministic mask-driven template caption baseline (no neural captioner)."""
    out = {}
    for pid, (r_px, b_px) in det_masks_meta.items():
        has_r = r_px >= 50
        has_b = b_px >= 50
        if not has_r and not has_b:
            out[pid] = "the scene is the same as before"
        elif has_b and has_r:
            out[pid] = "a road and some buildings have been built in this area"
        elif has_b:
            out[pid] = "some buildings have been built on the bareland" if b_px >= 600 else "a building has been built in this area"
        else:
            out[pid] = "a road has been built across the scene"
    return out


# --- Locked Val-Calibrated Mask-First Hybrid Verifier (Zero Test Tuning!) ---
def calibrated_mask_first_hybrid(
    claim_rec: dict,
    tau_accept: float = 0.60,
    tau_reject: float = 0.75,
) -> Tuple[str, float, bool, str]:
    """Apply the EXACT validation-locked Mask-First Hybrid routing policy.

    Returns: (decision, calibrated_accept_prob, vlm_triggered, route_reason)
    """
    m = claim_rec["dec_mask_only"]
    obj = claim_rec["claim"]["object"]
    act = claim_rec["claim"]["action"]
    trigs = set(claim_rec["triggers"])
    forced = set(claim_rec["forced_reasons"])
    any_px = claim_rec["mask_summary"]["any_change_pixels"]
    cls_px = claim_rec["mask_summary"]["class_change_pixels"]

    # Rule 1: Out-of-ontology or explicit exact numeric count -> deterministic defer (0 VLM calls)
    if "T_out_of_mask_ontology" in trigs or "T_exact_numeric_count" in trigs or "exact_count_without_independent_instance_evidence" in forced or "exact_numeric_count_unverified" in forced:
        return "defer", 0.50, False, "deterministic_defer_ontology_or_exact_count"

    # Rule 2: Strong BIT mask reject -> trust high-precision BIT reject (0 VLM calls)
    if m == "reject":
        return "reject", 0.05, False, "mask_first_reject"

    # Rule 3: Selective Direct Qwen3-VL Escalation ONLY on genuine BIT blind spots
    escalate_reason = None
    if act == "removed":
        escalate_reason = "vlm_escalate_direction_removed"
    elif act == "no_change" and (any_px >= 10 or "T_hedged_claim" in trigs):
        escalate_reason = "vlm_escalate_no_change_residual_or_hedged"
    elif act == "added" and obj in ("road", "building") and cls_px < 200:
        escalate_reason = "vlm_escalate_weak_mask_support"

    if not escalate_reason:
        # High-confidence BIT mask accept (0 VLM calls!)
        conf = 0.94 if (act == "no_change" and any_px == 0) else min(0.96, 0.84 + 0.12 * min(cls_px / 2000.0, 1.0))
        return "accept", round(conf, 4), False, "mask_first_accept"

    pA = float(claim_rec["raw_direct"]["probabilities"]["accept"])
    pR = float(claim_rec["raw_direct"]["probabilities"]["reject"])
    if pR >= tau_reject:
        return "reject", round(pA, 4), True, f"{escalate_reason}->vlm_reject"
    if pA >= tau_accept:
        return "accept", round(pA, 4), True, f"{escalate_reason}->vlm_accept"
    return "defer", round(pA, 4), True, f"{escalate_reason}->vlm_defer"


def compute_calibration_metrics(probs: List[float], y_true: List[int], n_bins: int = 10) -> dict:
    """Compute ECE, Brier score, and reliability bin coordinates for supported-claim probability."""
    p = np.array(probs, dtype=np.float64)
    y = np.array(y_true, dtype=np.float64)
    brier = float(np.mean((p - y) ** 2)) if len(p) else 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            conf = float(p[mask].mean())
            acc = float(y[mask].mean())
            ece += (cnt / len(p)) * abs(acc - conf)
            bin_rows.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "count": cnt, "mean_conf": round(conf, 4), "empirical_acc": round(acc, 4)})
    return {"ece": round(ece, 6), "brier_score": round(brier, 6), "reliability_bins": bin_rows}


def compute_aurc(probs: List[float], y_true: List[int]) -> float:
    """Area Under Risk-Coverage Curve (lower is better)."""
    if not probs:
        return 0.0
    order = np.argsort(-np.array(probs, dtype=np.float64))
    y_sorted = np.array(y_true, dtype=np.float64)[order]
    n = len(y_sorted)
    cum_err = np.cumsum(1.0 - y_sorted)
    cov = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    return round(float(np.trapezoid(risk, cov)), 6)


def evaluate_setting(claim_rows: List[dict], key: str) -> dict:
    total = len(claim_rows)
    acc = [c for c in claim_rows if c[key] == "accept"]
    rej = [c for c in claim_rows if c[key] == "reject"]
    dfn = [c for c in claim_rows if c[key] == "defer"]

    acc_sup = sum(1 for c in acc if c["proxy_gt"] == "supported")
    acc_con = sum(1 for c in acc if c["proxy_gt"] == "contradicted")
    acc_unv = sum(1 for c in acc if c["proxy_gt"] == "unverifiable")
    rej_con = sum(1 for c in rej if c["proxy_gt"] == "contradicted")
    rej_sup = sum(1 for c in rej if c["proxy_gt"] == "supported")
    tot_sup = sum(1 for c in claim_rows if c["proxy_gt"] == "supported")
    tot_con = sum(1 for c in claim_rows if c["proxy_gt"] == "contradicted")

    prec = acc_sup / max(len(acc), 1)
    rec = acc_sup / max(tot_sup, 1)
    return {
        "total_claims": total,
        "accepted": len(acc),
        "rejected": len(rej),
        "deferred": len(dfn),
        "accept_rate": round(len(acc) / max(total, 1), 4),
        "reject_rate": round(len(rej) / max(total, 1), 4),
        "defer_rate": round(len(dfn) / max(total, 1), 4),
        "proxy_supported_precision_strict": round(prec, 4),
        "proxy_selective_risk": round(1.0 - prec, 4) if acc else 0.0,
        "proxy_supported_recall": round(rec, 4),
        "proxy_contradicted_catch_rate": round(rej_con / max(tot_con, 1), 4),
        "proxy_false_accept_rate": round(acc_con / max(tot_con, 1), 4),
        "proxy_false_reject_rate": round(rej_sup / max(tot_sup, 1), 4),
        "breakdown": {
            "accepted_supported": acc_sup,
            "accepted_contradicted": acc_con,
            "accepted_unverifiable": acc_unv,
            "rejected_contradicted": rej_con,
            "rejected_supported": rej_sup,
        },
    }


def proxy_ground_truth_label(claim_dict: dict, gt_mask: np.ndarray) -> str:
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
    return "supported"


def select_stratified_test_pairs(test_rows: List[dict], ext_root: Path, seed: int = 2026) -> List[Tuple[dict, str]]:
    buckets: Dict[str, List[dict]] = {k: [] for k in ("unchanged", "building_only", "road_only", "mixed_both", "small_change")}
    for r in test_rows:
        lbl = np.array(Image.open(ext_root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        r_px = int((lbl == 128).sum())
        b_px = int((lbl == 255).sum())
        tot = r_px + b_px
        if tot == 0:
            buckets["unchanged"].append(r)
        elif tot < 655:
            buckets["small_change"].append(r)
        elif b_px >= 50 and r_px >= 50:
            buckets["mixed_both"].append(r)
        elif b_px >= 50:
            buckets["building_only"].append(r)
        elif r_px >= 50:
            buckets["road_only"].append(r)
        else:
            buckets["small_change"].append(r)

    rng = np.random.default_rng(seed)
    quotas = [("unchanged", 50), ("building_only", 35), ("road_only", 20), ("mixed_both", 25), ("small_change", 20)]
    selected = []
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


def save_blind_audit_strip(t1: np.ndarray, t2: np.ndarray, c1: np.ndarray, c2: np.ndarray, bbox: List[int], audit_id: str, claim_text: str, out_path: Path) -> None:
    """Create a crisp 4-panel blind inspection strip (T1 | T2+Box | Crop T1 | Crop T2) with ZERO model verdict shown."""
    sz = 256
    t1_im = Image.fromarray(t1).resize((sz, sz))
    t2_box = t2.copy()
    x0, y0, x1, y1 = bbox
    if not (x0 == 0 and y0 == 0 and x1 == t2.shape[1] and y1 == t2.shape[0]):
        cv2.rectangle(t2_box, (x0, y0), (x1 - 1, y1 - 1), (0, 230, 180), 2)
    t2_im = Image.fromarray(t2_box).resize((sz, sz))
    c1_im = Image.fromarray(c1).resize((sz, sz), Image.NEAREST if min(c1.shape[:2]) < 96 else Image.BILINEAR)
    c2_im = Image.fromarray(c2).resize((sz, sz), Image.NEAREST if min(c2.shape[:2]) < 96 else Image.BILINEAR)

    pad = 8
    header_h = 48
    canvas_w = sz * 4 + pad * 5
    canvas_h = header_h + sz + pad * 2
    sheet = Image.new("RGB", (canvas_w, canvas_h), (16, 38, 48))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{audit_id}  |  CLAIM: \"{claim_text[:110]}\"", fill=(217, 237, 133))
    labels = ["T1 (Before)", "T2 (After + Candidate Region)", "Zoom Crop T1", "Zoom Crop T2"]
    for i, (im, lbl) in enumerate(zip([t1_im, t2_im, c1_im, c2_im], labels)):
        x = pad + i * (sz + pad)
        draw.text((x + 4, 28), lbl, fill=(247, 245, 239))
        sheet.paste(im, (x, header_h))
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--extracted-root", required=True)
    ap.add_argument("--detector-ckpt", required=True)
    ap.add_argument("--bit-source-dir", required=True)
    ap.add_argument("--changeformer-source-dir", required=True)
    ap.add_argument("--rscama-source-dir", required=True)
    ap.add_argument("--c06-test-captions", required=True)
    ap.add_argument("--c00-test-captions", required=True)
    ap.add_argument("--c01-test-captions", default="")
    ap.add_argument("--c04-test-captions", default="")
    ap.add_argument("--probe-captions-json", default="")
    ap.add_argument("--c06-ckpt", default="")
    ap.add_argument("--c01-ckpt", default="")
    ap.add_argument("--c04-ckpt", default="")
    ap.add_argument("--val-claims-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_root = Path(args.extracted_root)
    man_dir = Path(args.manifests_dir)

    test_rows = read_jsonl(man_dir / "mci_test.jsonl")
    refs = json.loads((man_dir / "mci_references.json").read_text(encoding="utf-8"))
    assert len(test_rows) == 1858, len(test_rows)

    t0 = time.time()
    print("=== [1/5] Loading GT masks & running frozen D03_bit_lr3e4 on 1,858 Test pairs ===", flush=True)
    gt_masks_meta: Dict[str, Tuple[int, int]] = {}
    for r in test_rows:
        lbl = np.array(Image.open(ext_root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        gt_masks_meta[r["pair_id"]] = (int((lbl == 128).sum()), int((lbl == 255).sum()))

    det_ckpt = Path(args.detector_ckpt)
    det_sha = sha256_file(det_ckpt)
    det_model, _, _, _ = build_model("bit_resnet18", args.device, args.bit_source_dir, args.changeformer_source_dir)
    det_model.load_state_dict(torch.load(det_ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
    det_model.eval()

    det_masks_meta: Dict[str, Tuple[int, int]] = {}
    test_pred_masks: Dict[str, np.ndarray] = {}
    selected_test = select_stratified_test_pairs(test_rows, ext_root, seed=2026)
    selected_pids = {r["pair_id"] for r, _ in selected_test}

    has_native_bf16 = args.device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if has_native_bf16 else torch.float16
    with torch.inference_mode():
        for i in range(0, len(test_rows), 32):
            chunk = test_rows[i : i + 32]
            x1s, x2s = [], []
            for r in chunk:
                t1 = np.array(Image.open(ext_root / r["t1"]).convert("RGB"))
                t2 = np.array(Image.open(ext_root / r["t2"]).convert("RGB"))
                x1s.append(torch.from_numpy(t1.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0)
                x2s.append(torch.from_numpy(t2.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0)
            b1 = torch.stack(x1s).to(args.device)
            b2 = torch.stack(x2s).to(args.device)
            with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                preds = det_model(b1, b2).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for r, pm in zip(chunk, preds):
                pid = r["pair_id"]
                det_masks_meta[pid] = (int((pm == 1).sum()), int((pm == 2).sum()))
                if pid in selected_pids:
                    test_pred_masks[pid] = pm
    del det_model
    torch.cuda.empty_cache()

    print("=== [2/5] Evaluating All Captioner Models + Baselines on 1,858 Held-Out Test Pairs ===", flush=True)
    c06_caps = {rec["pair_id"]: rec["caption"] for rec in read_jsonl(Path(args.c06_test_captions))}
    c00_caps = {rec["pair_id"]: rec["caption"] for rec in read_jsonl(Path(args.c00_test_captions))}
    always_nc_caps = {r["pair_id"]: "the scene is the same as before" for r in test_rows}
    template_caps = build_bit_mask_template_captions(det_masks_meta)

    cap_models = [
        ("C06_rscama_frozen_clip", c06_caps),
        ("C00_rscama_released", c00_caps),
    ]
    if args.c04_test_captions and Path(args.c04_test_captions).is_file():
        c04_caps = {rec["pair_id"]: rec["caption"] for rec in read_jsonl(Path(args.c04_test_captions))}
        cap_models.append(("C04_rscama_adamw_cos", c04_caps))
    elif args.c04_ckpt and Path(args.c04_ckpt).is_file():
        c04_caps = generate_rscama_captions(Path(args.c04_ckpt), Path(args.rscama_source_dir), test_rows, ext_root, args.device)
        cap_models.append(("C04_rscama_adamw_cos", c04_caps))
    if args.c01_test_captions and Path(args.c01_test_captions).is_file():
        c01_caps = {rec["pair_id"]: rec["caption"] for rec in read_jsonl(Path(args.c01_test_captions))}
        cap_models.append(("C01_rscama_ref", c01_caps))
    elif args.c01_ckpt and Path(args.c01_ckpt).is_file():
        c01_caps = generate_rscama_captions(Path(args.c01_ckpt), Path(args.rscama_source_dir), test_rows, ext_root, args.device)
        cap_models.append(("C01_rscama_ref", c01_caps))
    cap_models.extend([
        ("bit_mask_template", template_caps),
        ("always_no_change", always_nc_caps),
    ])

    cap_summaries = {}
    cap_per_pair: Dict[str, List[dict]] = {}
    for name, pred_map in cap_models:
        tm = time.time()
        summ, pp = evaluate_caption_predictions(name, pred_map, test_rows, refs, gt_masks_meta, args.device, run_bertscore=True)
        cap_summaries[name] = summ
        cap_per_pair[name] = pp
        print(
            f"  {name:24s} | B4={summ['bleu4']:.4f} | RL={summ['rouge_l']:.4f} | MET={summ['meteor']:.4f} | "
            f"CIDEr={summ['cider_d']:.4f} | BERT={summ['bertscore']['f1']:.4f} | NC_FA={summ['unchanged_fa_rate']*100:.2f}% | "
            f"ObjHall={summ['object_hallucination_rate']*100:.2f}% ({time.time()-tm:.1f}s)",
            flush=True,
        )

    # 1,000-resample paired bootstrap 95% CIs & Wilcoxon p-values vs C06
    rng = np.random.default_rng(42)
    n = len(test_rows)
    boot_idx = rng.integers(0, n, size=(1000, n))
    cap_ci = {}
    cap_sig = {}
    c06_rl = np.array([x["rouge_l"] for x in cap_per_pair["C06_rscama_frozen_clip"]])
    c06_bf = np.array([x["bertscore_f1"] for x in cap_per_pair["C06_rscama_frozen_clip"]])
    c06_cid = np.array([x["cider"] for x in cap_per_pair["C06_rscama_frozen_clip"]])

    for name, pp in cap_per_pair.items():
        rl = np.array([x["rouge_l"] for x in pp])
        bf = np.array([x["bertscore_f1"] for x in pp])
        cid = np.array([x["cider"] for x in pp])
        b_rl = rl[boot_idx].mean(axis=1)
        b_bf = bf[boot_idx].mean(axis=1)
        b_cid = cid[boot_idx].mean(axis=1)
        cap_ci[name] = {
            "rouge_l_ci95": [round(float(np.percentile(b_rl, 2.5)), 5), round(float(np.percentile(b_rl, 97.5)), 5)],
            "bertscore_f1_ci95": [round(float(np.percentile(b_bf, 2.5)), 5), round(float(np.percentile(b_bf, 97.5)), 5)],
            "cider_d_ci95": [round(float(np.percentile(b_cid, 2.5)), 5), round(float(np.percentile(b_cid, 97.5)), 5)],
        }
        if name != "C06_rscama_frozen_clip":
            d_rl = c06_rl - rl
            d_bf = c06_bf - bf
            d_cid = c06_cid - cid
            cap_sig[f"C06_vs_{name}"] = {
                "rouge_l_diff": round(float(d_rl.mean()), 5),
                "rouge_l_wilcoxon_p": float(stats.wilcoxon(d_rl[d_rl != 0]).pvalue) if (d_rl != 0).any() else 1.0,
                "bertscore_diff": round(float(d_bf.mean()), 5),
                "bertscore_wilcoxon_p": float(stats.wilcoxon(d_bf[d_bf != 0]).pvalue) if (d_bf != 0).any() else 1.0,
                "cider_diff": round(float(d_cid.mean()), 5),
                "cider_wilcoxon_p": float(stats.wilcoxon(d_cid[d_cid != 0]).pvalue) if (d_cid != 0).any() else 1.0,
            }

    # WS4-Cap Error Mining on C06
    c06_pp = cap_per_pair["C06_rscama_frozen_clip"]
    cap_hallucinations = [x for x in c06_pp if (not x["gt_changed"]) and (not x["pred_no_change"])][:25]
    cap_missed = sorted(
        [x for x in c06_pp if x["gt_changed"] and x["pred_no_change"]],
        key=lambda x: -(x["gt_road_px"] + x["gt_bldg_px"]),
    )[:25]
    cap_obj_halls = [x for x in c06_pp if x["object_hallucination"]][:25]

    print("=== [3/5] Running 200-Pair Captioner Corruption Probe (C06) ===", flush=True)
    probe_rows = [r for r, _ in selected_test] + [r for r in test_rows if r["pair_id"] not in selected_pids][:50]
    probe_rows = probe_rows[:200]

    def _c_blur(a, b, s):
        return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.0))), np.array(Image.fromarray(b).filter(ImageFilter.GaussianBlur(1.0)))

    def _c_bright(a, b, s):
        return a, np.array(ImageEnhance.Brightness(Image.fromarray(b)).enhance(0.75))

    def _c_haze(a, b, s):
        h, w = b.shape[:2]
        rng_h = np.random.default_rng(s)
        cx, cy = float(rng_h.uniform(0.25 * w, 0.75 * w)), float(rng_h.uniform(0.25 * h, 0.75 * h))
        rx, ry = float(rng_h.uniform(0.20 * w, 0.28 * w)), float(rng_h.uniform(0.20 * h, 0.28 * h))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        alpha = (np.exp(-0.5 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)) * 0.72)[:, :, None]
        out = np.clip(b.astype(np.float32) * (1.0 - alpha) + np.array([238.0, 242.0, 246.0]) * alpha, 0, 255).astype(np.uint8)
        return a, out

    cap_probe = {}
    pre_probe = json.loads(Path(args.probe_captions_json).read_text(encoding="utf-8")) if (args.probe_captions_json and Path(args.probe_captions_json).is_file()) else {}
    if pre_probe:
        for cname in ("clean", "blur_both_r1.0", "brightness_t2_0.75", "cloud_haze_t2_15pct"):
            pmap = c06_caps if cname == "clean" else pre_probe[cname]
            psumm, _ = evaluate_caption_predictions(f"C06_{cname}", pmap, probe_rows, refs, gt_masks_meta, args.device, run_bertscore=False)
            cap_probe[cname] = psumm
            print(f"  [Probe 200] {cname:22s} -> B4={psumm['bleu4']:.4f}, RL={psumm['rouge_l']:.4f}, NC_FA={psumm['unchanged_fa_rate']*100:.2f}%, Miss={psumm['changed_miss_rate']*100:.2f}%", flush=True)
    elif args.c06_ckpt and Path(args.c06_ckpt).is_file():
        for cname, cfn in [("clean", None), ("blur_both_r1.0", _c_blur), ("brightness_t2_0.75", _c_bright), ("cloud_haze_t2_15pct", _c_haze)]:
            pmap = c06_caps if cfn is None else generate_rscama_captions(Path(args.c06_ckpt), Path(args.rscama_source_dir), probe_rows, ext_root, args.device, corrupt_fn=cfn)
            psumm, _ = evaluate_caption_predictions(f"C06_{cname}", pmap, probe_rows, refs, gt_masks_meta, args.device, run_bertscore=False)
            cap_probe[cname] = psumm
            print(f"  [Probe 200] {cname:22s} -> B4={psumm['bleu4']:.4f}, RL={psumm['rouge_l']:.4f}, NC_FA={psumm['unchanged_fa_rate']*100:.2f}%, Miss={psumm['changed_miss_rate']*100:.2f}%", flush=True)
    else:
        print("  [Probe 200] Skipping corrupted captioner probe (no precomputed probe or c06_ckpt)", flush=True)

    cap_payload = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "levir_mci_test_strict",
        "num_pairs": len(test_rows),
        "models": cap_summaries,
        "bootstrap_95ci": cap_ci,
        "paired_significance": cap_sig,
        "robustness_probe_200": cap_probe,
        "error_mining": {
            "unchanged_hallucinations": cap_hallucinations,
            "changed_misses_top25": cap_missed,
            "object_class_hallucinations": cap_obj_halls,
        },
    }
    (out_dir / "captioner_test_full.json").write_text(json.dumps(cap_payload, indent=2), encoding="utf-8")

    print("=== [4/5] Computing Val Calibration Metrics for Locked Mask-First Hybrid Verifier ===", flush=True)
    val_claims = read_jsonl(Path(args.val_claims_jsonl))
    val_hybrid_decs = []
    val_hybrid_probs = []
    val_vlm_calls = 0
    for c in val_claims:
        dec, prob, trig, rsn = calibrated_mask_first_hybrid(c, tau_accept=0.60, tau_reject=0.75)
        c["dec_m5_hybrid"] = dec
        val_hybrid_decs.append(dec)
        val_hybrid_probs.append(prob)
        val_vlm_calls += int(trig)
    val_y = [1 if c["proxy_gt"] == "supported" else 0 for c in val_claims]
    val_cal = {
        "num_claims": len(val_claims),
        "locked_params": {"tau_accept": 0.60, "tau_reject": 0.75, "mask_alarm_px": 50},
        "m5_hybrid_val_metrics": evaluate_setting(val_claims, "dec_m5_hybrid"),
        "vlm_call_rate": round(val_vlm_calls / max(len(val_claims), 1), 4),
        "hybrid_calibration": compute_calibration_metrics(val_hybrid_probs, val_y),
        "hybrid_aurc": compute_aurc(val_hybrid_probs, val_y),
        "direct_qwen_calibration": compute_calibration_metrics([c["raw_direct"]["probabilities"]["accept"] for c in val_claims], val_y),
        "direct_qwen_aurc": compute_aurc([c["raw_direct"]["probabilities"]["accept"] for c in val_claims], val_y),
    }
    print("  Val Locked Hybrid:", json.dumps(val_cal["m5_hybrid_val_metrics"]), "VLM rate:", val_cal["vlm_call_rate"], flush=True)

    print("=== [5/5] Blind Held-Out Test Verification Audit (150 Stratified Test Pairs, Qwen3-VL-4B-Instruct) ===", flush=True)
    inspector = VisualEvidenceInspector(min_crop_size=24)
    referee = Qwen3VLReferee(device=args.device)
    work_crops = out_dir / "_test_audit_crops"
    work_crops.mkdir(parents=True, exist_ok=True)

    test_claims: List[dict] = []
    pair_bundles: Dict[str, dict] = {}
    t_v = time.time()

    for idx, (r, stratum) in enumerate(selected_test):
        pid = r["pair_id"]
        safe_pid = pid.replace("/", "_")
        p1_path = ext_root / r["t1"]
        p2_path = ext_root / r["t2"]
        t1_rgb = np.array(Image.open(p1_path).convert("RGB"))
        t2_rgb = np.array(Image.open(p2_path).convert("RGB"))
        lbl = np.array(Image.open(ext_root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        gt_mask = np.zeros_like(lbl, dtype=np.uint8)
        gt_mask[lbl == 128] = 1
        gt_mask[lbl == 255] = 2

        pred_mask = test_pred_masks[pid]
        caption = c06_caps[pid]
        parsed = [(c, to_contract(c)) for c in extract_claims(caption, pair_id=pid, run_id="m5_test")]
        if not parsed:
            fallback_raw = {
                "pair_id": pid,
                "claim_id": f"m5_test:{pid}:c0",
                "source_span": caption,
                "claim": {"object": "scene", "action": "no_change", "location": None, "quantity": None, "quantity_unit": None},
                "parser": {"quantity_kind": None, "quantity_text": None, "hedged": False, "negated": True, "version": "fallback"},
            }
            parsed = [(fallback_raw, fallback_raw)]

        pair_claims = []
        for ci, (c_raw, contract) in enumerate(parsed):
            claim_id = f"m5_test:{pid}:c{ci}"
            bbox, localized, mask_summary = candidate_box_and_components(pred_mask, c_raw, min_component_px=16)
            padded_bbox, transform_meta = compute_contextual_bbox(bbox, t1_rgb.shape[:2], padding_fraction=0.25, min_size=32)
            x0, y0, x1, y1 = padded_bbox
            c1 = t1_rgb[y0:y1, x0:x1].copy()
            c2 = t2_rgb[y0:y1, x0:x1].copy()
            c1_path = work_crops / f"{safe_pid}_c{ci}_T1.png"
            c2_path = work_crops / f"{safe_pid}_c{ci}_T2.png"
            Image.fromarray(c1).save(c1_path)
            Image.fromarray(c2).save(c2_path)

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
            # Witness-only baseline
            if c_raw["claim"]["action"] == "no_change":
                dec_wit = "reject" if wit_dict.get("visible_change") else "accept"
            elif c_raw["claim"]["object"] not in ("road", "building", "scene", "region"):
                dec_wit = "defer"
            else:
                dec_wit = "accept" if wit_dict.get("visible_change") else "reject"

            direct_state = {
                "claim": contract["source_span"],
                "structured_claim": contract["claim"],
                "temporal_order": "T1_earlier_T2_later",
            }
            raw_direct = referee.decide(direct_state, [str(p1_path), str(p2_path)], mode="direct_full_image")
            reinsp_state = {
                "claim": contract["source_span"],
                "structured_claim": contract["claim"],
                "predicted_mask_summary": mask_summary,
                "region_localized": localized,
                "candidate_bbox": padded_bbox,
                "fallible_witness_observation": {
                    "visible_change": wit_dict["visible_change"],
                    "state_transition": wit_dict["state_transition"],
                    "observations": wit_dict["observations"],
                    "quality_flags": wit_dict["quality_flags"],
                    "visibility_flags": wit_dict["visibility_flags"],
                    "ssim_score": round(wit_dict["ssim_score"], 4),
                },
                "disagreement_triggers": triggers,
                "temporal_order": "T1_earlier_T2_later",
            }
            raw_reinsp = referee.decide(
                reinsp_state,
                [str(p1_path), str(p2_path), str(c1_path), str(c2_path)],
                mode="targeted_crop_context_reinspection",
            )
            gated_direct = apply_decision_gate(raw_direct, 0.70, forced_reasons)
            gated_reinsp = apply_decision_gate(raw_reinsp, 0.70, forced_reasons)
            dec_m4_trig = "defer" if forced_reasons else (gated_reinsp["outcome"] if triggers else dec_mask)

            rec = {
                "pair_id": pid,
                "stratum": stratum,
                "claim_id": claim_id,
                "caption": caption,
                "source_span": contract["source_span"],
                "claim": contract["claim"],
                "parser": c_raw.get("parser", {}),
                "localized": localized,
                "bbox": bbox,
                "padded_bbox": padded_bbox,
                "transform_meta": transform_meta,
                "crop_paths": [str(c1_path), str(c2_path)],
                "mask_summary": mask_summary,
                "witness": wit_dict,
                "triggers": triggers,
                "forced_reasons": forced_reasons,
                "proxy_gt": proxy_ground_truth_label(c_raw, gt_mask),
                "dec_unverified": "accept",
                "dec_mask_only": dec_mask,
                "dec_witness_only": dec_wit,
                "dec_direct_qwen_gated": gated_direct["outcome"],
                "dec_reinsp_all_gated": gated_reinsp["outcome"],
                "dec_m4_triggered_reinsp": dec_m4_trig,
                "raw_direct": raw_direct,
                "gated_direct": gated_direct,
                "raw_reinsp": raw_reinsp,
                "gated_reinsp": gated_reinsp,
            }
            h_dec, h_prob, h_trig, h_rsn = calibrated_mask_first_hybrid(rec, tau_accept=0.60, tau_reject=0.75)
            rec["dec_m5_hybrid"] = h_dec
            rec["hybrid_verifier_prob"] = h_prob
            rec["hybrid_vlm_triggered"] = h_trig
            rec["hybrid_route_reason"] = h_rsn

            test_claims.append(rec)
            pair_claims.append(rec)

        pair_bundles[pid] = {"row": r, "stratum": stratum, "caption": caption, "pred_mask": pred_mask, "claims": pair_claims}
        if (idx + 1) % 25 == 0 or (idx + 1) == len(selected_test):
            print(f"  Test Verified {idx+1}/{len(selected_test)} pairs ({len(test_claims)} claims, {time.time()-t_v:.1f}s)", flush=True)

    # Threshold sweep & calibration on Held-Out Test
    test_y = [1 if c["proxy_gt"] == "supported" else 0 for c in test_claims]
    sweep = {}
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        for c in test_claims:
            c["_tmp_d"] = apply_decision_gate(c["raw_direct"], th, c["forced_reasons"])["outcome"]
            c["_tmp_r"] = apply_decision_gate(c["raw_reinsp"], th, c["forced_reasons"])["outcome"]
            hd, _, _, _ = calibrated_mask_first_hybrid(c, tau_accept=th, tau_reject=0.75)
            c["_tmp_h"] = hd
        sweep[f"{th:.2f}"] = {
            "direct_qwen_gated": evaluate_setting(test_claims, "_tmp_d"),
            "reinsp_all_gated": evaluate_setting(test_claims, "_tmp_r"),
            "m5_mask_first_hybrid": evaluate_setting(test_claims, "_tmp_h"),
        }

    hybrid_vlm_calls = sum(1 for c in test_claims if c["hybrid_vlm_triggered"])
    m4_vlm_calls = sum(1 for c in test_claims if c["triggers"])

    ver_summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": {
            "split": "levir_mci_test_stratified_blind",
            "num_pairs": len(selected_test),
            "strata_counts": dict(Counter(s for _, s in selected_test)),
            "total_claims": len(test_claims),
            "proxy_gt_counts": dict(Counter(c["proxy_gt"] for c in test_claims)),
        },
        "models": {
            "detector": {"arch": "bit_resnet18", "checkpoint": det_ckpt.name, "sha256": det_sha},
            "captioner": {
                "run_id": "C06_rscama_frozen_clip",
                "checkpoint": Path(args.c06_ckpt).name if args.c06_ckpt else "best.pth",
                "sha256": sha256_file(Path(args.c06_ckpt)) if (args.c06_ckpt and Path(args.c06_ckpt).is_file()) else "c5b80d6bf6b6b0b2628c48d47df07ad77bc086ff0004bdd7160d0b0ff5545796",
            },
            "referee": referee.metadata,
        },
        "validation_calibration_locked": val_cal,
        "test_vlm_efficiency": {
            "direct_qwen_vlm_call_rate": 1.0,
            "reinsp_all_vlm_call_rate": 1.0,
            "m4_triggered_vlm_call_rate": round(m4_vlm_calls / max(len(test_claims), 1), 4),
            "m5_hybrid_vlm_calls": hybrid_vlm_calls,
            "m5_hybrid_vlm_call_rate": round(hybrid_vlm_calls / max(len(test_claims), 1), 4),
            "vlm_savings_vs_unconditional_pct": round((1.0 - hybrid_vlm_calls / max(len(test_claims), 1)) * 100, 2),
            "hybrid_route_counts": dict(Counter(c["hybrid_route_reason"] for c in test_claims)),
        },
        "test_settings_comparison": {
            "1_unverified_captioner": evaluate_setting(test_claims, "dec_unverified"),
            "2_mask_only_checker": evaluate_setting(test_claims, "dec_mask_only"),
            "3_witness_only_checker": evaluate_setting(test_claims, "dec_witness_only"),
            "4_direct_qwen3vl_gated": evaluate_setting(test_claims, "dec_direct_qwen_gated"),
            "5_reinspection_all_gated": evaluate_setting(test_claims, "dec_reinsp_all_gated"),
            "6_m4_triggered_reinspection": evaluate_setting(test_claims, "dec_m4_triggered_reinsp"),
            "7_m5_calibrated_mask_first_hybrid": evaluate_setting(test_claims, "dec_m5_hybrid"),
        },
        "test_calibration": {
            "m5_hybrid": compute_calibration_metrics([c["hybrid_verifier_prob"] for c in test_claims], test_y),
            "m5_hybrid_aurc": compute_aurc([c["hybrid_verifier_prob"] for c in test_claims], test_y),
            "direct_qwen": compute_calibration_metrics([c["raw_direct"]["probabilities"]["accept"] for c in test_claims], test_y),
            "direct_qwen_aurc": compute_aurc([c["raw_direct"]["probabilities"]["accept"] for c in test_claims], test_y),
            "reinsp_all": compute_calibration_metrics([c["raw_reinsp"]["probabilities"]["accept"] for c in test_claims], test_y),
            "reinsp_all_aurc": compute_aurc([c["raw_reinsp"]["probabilities"]["accept"] for c in test_claims], test_y),
        },
        "test_threshold_sweep": sweep,
    }
    (out_dir / "test_verification_summary.json").write_text(json.dumps(ver_summary, indent=2), encoding="utf-8")

    clean_claims = [{k: v for k, v in c.items() if not k.startswith("_tmp_")} for c in test_claims]
    with (out_dir / "test_verification_claims.jsonl").open("w", encoding="utf-8") as f:
        for c in clean_claims:
            f.write(json.dumps(c) + "\n")

    # Export EXACT 120-claim Blind Human-Audit Packet matching locked Sujal schema!
    # Stratify to include ALL rejects & defers & contradicted cases + balanced accepts = 120 claims
    non_acc = [c for c in clean_claims if c["dec_m5_hybrid"] != "accept" or c["proxy_gt"] != "supported"]
    acc_clean = [c for c in clean_claims if c["dec_m5_hybrid"] == "accept" and c["proxy_gt"] == "supported"]
    rng_a = np.random.default_rng(2026)
    if len(non_acc) >= 120:
        audit_120 = non_acc[:120]
    else:
        need = 120 - len(non_acc)
        idx_a = sorted(rng_a.choice(len(acc_clean), size=min(need, len(acc_clean)), replace=False).tolist())
        audit_120 = non_acc + [acc_clean[i] for i in idx_a]
    # Deterministic shuffle so raters cannot guess accept vs reject from row order!
    perm = rng_a.permutation(len(audit_120)).tolist()
    audit_120 = [audit_120[i] for i in perm]

    audit_dir = out_dir / "blind_human_audit_packet"
    strips_dir = audit_dir / "blind_audit_strips"
    strips_dir.mkdir(parents=True, exist_ok=True)

    master_cols = [
        "audit_id", "pair_id", "stratum", "claim_id", "caption", "claim_span",
        "object", "action", "location", "quantity", "proxy_label",
        "unverified_decision", "mask_only_decision", "direct_vlm_decision",
        "reinsp_all_decision", "hybrid_verifier_decision", "hybrid_verifier_prob", "vlm_triggered",
    ]
    blind_cols = [
        "audit_id", "pair_id", "stratum", "claim_id", "caption", "claim_span",
        "object", "action", "location", "quantity", "human_label", "error_category", "notes",
    ]

    master_rows = []
    blind_rows = []
    for k, c in enumerate(audit_120, 1):
        aid = f"M5T_{k:03d}"
        loc_str = json.dumps(c["claim"]["location"]) if c["claim"].get("location") else ""
        qty_str = "" if c["claim"].get("quantity") is None else str(c["claim"]["quantity"])
        master_rows.append([
            aid, c["pair_id"], c["stratum"], c["claim_id"], c["caption"], c["source_span"],
            c["claim"]["object"], c["claim"]["action"], loc_str, qty_str, c["proxy_gt"],
            c["dec_unverified"], c["dec_mask_only"], c["dec_direct_qwen_gated"],
            c["dec_reinsp_all_gated"], c["dec_m5_hybrid"], c["hybrid_verifier_prob"], str(c["hybrid_vlm_triggered"]),
        ])
        blind_rows.append([
            aid, c["pair_id"], c["stratum"], c["claim_id"], c["caption"], c["source_span"],
            c["claim"]["object"], c["claim"]["action"], loc_str, qty_str, "", "", "",
        ])
        # Generate blind visual strip (NO verdict leaked!)
        b = pair_bundles[c["pair_id"]]
        t1_rgb = np.array(Image.open(ext_root / b["row"]["t1"]).convert("RGB"))
        t2_rgb = np.array(Image.open(ext_root / b["row"]["t2"]).convert("RGB"))
        c1_rgb = np.array(Image.open(c["crop_paths"][0]).convert("RGB"))
        c2_rgb = np.array(Image.open(c["crop_paths"][1]).convert("RGB"))
        save_blind_audit_strip(t1_rgb, t2_rgb, c1_rgb, c2_rgb, c["padded_bbox"], aid, c["source_span"], strips_dir / f"{aid}.png")

    def write_csv(path: Path, header: List[str], rows: List[List[Any]]):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write_csv(audit_dir / "test_audit_master_key.csv", master_cols, master_rows)
    write_csv(audit_dir / "full_blind_120.csv", blind_cols, blind_rows)
    # 4-person split + 2-person full-120 templates (SSSM & SRP can rate all 120 for 100% dual coverage!)
    write_csv(audit_dir / "rater_SSSM_60.csv", blind_cols, blind_rows[0:30] + blind_rows[90:120])
    write_csv(audit_dir / "rater_PA_60.csv", blind_cols, blind_rows[0:60])
    write_csv(audit_dir / "rater_SS_60.csv", blind_cols, blind_rows[30:90])
    write_csv(audit_dir / "rater_SRP_60.csv", blind_cols, blind_rows[60:120])

    # Export 10 Authentic Held-Out Test E2E Traces (Evidence Review UI compatible!)
    traces_root = out_dir / "test_e2e_traces"
    if traces_root.exists():
        shutil.rmtree(traces_root)
    traces_root.mkdir(parents=True)

    used_pids = set()
    chosen_traces = []
    trace_cats = [
        ("test_accepted_building", lambda b: b["stratum"] in ("building_only", "mixed_both") and any(c["dec_m5_hybrid"] == "accept" and c["claim"]["object"] == "building" for c in b["claims"])),
        ("test_accepted_road", lambda b: b["stratum"] in ("road_only", "mixed_both") and any(c["dec_m5_hybrid"] == "accept" and c["claim"]["object"] == "road" for c in b["claims"])),
        ("test_accepted_unchanged", lambda b: b["stratum"] == "unchanged" and any(c["dec_m5_hybrid"] == "accept" and c["claim"]["action"] == "no_change" for c in b["claims"])),
        ("test_rejected_hallucination", lambda b: any(c["dec_m5_hybrid"] == "reject" and c["proxy_gt"] == "contradicted" for c in b["claims"])),
        ("test_rejected_second", lambda b: any(c["dec_m5_hybrid"] == "reject" for c in b["claims"])),
        ("test_deferred_out_of_ontology", lambda b: any("T_out_of_mask_ontology" in c["triggers"] for c in b["claims"])),
        ("test_deferred_exact_count", lambda b: any("exact_count_without_independent_instance_evidence" in c["forced_reasons"] for c in b["claims"])),
        ("test_vlm_escalated_case", lambda b: any(c["hybrid_vlm_triggered"] for c in b["claims"])),
        ("test_small_change_case", lambda b: b["stratum"] == "small_change"),
        ("test_mixed_scene_case", lambda b: b["stratum"] == "mixed_both"),
    ]
    for label, fn in trace_cats:
        for pid, b in pair_bundles.items():
            if pid not in used_pids and b["claims"] and fn(b):
                used_pids.add(pid)
                chosen_traces.append((label, pid))
                break

    trace_index = []
    for label, pid in chosen_traces:
        b = pair_bundles[pid]
        r = b["row"]
        folder = traces_root / f"{label}__{pid.replace('/', '_')}"
        crops_dir = folder / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ext_root / r["t1"], folder / "T1.png")
        shutil.copy2(ext_root / r["t2"], folder / "T2.png")
        Image.fromarray(b["pred_mask"]).save(folder / "detector_response.mask.png")
        t2_rgb = np.array(Image.open(folder / "T2.png").convert("RGB"))
        first_box = b["claims"][0]["padded_bbox"] if b["claims"] else None
        Image.fromarray(make_mask_overlay(t2_rgb, b["pred_mask"], first_box)).save(folder / "mask_overlay.png")

        rep_claims = []
        for idx_c, c in enumerate(b["claims"]):
            c1n, c2n = f"claim_{idx_c}_crop_T1.png", f"claim_{idx_c}_crop_T2.png"
            shutil.copy2(c["crop_paths"][0], crops_dir / c1n)
            shutil.copy2(c["crop_paths"][1], crops_dir / c2n)
            rep_claims.append({
                "pair_id": pid,
                "claim_id": c["claim_id"],
                "source_span": c["source_span"],
                "claim": c["claim"],
                "candidate_bbox": c["padded_bbox"],
                "parser_provenance": {"parser": c["parser"]},
                "evidence": {
                    **c["witness"],
                    "crop_paths": [f"crops/{c1n}", f"crops/{c2n}"],
                    "transform_record": c["transform_meta"],
                    "mask_summary": c["mask_summary"],
                    "disagreement_triggers": c["triggers"],
                },
                "decision": {
                    "outcome": c["dec_m5_hybrid"],
                    "probabilities": {"accept": c["hybrid_verifier_prob"], "reject": round(1.0 - c["hybrid_verifier_prob"], 4) if c["dec_m5_hybrid"] == "reject" else 0.0, "defer": 0.0},
                    "route_reason": c["hybrid_route_reason"],
                    "vlm_triggered": c["hybrid_vlm_triggered"],
                    "reason_codes": c["forced_reasons"],
                },
                "direct_qwen_decision": c["gated_direct"],
                "reinspection_decision": c["gated_reinsp"],
                "mask_only_decision": c["dec_mask_only"],
            })

        acc_spans = list(dict.fromkeys(rc["source_span"] for rc in rep_claims if rc["decision"]["outcome"] == "accept"))
        report = {
            "pair_id": pid,
            "trace_category": label,
            "stratum": b["stratum"],
            "execution_mode": "m5_calibrated_mask_first_hybrid_test",
            "input_sha256": [sha256_file(folder / "T1.png"), sha256_file(folder / "T2.png")],
            "generated_caption": b["caption"],
            "verified_description": "; ".join(acc_spans) if acc_spans else "The image pair requires review before a change description can be confirmed.",
            "review_required": (not acc_spans) or any(rc["decision"]["outcome"] == "defer" for rc in rep_claims),
            "claims": rep_claims,
            "errors": [],
            "model_metadata": ver_summary["models"],
        }
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (folder / "review_inputs.json").write_text(json.dumps({"images": ["T1.png", "T2.png"], "mask": "detector_response.mask.png", "overlay": "mask_overlay.png"}, indent=2), encoding="utf-8")
        trace_index.append({"category": label, "pair_id": pid, "folder": folder.name, "caption": b["caption"], "outcomes": [rc["decision"]["outcome"] for rc in rep_claims]})

    (traces_root / "trace_index.json").write_text(json.dumps(trace_index, indent=2), encoding="utf-8")
    shutil.rmtree(work_crops, ignore_errors=True)
    print(f"=== M5 CAPTIONER + HYBRID VERIFIER SUITE COMPLETE in {time.time()-t0:.1f}s ===", flush=True)
    print(json.dumps(ver_summary["test_settings_comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
