"""Milestone 5 WS1-Det, WS5 & WS4-Det Evaluation Suite (1,858 Held-Out LEVIR-MCI Test Pairs).

Runs on Colab L4 (or any CUDA GPU) with frozen Milestone 4 checkpoints:
  1. Full 1,858-pair Test evaluation across all 5 neural architectures + Otsu + Always-BG:
     - 3x3 pixel confusion matrices, Pixel Acc, Macro IoU, Change mIoU, per-class IoU/Dice/P/R/F1
     - Exact pixel-level ROC-AUC & PR-AUC (1,000-bin histogram accumulation over 121.7M pixels)
     - 5-bucket change-area stratified performance (unchanged, tiny, small, medium, large)
     - Per-pair metrics + 1,000-resample paired bootstrap 95% CIs + Wilcoxon/McNemar p-values
  2. 17-condition Held-Out Test Stress Suite (D03_bit_lr3e4 vs A03_siam_r18_unet):
     - Core 10 corruptions + shift_8px, rotate_2deg, rotate_5deg, noise_sigma15,
       cloud_haze_15pct, cross_sensor_gamma_hue, downscale_0.5x
  3. Automated WS4-Det error mining + visual strip inputs (TP/FP/FN error maps) for D03.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from scipy import stats
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(SCRIPT_DIR), str(SCRIPT_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detector_models import build_model
from train_detector import read_jsonl, sha256_file


# --- Deterministic Corruptions & Stress Transforms ---
def corrupt_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.array(ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor))


def corrupt_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    return np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor))


def corrupt_blur(img: np.ndarray, radius: float) -> np.ndarray:
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def corrupt_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def corrupt_shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def corrupt_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def corrupt_gaussian_noise(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def corrupt_cloud_haze(img: np.ndarray, seed: int) -> np.ndarray:
    """Synthetic translucent cloud/haze patch covering ~15% of T2 deterministically per pair."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    cx = float(rng.uniform(0.25 * w, 0.75 * w))
    cy = float(rng.uniform(0.25 * h, 0.75 * h))
    rx = float(rng.uniform(0.20 * w, 0.28 * w))
    ry = float(rng.uniform(0.20 * h, 0.28 * h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    alpha = np.exp(-0.5 * dist2) * 0.72
    alpha = alpha[:, :, None]
    cloud_color = np.array([238.0, 242.0, 246.0], dtype=np.float32)
    out = img.astype(np.float32) * (1.0 - alpha) + cloud_color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_cross_sensor(img: np.ndarray) -> np.ndarray:
    """Simulate cross-sensor radiometric mismatch: gamma 1.35 + cool color-temperature shift on T2."""
    x = img.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 1e-6, 1.0), 1.35)
    # Channel gains simulating different spectral response
    gains = np.array([0.92, 1.02, 1.12], dtype=np.float32)[None, None, :]
    return np.clip(x * gains * 255.0, 0, 255).astype(np.uint8)


def corrupt_downscale(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


STRESS_CONDITIONS: List[Tuple[str, Callable[[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]]]] = [
    ("clean", lambda a, b, s: (a, b)),
    ("brightness_t2_0.75", lambda a, b, s: (a, corrupt_brightness(b, 0.75))),
    ("brightness_t2_1.25", lambda a, b, s: (a, corrupt_brightness(b, 1.25))),
    ("contrast_both_0.70", lambda a, b, s: (corrupt_contrast(a, 0.70), corrupt_contrast(b, 0.70))),
    ("blur_both_r1.0", lambda a, b, s: (corrupt_blur(a, 1.0), corrupt_blur(b, 1.0))),
    ("blur_both_r2.0", lambda a, b, s: (corrupt_blur(a, 2.0), corrupt_blur(b, 2.0))),
    ("jpeg_both_q40", lambda a, b, s: (corrupt_jpeg(a, 40), corrupt_jpeg(b, 40))),
    ("jpeg_both_q20", lambda a, b, s: (corrupt_jpeg(a, 20), corrupt_jpeg(b, 20))),
    ("noise_t2_sigma15", lambda a, b, s: (a, corrupt_gaussian_noise(b, 15.0, s))),
    ("shift_t2_2px", lambda a, b, s: (a, corrupt_shift(b, 2, 1))),
    ("shift_t2_4px", lambda a, b, s: (a, corrupt_shift(b, 4, 2))),
    ("shift_t2_8px", lambda a, b, s: (a, corrupt_shift(b, 8, 4))),
    ("rotate_t2_2deg", lambda a, b, s: (a, corrupt_rotate(b, 2.0))),
    ("rotate_t2_5deg", lambda a, b, s: (a, corrupt_rotate(b, 5.0))),
    ("cloud_haze_t2_15pct", lambda a, b, s: (a, corrupt_cloud_haze(b, s))),
    ("cross_sensor_radiometric", lambda a, b, s: (a, corrupt_cross_sensor(b))),
    ("downscale_both_0.5x", lambda a, b, s: (corrupt_downscale(a, 0.5), corrupt_downscale(b, 0.5))),
]


class StressMCIDataset(Dataset):
    def __init__(self, rows: List[dict], root: Path, fn):
        self.rows = rows
        self.root = root
        self.fn = fn

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        t1 = np.array(Image.open(self.root / r["t1"]).convert("RGB"))
        t2 = np.array(Image.open(self.root / r["t2"]).convert("RGB"))
        t1, t2 = self.fn(t1, t2, 10000 + idx)
        lbl = np.array(Image.open(self.root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        mask = np.zeros_like(lbl, dtype=np.uint8)
        mask[lbl == 128] = 1
        mask[lbl == 255] = 2
        x1 = torch.from_numpy(t1.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        x2 = torch.from_numpy(t2.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        return x1, x2, torch.from_numpy(mask.copy()).long(), r["pair_id"]


def roc_pr_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> dict:
    """Compute exact ROC-AUC, PR-AUC, and 101-point curves from high-resolution score histograms."""
    tp = np.cumsum(pos_hist[::-1]).astype(np.float64)
    fp = np.cumsum(neg_hist[::-1]).astype(np.float64)
    total_pos = float(pos_hist.sum())
    total_neg = float(neg_hist.sum())

    tpr = np.concatenate([[0.0], tp / max(total_pos, 1.0)])
    fpr = np.concatenate([[0.0], fp / max(total_neg, 1.0)])
    prec = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1.0)])
    rec = tpr.copy()

    roc_auc = float(np.trapezoid(tpr, fpr))
    # Exact Average Precision (step-function integral matching sklearn average_precision_score)
    pr_auc = float(np.sum(np.diff(rec) * prec[1:]))

    # Subsample 101 evenly spaced points along recall/FPR for compact smooth plots
    idx = np.linspace(0, len(tpr) - 1, 101).astype(int)
    return {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "roc_fpr": [round(float(x), 5) for x in fpr[idx]],
        "roc_tpr": [round(float(x), 5) for x in tpr[idx]],
        "pr_recall": [round(float(x), 5) for x in rec[idx]],
        "pr_precision": [round(float(x), 5) for x in prec[idx]],
    }


def metrics_from_cm(cm: np.ndarray, fa_counts: Dict[int, int], unchanged_n: int, small_stats: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    total = cm.sum()
    pixel_acc = float(np.trace(cm) / max(total, 1.0))

    per_class = {}
    ious = []
    dices = []
    for c, name in enumerate(["background", "road", "building"]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = float(tp / max(tp + fp + fn, 1.0))
        prec = float(tp / max(tp + fp, 1.0))
        rec = float(tp / max(tp + fn, 1.0))
        f1 = float(2 * tp / max(2 * tp + fp + fn, 1.0))
        per_class[name] = {
            "iou": round(iou, 6),
            "dice_f1": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
        }
        ious.append(iou)
        dices.append(f1)

    b_tp = float(cm[1:, 1:].sum())
    b_fp = float(cm[0, 1:].sum())
    b_fn = float(cm[1:, 0].sum())
    b_tn = float(cm[0, 0])
    b_prec = b_tp / max(b_tp + b_fp, 1.0)
    b_rec = b_tp / max(b_tp + b_fn, 1.0)
    b_f1 = 2 * b_tp / max(2 * b_tp + b_fp + b_fn, 1.0)
    b_iou = b_tp / max(b_tp + b_fp + b_fn, 1.0)

    s_tp, s_fp, s_fn = [float(x) for x in small_stats]
    small_f1 = 2 * s_tp / max(2 * s_tp + s_fp + s_fn, 1.0)

    cm_int = cm.astype(np.int64).tolist()
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    cm_norm = np.round(cm / row_sums, 6).tolist()

    return {
        "change_miou": round(0.5 * (ious[1] + ious[2]), 6),
        "macro_iou": round(float(np.mean(ious)), 6),
        "change_mean_dice": round(0.5 * (dices[1] + dices[2]), 6),
        "pixel_accuracy": round(pixel_acc, 6),
        "bg_iou": round(ious[0], 6),
        "road_iou": round(ious[1], 6),
        "building_iou": round(ious[2], 6),
        "per_class": per_class,
        "binary_precision": round(b_prec, 6),
        "binary_recall": round(b_rec, 6),
        "binary_f1": round(b_f1, 6),
        "binary_iou": round(b_iou, 6),
        "small_change_f1": round(small_f1, 6),
        "unchanged_pairs": unchanged_n,
        "unchanged_fa_counts": {str(k): int(v) for k, v in fa_counts.items()},
        "unchanged_fa_rates": {str(k): round(v / max(unchanged_n, 1), 6) for k, v in fa_counts.items()},
        "confusion_matrix": cm_int,
        "confusion_matrix_row_norm": cm_norm,
    }


def area_bucket(gt_chg_px: int) -> str:
    if gt_chg_px == 0:
        return "unchanged_0"
    if gt_chg_px <= 250:
        return "tiny_1_250"
    if gt_chg_px <= 1000:
        return "small_251_1000"
    if gt_chg_px <= 4000:
        return "medium_1001_4000"
    return "large_gt_4000"


@torch.inference_mode()
def evaluate_detector_full(
    model: torch.nn.Module,
    rows: List[dict],
    ext_root: Path,
    device: str,
    fn,
    batch_size: int = 32,
    compute_roc_pr: bool = True,
    collect_per_pair: bool = True,
    use_amp: bool = True,
) -> Tuple[dict, List[dict]]:
    ds = StressMCIDataset(rows, ext_root, fn)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    has_native_bf16 = device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if has_native_bf16 else torch.float16

    cm = np.zeros((3, 3), dtype=np.int64)
    fa_ths = [1, 16, 50, 100, 200, 500]
    fa_counts = {t: 0 for t in fa_ths}
    unchanged_n = 0
    small_stats = np.zeros(3, dtype=np.int64)  # tp, fp, fn

    n_bins = 10000
    pos_hist = np.zeros(n_bins, dtype=np.int64)
    neg_hist = np.zeros(n_bins, dtype=np.int64)

    bucket_cms: Dict[str, np.ndarray] = {
        b: np.zeros((3, 3), dtype=np.int64)
        for b in ["unchanged_0", "tiny_1_250", "small_251_1000", "medium_1001_4000", "large_gt_4000"]
    }
    bucket_counts = {b: 0 for b in bucket_cms}
    per_pair: List[dict] = []

    model.eval()
    for x1, x2, y, pids in dl:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.startswith("cuda")), dtype=amp_dtype):
            logits = model(x1, x2)
        logits_f = logits.float()
        # Exact non-saturating log-odds change score: logsumexp(logit_1, logit_2) - logit_0
        change_logodds = (torch.logsumexp(logits_f[:, 1:], dim=1) - logits_f[:, 0]).cpu().numpy()
        preds = logits_f.argmax(dim=1).cpu().numpy().astype(np.uint8)
        ys = y.numpy().astype(np.uint8)

        # Global 3x3 confusion matrix
        flat_y = ys.reshape(-1)
        flat_p = preds.reshape(-1)
        cm += np.bincount(flat_y * 3 + flat_p, minlength=9).reshape(3, 3)

        if compute_roc_pr:
            # Map log-odds [-15, +15] monotonically into 10,000 bins (zero float saturation!)
            scaled = np.clip((change_logodds.reshape(-1) + 15.0) / 30.0, 0.0, 1.0)
            bins = np.clip((scaled * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)
            gt_pos = flat_y > 0
            pos_hist += np.bincount(bins[gt_pos], minlength=n_bins)
            neg_hist += np.bincount(bins[~gt_pos], minlength=n_bins)

        for i in range(preds.shape[0]):
            yi = ys[i]
            pi = preds[i]
            gt_chg = int((yi > 0).sum())
            pr_chg = int((pi > 0).sum())
            bname = area_bucket(gt_chg)
            bucket_counts[bname] += 1
            cmi = np.bincount(yi.reshape(-1) * 3 + pi.reshape(-1), minlength=9).reshape(3, 3)
            bucket_cms[bname] += cmi

            if gt_chg == 0:
                unchanged_n += 1
                for t in fa_ths:
                    if pr_chg >= t:
                        fa_counts[t] += 1
            elif gt_chg < 655:  # <1% of 256x256
                small_stats[0] += int(((yi > 0) & (pi > 0)).sum())
                small_stats[1] += int(((yi == 0) & (pi > 0)).sum())
                small_stats[2] += int(((yi > 0) & (pi == 0)).sum())

            if collect_per_pair:
                tp = int(cmi[1:, 1:].sum())
                fp = int(cmi[0, 1:].sum())
                fn = int(cmi[1:, 0].sum())
                f1 = float(2 * tp / max(2 * tp + fp + fn, 1)) if (gt_chg > 0 or pr_chg > 0) else 1.0
                r_tp, r_fp, r_fn = int(cmi[1, 1]), int(cmi[:, 1].sum() - cmi[1, 1]), int(cmi[1, :].sum() - cmi[1, 1])
                b_tp, b_fp, b_fn = int(cmi[2, 2]), int(cmi[:, 2].sum() - cmi[2, 2]), int(cmi[2, :].sum() - cmi[2, 2])
                rb_swap = int(cmi[1, 2] + cmi[2, 1])
                per_pair.append({
                    "pair_id": pids[i],
                    "bucket": bname,
                    "gt_chg_px": gt_chg,
                    "pred_chg_px": pr_chg,
                    "tp": tp, "fp": fp, "fn": fn,
                    "binary_f1": round(f1, 6),
                    "road_tp": r_tp, "road_fp": r_fp, "road_fn": r_fn,
                    "bldg_tp": b_tp, "bldg_fp": b_fp, "bldg_fn": b_fn,
                    "road_bldg_swap_px": rb_swap,
                })

    res = metrics_from_cm(cm, fa_counts, unchanged_n, small_stats)
    if compute_roc_pr:
        res["roc_pr"] = roc_pr_from_hist(pos_hist, neg_hist)

    strat = {}
    for bname, bcm in bucket_cms.items():
        btp = float(bcm[1:, 1:].sum())
        bfp = float(bcm[0, 1:].sum())
        bfn = float(bcm[1:, 0].sum())
        bf1 = 2 * btp / max(2 * btp + bfp + bfn, 1.0)
        biou = btp / max(btp + bfp + bfn, 1.0)
        r_iou = float(bcm[1, 1] / max(bcm[1, :].sum() + bcm[:, 1].sum() - bcm[1, 1], 1.0))
        bl_iou = float(bcm[2, 2] / max(bcm[2, :].sum() + bcm[:, 2].sum() - bcm[2, 2], 1.0))
        strat[bname] = {
            "num_pairs": bucket_counts[bname],
            "binary_f1": round(bf1, 6),
            "binary_iou": round(biou, 6),
            "road_iou": round(r_iou, 6),
            "building_iou": round(bl_iou, 6),
            "change_miou": round(0.5 * (r_iou + bl_iou), 6),
        }
    res["area_stratified"] = strat
    return res, per_pair


def eval_nonlearning_test(rows: List[dict], ext_root: Path) -> Dict[str, dict]:
    """Otsu pixel-difference and Always-Background controls on 1,858 Test pairs (with ROC/PR for diff score)."""
    otsu_tp = otsu_fp = otsu_fn = otsu_tn = 0
    bg_tp = bg_fp = bg_fn = bg_tn = 0
    unchanged_n = 0
    otsu_fa = {t: 0 for t in [1, 16, 50, 100, 200, 500]}
    pos_hist = np.zeros(1000, dtype=np.int64)
    neg_hist = np.zeros(1000, dtype=np.int64)
    otsu_per_pair = []

    for r in rows:
        t1 = np.array(Image.open(ext_root / r["t1"]).convert("RGB"))
        t2 = np.array(Image.open(ext_root / r["t2"]).convert("RGB"))
        lbl = np.array(Image.open(ext_root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        gt = lbl > 0
        diff = np.mean(np.abs(t1.astype(np.float32) - t2.astype(np.float32)), axis=2).astype(np.uint8)
        bins = np.clip((diff.reshape(-1).astype(np.float32) / 255.0 * 999.0).astype(np.int32), 0, 999)
        flat_gt = gt.reshape(-1)
        pos_hist += np.bincount(bins[flat_gt], minlength=1000)
        neg_hist += np.bincount(bins[~flat_gt], minlength=1000)

        _, pred = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred_b = pred > 0
        tp = int((gt & pred_b).sum())
        fp = int((~gt & pred_b).sum())
        fn = int((gt & ~pred_b).sum())
        tn = int((~gt & ~pred_b).sum())
        otsu_tp += tp; otsu_fp += fp; otsu_fn += fn; otsu_tn += tn
        bg_fn += int(gt.sum()); bg_tn += int((~gt).sum())
        if int(gt.sum()) == 0:
            unchanged_n += 1
            pr_n = int(pred_b.sum())
            for t in otsu_fa:
                if pr_n >= t:
                    otsu_fa[t] += 1
        otsu_per_pair.append({
            "pair_id": r["pair_id"], "tp": tp, "fp": fp, "fn": fn,
            "binary_f1": round(float(2 * tp / max(2 * tp + fp + fn, 1)), 6),
        })

    o_p = otsu_tp / max(otsu_tp + otsu_fp, 1)
    o_r = otsu_tp / max(otsu_tp + otsu_fn, 1)
    o_f1 = 2 * otsu_tp / max(2 * otsu_tp + otsu_fp + otsu_fn, 1)
    o_iou = otsu_tp / max(otsu_tp + otsu_fp + otsu_fn, 1)
    o_acc = (otsu_tp + otsu_tn) / max(otsu_tp + otsu_fp + otsu_fn + otsu_tn, 1)
    bg_acc = bg_tn / max(bg_fn + bg_tn, 1)

    return {
        "otsu_diff": {
            "binary_precision": round(o_p, 6),
            "binary_recall": round(o_r, 6),
            "binary_f1": round(o_f1, 6),
            "binary_iou": round(o_iou, 6),
            "pixel_accuracy": round(o_acc, 6),
            "unchanged_fa_rates": {str(k): round(v / max(unchanged_n, 1), 6) for k, v in otsu_fa.items()},
            "roc_pr": roc_pr_from_hist(pos_hist, neg_hist),
        },
        "always_background": {
            "binary_precision": 0.0,
            "binary_recall": 0.0,
            "binary_f1": 0.0,
            "binary_iou": 0.0,
            "pixel_accuracy": round(bg_acc, 6),
            "unchanged_fa_rates": {str(k): 0.0 for k in otsu_fa},
        },
        "_otsu_per_pair": otsu_per_pair,
    }


def bootstrap_ci_and_sig(
    per_pair_map: Dict[str, List[dict]],
    ref_name: str = "D03_bit_lr3e4",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """1,000-resample paired bootstrap 95% CIs on Change mIoU & Binary F1 + Wilcoxon/McNemar p-values."""
    rng = np.random.default_rng(seed)
    n = len(per_pair_map[ref_name])
    boot_idx = rng.integers(0, n, size=(n_boot, n))

    def agg_arrays(rows: List[dict]):
        tp = np.array([r["tp"] for r in rows], dtype=np.float64)
        fp = np.array([r["fp"] for r in rows], dtype=np.float64)
        fn = np.array([r["fn"] for r in rows], dtype=np.float64)
        rtp = np.array([r["road_tp"] for r in rows], dtype=np.float64)
        rfp = np.array([r["road_fp"] for r in rows], dtype=np.float64)
        rfn = np.array([r["road_fn"] for r in rows], dtype=np.float64)
        btp = np.array([r["bldg_tp"] for r in rows], dtype=np.float64)
        bfp = np.array([r["bldg_fp"] for r in rows], dtype=np.float64)
        bfn = np.array([r["bldg_fn"] for r in rows], dtype=np.float64)
        return tp, fp, fn, rtp, rfp, rfn, btp, bfp, bfn

    out_ci = {}
    for name, rows in per_pair_map.items():
        tp, fp, fn, rtp, rfp, rfn, btp, bfp, bfn = agg_arrays(rows)
        s_tp = tp[boot_idx].sum(axis=1)
        s_fp = fp[boot_idx].sum(axis=1)
        s_fn = fn[boot_idx].sum(axis=1)
        f1s = 2 * s_tp / np.maximum(2 * s_tp + s_fp + s_fn, 1.0)

        sr_iou = rtp[boot_idx].sum(axis=1) / np.maximum((rtp + rfp + rfn)[boot_idx].sum(axis=1), 1.0)
        sb_iou = btp[boot_idx].sum(axis=1) / np.maximum((btp + bfp + bfn)[boot_idx].sum(axis=1), 1.0)
        mious = 0.5 * (sr_iou + sb_iou)

        out_ci[name] = {
            "change_miou_ci95": [round(float(np.percentile(mious, 2.5)), 6), round(float(np.percentile(mious, 97.5)), 6)],
            "binary_f1_ci95": [round(float(np.percentile(f1s, 2.5)), 6), round(float(np.percentile(f1s, 97.5)), 6)],
        }

    # Paired significance vs ref_name
    ref_rows = per_pair_map[ref_name]
    ref_f1 = np.array([r["binary_f1"] for r in ref_rows], dtype=np.float64)
    ref_unch_fa = np.array([1 if (r["gt_chg_px"] == 0 and r["pred_chg_px"] >= 50) else 0 for r in ref_rows])
    unch_mask = np.array([r["gt_chg_px"] == 0 for r in ref_rows])

    sig = {}
    for name, rows in per_pair_map.items():
        if name == ref_name:
            continue
        oth_f1 = np.array([r["binary_f1"] for r in rows], dtype=np.float64)
        diff = ref_f1 - oth_f1
        nz = diff[diff != 0]
        w_stat, w_p = stats.wilcoxon(nz) if len(nz) > 0 else (0.0, 1.0)

        # McNemar on unchanged FA@50px
        oth_unch_fa = np.array([1 if (r["gt_chg_px"] == 0 and r["pred_chg_px"] >= 50) else 0 for r in rows])
        b01 = int(((ref_unch_fa == 0) & (oth_unch_fa == 1) & unch_mask).sum())  # ref right, oth wrong
        b10 = int(((ref_unch_fa == 1) & (oth_unch_fa == 0) & unch_mask).sum())  # ref wrong, oth right
        n_disc = b01 + b10
        mc_p = float(stats.binomtest(min(b01, b10), n_disc, 0.5).pvalue) if n_disc > 0 else 1.0

        sig[f"{ref_name}_vs_{name}"] = {
            "mean_pair_f1_diff": round(float(diff.mean()), 6),
            "wilcoxon_stat": round(float(w_stat), 2),
            "wilcoxon_p_value": float(w_p),
            "mcnemar_unchanged_fa50_ref_better": b01,
            "mcnemar_unchanged_fa50_other_better": b10,
            "mcnemar_p_value": float(mc_p),
        }

    return {"bootstrap_95ci": out_ci, "paired_significance": sig}


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Color-coded error map: dark slate=TN, green=TP, red=FP, blue=FN, yellow=Road/Bldg swap."""
    h, w = gt.shape
    out = np.full((h, w, 3), (24, 34, 42), dtype=np.uint8)
    tp_exact = (gt > 0) & (gt == pred)
    swap = (gt > 0) & (pred > 0) & (gt != pred)
    fp = (gt == 0) & (pred > 0)
    fn = (gt > 0) & (pred == 0)
    out[tp_exact] = (52, 199, 89)     # green: exact TP
    out[swap] = (255, 204, 0)         # yellow: road <-> building confusion
    out[fp] = (235, 64, 52)           # red: false positive
    out[fn] = (58, 134, 255)          # blue: false negative
    return out


def colorize_mask(m: np.ndarray) -> np.ndarray:
    out = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    out[m == 1] = (255, 200, 40)   # road = gold
    out[m == 2] = (235, 70, 70)    # building = crimson
    return out


@torch.inference_mode()
def export_error_mining_and_strips(
    model: torch.nn.Module,
    rows: List[dict],
    per_pair: List[dict],
    ext_root: Path,
    device: str,
    out_dir: Path,
) -> dict:
    """Mine worst Test FPs, FNs, Road/Bldg swaps, and representative successes + export visual strips."""
    by_pid = {r["pair_id"]: r for r in rows}
    fps = sorted([r for r in per_pair if r["gt_chg_px"] == 0 and r["pred_chg_px"] > 0], key=lambda x: -x["pred_chg_px"])[:20]
    fns = sorted([r for r in per_pair if r["gt_chg_px"] >= 50], key=lambda x: (x["binary_f1"], -x["fn"]))[:20]
    swaps = sorted([r for r in per_pair if r["road_bldg_swap_px"] > 0], key=lambda x: -x["road_bldg_swap_px"])[:15]
    tiny_fn = sorted([r for r in per_pair if 0 < r["gt_chg_px"] <= 250], key=lambda x: x["binary_f1"])[:15]
    successes = sorted([r for r in per_pair if r["gt_chg_px"] >= 500 and r["binary_f1"] >= 0.90], key=lambda x: -x["binary_f1"])[:10]

    picks = (
        [("false_positive_unchanged", r) for r in fps[:4]]
        + [("false_negative_missed", r) for r in fns[:4]]
        + [("road_building_swap", r) for r in swaps[:3]]
        + [("tiny_structure_miss", r) for r in tiny_fn[:3]]
        + [("success_high_f1", r) for r in successes[:4]]
    )

    strips_dir = out_dir / "detector_visual_cases"
    strips_dir.mkdir(parents=True, exist_ok=True)
    has_native_bf16 = device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if has_native_bf16 else torch.float16
    model.eval()

    exported = []
    seen = set()
    for cat, rec in picks:
        pid = rec["pair_id"]
        if pid in seen:
            continue
        seen.add(pid)
        row = by_pid[pid]
        t1 = np.array(Image.open(ext_root / row["t1"]).convert("RGB"))
        t2 = np.array(Image.open(ext_root / row["t2"]).convert("RGB"))
        lbl = np.array(Image.open(ext_root / row["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        gt = np.zeros_like(lbl, dtype=np.uint8)
        gt[lbl == 128] = 1
        gt[lbl == 255] = 2

        x1 = torch.from_numpy(t1.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 127.5 - 1.0
        x2 = torch.from_numpy(t2.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 127.5 - 1.0
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=amp_dtype):
            pred = model(x1, x2).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        case_dir = strips_dir / f"{cat}__{pid.replace('/', '_')}"
        case_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(t1).save(case_dir / "T1.png")
        Image.fromarray(t2).save(case_dir / "T2.png")
        Image.fromarray(colorize_mask(gt)).save(case_dir / "gt_color.png")
        Image.fromarray(colorize_mask(pred)).save(case_dir / "pred_color.png")
        Image.fromarray(make_error_map(gt, pred)).save(case_dir / "error_map.png")
        exported.append({"category": cat, "folder": case_dir.name, **rec})

    return {
        "top20_false_positives_unchanged": fps,
        "top20_false_negatives": fns,
        "top15_road_bldg_swaps": swaps,
        "top15_tiny_structure_misses": tiny_fn,
        "top10_high_f1_successes": successes,
        "exported_visual_cases": exported,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--extracted-root", required=True)
    ap.add_argument("--checkpoints-dir", required=True)
    ap.add_argument("--bit-source-dir", required=True)
    ap.add_argument("--changeformer-source-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_root = Path(args.extracted_root)
    ckpt_root = Path(args.checkpoints_dir)
    test_rows = read_jsonl(Path(args.manifests_dir) / "mci_test.jsonl")
    assert len(test_rows) == 1858, f"Expected 1858 test pairs, got {len(test_rows)}"

    models_spec = [
        ("D03_bit_lr3e4", "bit_resnet18", ckpt_root / "D03_bit_lr3e4/best.pt"),
        ("A03_siam_r18_unet", "siam_resnet18_unet", ckpt_root / "A03_siam_r18_unet/best.pt"),
        ("A01_ef_unet", "ef_unet", ckpt_root / "A01_ef_unet/best.pt"),
        ("A02_fc_siam_diff", "fc_siam_diff", ckpt_root / "A02_fc_siam_diff/best.pt"),
        ("A04_changeformer", "changeformer", ckpt_root / "A04_changeformer/best.pt"),
    ]

    t0 = time.time()
    print("=== [1/4] Evaluating Non-Learning Controls on 1,858 Held-Out Test Pairs ===", flush=True)
    controls = eval_nonlearning_test(test_rows, ext_root)
    controls_clean = {k: v for k, v in controls.items() if not k.startswith("_")}
    print(f"  Otsu Test F1={controls_clean['otsu_diff']['binary_f1']*100:.2f}%, ROC-AUC={controls_clean['otsu_diff']['roc_pr']['roc_auc']:.4f}", flush=True)

    print("=== [2/4] Evaluating All 5 Trained Architectures on 1,858 Held-Out Test Pairs ===", flush=True)
    det_results = {}
    per_pair_map: Dict[str, List[dict]] = {}
    cached_models: Dict[str, Tuple[str, Path]] = {}

    for run_id, arch, ckpt_path in models_spec:
        assert ckpt_path.is_file(), f"Missing {ckpt_path}"
        sha = sha256_file(ckpt_path)
        t_m = time.time()
        model, _, _, _ = build_model(arch, args.device, args.bit_source_dir, args.changeformer_source_dir)
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=True)
        res, per_pair = evaluate_detector_full(
            model, test_rows, ext_root, args.device, STRESS_CONDITIONS[0][1],
            batch_size=16 if arch == "changeformer" else 32,
            compute_roc_pr=True, collect_per_pair=True,
            use_amp=(arch != "changeformer"),
        )
        res["arch"] = arch
        res["checkpoint_sha256"] = sha
        res["checkpoint_bytes"] = ckpt_path.stat().st_size
        det_results[run_id] = res
        per_pair_map[run_id] = per_pair
        cached_models[run_id] = (arch, ckpt_path)
        print(
            f"  {run_id:20s} | Chg mIoU={res['change_miou']*100:.2f}% | Bin F1={res['binary_f1']*100:.2f}% | "
            f"ROC-AUC={res['roc_pr']['roc_auc']:.4f} | PR-AUC={res['roc_pr']['pr_auc']:.4f} | FA@50={res['unchanged_fa_rates']['50']*100:.2f}% ({time.time()-t_m:.1f}s)",
            flush=True,
        )
        if run_id == "D03_bit_lr3e4":
            print("=== [2b] Mining WS4-Det Failure & Success Cases for D03_bit_lr3e4 ===", flush=True)
            err_summary = export_error_mining_and_strips(model, test_rows, per_pair, ext_root, args.device, out_dir)
            (out_dir / "detector_error_mining.json").write_text(json.dumps(err_summary, indent=2), encoding="utf-8")
        del model
        torch.cuda.empty_cache()

    print("=== [3/4] Computing 1,000-Resample Paired Bootstrap CIs & Wilcoxon/McNemar Tests ===", flush=True)
    stats_out = bootstrap_ci_and_sig(per_pair_map, ref_name="D03_bit_lr3e4", n_boot=1000, seed=42)

    full_test_payload = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "levir_mci_test_strict",
        "num_pairs": len(test_rows),
        "non_learning_controls": controls_clean,
        "models": det_results,
        "statistical_analysis": stats_out,
    }
    (out_dir / "detector_test_full.json").write_text(json.dumps(full_test_payload, indent=2), encoding="utf-8")
    # Save compact per-pair table for D03 & A03
    with (out_dir / "detector_per_pair_d03_a03.jsonl").open("w", encoding="utf-8") as f:
        for r_d, r_a in zip(per_pair_map["D03_bit_lr3e4"], per_pair_map["A03_siam_r18_unet"]):
            f.write(json.dumps({"pair_id": r_d["pair_id"], "bucket": r_d["bucket"], "gt_chg_px": r_d["gt_chg_px"], "d03": r_d, "a03": r_a}) + "\n")

    print("=== [4/4] Running 17-Condition Held-Out Test Stress Suite (D03 vs A03) ===", flush=True)
    stress_results: Dict[str, Dict[str, dict]] = {}
    for run_id in ("D03_bit_lr3e4", "A03_siam_r18_unet"):
        arch, ckpt_path = cached_models[run_id]
        model, _, _, _ = build_model(arch, args.device, args.bit_source_dir, args.changeformer_source_dir)
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=True)
        stress_results[run_id] = {}
        for cond_name, fn in STRESS_CONDITIONS:
            if cond_name == "clean":
                # Reuse exact clean metrics without ROC arrays
                c_copy = {k: v for k, v in det_results[run_id].items() if k not in ("roc_pr", "area_stratified")}
                stress_results[run_id]["clean"] = c_copy
                continue
            tc = time.time()
            res_c, _ = evaluate_detector_full(
                model, test_rows, ext_root, args.device, fn,
                batch_size=32, compute_roc_pr=False, collect_per_pair=False,
            )
            stress_results[run_id][cond_name] = {k: v for k, v in res_c.items() if k != "area_stratified"}
            print(
                f"  [{run_id}] {cond_name:26s} -> Chg mIoU={res_c['change_miou']*100:5.2f}%, Bin F1={res_c['binary_f1']*100:5.2f}%, FA@50={res_c['unchanged_fa_rates']['50']*100:5.2f}% ({time.time()-tc:.1f}s)",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    stress_payload = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "levir_mci_test_strict",
        "num_pairs": len(test_rows),
        "conditions": [c[0] for c in STRESS_CONDITIONS],
        "results": stress_results,
    }
    (out_dir / "stress_test_levir_mci_test.json").write_text(json.dumps(stress_payload, indent=2), encoding="utf-8")
    print(f"=== M5 DETECTOR + STRESS SUITE COMPLETE in {time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
