"""Evaluate frozen detector checkpoints on:
  1. Full LEVIR-MCI validation (1,242) & held-out test (1,858)
  2. Non-learning controls (Always-Background, Otsu pixel-difference)
  3. Corruption robustness ladder on LEVIR-MCI val (1,242):
       brightness, contrast, Gaussian blur, JPEG compression, translation shift
  4. Cross-dataset transfer on SECOND-CC val (530) & test (1,142) (binary any-change)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.models.detector_architectures import build_model
from scripts.train_detector_full import ChangeLoss, compute_metrics_from_stats, read_jsonl, sha256_file


# --- Deterministic Corruption Ladder ---
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


def corrupt_shift_t2(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


CORRUPTIONS: List[Tuple[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]] = [
    ("clean", lambda a, b: (a, b)),
    ("brightness_t2_0.75", lambda a, b: (a, corrupt_brightness(b, 0.75))),
    ("brightness_t2_1.25", lambda a, b: (a, corrupt_brightness(b, 1.25))),
    ("contrast_both_0.70", lambda a, b: (corrupt_contrast(a, 0.70), corrupt_contrast(b, 0.70))),
    ("blur_both_r1.0", lambda a, b: (corrupt_blur(a, 1.0), corrupt_blur(b, 1.0))),
    ("blur_both_r2.0", lambda a, b: (corrupt_blur(a, 2.0), corrupt_blur(b, 2.0))),
    ("jpeg_both_q40", lambda a, b: (corrupt_jpeg(a, 40), corrupt_jpeg(b, 40))),
    ("jpeg_both_q20", lambda a, b: (corrupt_jpeg(a, 20), corrupt_jpeg(b, 20))),
    ("shift_t2_2px", lambda a, b: (a, corrupt_shift_t2(b, 2, 1))),
    ("shift_t2_4px", lambda a, b: (a, corrupt_shift_t2(b, 4, 2))),
]


class CorruptedMCIDataset(Dataset):
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
        t1, t2 = self.fn(t1, t2)
        lbl = np.array(Image.open(self.root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        mask = np.zeros_like(lbl, dtype=np.uint8)
        mask[lbl == 128] = 1
        mask[lbl == 255] = 2
        x1 = torch.from_numpy(t1.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        x2 = torch.from_numpy(t2.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        return x1, x2, torch.from_numpy(mask.copy()).long(), r["pair_id"]


class SecondCCTransferDataset(Dataset):
    """Loads SECOND-CC pairs; ground truth binary any-change = (sem_t1 != sem_t2) & non-unchanged."""

    def __init__(self, rows: List[dict], root: Path):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        t1 = np.array(Image.open(self.root / r["t1"]).convert("RGB"))
        t2 = np.array(Image.open(self.root / r["t2"]).convert("RGB"))
        # SECOND semantic masks: white (255,255,255) = unchanged; any non-white pixel in sem_A or sem_B is changed
        s1 = np.array(Image.open(self.root / r["labels"][0]).convert("RGB"))
        s2 = np.array(Image.open(self.root / r["labels"][1]).convert("RGB"))
        gt_chg = (~((s1[:, :, 0] == 255) & (s1[:, :, 1] == 255) & (s1[:, :, 2] == 255))) | (
            ~((s2[:, :, 0] == 255) & (s2[:, :, 1] == 255) & (s2[:, :, 2] == 255))
        )
        # Building class in SECOND color palette is (128, 0, 0) RGB (dark red)
        b1 = (s1[:, :, 0] == 128) & (s1[:, :, 1] == 0) & (s1[:, :, 2] == 0)
        b2 = (s2[:, :, 0] == 128) & (s2[:, :, 1] == 0) & (s2[:, :, 2] == 0)
        gt_bldg = b1 | b2
        x1 = torch.from_numpy(t1.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        x2 = torch.from_numpy(t2.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        return x1, x2, torch.from_numpy(gt_chg.astype(np.uint8)).long(), torch.from_numpy(gt_bldg.astype(np.uint8)).long(), r["pair_id"]


def eval_nonlearning_controls(rows: List[dict], root: Path) -> Dict[str, dict]:
    """Always-Background and Otsu pixel-difference controls on full partition."""
    otsu_tp = otsu_fp = otsu_fn = otsu_tn = 0
    bg_tp = bg_fp = bg_fn = bg_tn = 0
    unchanged_total = 0
    otsu_fa = {1: 0, 50: 0, 100: 0}

    for r in rows:
        t1 = np.array(Image.open(root / r["t1"]).convert("RGB"))
        t2 = np.array(Image.open(root / r["t2"]).convert("RGB"))
        lbl = np.array(Image.open(root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        gt = (lbl > 0)
        diff = np.mean(np.abs(t1.astype(np.float32) - t2.astype(np.float32)), axis=2).astype(np.uint8)
        _, pred_otsu = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        po = pred_otsu.astype(bool)

        otsu_tp += int((gt & po).sum())
        otsu_fp += int(((~gt) & po).sum())
        otsu_fn += int((gt & (~po)).sum())
        otsu_tn += int(((~gt) & (~po)).sum())

        bg_fn += int(gt.sum())
        bg_tn += int((~gt).sum())

        if not gt.any():
            unchanged_total += 1
            pc = int(po.sum())
            for t in otsu_fa:
                if pc >= t:
                    otsu_fa[t] += 1

    def bin_summary(tp, fp, fn, tn, fa_dict):
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        return {
            "binary_precision": prec,
            "binary_recall": rec,
            "binary_f1": f1,
            "binary_iou": iou,
            "unchanged_pairs": unchanged_total,
            "unchanged_fa_counts": {str(k): v for k, v in fa_dict.items()},
            "unchanged_fa_rates": {str(k): v / max(unchanged_total, 1) for k, v in fa_dict.items()},
        }

    return {
        "otsu_control": bin_summary(otsu_tp, otsu_fp, otsu_fn, otsu_tn, otsu_fa),
        "always_background": bin_summary(bg_tp, bg_fp, bg_fn, bg_tn, {1: 0, 50: 0, 100: 0}),
    }


@torch.inference_mode()
def eval_mci_loader(model, loader, device, amp_dtype):
    model.eval()
    cm = torch.zeros((3, 3), dtype=torch.int64, device=device)
    fa_thresholds = [1, 16, 50, 100, 200]
    unchanged_total = 0
    unchanged_fa = {t: 0 for t in fa_thresholds}
    small_tp = small_fp = small_fn = 0
    for x1, x2, y, _ in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            pred = model(x1, x2).argmax(dim=1)
        cm += torch.bincount((y * 3 + pred).reshape(-1), minlength=9).reshape(3, 3)
        gt_chg = (y > 0).reshape(y.shape[0], -1)
        pr_chg = (pred > 0).reshape(pred.shape[0], -1)
        gt_px = gt_chg.sum(dim=1).cpu().tolist()
        pr_px = pr_chg.sum(dim=1).cpu().tolist()
        tp_v = (gt_chg & pr_chg).sum(dim=1).cpu().tolist()
        fp_v = ((~gt_chg) & pr_chg).sum(dim=1).cpu().tolist()
        fn_v = (gt_chg & (~pr_chg)).sum(dim=1).cpu().tolist()
        for i in range(len(gt_px)):
            if gt_px[i] == 0:
                unchanged_total += 1
                for t in fa_thresholds:
                    if pr_px[i] >= t:
                        unchanged_fa[t] += 1
            elif gt_px[i] < 655:
                small_tp += tp_v[i]
                small_fp += fp_v[i]
                small_fn += fn_v[i]
    return compute_metrics_from_stats(cm, unchanged_total, unchanged_fa, small_tp, small_fp, small_fn)


@torch.inference_mode()
def eval_second_cc(model, loader, device, amp_dtype):
    model.eval()
    tp = fp = fn = tn = 0
    b_tp = b_fp = b_fn = 0
    unchanged_total = 0
    fa = {1: 0, 50: 0, 100: 0}
    for x1, x2, gt_chg, gt_bldg, _ in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        gt_chg = gt_chg.to(device).bool()
        gt_bldg = gt_bldg.to(device).bool()
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            pred = model(x1, x2).argmax(dim=1)
        pr_chg = pred > 0
        pr_bldg = pred == 2
        tp += int((gt_chg & pr_chg).sum().item())
        fp += int(((~gt_chg) & pr_chg).sum().item())
        fn += int((gt_chg & (~pr_chg)).sum().item())
        tn += int(((~gt_chg) & (~pr_chg)).sum().item())
        b_tp += int((gt_bldg & pr_bldg).sum().item())
        b_fp += int(((~gt_bldg) & pr_bldg).sum().item())
        b_fn += int((gt_bldg & (~pr_bldg)).sum().item())
        for i in range(gt_chg.shape[0]):
            if not gt_chg[i].any():
                unchanged_total += 1
                pc = int(pr_chg[i].sum().item())
                for t in fa:
                    if pc >= t:
                        fa[t] += 1
    return {
        "any_change_precision": tp / max(tp + fp, 1),
        "any_change_recall": tp / max(tp + fn, 1),
        "any_change_f1": (2 * tp) / max(2 * tp + fp + fn, 1),
        "any_change_iou": tp / max(tp + fp + fn, 1),
        "building_change_precision": b_tp / max(b_tp + b_fp, 1),
        "building_change_recall": b_tp / max(b_tp + b_fn, 1),
        "building_change_f1": (2 * b_tp) / max(2 * b_tp + b_fp + b_fn, 1),
        "building_change_iou": b_tp / max(b_tp + b_fp + b_fn, 1),
        "unchanged_pairs": unchanged_total,
        "unchanged_fa_counts": {str(k): v for k, v in fa.items()},
        "unchanged_fa_rates": {str(k): v / max(unchanged_total, 1) for k, v in fa.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bit-source-dir", default="/media/shoaib/STUDYLINUX/IIT/DegreeTerm3/DSAI-Project/models/external/BIT_CD")
    ap.add_argument("--changeformer-source-dir", default="/media/shoaib/STUDYLINUX/IIT/DegreeTerm3/DSAI-Project/models/external/ChangeFormer")
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--extracted-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifests = Path(args.manifests_dir)
    ext = Path(args.extracted_root)
    ckpt_path = Path(args.checkpoint)

    amp_dtype = torch.bfloat16 if (args.device.startswith("cuda") and torch.cuda.is_bf16_supported()) else None
    model, _, _, _ = build_model(args.arch, args.device, args.bit_source_dir, args.changeformer_source_dir)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    val_rows = read_jsonl(manifests / "mci_val.jsonl")
    test_rows = read_jsonl(manifests / "mci_test.jsonl")

    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arch": args.arch,
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "levir_mci_val_corruptions": {},
        "levir_mci_test": None,
        "nonlearning_controls": {},
        "second_cc_transfer": {},
    }

    print("=== Evaluating corruption robustness ladder on LEVIR-MCI Val (1,242 pairs) ===", flush=True)
    for cname, cfn in CORRUPTIONS:
        ds = CorruptedMCIDataset(val_rows, ext, cfn)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        m = eval_mci_loader(model, dl, args.device, amp_dtype)
        result["levir_mci_val_corruptions"][cname] = m
        print(f"  [{cname:20s}] chg_mIoU={m['change_miou']*100:.2f}% binF1={m['binary_f1']*100:.2f}% FA@50px={m['unchanged_fa_rates']['50']*100:.1f}%", flush=True)

    print("=== Evaluating frozen checkpoint on held-out LEVIR-MCI Test (1,858 pairs) ===", flush=True)
    test_ds = CorruptedMCIDataset(test_rows, ext, lambda a, b: (a, b))
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    tm = eval_mci_loader(model, test_dl, args.device, amp_dtype)
    result["levir_mci_test"] = tm
    print(f"  [LEVIR-MCI TEST] chg_mIoU={tm['change_miou']*100:.2f}% (road={tm['road_iou']*100:.2f}% bldg={tm['building_iou']*100:.2f}%) binF1={tm['binary_f1']*100:.2f}% FA@50px={tm['unchanged_fa_rates']['50']*100:.1f}%", flush=True)

    print("=== Evaluating non-learning controls (Otsu, Always-BG) ===", flush=True)
    result["nonlearning_controls"]["levir_mci_val"] = eval_nonlearning_controls(val_rows, ext)
    result["nonlearning_controls"]["levir_mci_test"] = eval_nonlearning_controls(test_rows, ext)

    for split in ("val", "test"):
        sp = manifests / f"second_cc_{split}.jsonl"
        if sp.exists() and (ext / "SECOND-CC-AUG" / split).exists():
            sc_rows = read_jsonl(sp)
            sc_ds = SecondCCTransferDataset(sc_rows, ext)
            sc_dl = DataLoader(sc_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            sm = eval_second_cc(model, sc_dl, args.device, amp_dtype)
            result["second_cc_transfer"][split] = {"num_pairs": len(sc_rows), **sm}
            print(f"  [SECOND-CC {split:4s}] n={len(sc_rows)} anyF1={sm['any_change_f1']*100:.2f}% bldgF1={sm['building_change_f1']*100:.2f}%", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SAVED_EVAL_REPORT:", out_path, flush=True)


if __name__ == "__main__":
    main()
