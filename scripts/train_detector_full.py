"""Full-data, resumable, auditable 3-class change detection trainer for TerraWatch M4 recovery.

Guarantees:
- Consumes ALL 6,444 strict LEVIR-MCI training pairs and evaluates ALL 1,242 strict validation pairs
  (asserts exact pair-ID counts and zero train/val/test overlap).
- Logs train loss, val loss, change-class mIoU (predeclared primary selection metric),
  all-class macro-IoU, per-class IoUs, binary precision/recall/F1/IoU,
  unchanged-pair false-alarm rates at multiple pixel thresholds, small-change F1,
  LR, trainable/frozen param counts, epoch duration, throughput, and peak VRAM.
- Saves both best.pt and resumable last.pt with SHA-256 digests and full run.json receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.models.detector_architectures import build_model


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_photometric_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild photometric jitter: brightness +/-15%, contrast +/-15%, per-channel gain +/-5%."""
    arr = img.astype(np.float32)
    b = rng.uniform(0.85, 1.15)
    c = rng.uniform(0.85, 1.15)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * c + mean * b
    gain = rng.uniform(0.95, 1.05, size=(1, 1, 3)).astype(np.float32)
    arr = arr * gain
    return np.clip(arr, 0, 255).astype(np.uint8)


def _apply_paired_scale(
    t1: np.ndarray, t2: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mild paired random resized crop (scale in [0.85, 1.0]), bilinear for RGB, nearest for mask."""
    h, w = mask.shape
    scale = rng.uniform(0.85, 1.0)
    nh = max(32, int(round(h * scale)))
    nw = max(32, int(round(w * scale)))
    if nh == h and nw == w:
        return t1, t2, mask
    y0 = int(rng.integers(0, h - nh + 1))
    x0 = int(rng.integers(0, w - nw + 1))
    t1_c = np.array(Image.fromarray(t1[y0 : y0 + nh, x0 : x0 + nw]).resize((w, h), Image.Resampling.BILINEAR))
    t2_c = np.array(Image.fromarray(t2[y0 : y0 + nh, x0 : x0 + nw]).resize((w, h), Image.Resampling.BILINEAR))
    m_c = np.array(Image.fromarray(mask[y0 : y0 + nh, x0 : x0 + nw]).resize((w, h), Image.Resampling.NEAREST))
    return t1_c, t2_c, m_c


class ExtractedMCIDataset(Dataset):
    """Fast NVMe dataset reading extracted LEVIR-MCI PNGs with strict manifest order."""

    def __init__(
        self,
        rows: List[dict],
        extracted_root: Path,
        aug: str = "none",
        base_seed: int = 42,
    ):
        self.rows = rows
        self.root = extracted_root
        self.aug = aug
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        t1 = np.array(Image.open(self.root / r["t1"]).convert("RGB"))
        t2 = np.array(Image.open(self.root / r["t2"]).convert("RGB"))
        lbl = np.array(Image.open(self.root / r["labels"][0]))
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        # Exact lossless mapping: 0 -> 0 (bg), 128 -> 1 (road), 255 -> 2 (building)
        mask = np.zeros_like(lbl, dtype=np.uint8)
        mask[lbl == 128] = 1
        mask[lbl == 255] = 2

        if self.aug != "none":
            rng = np.random.default_rng(self.base_seed * 1000003 + self.epoch * 10007 + idx)
            if self.aug == "flip":
                if rng.random() < 0.5:
                    t1 = np.ascontiguousarray(t1[:, ::-1])
                    t2 = np.ascontiguousarray(t2[:, ::-1])
                    mask = np.ascontiguousarray(mask[:, ::-1])
            elif self.aug in ("flip_photometric", "flip_photo_scale"):
                if rng.random() < 0.5:
                    t1 = np.ascontiguousarray(t1[:, ::-1])
                    t2 = np.ascontiguousarray(t2[:, ::-1])
                    mask = np.ascontiguousarray(mask[:, ::-1])
                if rng.random() < 0.5:
                    t1 = np.ascontiguousarray(t1[::-1, :])
                    t2 = np.ascontiguousarray(t2[::-1, :])
                    mask = np.ascontiguousarray(mask[::-1, :])
                if rng.random() < 0.5:
                    t1 = _apply_photometric_jitter(t1, rng)
                if rng.random() < 0.5:
                    t2 = _apply_photometric_jitter(t2, rng)
                if self.aug == "flip_photo_scale" and rng.random() < 0.5:
                    t1, t2, mask = _apply_paired_scale(t1, t2, mask, rng)
            else:
                raise ValueError(f"Unknown augmentation mode: {self.aug}")

        x1 = torch.from_numpy(t1.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        x2 = torch.from_numpy(t2.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0
        y = torch.from_numpy(mask.copy()).long()
        return x1, x2, y, r["pair_id"]


class ChangeLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "focal_dice",
        class_weights: Tuple[float, float, float] = (0.10, 0.45, 0.45),
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        w = self.weights.to(logits_f.device)
        if self.loss_type == "weighted_ce":
            return F.cross_entropy(logits_f, target, weight=w, label_smoothing=self.label_smoothing)

        if self.loss_type == "ce_dice":
            ce = F.cross_entropy(logits_f, target, weight=w, label_smoothing=self.label_smoothing)
            dice = self._soft_dice(logits_f, target)
            return 0.5 * ce + 0.5 * dice

        if self.loss_type == "focal_dice":
            ce_none = F.cross_entropy(logits_f, target, weight=w, reduction="none")
            pt = logits_f.softmax(dim=1).gather(1, target[:, None]).squeeze(1).clamp(1e-6, 1.0)
            focal = ((1.0 - pt) ** self.focal_gamma * ce_none).mean()
            dice = self._soft_dice(logits_f, target)
            return 0.5 * focal + 0.5 * dice

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    @staticmethod
    def _soft_dice(logits_f: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = logits_f.softmax(dim=1)
        truth = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        dice_per_class = (2.0 * (prob * truth).sum(dims) + 1.0) / (prob.sum(dims) + truth.sum(dims) + 1.0)
        return 1.0 - dice_per_class.mean()


def compute_metrics_from_stats(
    cm: torch.Tensor,
    unchanged_total: int,
    unchanged_fa: Dict[int, int],
    small_tp: int,
    small_fp: int,
    small_fn: int,
) -> Dict[str, object]:
    cm_cpu = cm.cpu().to(torch.float64)
    tp = cm_cpu.diag()
    union = cm_cpu.sum(0) + cm_cpu.sum(1) - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1.0), torch.zeros_like(tp))
    bg_iou, road_iou, bldg_iou = [float(x) for x in iou.tolist()]
    change_miou = 0.5 * (road_iou + bldg_iou)
    macro_iou = float(iou[union > 0].mean().item()) if (union > 0).any() else 0.0

    # Binary any-change (classes 1 or 2 vs 0)
    bin_tn = float(cm_cpu[0, 0].item())
    bin_fp = float(cm_cpu[0, 1:].sum().item())
    bin_fn = float(cm_cpu[1:, 0].sum().item())
    bin_tp = float(cm_cpu[1:, 1:].sum().item())
    bin_prec = bin_tp / max(bin_tp + bin_fp, 1.0)
    bin_rec = bin_tp / max(bin_tp + bin_fn, 1.0)
    bin_f1 = (2.0 * bin_tp) / max(2.0 * bin_tp + bin_fp + bin_fn, 1.0)
    bin_iou = bin_tp / max(bin_tp + bin_fp + bin_fn, 1.0)

    small_f1 = (2.0 * small_tp) / max(2.0 * small_tp + small_fp + small_fn, 1.0)

    return {
        "change_miou": change_miou,
        "macro_iou": macro_iou,
        "bg_iou": bg_iou,
        "road_iou": road_iou,
        "building_iou": bldg_iou,
        "binary_precision": bin_prec,
        "binary_recall": bin_rec,
        "binary_f1": bin_f1,
        "binary_iou": bin_iou,
        "small_change_f1": small_f1,
        "unchanged_pairs": unchanged_total,
        "unchanged_fa_counts": {str(k): int(v) for k, v in unchanged_fa.items()},
        "unchanged_fa_rates": {
            str(k): (float(v) / unchanged_total if unchanged_total > 0 else 0.0)
            for k, v in unchanged_fa.items()
        },
        "confusion_matrix": cm.cpu().tolist(),
    }


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: ChangeLoss,
    device: str,
    amp_dtype: Optional[torch.dtype],
    save_pair_stats: bool = False,
) -> Tuple[float, Dict[str, object], List[dict]]:
    model.eval()
    cm = torch.zeros((3, 3), dtype=torch.int64, device=device)
    total_loss = 0.0
    n_batches = 0
    fa_thresholds = [1, 16, 50, 100, 200]
    unchanged_total = 0
    unchanged_fa = {t: 0 for t in fa_thresholds}
    small_tp = small_fp = small_fn = 0
    pair_rows: List[dict] = []

    for x1, x2, y, pair_ids in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            logits = model(x1, x2)
            loss = criterion(logits, y)
        total_loss += float(loss.item())
        n_batches += 1
        pred = logits.argmax(dim=1)
        cm += torch.bincount((y * 3 + pred).reshape(-1), minlength=9).reshape(3, 3)

        # Per-pair diagnostics
        gt_chg = (y > 0).reshape(y.shape[0], -1)
        pr_chg = (pred > 0).reshape(pred.shape[0], -1)
        gt_px = gt_chg.sum(dim=1).cpu().tolist()
        pr_px = pr_chg.sum(dim=1).cpu().tolist()
        tp_vec = (gt_chg & pr_chg).sum(dim=1).cpu().tolist()
        fp_vec = ((~gt_chg) & pr_chg).sum(dim=1).cpu().tolist()
        fn_vec = (gt_chg & (~pr_chg)).sum(dim=1).cpu().tolist()

        for i, pid in enumerate(pair_ids):
            g_cnt, p_cnt = int(gt_px[i]), int(pr_px[i])
            if g_cnt == 0:
                unchanged_total += 1
                for t in fa_thresholds:
                    if p_cnt >= t:
                        unchanged_fa[t] += 1
            elif g_cnt < 655:  # < 1% of 256x256
                small_tp += int(tp_vec[i])
                small_fp += int(fp_vec[i])
                small_fn += int(fn_vec[i])
            if save_pair_stats:
                pair_rows.append({
                    "pair_id": pid,
                    "gt_change_px": g_cnt,
                    "pred_change_px": p_cnt,
                    "tp_px": int(tp_vec[i]),
                    "fp_px": int(fp_vec[i]),
                    "fn_px": int(fn_vec[i]),
                    "gt_road_px": int((y[i] == 1).sum().item()),
                    "gt_bldg_px": int((y[i] == 2).sum().item()),
                    "pred_road_px": int((pred[i] == 1).sum().item()),
                    "pred_bldg_px": int((pred[i] == 2).sum().item()),
                })

    metrics = compute_metrics_from_stats(cm, unchanged_total, unchanged_fa, small_tp, small_fp, small_fn)
    return total_loss / max(n_batches, 1), metrics, pair_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arch", default="bit_resnet18", choices=["bit_resnet18", "ef_unet", "fc_siam_diff", "siam_resnet18_unet", "changeformer"])
    ap.add_argument("--bit-source-dir", default="/media/shoaib/STUDYLINUX/IIT/DegreeTerm3/DSAI-Project/models/external/BIT_CD")
    ap.add_argument("--changeformer-source-dir", default="/media/shoaib/STUDYLINUX/IIT/DegreeTerm3/DSAI-Project/models/external/ChangeFormer")
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--extracted-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "constant", "step"])
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--loss", default="focal_dice", choices=["focal_dice", "ce_dice", "weighted_ce"])
    ap.add_argument("--class-weights", default="0.10,0.45,0.45")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--aug", default="flip", choices=["none", "flip", "flip_photometric", "flip_photo_scale"])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-subset-diagnostic", type=int, default=0, help="ONLY for unit/resume tests; 0 enforces full 6444/1242.")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    manifests_dir = Path(args.manifests_dir)
    extracted_root = Path(args.extracted_root)
    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(manifests_dir / "mci_train.jsonl")
    val_rows = read_jsonl(manifests_dir / "mci_val.jsonl")
    test_rows = read_jsonl(manifests_dir / "mci_test.jsonl")

    # Full-data governance assertions
    if args.allow_subset_diagnostic > 0:
        train_rows = train_rows[: args.allow_subset_diagnostic]
        val_rows = val_rows[: max(8, args.allow_subset_diagnostic // 2)]
    else:
        assert len(train_rows) == 6444, f"Expected 6444 strict MCI train pairs, got {len(train_rows)}"
        assert len(val_rows) == 1242, f"Expected 1242 strict MCI val pairs, got {len(val_rows)}"
        assert len(test_rows) == 1858, f"Expected 1858 strict MCI test pairs, got {len(test_rows)}"
        t_ids = {r["pair_id"] for r in train_rows}
        v_ids = {r["pair_id"] for r in val_rows}
        te_ids = {r["pair_id"] for r in test_rows}
        assert not (t_ids & v_ids) and not (t_ids & te_ids) and not (v_ids & te_ids)

    c_weights = tuple(float(x) for x in args.class_weights.split(","))
    assert len(c_weights) == 3

    amp_dtype: Optional[torch.dtype] = None
    use_scaler = False
    if args.device.startswith("cuda"):
        if args.amp == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif args.amp == "fp16":
            amp_dtype = torch.float16
            use_scaler = True

    model, bb_params, head_params, bb_module = build_model(
        args.arch,
        args.device,
        bit_source_dir=args.bit_source_dir,
        changeformer_source_dir=args.changeformer_source_dir,
    )
    criterion = ChangeLoss(args.loss, c_weights, args.focal_gamma, args.label_smoothing).to(args.device)

    # Eval-only branch (for held-out test set or checkpoint verification)
    if args.eval_only:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        eval_rows = test_rows if args.eval_split == "test" else val_rows
        eval_ds = ExtractedMCIDataset(eval_rows, extracted_root, aug="none", base_seed=args.seed)
        eval_loader = DataLoader(eval_ds, batch_size=max(args.batch_size, 16), shuffle=False, num_workers=args.num_workers, pin_memory=True)
        loss, metrics, pair_rows = evaluate_loader(model, eval_loader, criterion, args.device, amp_dtype, save_pair_stats=True)
        report = {
            "run_id": args.run_id,
            "arch": args.arch,
            "checkpoint": str(ckpt_path.resolve()),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "split": args.eval_split,
            "num_pairs": len(eval_rows),
            "loss": loss,
            "metrics": metrics,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / f"eval_{args.eval_split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (out_dir / f"eval_{args.eval_split}_pairs.jsonl").open("w", encoding="utf-8") as f:
            for pr in pair_rows:
                f.write(json.dumps(pr) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return

    param_groups = []
    if bb_params:
        param_groups.append({"params": bb_params, "lr": args.backbone_lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr, "name": "head"})

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)

    def lr_lambda(ep: int) -> float:
        if args.warmup_epochs > 0 and ep < args.warmup_epochs:
            return float(ep + 1) / float(args.warmup_epochs)
        if args.scheduler == "constant":
            return 1.0
        if args.scheduler == "step":
            f = 1.0
            if ep >= 25:
                f *= 0.1
            if ep >= 40:
                f *= 0.1
            return f
        # cosine
        eff_ep = max(0, ep - args.warmup_epochs)
        eff_tot = max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(eff_ep, eff_tot) / eff_tot))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    train_ds = ExtractedMCIDataset(train_rows, extracted_root, aug=args.aug, base_seed=args.seed)
    val_ds = ExtractedMCIDataset(val_rows, extracted_root, aug="none", base_seed=args.seed)
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 16),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    start_epoch = 0
    best_score = -1.0
    best_epoch = -1
    stale = 0
    history_path = out_dir / "history.jsonl"
    last_ckpt = out_dir / "last.pt"
    best_ckpt = out_dir / "best.pt"

    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if use_scaler and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt["best_score"])
        best_epoch = int(ckpt["best_epoch"])
        stale = int(ckpt["stale"])
        random.setstate(ckpt["py_rng"])
        np.random.set_state(ckpt["np_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        print(f"[RESUME] Resumed {args.run_id} from epoch {start_epoch} (best={best_score:.4f} @ ep {best_epoch})", flush=True)
    else:
        if history_path.exists():
            history_path.unlink()

    total_params = sum(p.numel() for p in model.parameters())
    bb_param_count = sum(p.numel() for p in bb_params)
    head_param_count = sum(p.numel() for p in head_params)

    run_meta = {
        "run_id": args.run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config": vars(args),
        "effective_batch_size": args.batch_size * args.accum_steps,
        "amp_actual": str(amp_dtype) if amp_dtype is not None else "fp32",
        "primary_selection_metric": "val_change_miou",
        "data_scope": {
            "dataset": "levir_mci_strict",
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "train_ids_sha256": sha256_text("\n".join(sorted(r["pair_id"] for r in train_rows))),
            "val_ids_sha256": sha256_text("\n".join(sorted(r["pair_id"] for r in val_rows))),
            "is_full_data": args.allow_subset_diagnostic == 0,
        },
        "model_params": {
            "total": total_params,
            "backbone": bb_param_count,
            "head": head_param_count,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hostname": platform.node(),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    t_run_start = time.time()
    stopped_early = False

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        freeze = (bb_module is not None) and (epoch < args.freeze_epochs)
        if bb_module is not None:
            for p in bb_params:
                p.requires_grad = not freeze

        model.train()
        if freeze and bb_module is not None:
            bb_module.eval()

        train_ds.set_epoch(epoch)
        gen = torch.Generator()
        gen.manual_seed(args.seed + epoch * 1009)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=gen,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        steps = len(train_loader)
        loss_sum = 0.0
        opt_updates = 0
        seen_samples = 0

        for step, (x1, x2, y, _) in enumerate(train_loader):
            x1 = x1.to(args.device, non_blocking=True)
            x2 = x2.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            seen_samples += x1.shape[0]

            with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
                logits = model(x1, x2)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss.item()}")

            loss_sum += float(loss.item())
            # Exact accumulation tail handling
            divisor = min(args.accum_steps, steps - (step // args.accum_steps) * args.accum_steps)
            scaled_loss = loss / divisor
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == steps:
                if args.grad_clip > 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip
                    )
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_updates += 1

        assert seen_samples == len(train_rows), f"Expected {len(train_rows)} train samples, saw {seen_samples}"

        val_loss, val_metrics, _ = evaluate_loader(model, val_loader, criterion, args.device, amp_dtype)
        lrs = {g.get("name", f"g{i}"): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)}
        scheduler.step()

        epoch_sec = time.time() - t0
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        )
        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)

        score = float(val_metrics["change_miou"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(steps, 1),
            "val_loss": val_loss,
            "val_change_miou": score,
            "val_macro_iou": val_metrics["macro_iou"],
            "val_bg_iou": val_metrics["bg_iou"],
            "val_road_iou": val_metrics["road_iou"],
            "val_building_iou": val_metrics["building_iou"],
            "val_binary_precision": val_metrics["binary_precision"],
            "val_binary_recall": val_metrics["binary_recall"],
            "val_binary_f1": val_metrics["binary_f1"],
            "val_binary_iou": val_metrics["binary_iou"],
            "val_small_change_f1": val_metrics["small_change_f1"],
            "val_unchanged_fa_rates": val_metrics["unchanged_fa_rates"],
            "val_unchanged_fa_counts": val_metrics["unchanged_fa_counts"],
            "lrs": lrs,
            "backbone_frozen": freeze,
            "trainable_params": trainable_now,
            "samples_consumed": seen_samples,
            "optimizer_updates": opt_updates,
            "epoch_sec": round(epoch_sec, 2),
            "samples_per_sec": round(seen_samples / max(epoch_sec, 1e-6), 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "best_so_far": best_score,
            "improved": improved,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[{args.run_id}] ep={epoch:02d}/{args.epochs} "
            f"tr_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"chg_mIoU={score*100:.2f}% (road={val_metrics['road_iou']*100:.2f}% bldg={val_metrics['building_iou']*100:.2f}%) "
            f"binF1={val_metrics['binary_f1']*100:.2f}% FA@1px={val_metrics['unchanged_fa_rates']['1']*100:.1f}% "
            f"FA@50px={val_metrics['unchanged_fa_rates']['50']*100:.1f}% "
            f"t={epoch_sec:.1f}s vram={peak_vram_mb:.0f}MB {'*BEST*' if improved else ''}",
            flush=True,
        )

        ckpt_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "config": vars(args),
            "val_metrics": val_metrics,
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(ckpt_payload, last_ckpt)
        if improved:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_change_miou": score,
                    "val_metrics": val_metrics,
                    "config": vars(args),
                },
                best_ckpt,
            )

        if (epoch + 1) >= args.min_epochs and stale >= args.patience:
            stopped_early = True
            print(f"[{args.run_id}] Early stopping at epoch {epoch} (stale={stale}, best={best_score:.4f} @ ep {best_epoch})", flush=True)
            break

    # Reload best checkpoint and verify exact metrics + save per-pair val stats
    best_state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"], strict=True)
    reload_loss, reload_metrics, val_pair_rows = evaluate_loader(
        model, val_loader, criterion, args.device, amp_dtype, save_pair_stats=True
    )
    with (out_dir / "val_best_pairs.jsonl").open("w", encoding="utf-8") as f:
        for pr in val_pair_rows:
            f.write(json.dumps(pr) + "\n")

    final_receipt = dict(run_meta)
    final_receipt.update({
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_wall_sec": round(time.time() - t_run_start, 2),
        "epochs_completed": epoch + 1 if args.epochs > start_epoch else start_epoch,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_change_miou": best_score,
        "reloaded_best_val_loss": reload_loss,
        "reloaded_best_val_metrics": reload_metrics,
        "best_checkpoint": str(best_ckpt.resolve()),
        "best_checkpoint_sha256": sha256_file(best_ckpt),
        "last_checkpoint": str(last_ckpt.resolve()),
        "last_checkpoint_sha256": sha256_file(last_ckpt),
        "history_sha256": sha256_file(history_path),
    })
    (out_dir / "run.json").write_text(json.dumps(final_receipt, indent=2), encoding="utf-8")
    print(f"[{args.run_id}] DONE. Best val_change_miou={best_score*100:.2f}% @ ep {best_epoch}, sha256={final_receipt['best_checkpoint_sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
