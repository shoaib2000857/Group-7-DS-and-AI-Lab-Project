"""Bounded, local BIT-R18 learning pilot over LEVIR-MCI ZIP members.

This deliberately short pilot is a pipeline/fit diagnostic, not convergence or
final model-selection evidence. All outputs (including weights) stay local in
the ignored runs/ directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "POC/pipeline/cache/BIT_CD"))
from POC.pipeline.workers import build_detector
from benchmark_pixel_control import PREFIX, ids_for, sample_ids, sha256_file


class ZipPairs:
    def __init__(self, zf, split, ids, augment=False, seed=42):
        self.zf, self.split, self.ids = zf, split, ids
        self.augment, self.seed = augment, seed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pair_id = self.ids[i]
        base = f"{PREFIX}/{self.split}"
        images = []
        for date in ("A", "B"):
            with Image.open(io.BytesIO(self.zf.read(f"{base}/{date}/{pair_id}.png"))) as im:
                images.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        with Image.open(io.BytesIO(self.zf.read(f"{base}/label/{pair_id}.png"))) as im:
            raw = np.asarray(im.convert("RGB"), dtype=np.uint8)[..., 0]
        target = np.zeros(raw.shape, dtype=np.uint8)
        target[raw == 128] = 1
        target[raw == 255] = 2
        rng = np.random.default_rng(self.seed + i)
        if self.augment and rng.random() < 0.5:
            images = [np.flip(x, axis=1).copy() for x in images]
            target = np.flip(target, axis=1).copy()
        if self.augment and rng.random() < 0.5:
            images = [np.flip(x, axis=0).copy() for x in images]
            target = np.flip(target, axis=0).copy()
        x = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).float() / 127.5 - 1
        y = torch.from_numpy(target.astype(np.int64))
        return pair_id, x, y


def batches(ds, batch_size, order=None):
    order = list(range(len(ds))) if order is None else list(order)
    for start in range(0, len(order), batch_size):
        records = [ds[i] for i in order[start:start + batch_size]]
        yield ([r[0] for r in records], torch.stack([r[1] for r in records]),
               torch.stack([r[2] for r in records]))


def loss_fn(logits, target, weights):
    ce = torch.nn.functional.cross_entropy(logits, target, weight=weights, reduction="none")
    pt = logits.softmax(1).gather(1, target[:, None]).squeeze(1)
    focal = ((1 - pt) ** 2 * ce).mean()
    probs = logits.softmax(1)
    truth = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = 1 - ((2 * (probs * truth).sum(dims) + 1) /
                (probs.sum(dims) + truth.sum(dims) + 1)).mean()
    return 0.5 * focal + 0.5 * dice


def confusion_for(model, ds, ids, batch_size, device):
    cm_total = torch.zeros((3, 3), dtype=torch.int64, device=device)
    records, elapsed = [], []
    model.eval()
    with torch.inference_mode():
        for pair_ids, x, y in batches(ds, batch_size):
            x, y = x.to(device), y.to(device)
            start = time.perf_counter()
            pred = model(x[:, 0], x[:, 1]).argmax(1)
            torch.cuda.synchronize()
            latency = (time.perf_counter() - start) * 1000 / len(pair_ids)
            cm = torch.bincount((y * 3 + pred).flatten(), minlength=9).reshape(3, 3)
            cm_total += cm
            for j, pair_id in enumerate(pair_ids):
                yj, pj = y[j], pred[j]
                row = {"pair_id": pair_id, "latency_ms": latency}
                for c, name in enumerate(("background", "road", "building")):
                    row[name + "_tp"] = int(((yj == c) & (pj == c)).sum().item())
                    row[name + "_truth"] = int((yj == c).sum().item())
                    row[name + "_pred"] = int((pj == c).sum().item())
                row["truth_changed"] = bool((yj != 0).any().item())
                row["pred_changed"] = bool((pj != 0).any().item())
                records.append(row)
                elapsed.append(latency)
    cm_np = cm_total.cpu().numpy()
    tp = np.diag(cm_np).astype(float)
    union = cm_np.sum(0) + cm_np.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    f1 = np.divide(2 * tp, cm_np.sum(0) + cm_np.sum(1),
                   out=np.zeros_like(tp), where=(cm_np.sum(0) + cm_np.sum(1)) > 0)
    changed = [r for r in records if r["truth_changed"]]
    unchanged = [r for r in records if not r["truth_changed"]]
    binary_tp = int(cm_np[1:, 1:].sum())
    binary_fp = int(cm_np[0, 1:].sum())
    binary_fn = int(cm_np[1:, 0].sum())
    bp = binary_tp / (binary_tp + binary_fp) if binary_tp + binary_fp else 0
    br = binary_tp / (binary_tp + binary_fn) if binary_tp + binary_fn else 0
    return {
        "pairs": len(records), "changed_pairs": len(changed), "unchanged_pairs": len(unchanged),
        "unchanged_pair_false_alarm_rate": (sum(r["pred_changed"] for r in unchanged) / len(unchanged) if unchanged else None),
        "confusion_matrix_rows_true_cols_pred": cm_np.tolist(),
        "class_iou": {name: float(iou[i]) for i, name in enumerate(("background", "road", "building"))},
        "class_f1": {name: float(f1[i]) for i, name in enumerate(("background", "road", "building"))},
        "macro_iou_present_classes": float(iou[union > 0].mean()) if (union > 0).any() else 0.0,
        "binary_any_change": {"precision": bp, "recall": br,
            "f1": 2 * bp * br / (bp + br) if bp + br else 0.0,
            "iou": binary_tp / (binary_tp + binary_fp + binary_fn) if binary_tp + binary_fp + binary_fn else 0.0},
        "median_pair_inference_ms": float(np.median(elapsed)),
        "mean_pair_inference_ms": float(np.mean(elapsed)), "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ROOT / "data/raw/LEVIR-MCI-dataset.zip")
    ap.add_argument("--train-pairs", type=int, default=100)
    ap.add_argument("--validation-pairs", type=int, default=25)
    ap.add_argument("--test-pairs", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not (ROOT / "POC/pipeline/cache/BIT_CD/models/networks.py").is_file():
        raise FileNotFoundError("run POC/pipeline/setup_sources.py first")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) if args.device == "cuda" else None
    torch.set_num_threads(4)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or ROOT / "POC/milestone3/sarthak/runs" / f"bit-pilot-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    with zipfile.ZipFile(args.archive) as zf:
        train_ids = sample_ids(zf, "train", args.train_pairs, args.seed)
        val_ids = sample_ids(zf, "val", args.validation_pairs, args.seed + 1)
        test_ids = sample_ids(zf, "test", args.test_pairs, 20261110)
        train_ds = ZipPairs(zf, "train", train_ids, augment=True, seed=args.seed)
        val_ds = ZipPairs(zf, "val", val_ids)
        test_ds = ZipPairs(zf, "test", test_ids)
        model = build_detector({"source_dir": str(ROOT / "POC/pipeline/cache/BIT_CD")}, str(device))
        model.resnet.eval()
        for param in model.resnet.parameters():
            param.requires_grad = False
        optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4, weight_decay=0.01)
        weights = torch.tensor([0.10, 0.45, 0.45], device=device)
        history, best, stale = [], -1.0, 0
        for epoch in range(args.epochs):
            model.train(); model.resnet.eval()
            order = np.random.default_rng(args.seed + epoch).permutation(len(train_ds))
            losses = []
            for _, x, y in batches(train_ds, args.batch_size, order):
                x, y = x.to(device), y.to(device)
                optim.zero_grad(set_to_none=True)
                logits = model(x[:, 0], x[:, 1])
                loss = loss_fn(logits, y, weights)
                loss.backward(); optim.step(); losses.append(float(loss.detach().cpu()))
            val = confusion_for(model, val_ds, val_ids, args.batch_size, device)
            row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   "validation_macro_iou_present_classes": val["macro_iou_present_classes"],
                   "validation_class_iou": val["class_iou"], "optimizer_steps": len(losses)}
            history.append(row)
            if row["validation_macro_iou_present_classes"] > best:
                best = row["validation_macro_iou_present_classes"]
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "validation_macro_iou": best, "seed": args.seed}, out / "best.pt")
                stale = 0
            else:
                stale += 1
            print(json.dumps(row), flush=True)
        model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
        test = confusion_for(model, test_ds, test_ids, args.batch_size, device)
        test_ids_for_control = sample_ids(zf, "test", args.test_pairs, 20261110)
        if test_ids != test_ids_for_control:
            raise AssertionError("pilot test IDs differ from the pixel-control test population")
        record = {
            "run": "Sarthak Saini / M3 BIT-R18 short learning pilot",
            "run_started_utc": stamp, "seed": args.seed, "epochs": args.epochs,
            "train_pairs": len(train_ids), "validation_pairs": len(val_ids), "test_pairs": len(test_ids),
            "optimizer_steps": sum(x["optimizer_steps"] for x in history), "batch_size": args.batch_size,
            "image_size": 256, "device": torch.cuda.get_device_name(0) if args.device == "cuda" else str(device),
            "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
            "bit_source_commit": "adcd7aea6f234586ffffdd4e9959404f96271711",
            "backbone_initialization": "official ImageNet ResNet-18 weights; frozen for this short pilot",
            "output_classes": ["background", "road", "building"],
            "loss": "0.5 class-weighted focal(gamma=2, weights=[0.10,0.45,0.45]) + 0.5 soft Dice",
            "optimizer": "AdamW(lr=1e-4, weight_decay=0.01); backbone frozen",
            "source_archive_sha256": sha256_file(args.archive),
            "train_ids": train_ids, "validation_ids": val_ids, "test_ids": test_ids,
            "history": history, "test": {k: v for k, v in test.items() if k != "records"},
            "limitations": [
                "Two epochs over 100 fitting pairs are only an implementation/fit smoke pilot, not a converged model.",
                "Test pairs are held out from this pilot; this sample is too small for a final leaderboard claim.",
                "The pretrained BIT detector selection remains architectural; this pilot does not compare all nine candidates.",
            ],
        }
    (out / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "test": record["test"]}, indent=2))


if __name__ == "__main__":
    main()
