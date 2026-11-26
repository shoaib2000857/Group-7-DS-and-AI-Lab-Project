"""Device-agnostic compute profiler for the TerraWatch inference stages (detectors, captioner, witness, referee)."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

M4_EVIDENCE = ROOT / "docs/milestone4/evidence/full_data_m4"
PAIRS_DIR = ROOT / "POC/execution/data/pairs"
DETECTORS = {
    "A01_ef_unet": "ef_unet",
    "A02_fc_siam_diff": "fc_siam_diff",
    "A03_siam_r18_unet": "siam_resnet18_unet",
    "A04_changeformer": "changeformer",
    "D03_bit_lr3e4": "bit_resnet18",
}
CAPTIONER_ID = "C06_rscama_frozen_clip"
REFEREE_ID = "Qwen3-VL-4B-Instruct"
ALL_STAGES = ("detectors", "captioner", "witness", "referee")


def percentile_summary(values_ms) -> dict:
    a = np.asarray(values_ms, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def cost_per_1000_pairs(stage_latency_ms: dict, hourly_usd: float, claims_per_pair: float, vlm_calls_per_claim: float) -> float:
    base = sum(v for k, v in stage_latency_ms.items() if k != "referee") / 1000.0
    referee = claims_per_pair * vlm_calls_per_claim * stage_latency_ms.get("referee", 0.0) / 1000.0
    return 1000.0 * (base + referee) * hourly_usd / 3600.0


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class MemTracker:
    def __init__(self, device: str):
        self.device, self.peak = device, 0

    def start(self):
        import gc
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        self.peak = 0

    def poll(self):
        if self.device == "mps":
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())

    def result(self) -> dict:
        if self.device.startswith("cuda"):
            return {"peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "peak_mem_kind": "cuda_max_memory_allocated"}
        if self.device == "mps":
            return {"peak_mem_mb": self.peak / 2**20, "peak_mem_kind": "mps_driver_allocated_memory_max_polled_per_iter"}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == "darwin" else rss * 1024
        return {"peak_mem_mb": rss / 2**20, "peak_mem_kind": "process_rss_peak_cumulative"}


def time_fn(fn, device, warmup, iters, mem: MemTracker) -> list:
    for _ in range(warmup):
        fn()
    sync(device)
    mem.start()
    out = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.append((time.perf_counter() - t0) * 1000.0)
        mem.poll()
    return out


def flops_of(fn) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())


def count_params(*modules) -> dict:
    mods = [p for m in modules for p in m.parameters()]
    return {"params_total": sum(p.numel() for p in mods), "params_trainable": sum(p.numel() for p in mods if p.requires_grad)}


def find_checkpoint(ckpt_dir, run_id, exts):
    if not ckpt_dir:
        return None
    d = Path(ckpt_dir).expanduser()
    for ext in exts:
        for cand in (d / f"{run_id}_best{ext}", d / run_id / f"best{ext}"):
            if cand.is_file():
                return cand
    return None


def checkpoint_info(path) -> dict:
    if path is None:
        return {"weights": "random_init", "checkpoint_file": None, "checkpoint_size_mb": None}
    return {"weights": "checkpoint", "checkpoint_file": str(path), "checkpoint_size_mb": path.stat().st_size / 2**20}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def module_version(name):
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


def env_block(device: str) -> dict:
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0)
    elif device == "mps":
        name = "Apple MPS (unified memory)"
    else:
        name = "cpu"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": module_version("torchvision"),
        "transformers": module_version("transformers"),
        "triton": module_version("triton"),
        "cuda_version": torch.version.cuda,
        "device": device,
        "device_name": name,
        "os": platform.platform(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }


def load_pairs(n_max: int) -> list:
    import cv2
    out = []
    for split in ("val", "train"):
        for a in sorted((PAIRS_DIR / split / "A").glob("*.png")):
            b, lab = PAIRS_DIR / split / "B" / a.name, PAIRS_DIR / split / "label" / a.name
            if b.is_file() and lab.is_file():
                out.append({"id": f"{split}/{a.stem}", "t1": a, "t2": b, "label": lab})
            if len(out) >= n_max:
                break
        if len(out) >= n_max:
            break
    if not out:
        raise FileNotFoundError(f"no pairs under {PAIRS_DIR}")
    for p in out:
        p["t1_rgb"] = cv2.cvtColor(cv2.imread(str(p["t1"])), cv2.COLOR_BGR2RGB)
        p["t2_rgb"] = cv2.cvtColor(cv2.imread(str(p["t2"])), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(str(p["label"]), cv2.IMREAD_UNCHANGED)
        lab = lab[..., 0] if lab.ndim == 3 else lab
        p["mask"] = np.where(lab == 128, 1, np.where(lab == 255, 2, 0)).astype(np.uint8)
    return out


def to_tensor(rgb, device):
    return (torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 127.5 - 1).to(device)


def bench(fn, device, warmup, iters, mem) -> dict:
    lat = time_fn(fn, device, warmup, iters, mem)
    return {**percentile_summary(lat), "warmup": warmup, "iters": iters, **mem.result()}


def detector_stage(run_id, arch, args, pairs, ov) -> dict:
    from src.models.detector_architectures import build_model
    device = args.device
    run = read_json(M4_EVIDENCE / "detector_runs" / run_id / "run.json")
    expected = run["model_params"]["total"]
    model, *_ = build_model(arch, "cpu", bit_source_dir=str(ROOT / "models/external/BIT_CD"),
                            changeformer_source_dir=str(ROOT / "models/external/ChangeFormer"))
    ck = find_checkpoint(args.ckpt_dir, run_id, (".pt", ".pth"))
    if ck:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=True)
    model = model.to(device).eval()
    rec = {"arch": arch, **count_params(model), "params_expected_run_json": expected,
           "params_source": f"docs/milestone4/evidence/full_data_m4/detector_runs/{run_id}/run.json:model_params.total",
           **checkpoint_info(ck), "input": "2x[1,3,256,256] RGB scaled to [-1,1]"}
    rec["params_match_run_json"] = rec["params_total"] == expected
    tens = [(to_tensor(p["t1_rgb"], device), to_tensor(p["t2_rgb"], device)) for p in pairs]
    idx = {"i": 0}

    def one():
        a, b = tens[idx["i"] % len(tens)]
        idx["i"] += 1
        model(a, b)

    n = args.batch_size
    bat = [(torch.cat([tens[(j + k) % len(tens)][0] for k in range(n)]), torch.cat([tens[(j + k) % len(tens)][1] for k in range(n)])) for j in range(min(4, len(tens)))]
    bidx = {"i": 0}

    def batched():
        a, b = bat[bidx["i"] % len(bat)]
        bidx["i"] += 1
        model(a, b)

    with torch.inference_mode():
        rec["gflops_per_pair"] = flops_of(lambda: model(*tens[0])) / 1e9
        mem = MemTracker(device)
        rec["batch1"] = bench(one, device, ov["warmup"], ov["iters"], mem)
        rec["batch1"]["throughput_pairs_per_s"] = 1000.0 / rec["batch1"]["mean_ms"]
        rec[f"batch{n}"] = bench(batched, device, ov["batch_warmup"], ov["batch_iters"], MemTracker(device))
        rec[f"batch{n}"]["throughput_pairs_per_s"] = n * 1000.0 / rec[f"batch{n}"]["mean_ms"]
    rec["distinct_pairs_used"] = min(len(pairs), ov["warmup"] + ov["iters"])
    rec["latency_ms"] = rec["batch1"]["mean_ms"]
    rec["peak_mem_mb"], rec["peak_mem_kind"] = rec["batch1"]["peak_mem_mb"], rec["batch1"]["peak_mem_kind"]
    del model
    return rec


@contextlib.contextmanager
def cuda_alias(device):
    if device.startswith("cuda"):
        yield
        return
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self.to(device)
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def set_scan_impl(impl):
    import transformers.models.mamba.modeling_mamba as mm
    mm.selective_scan_fn = impl
    blk = sys.modules.get("model.mamba_block")
    if blk is not None:
        blk.selective_scan_fn = impl


def captioner_stage(args, pairs, ov) -> dict:
    from src.models.captioner import build_components
    device = args.device
    cfg = read_json(M4_EVIDENCE / f"captioner_runs/{CAPTIONER_ID}/run.json")
    train = cfg["trainability"]
    expected = {k: train[k]["total"] for k in ("encoder", "fusion", "decoder")}
    ck = find_checkpoint(args.ckpt_dir, CAPTIONER_ID, (".pth", ".pt"))
    state = torch.load(ck, map_location="cpu", weights_only=False) if ck else None
    vocab = state["vocab"] if state and "vocab" in state else {w: i for i, w in enumerate(["<NULL>", "<UNK>", "<START>", "<END>"] + [f"w{i}" for i in range(cfg["vocab_size"] - 4)])}
    (encoder, fusion, decoder), processor = build_components(str(ROOT / "models/external/RSCaMa"), vocab, "cpu")
    if state:
        for m, key in zip((encoder, fusion, decoder), ("encoder_dict", "encoder_trans_dict", "decoder_dict")):
            m.load_state_dict(state[key], strict=True)
    if cfg["config"]["freeze_encoder"]:
        for p in encoder.parameters():
            p.requires_grad_(False)
    parts = {"encoder": encoder, "fusion": fusion, "decoder": decoder}
    for m in parts.values():
        m.to(device).eval()
    rec = {"components": {k: {**count_params(m), "params_expected_run_json": expected[k]} for k, m in parts.items()},
           **count_params(*parts.values()), "params_expected_run_json": sum(expected.values()),
           "params_source": f"docs/milestone4/evidence/full_data_m4/captioner_runs/{CAPTIONER_ID}/run.json:trainability",
           "vocab_size": len(vocab), "vocab_source": "checkpoint" if state and "vocab" in state else "placeholder_tokens_sized_from_run.json", **checkpoint_info(ck),
           "input": "2x[1,3,224,224] CLIP-preprocessed", "freeze_encoder_from_config": cfg["config"]["freeze_encoder"]}
    rec["params_match_run_json"] = rec["params_total"] == rec["params_expected_run_json"] and all(
        rec["components"][k]["params_total"] == expected[k] for k in parts)
    from PIL import Image
    inputs = []
    for p in pairs[: ov["warmup"] + ov["iters"]]:
        with Image.open(p["t1"]) as a, Image.open(p["t2"]) as b:
            inputs.append((processor(a.convert("RGB")).unsqueeze(0).to(device), processor(b.convert("RGB")).unsqueeze(0).to(device)))
    idx = {"i": 0, "steps": []}

    def one():
        a, b = inputs[idx["i"] % len(inputs)]
        idx["i"] += 1
        first, second = encoder(a, b)
        seqs = decoder.sample(fusion(first, second), k=1)
        idx["steps"].append(len(seqs) - 1)

    def run_variant():
        idx["steps"] = []
        r = bench(one, device, ov["warmup"], ov["iters"], MemTracker(device))
        r["decode_steps_mean"] = float(np.mean(idx["steps"]))
        r["distinct_pairs_used"] = min(len(inputs), ov["warmup"] + ov["iters"])
        return r

    with cuda_alias(device), torch.inference_mode():
        rec["gflops_per_pair"] = flops_of(lambda: decoder.sample(fusion(*encoder(*inputs[0])), k=1)) / 1e9
        rec["flops_note"] = "FlopCounterMode counts matmul/conv/attention ops only; selective-scan elementwise ops are not counted"
        rec["ref_scan"] = run_variant()
        end_id = decoder.word_vocab["<END>"]
        decoder.word_vocab["<END>"] = -1
        rec["ref_scan_full_len"] = run_variant()
        decoder.word_vocab["<END>"] = end_id
        primary = "ref_scan" if ck else "ref_scan_full_len"
        rec["primary_variant"] = primary
        rec["primary_note"] = "trained checkpoint: natural decode length" if ck else "random init stops decoding at a different length than trained weights; full_len disables <END> so all max_lengths steps run (upper bound)"
        rec["latency_ms"] = rec[primary]["mean_ms"]
        rec["peak_mem_mb"], rec["peak_mem_kind"] = rec[primary]["peak_mem_mb"], rec[primary]["peak_mem_kind"]
        if device.startswith("cuda"):
            if not ck:
                decoder.word_vocab["<END>"] = -1
            rec["triton_scan"] = triton_variant(run_variant)
            decoder.word_vocab["<END>"] = end_id
    del encoder, fusion, decoder
    return rec


def triton_variant(run_variant) -> dict:
    try:
        spec = importlib.util.spec_from_file_location("train_captioner_full", ROOT / "scripts/train_captioner_full.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not getattr(mod, "_HAS_TRITON", False):
            return {"skipped": "triton not importable"}
        import transformers.models.mamba.modeling_mamba as mm
        previous = mm.selective_scan_fn
        set_scan_impl(mod.selective_scan_ref)
        try:
            return run_variant()
        finally:
            set_scan_impl(previous)
    except Exception as e:
        return {"skipped": f"{type(e).__name__}: {e}"}


def witness_stage(args, pairs, ov) -> dict:
    from src.evidence.visual_inspector import VisualEvidenceInspector
    from src.inference.verification_pipeline import candidate_box_and_components, compute_contextual_bbox, make_mask_overlay
    pipe = read_json(ROOT / "configs/pipeline.json")["witness"]
    inspector = VisualEvidenceInspector()
    objs = ["building", "road"]
    idx = {"i": 0}

    def one():
        p = pairs[idx["i"] % len(pairs)]
        claim = {"claim": {"object": objs[idx["i"] % 2], "location": None}}
        idx["i"] += 1
        box, localized, _ = candidate_box_and_components(p["mask"], claim)
        padded, _ = compute_contextual_bbox(box, p["mask"].shape, pipe["padding_fraction"], pipe["min_crop_size"])
        x0, y0, x1, y1 = padded
        make_mask_overlay(p["t2_rgb"], p["mask"], box)
        inspector.inspect_crop_pair(p["t1_rgb"][y0:y1, x0:x1], p["t2_rgb"][y0:y1, x0:x1], claim["claim"]["object"])

    rec = {"device": "cpu (OpenCV/numpy)", "unit": "one claim: candidate box + contextual crop + overlay + structural inspection",
           "masks": "ground-truth LEVIR-MCI label masks (0/1/2) from POC/execution/data/pairs/*/label",
           "witness_config": pipe, "weights": "n/a", "params_total": 0, "gflops_per_pair": None}
    mem = MemTracker("cpu")
    rec.update(bench(one, "cpu", ov["warmup"], ov["iters"], mem))
    rec["distinct_pairs_used"] = min(len(pairs), ov["warmup"] + ov["iters"])
    rec["latency_ms"] = rec["mean_ms"]
    return rec


def referee_stage(args, pairs, ov) -> dict:
    if not args.device.startswith("cuda"):
        return {"skipped": "requires cuda"}
    import cv2
    from src.inference.verification_pipeline import Qwen3VLReferee, candidate_box_and_components, compute_contextual_bbox
    ref_cfg = read_json(ROOT / "configs/pipeline.json")["referee"]
    device = args.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    referee = Qwen3VLReferee(repo=ref_cfg["repo"], revision=ref_cfg["revision"], device=device, max_image_side=ref_cfg["max_image_side"])
    sync(device)
    rec = {"repo": ref_cfg["repo"], "revision": ref_cfg["revision"], "weights": "hf_pretrained", "checkpoint_size_mb": None,
           "load_s": time.perf_counter() - t0, "load_peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
           "params_total": sum(p.numel() for p in referee.model.parameters()),
           "params_trainable": sum(p.numel() for p in referee.model.parameters() if p.requires_grad),
           "dtype": referee.metadata["dtype"], "gflops_per_pair": None}
    tmp = Path(tempfile.mkdtemp(prefix="m5_referee_"))
    cases = []
    for i, p in enumerate(pairs):
        claim = {"claim": {"object": "building", "location": None}}
        box, _, summary = candidate_box_and_components(p["mask"], claim)
        x0, y0, x1, y1 = compute_contextual_bbox(box, p["mask"].shape, 0.25, 32)[0]
        paths = {"t1": p["t1"], "t2": p["t2"]}
        for k, rgb in (("c1", p["t1_rgb"][y0:y1, x0:x1]), ("c2", p["t2_rgb"][y0:y1, x0:x1])):
            paths[k] = tmp / f"{i}_{k}.png"
            cv2.imwrite(str(paths[k]), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        state = {"claim": claim["claim"], "mask_summary": summary, "claim_text": "new buildings were constructed"}
        cases.append((state, paths))
    variants = {"full_pair_2img": ("t1", "t2"), "pair_plus_crops_4img": ("t1", "t2", "c1", "c2")}
    for name, keys in variants.items():
        idx, tokens = {"i": 0}, []

        def one():
            state, paths = cases[idx["i"] % len(cases)]
            idx["i"] += 1
            out = referee.decide(state, [str(paths[k]) for k in keys])
            tokens.append(out["raw"]["input_tokens"])

        r = bench(one, device, ov["warmup"], ov["iters"], MemTracker(device))
        r["input_tokens_mean"], r["input_tokens_min"], r["input_tokens_max"] = float(np.mean(tokens[-ov["iters"]:])), int(min(tokens)), int(max(tokens))
        r["distinct_pairs_used"] = min(len(cases), ov["warmup"] + ov["iters"])
        rec[name] = r
    rec["latency_ms"] = rec["pair_plus_crops_4img"]["mean_ms"]
    rec["peak_mem_mb"], rec["peak_mem_kind"] = rec["pair_plus_crops_4img"]["peak_mem_mb"], rec["pair_plus_crops_4img"]["peak_mem_kind"]
    return rec


def parse_overrides(spec: str, base: dict) -> dict:
    out = {s: dict(base) for s in ALL_STAGES}
    for item in filter(None, (spec or "").split(";")):
        stage, kv = item.split(":", 1)
        for pair in kv.split(","):
            k, v = pair.split("=")
            out[stage][k] = int(v)
    return out


def safe(label, fn):
    try:
        return fn()
    except Exception as e:
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=6), "stage": label}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hardware-label", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--stages", default="detectors,captioner,witness")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--batch-warmup", type=int, default=10)
    ap.add_argument("--batch-iters", type=int, default=50)
    ap.add_argument("--stage-overrides", default="", help="e.g. 'captioner:warmup=3,iters=20;referee:warmup=3,iters=20'")
    ap.add_argument("--num-pairs", type=int, default=250)
    ap.add_argument("--only-detectors", default=None)
    args = ap.parse_args(argv)
    stages = ALL_STAGES if args.stages == "all" else tuple(s.strip() for s in args.stages.split(","))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("mps unavailable")
    base = {"warmup": args.warmup, "iters": args.iters, "batch_warmup": args.batch_warmup, "batch_iters": args.batch_iters}
    ov = parse_overrides(args.stage_overrides, base)
    pairs = load_pairs(args.num_pairs)
    result = {"env": env_block(args.device),
              "settings": {"hardware_label": args.hardware_label, "stages": list(stages), "ckpt_dir": args.ckpt_dir, "batch_size": args.batch_size,
                           "per_stage": {s: ov[s] for s in stages}, "pairs_loaded": len(pairs), "pairs_source": str(PAIRS_DIR.relative_to(ROOT)),
                           "pair_ids": [p["id"] for p in pairs[:3]] + ["..."], "argv": sys.argv[1:]},
              "stages": {}}
    out_path = Path(args.out) / f"compute_profile_{args.hardware_label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if "detectors" in stages:
        wanted = set(args.only_detectors.split(",")) if args.only_detectors else set(DETECTORS)
        for run_id, arch in DETECTORS.items():
            if run_id in wanted:
                result["stages"][run_id] = {"stage": "detector", **safe(run_id, lambda: detector_stage(run_id, arch, args, pairs, ov["detectors"]))}
                dump()
    if "captioner" in stages:
        result["stages"][CAPTIONER_ID] = {"stage": "captioner", **safe(CAPTIONER_ID, lambda: captioner_stage(args, pairs, ov["captioner"]))}
        dump()
    if "witness" in stages:
        result["stages"]["witness_opencv"] = {"stage": "witness", **safe("witness", lambda: witness_stage(args, pairs, ov["witness"]))}
        dump()
    if "referee" in stages:
        result["stages"][REFEREE_ID] = {"stage": "referee", **safe("referee", lambda: referee_stage(args, pairs, ov["referee"]))}
        dump()
    dump()
    print(out_path)
    return result


if __name__ == "__main__":
    main()
