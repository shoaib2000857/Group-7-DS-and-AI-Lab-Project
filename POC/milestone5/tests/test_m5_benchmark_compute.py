import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/m5_benchmark_compute.py"
spec = importlib.util.spec_from_file_location("m5_benchmark_compute", SCRIPT)
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def test_percentiles_match_numpy():
    a = np.random.default_rng(0).uniform(1, 100, 1000)
    s = bm.percentile_summary(a)
    assert s["n"] == 1000
    assert s["mean_ms"] == pytest.approx(a.mean())
    for key, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99)):
        assert s[key] == pytest.approx(np.percentile(a, q))
    assert s["min_ms"] == a.min() and s["max_ms"] == a.max()


def test_percentiles_hand_computed():
    s = bm.percentile_summary([10, 20, 30, 40, 50])
    assert s["mean_ms"] == 30
    assert s["p50_ms"] == 30
    assert s["p95_ms"] == pytest.approx(48.0)
    assert s["p99_ms"] == pytest.approx(49.6)
    assert (s["min_ms"], s["max_ms"]) == (10, 50)


def test_cost_per_1000_pairs_hand_computed():
    lat = {"detector": 50.0, "captioner": 450.0, "witness": 10.0, "referee": 800.0}
    cost = bm.cost_per_1000_pairs(lat, hourly_usd=0.36, claims_per_pair=2.0, vlm_calls_per_claim=1.5)
    seconds = 0.51 + 2.0 * 1.5 * 0.8
    assert cost == pytest.approx(1000 * seconds * 0.36 / 3600)
    assert cost == pytest.approx(0.291)


def test_cost_without_vlm_calls_drops_referee():
    lat = {"detector": 50.0, "captioner": 450.0, "witness": 10.0, "referee": 800.0}
    cost = bm.cost_per_1000_pairs(lat, 0.36, 2.0, 0.0)
    assert cost == pytest.approx(1000 * 0.51 * 0.36 / 3600)
    assert cost == pytest.approx(bm.cost_per_1000_pairs({k: v for k, v in lat.items() if k != "referee"}, 0.36, 2.0, 5.0))


def test_flop_counter_conv_matches_formula():
    conv = nn.Conv2d(3, 8, 3, bias=False)
    x = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        measured = bm.flops_of(lambda: conv(x))
    hout = wout = 32 - 3 + 1
    assert measured == 2 * 8 * 3 * 3 * 3 * hout * wout


def test_bit_param_count():
    src = bm.ROOT / "models/external/BIT_CD/models/networks.py"
    if not src.is_file():
        pytest.skip("models/external/BIT_CD missing")
    from src.models.detector_architectures import build_model
    model, *_ = build_model("bit_resnet18", "cpu", bit_source_dir=str(src.parents[1]))
    assert sum(p.numel() for p in model.parameters()) == 11944043


def test_referee_skipped_off_cuda():
    assert bm.referee_stage(type("A", (), {"device": "cpu"})(), [], {}) == {"skipped": "requires cuda"}


def test_override_parsing():
    ov = bm.parse_overrides("captioner:warmup=3,iters=10", {"warmup": 50, "iters": 200})
    assert ov["captioner"] == {"warmup": 3, "iters": 10}
    assert ov["detectors"] == {"warmup": 50, "iters": 200}
    assert math.isfinite(bm.percentile_summary([1.0])["p99_ms"])
