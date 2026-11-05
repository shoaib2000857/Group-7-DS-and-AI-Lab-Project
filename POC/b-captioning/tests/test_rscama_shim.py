"""The pure-torch kernel shim matches an independent per-element recurrence."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from caption_rscama import causal_conv1d_ref, selective_scan_ref  # noqa: E402


def naive_scan(u, delta, A, B, C, D, z, delta_bias):
    b, d, l = u.shape
    n = A.shape[1]
    delta = F.softplus(delta + delta_bias[:, None])
    y = torch.zeros(b, d, l)
    for bi in range(b):
        for di in range(d):
            h = torch.zeros(n)
            for t in range(l):
                h = torch.exp(delta[bi, di, t] * A[di]) * h + delta[bi, di, t] * B[bi, :, t] * u[bi, di, t]
                y[bi, di, t] = (h * C[bi, :, t]).sum() + D[di] * u[bi, di, t]
    return y * F.silu(z)


def test_selective_scan_matches_naive():
    g = torch.Generator().manual_seed(0)
    b, d, n, l = 2, 3, 4, 5
    u, delta, z = (torch.randn(b, d, l, generator=g) for _ in range(3))
    A = -torch.rand(d, n, generator=g)
    B, C = torch.randn(b, n, l, generator=g), torch.randn(b, n, l, generator=g)
    D, bias = torch.randn(d, generator=g), torch.randn(d, generator=g)
    out, last = selective_scan_ref(u, delta, A, B, C, D, z, bias, delta_softplus=True, return_last_state=True)
    assert torch.allclose(out, naive_scan(u, delta, A, B, C, D, z, bias), atol=1e-5)
    assert last.shape == (b, d, n)


def test_causal_conv_is_causal_and_depthwise():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 2, 6, generator=g)
    w = torch.randn(2, 3, generator=g)
    y = causal_conv1d_ref(x, w, None, activation=None)
    for c in range(2):
        for t in range(6):
            ref = sum(w[c, k] * x[0, c, t - (2 - k)] for k in range(3) if t - (2 - k) >= 0)
            assert torch.isclose(y[0, c, t], ref, atol=1e-5)
