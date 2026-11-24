"""Full-data, resumable, auditable RSCaMa captioner trainer for TerraWatch M4 recovery.

Guarantees:
- Uses ALL 6,444 strict LEVIR-MCI training pairs (32,220 training captions) and ALL 1,242 validation pairs.
- Builds vocabulary strictly from training captions (zero val/test lexical leakage).
- Logs exact trainable vs frozen parameter counts per component (CLIP encoder, AttentiveEncoder, Decoder).
- Records train CE loss, validation CE loss, corpus BLEU-4 (primary selection metric),
  BLEU-1..3, ROUGE-L, unchanged-pair false-alarm rate, changed-pair BLEU-4, LR, time, peak VRAM.
- Saves best.pth, resumable last.pth, vocab.json, val_captions.jsonl, history.jsonl, run.json with SHA-256.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


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


# --- Compatibility patches for RSCaMa Mamba & CLIP ---
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ssm_fwd_kernel(
        u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, db_ptr,
        out_ptr, last_ptr,
        D_sz, L: tl.constexpr, N: tl.constexpr,
        HAS_D: tl.constexpr, HAS_Z: tl.constexpr, HAS_DB: tl.constexpr, SOFTPLUS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D_sz
        d = pid % D_sz
        ns = tl.arange(0, N)
        a_val = tl.load(A_ptr + d * N + ns).to(tl.float32)
        db_val = tl.load(db_ptr + d).to(tl.float32) if HAS_DB else 0.0
        d_val = tl.load(D_ptr + d).to(tl.float32) if HAS_D else 0.0
        h = tl.zeros([N], dtype=tl.float32)
        base_bd = (b * D_sz + d) * L
        base_b = b * N * L
        for t in range(L):
            u_t = tl.load(u_ptr + base_bd + t).to(tl.float32)
            dt_t = tl.load(dt_ptr + base_bd + t).to(tl.float32) + db_val
            if SOFTPLUS:
                dt_t = tl.where(dt_t > 20.0, dt_t, tl.log(1.0 + tl.exp(dt_t)))
            b_t = tl.load(B_ptr + base_b + ns * L + t).to(tl.float32)
            c_t = tl.load(C_ptr + base_b + ns * L + t).to(tl.float32)
            da = tl.exp(dt_t * a_val)
            h = da * h + (dt_t * u_t) * b_t
            y_t = tl.sum(h * c_t, axis=0)
            if HAS_D:
                y_t = y_t + u_t * d_val
            if HAS_Z:
                z_t = tl.load(z_ptr + base_bd + t).to(tl.float32)
                y_t = y_t * (z_t * tl.sigmoid(z_t))
            tl.store(out_ptr + base_bd + t, y_t)
        tl.store(last_ptr + (b * D_sz + d) * N + ns, h)

    @triton.jit
    def _ssm_replay_h_kernel(
        u_ptr, dt_ptr, A_ptr, B_ptr, db_ptr, h_all_ptr,
        D_sz, L: tl.constexpr, N: tl.constexpr,
        HAS_DB: tl.constexpr, SOFTPLUS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D_sz
        d = pid % D_sz
        ns = tl.arange(0, N)
        a_val = tl.load(A_ptr + d * N + ns).to(tl.float32)
        db_val = tl.load(db_ptr + d).to(tl.float32) if HAS_DB else 0.0
        h = tl.zeros([N], dtype=tl.float32)
        base_bd = (b * D_sz + d) * L
        base_b = b * N * L
        base_h = (b * D_sz + d) * L * N
        for t in range(L):
            u_t = tl.load(u_ptr + base_bd + t).to(tl.float32)
            dt_t = tl.load(dt_ptr + base_bd + t).to(tl.float32) + db_val
            if SOFTPLUS:
                dt_t = tl.where(dt_t > 20.0, dt_t, tl.log(1.0 + tl.exp(dt_t)))
            b_t = tl.load(B_ptr + base_b + ns * L + t).to(tl.float32)
            da = tl.exp(dt_t * a_val)
            h = da * h + (dt_t * u_t) * b_t
            tl.store(h_all_ptr + base_h + t * N + ns, h)

    @triton.jit
    def _ssm_bwd_kernel(
        u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, db_ptr,
        h_all_ptr, dout_ptr, dlast_ptr,
        du_ptr, ddt_ptr, dA_ptr, dB_ptr, dC_ptr, dD_ptr, dz_ptr, ddb_ptr,
        D_sz, L: tl.constexpr, N: tl.constexpr,
        HAS_D: tl.constexpr, HAS_Z: tl.constexpr, HAS_DB: tl.constexpr, SOFTPLUS: tl.constexpr,
        HAS_DLAST: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D_sz
        d = pid % D_sz
        ns = tl.arange(0, N)
        a_val = tl.load(A_ptr + d * N + ns).to(tl.float32)
        db_val = tl.load(db_ptr + d).to(tl.float32) if HAS_DB else 0.0
        d_val = tl.load(D_ptr + d).to(tl.float32) if HAS_D else 0.0
        base_bd = (b * D_sz + d) * L
        base_b = b * N * L
        base_h = (b * D_sz + d) * L * N

        dh = tl.load(dlast_ptr + (b * D_sz + d) * N + ns).to(tl.float32) if HAS_DLAST else tl.zeros([N], dtype=tl.float32)
        dA_acc = tl.zeros([N], dtype=tl.float32)
        dD_acc = 0.0
        ddb_acc = 0.0

        for rev in range(L):
            t = L - 1 - rev
            u_t = tl.load(u_ptr + base_bd + t).to(tl.float32)
            dt_raw = tl.load(dt_ptr + base_bd + t).to(tl.float32) + db_val
            if SOFTPLUS:
                dt_t = tl.where(dt_raw > 20.0, dt_raw, tl.log(1.0 + tl.exp(dt_raw)))
                dt_grad_scale = tl.sigmoid(dt_raw)
            else:
                dt_t = dt_raw
                dt_grad_scale = 1.0
            b_t = tl.load(B_ptr + base_b + ns * L + t).to(tl.float32)
            c_t = tl.load(C_ptr + base_b + ns * L + t).to(tl.float32)
            h_t = tl.load(h_all_ptr + base_h + t * N + ns).to(tl.float32)
            h_prev = tl.load(h_all_ptr + base_h + (t - 1) * N + ns).to(tl.float32) if t > 0 else tl.zeros([N], dtype=tl.float32)

            dout_t = tl.load(dout_ptr + base_bd + t).to(tl.float32)
            y_base = tl.sum(h_t * c_t, axis=0)
            if HAS_D:
                y_base = y_base + u_t * d_val
            if HAS_Z:
                z_t = tl.load(z_ptr + base_bd + t).to(tl.float32)
                sig_z = tl.sigmoid(z_t)
                silu_z = z_t * sig_z
                dsilu_z = sig_z * (1.0 + z_t * (1.0 - sig_z))
                dz_t = dout_t * y_base * dsilu_z
                tl.store(dz_ptr + base_bd + t, dz_t)
                dy_t = dout_t * silu_z
            else:
                dy_t = dout_t

            if HAS_D:
                dD_acc += dy_t * u_t
                du_t = dy_t * d_val
            else:
                du_t = 0.0

            dC_t = dy_t * h_t
            tl.atomic_add(dC_ptr + base_b + ns * L + t, dC_t)
            dh = dh + dy_t * c_t

            da = tl.exp(dt_t * a_val)
            dh_da = dh * h_prev * da
            dA_acc = dA_acc + dh_da * dt_t
            ddt_t = tl.sum(dh_da * a_val + dh * u_t * b_t, axis=0)
            ddt_raw = ddt_t * dt_grad_scale
            du_t = du_t + tl.sum(dh * dt_t * b_t, axis=0)
            dB_t = dh * (dt_t * u_t)
            tl.atomic_add(dB_ptr + base_b + ns * L + t, dB_t)

            tl.store(du_ptr + base_bd + t, du_t)
            tl.store(ddt_ptr + base_bd + t, ddt_raw)
            if HAS_DB:
                ddb_acc += ddt_raw
            dh = dh * da

        tl.atomic_add(dA_ptr + d * N + ns, dA_acc)
        if HAS_D:
            tl.atomic_add(dD_ptr + d, dD_acc)
        if HAS_DB:
            tl.atomic_add(ddb_ptr + d, ddb_acc)

    class _TritonSelectiveScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, dt, A, B, C, D, z, db, delta_softplus: bool):
            u_c, dt_c = u.contiguous(), dt.contiguous()
            A_c, B_c, C_c = A.contiguous(), B.contiguous(), C.contiguous()
            D_c = D.contiguous() if D is not None else u_c
            z_c = z.contiguous() if z is not None else u_c
            db_c = db.contiguous() if db is not None else u_c
            B_sz, D_sz, L = u_c.shape
            N = A_c.shape[1]
            out = torch.empty((B_sz, D_sz, L), device=u.device, dtype=torch.float32)
            last = torch.empty((B_sz, D_sz, N), device=u.device, dtype=torch.float32)
            _ssm_fwd_kernel[(B_sz * D_sz,)](
                u_c, dt_c, A_c, B_c, C_c, D_c, z_c, db_c, out, last,
                D_sz, L, N, D is not None, z is not None, db is not None, delta_softplus, num_warps=1,
            )
            ctx.save_for_backward(
                u_c, dt_c, A_c, B_c, C_c,
                D if D is not None else None,
                z if z is not None else None,
                db if db is not None else None,
            )
            ctx.delta_softplus = delta_softplus
            ctx.has_D = D is not None
            ctx.has_z = z is not None
            ctx.has_db = db is not None
            return out.to(u.dtype), last

        @staticmethod
        def backward(ctx, dout, dlast):
            u_c, dt_c, A_c, B_c, C_c, D_s, z_s, db_s = ctx.saved_tensors
            B_sz, D_sz, L = u_c.shape
            N = A_c.shape[1]
            D_c = D_s.contiguous() if ctx.has_D else u_c
            z_c = z_s.contiguous() if ctx.has_z else u_c
            db_c = db_s.contiguous() if ctx.has_db else u_c
            dout_c = dout.contiguous()
            has_dlast = dlast is not None and bool(dlast.abs().max().item() > 0)
            dlast_c = dlast.contiguous() if has_dlast else u_c

            h_all = torch.empty((B_sz, D_sz, L, N), device=u_c.device, dtype=torch.float32)
            _ssm_replay_h_kernel[(B_sz * D_sz,)](
                u_c, dt_c, A_c, B_c, db_c, h_all, D_sz, L, N, ctx.has_db, ctx.delta_softplus, num_warps=1,
            )
            du = torch.empty((B_sz, D_sz, L), device=u_c.device, dtype=torch.float32)
            ddt = torch.empty((B_sz, D_sz, L), device=u_c.device, dtype=torch.float32)
            dA = torch.zeros((D_sz, N), device=u_c.device, dtype=torch.float32)
            dB = torch.zeros((B_sz, N, L), device=u_c.device, dtype=torch.float32)
            dC = torch.zeros((B_sz, N, L), device=u_c.device, dtype=torch.float32)
            dD = torch.zeros((D_sz,), device=u_c.device, dtype=torch.float32) if ctx.has_D else u_c
            dz = torch.empty((B_sz, D_sz, L), device=u_c.device, dtype=torch.float32) if ctx.has_z else u_c
            ddb = torch.zeros((D_sz,), device=u_c.device, dtype=torch.float32) if ctx.has_db else u_c

            _ssm_bwd_kernel[(B_sz * D_sz,)](
                u_c, dt_c, A_c, B_c, C_c, D_c, z_c, db_c, h_all, dout_c, dlast_c,
                du, ddt, dA, dB, dC, dD, dz, ddb,
                D_sz, L, N, ctx.has_D, ctx.has_z, ctx.has_db, ctx.delta_softplus, has_dlast, num_warps=1,
            )
            del h_all
            return (
                du.to(u_c.dtype), ddt.to(dt_c.dtype), dA.to(A_c.dtype), dB.to(B_c.dtype), dC.to(C_c.dtype),
                dD.to(D_s.dtype) if ctx.has_D else None,
                dz.to(z_s.dtype) if ctx.has_z else None,
                ddb.to(db_s.dtype) if ctx.has_db else None,
                None,
            )

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _parallel_scan_core(u, delta, A, B, C, D, z, delta_bias, delta_softplus: bool):
    u_f, delta_f = u.float(), delta.float()
    if delta_bias is not None:
        delta_f = delta_f + delta_bias[..., None].float()
    if delta_softplus:
        delta_f = F.softplus(delta_f)
    B_f, C_f, A_f = B.float(), C.float(), A.float()
    L = u_f.shape[2]
    # Exact Hillis-Steele parallel associative scan: ceil(log2(L)) vectorized ops instead of L sequential steps
    a = torch.exp(delta_f.unsqueeze(-1) * A_f.view(1, A_f.shape[0], 1, A_f.shape[1]))
    b = (delta_f * u_f).unsqueeze(-1) * B_f.permute(0, 2, 1).unsqueeze(1)
    step = 1
    while step < L:
        a_hi = a[:, :, step:]
        b = torch.cat([b[:, :, :step], a_hi * b[:, :, :-step] + b[:, :, step:]], dim=2)
        if step * 2 < L:
            a = torch.cat([a[:, :, :step], a_hi * a[:, :, :-step]], dim=2)
        step *= 2
    y = (b * C_f.permute(0, 2, 1).unsqueeze(1)).sum(dim=-1)
    out = y if D is None else y + u_f * D.float()[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(u.dtype), b[:, :, -1]


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    if (
        _HAS_TRITON
        and u.is_cuda
        and u.ndim == 3
        and B.ndim == 3
        and C.ndim == 3
        and A.shape[1] > 0
        and (A.shape[1] & (A.shape[1] - 1)) == 0
    ):
        out, last_x = _TritonSelectiveScan.apply(u, delta, A, B, C, D, z, delta_bias, bool(delta_softplus))
    elif torch.is_grad_enabled() and u.requires_grad:
        out, last_x = torch.utils.checkpoint.checkpoint(
            _parallel_scan_core, u, delta, A, B, C, D, z, delta_bias, delta_softplus, use_reentrant=False
        )
    else:
        out, last_x = _parallel_scan_core(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    return (out, last_x) if return_last_state else out


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    L = x.shape[-1]
    y = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[1] - 1, groups=x.shape[1])[..., :L]
    return F.silu(y) if activation in ("silu", "swish") else y


def install_rscama_patches(device: str) -> str:
    import logging
    import transformers.models.mamba.modeling_mamba as mm
    kernel_mode = "triton_fused_exact" if (_HAS_TRITON and str(device).startswith("cuda")) else "reference_python_shim"
    mm.selective_scan_fn = selective_scan_ref
    mm.causal_conv1d_fn = causal_conv1d_ref
    if not hasattr(mm, "is_fast_path_available"):
        mm.is_fast_path_available = False
    if not hasattr(mm, "logger"):
        mm.logger = logging.getLogger("transformers.models.mamba.modeling_mamba")
    for name in ("causal_conv1d_update", "selective_state_update", "mamba_inner_fn"):
        if not hasattr(mm, name) or getattr(mm, name) is None:
            setattr(mm, name, None)

    import clip.model as cm

    def vit_forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        featuremap = x[:, 1:, :]
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x, featuremap

    cm.VisionTransformer.forward = vit_forward
    torch.Tensor.cuda = lambda t, *a, **k: t.to(device)
    torch.nn.Module.cuda = lambda m, *a, **k: m.to(device)
    return kernel_mode


def upstream_tokenizer(source_dir: Path):
    source = (source_dir / "preprocess_data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in {"tokenize", "build_vocab", "encode"})
        or (isinstance(node, ast.Assign) and any(getattr(x, "id", "") == "SPECIAL_TOKENS" for x in node.targets))
    ]
    ns: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "upstream_tokenizer", "exec"), ns)

    def tokenize(caption: str):
        return ns["tokenize"](
            caption, add_start_token=True, add_end_token=True,
            punct_to_keep=[";", ","], punct_to_remove=["?", "."]
        )
    return tokenize, ns["build_vocab"], ns["encode"]


def build_rscama_components(source_dir: Path, vocabulary: dict, device: str):
    source = source_dir.resolve()
    if not (source / "model/model_encoder_attMamba.py").is_file():
        raise FileNotFoundError(f"Missing RSCaMa source at {source}")
    clip_dir = source.parent / "CLIP"
    if clip_dir.is_dir() and str(clip_dir) not in sys.path:
        sys.path.insert(0, str(clip_dir))
    kernel_mode = install_rscama_patches(device)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    prev = Path.cwd()
    try:
        os.chdir(source)
        import clip
        from model.mamba_block import MambaMixer
        from model.model_encoder_attMamba import Encoder, AttentiveEncoder
        from model.model_decoder import DecoderTransformer

        MambaMixer.forward = lambda self, h, h2, cache_params=None, cache_params_2=None: self.cuda_kernels_forward(h, h2)
        _, processor = clip.load("ViT-B/32", device="cpu", jit=False)
        encoder = Encoder("CLIP-ViT-B/32")
        fusion = AttentiveEncoder(n_layers=3, feature_size=[7, 7, 768], heads=8, dropout=0.1)
        decoder = DecoderTransformer(
            decoder_type="transformer_decoder",
            embed_dim=768,
            vocab_size=len(vocabulary),
            max_lengths=42,
            word_vocab=vocabulary,
            n_head=8,
            n_layers=1,
            dropout=0.1,
        )
    finally:
        os.chdir(prev)
    return (encoder, fusion, decoder), processor, kernel_mode


# --- Fast pure-Python caption metrics (BLEU-1..4, ROUGE-L, no-change FA) ---
def _lcs_len(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def compute_rouge_l(references_list: List[List[List[str]]], hypotheses: List[List[str]]) -> float:
    scores = []
    beta2 = 1.2 ** 2
    for refs, hyp in zip(references_list, hypotheses):
        best = 0.0
        for ref in refs:
            if not hyp or not ref:
                continue
            lcs = _lcs_len(ref, hyp)
            prec = lcs / len(hyp)
            rec = lcs / len(ref)
            if prec + rec > 0:
                f = ((1 + beta2) * prec * rec) / (rec + beta2 * prec)
                if f > best:
                    best = f
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def compute_corpus_bleu(references_list: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
    from nltk.translate.bleu_score import corpus_bleu
    b1 = float(corpus_bleu(references_list, hypotheses, weights=(1, 0, 0, 0)))
    b2 = float(corpus_bleu(references_list, hypotheses, weights=(0.5, 0.5, 0, 0)))
    b3 = float(corpus_bleu(references_list, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0)))
    b4 = float(corpus_bleu(references_list, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)))
    return {"bleu1": b1, "bleu2": b2, "bleu3": b3, "bleu4": b4}


NO_CHANGE_RE = re.compile(
    r"\b(?:no difference|no change|no obvious|nothing has changed|almost nothing|same as before|seem identical|unchanged|identical|remains? the same)\b",
    re.I,
)


class CaptionPairDataset(Dataset):
    def __init__(self, rows: List[dict], extracted_root: Path, refs: dict, processor):
        self.rows = rows
        self.root = extracted_root
        self.refs = refs
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        with Image.open(self.root / r["t1"]) as im1:
            t1 = self.processor(im1.convert("RGB"))
        with Image.open(self.root / r["t2"]) as im2:
            t2 = self.processor(im2.convert("RGB"))
        return t1, t2, r["pair_id"]


def encode_batch_captions(pair_ids: List[str], refs: dict, tokenize, encode, vocab: dict, device: str):
    encoded = []
    repeats = []
    for pid in pair_ids:
        caps = refs[pid]["captions"]
        repeats.append(len(caps))
        for cap in caps:
            toks = encode(tokenize(cap), vocab, allow_unk=True)[:42]
            toks[-1] = vocab["<END>"]
            encoded.append((toks + [vocab["<NULL>"]] * (42 - len(toks)), len(toks)))
    tokens = torch.tensor([v for v, _ in encoded], dtype=torch.long, device=device)
    lengths = torch.tensor([l for _, l in encoded], dtype=torch.long, device=device)
    repeats_t = torch.tensor(repeats, dtype=torch.long, device=device)
    return tokens, lengths, repeats_t


def inspect_trainability(encoder, fusion, decoder) -> dict:
    def stats(m):
        tot = sum(p.numel() for p in m.parameters())
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        tr_names = [n for n, p in m.named_parameters() if p.requires_grad]
        return {"total": tot, "trainable": tr, "frozen": tot - tr, "trainable_tensors": len(tr_names), "sample_trainable": tr_names[:8]}
    return {
        "encoder": stats(encoder),
        "fusion": stats(fusion),
        "decoder": stats(decoder),
    }


@torch.inference_mode()
def evaluate_captioner(
    models,
    val_loader: DataLoader,
    refs: dict,
    tokenize,
    encode,
    vocab: dict,
    inverse: dict,
    skip_ids: set,
    device: str,
    label_smoothing: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
):
    encoder, fusion, decoder = models
    for m in models:
        m.eval()

    val_ce_sum = 0.0
    val_tok_count = 0
    hypotheses = []
    references = []
    changed_hyps, changed_refs = [], []
    unchanged_total = 0
    unchanged_fa = 0
    changed_total = 0
    changed_missed = 0
    caption_records = []

    for t1, t2, pair_ids in val_loader:
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        tokens, lengths, repeats_t = encode_batch_captions(pair_ids, refs, tokenize, encode, vocab, device)

        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            f1, f2 = encoder(t1, t2)
            fused = fusion(f1, f2)
            fused_rep = fused.repeat_interleave(repeats_t, dim=0)
            scores, sorted_tokens, decode_lengths, _ = decoder(fused_rep, tokens, lengths)
            pred_packed = pack_padded_sequence(scores.float(), decode_lengths, batch_first=True).data
            tgt_packed = pack_padded_sequence(sorted_tokens[:, 1:], decode_lengths, batch_first=True).data
            ce_sum = F.cross_entropy(pred_packed, tgt_packed.long(), reduction="sum", label_smoothing=label_smoothing)
        val_ce_sum += float(ce_sum.item())
        val_tok_count += int(tgt_packed.numel())

        # Greedy decoding per pair (matches upstream decoder.sample(..., k=1))
        fused_eval = fused.float()
        for i, pid in enumerate(pair_ids):
            single_feat = fused_eval[i : i + 1]
            toks = decoder.sample(single_feat, k=1)
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            while toks and isinstance(toks[0], list):
                toks = toks[0]
            hyp = [inverse[int(tk)] for tk in toks if int(tk) not in skip_ids]
            ref_toks = [tokenize(c)[1:-1] for c in refs[pid]["captions"]]
            hyp_text = " ".join(hyp)
            hypotheses.append(hyp)
            references.append(ref_toks)

            chg_flag = int(refs[pid].get("source_changeflag", 1))
            says_nc = bool(NO_CHANGE_RE.search(hyp_text))
            if chg_flag == 0:
                unchanged_total += 1
                if not says_nc:
                    unchanged_fa += 1
            else:
                changed_total += 1
                changed_hyps.append(hyp)
                changed_refs.append(ref_toks)
                if says_nc:
                    changed_missed += 1

            caption_records.append({
                "pair_id": pid,
                "caption": hyp_text,
                "source_changeflag": chg_flag,
                "says_no_change": says_nc,
            })

    bleus = compute_corpus_bleu(references, hypotheses)
    rouge_l = compute_rouge_l(references, hypotheses)
    chg_bleu4 = compute_corpus_bleu(changed_refs, changed_hyps)["bleu4"] if changed_hyps else 0.0

    metrics = {
        "val_ce_loss": val_ce_sum / max(val_tok_count, 1),
        "val_bleu1": bleus["bleu1"],
        "val_bleu2": bleus["bleu2"],
        "val_bleu3": bleus["bleu3"],
        "val_bleu4": bleus["bleu4"],
        "val_rouge_l": rouge_l,
        "val_changed_bleu4": chg_bleu4,
        "val_unchanged_pairs": unchanged_total,
        "val_unchanged_fa_count": unchanged_fa,
        "val_unchanged_fa_rate": unchanged_fa / max(unchanged_total, 1),
        "val_changed_pairs": changed_total,
        "val_changed_missed_count": changed_missed,
        "val_changed_missed_rate": changed_missed / max(changed_total, 1),
    }
    return metrics, caption_records


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--extracted-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-encoder", action="store_true", help="Freeze entire CLIP encoder instead of upstream fine_tune(True)")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--save-last", action="store_true")
    ap.add_argument("--vocab-threshold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    manifests_dir = Path(args.manifests_dir)
    extracted_root = Path(args.extracted_root)
    source_dir = Path(args.source_dir)
    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(manifests_dir / "mci_train.jsonl")
    val_rows = read_jsonl(manifests_dir / "mci_val.jsonl")
    test_rows = read_jsonl(manifests_dir / "mci_test.jsonl")
    refs = json.loads((manifests_dir / "mci_references.json").read_text(encoding="utf-8"))

    assert len(train_rows) == 6444, len(train_rows)
    assert len(val_rows) == 1242, len(val_rows)
    assert len(test_rows) == 1858, len(test_rows)
    assert not ({r["pair_id"] for r in train_rows} & {r["pair_id"] for r in val_rows})
    assert not ({r["pair_id"] for r in train_rows} & {r["pair_id"] for r in test_rows})

    tokenize, build_vocab, encode = upstream_tokenizer(source_dir)

    if args.eval_only:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vocab = ckpt.get("vocab")
        if vocab is None:
            vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
        models, processor, kernel_mode = build_rscama_components(source_dir, vocab, args.device)
        for m, k in zip(models, ["encoder_dict", "encoder_trans_dict", "decoder_dict"]):
            m.load_state_dict(ckpt[k], strict=True)
            m.to(args.device).eval()
        inverse = {int(v): w for w, v in vocab.items()}
        skip_ids = {vocab[w] for w in ("<START>", "<END>", "<NULL>")}
        eval_rows = test_rows if args.eval_split == "test" else val_rows
        eval_ds = CaptionPairDataset(eval_rows, extracted_root, refs, processor)
        eval_loader = DataLoader(eval_ds, batch_size=max(args.batch_size, 16), shuffle=False, num_workers=args.num_workers, pin_memory=True)
        metrics, cap_records = evaluate_captioner(models, eval_loader, refs, tokenize, encode, vocab, inverse, skip_ids, args.device)
        report = {
            "run_id": args.run_id,
            "checkpoint": str(ckpt_path.resolve()),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "split": args.eval_split,
            "num_pairs": len(eval_rows),
            "vocab_size": len(vocab),
            "kernel_mode": kernel_mode,
            "metrics": metrics,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / f"eval_{args.eval_split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (out_dir / f"eval_{args.eval_split}_captions.jsonl").open("w", encoding="utf-8") as f:
            for rec in cap_records:
                f.write(json.dumps(rec) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return

    # Training-only vocabulary from all 6,444 strict training pairs (32,220 captions)
    train_seqs = [
        (r["pair_id"], [tokenize(c) for c in refs[r["pair_id"]]["captions"]])
        for r in train_rows
    ]
    total_train_caps = sum(len(caps) for _, caps in train_seqs)
    assert total_train_caps == 32220, total_train_caps
    vocab = build_vocab(train_seqs, args.vocab_threshold)
    assert set(vocab.values()) == set(range(len(vocab)))
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    models, processor, kernel_mode = build_rscama_components(source_dir, vocab, args.device)
    encoder, fusion, decoder = models
    if args.freeze_encoder:
        encoder.fine_tune(False)
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        encoder.fine_tune(True)
    # Freeze unused Mamba token-embedding tables inside AttentiveEncoder (CaMalayer only passes inputs_embeds)
    for name, p in fusion.named_parameters():
        if "embeddings" in name:
            p.requires_grad = False
    for m in models:
        m.to(args.device)

    trainability = inspect_trainability(encoder, fusion, decoder)
    trainable_params = [p for m in models for p in m.parameters() if p.requires_grad]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    amp_dtype: Optional[torch.dtype] = None
    use_scaler = False
    if args.device.startswith("cuda"):
        if args.amp == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif args.amp == "fp16":
            amp_dtype = torch.float16
            use_scaler = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1.0)

    inverse = {int(v): w for w, v in vocab.items()}
    skip_ids = {vocab[w] for w in ("<START>", "<END>", "<NULL>")}

    train_ds = CaptionPairDataset(train_rows, extracted_root, refs, processor)
    val_ds = CaptionPairDataset(val_rows, extracted_root, refs, processor)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch_size, 16), shuffle=False, num_workers=args.num_workers, pin_memory=True)

    start_epoch = 0
    best_score = -1.0
    best_epoch = -1
    stale = 0
    history_path = out_dir / "history.jsonl"
    last_ckpt = out_dir / "last.pth"
    best_ckpt = out_dir / "best.pth"

    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder_dict"], strict=True)
        fusion.load_state_dict(ckpt["encoder_trans_dict"], strict=True)
        decoder.load_state_dict(ckpt["decoder_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt["best_score"])
        best_epoch = int(ckpt["best_epoch"])
        stale = int(ckpt["stale"])
        print(f"[RESUME] Resumed {args.run_id} from epoch {start_epoch} (best BLEU-4={best_score:.4f} @ ep {best_epoch})", flush=True)
    else:
        if history_path.exists():
            history_path.unlink()

    run_meta = {
        "run_id": args.run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config": vars(args),
        "effective_batch_size": args.batch_size * args.accum_steps,
        "kernel_mode": kernel_mode,
        "primary_selection_metric": "val_bleu4",
        "vocab_size": len(vocab),
        "vocab_scope": "strict_levir_mci_train_only",
        "data_scope": {
            "train_pairs": len(train_rows),
            "train_captions": total_train_caps,
            "val_pairs": len(val_rows),
            "train_ids_sha256": sha256_text("\n".join(sorted(r["pair_id"] for r in train_rows))),
            "val_ids_sha256": sha256_text("\n".join(sorted(r["pair_id"] for r in val_rows))),
        },
        "trainability": trainability,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hostname": platform.node(),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[{args.run_id}] Vocab={len(vocab)} | Trainable params: enc={trainability['encoder']['trainable']:,} fus={trainability['fusion']['trainable']:,} dec={trainability['decoder']['trainable']:,}", flush=True)

    t_run_start = time.time()
    stopped_early = False

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        for m in models:
            m.train()
        if args.freeze_encoder:
            encoder.eval()

        gen = torch.Generator()
        gen.manual_seed(args.seed + epoch * 1009)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=gen,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        steps = len(train_loader)
        tok_loss_sum = 0.0
        tok_total = 0
        seen_pairs = 0

        for step, (t1, t2, pair_ids) in enumerate(train_loader):
            t1 = t1.to(args.device, non_blocking=True)
            t2 = t2.to(args.device, non_blocking=True)
            seen_pairs += t1.shape[0]
            tokens, lengths, repeats_t = encode_batch_captions(pair_ids, refs, tokenize, encode, vocab, args.device)

            with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
                f1, f2 = encoder(t1, t2)
                fused = fusion(f1, f2).repeat_interleave(repeats_t, dim=0)
                scores, sorted_tokens, decode_lengths, _ = decoder(fused, tokens, lengths)
                pred_packed = pack_padded_sequence(scores.float(), decode_lengths, batch_first=True).data
                tgt_packed = pack_padded_sequence(sorted_tokens[:, 1:], decode_lengths, batch_first=True).data
                loss = F.cross_entropy(pred_packed, tgt_packed.long(), label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                print(f"[{args.run_id}] DIVERGED: Non-finite loss at epoch {epoch} step {step} — saving divergence receipt cleanly.", flush=True)
                final_receipt = dict(run_meta)
                final_receipt.update({
                    "status": "diverged_nonfinite",
                    "diverged_epoch": epoch,
                    "diverged_step": step,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "total_wall_sec": round(time.time() - t_run_start, 2),
                    "epochs_completed": epoch,
                    "stopped_early": True,
                    "best_epoch": best_epoch,
                    "best_val_bleu4": best_score,
                    "best_checkpoint": str(best_ckpt.resolve()) if best_ckpt.exists() else None,
                    "best_checkpoint_sha256": sha256_file(best_ckpt) if best_ckpt.exists() else None,
                    "history_sha256": sha256_file(history_path) if history_path.exists() else None,
                })
                (out_dir / "run.json").write_text(json.dumps(final_receipt, indent=2), encoding="utf-8")
                return

            n_tok = int(tgt_packed.numel())
            tok_loss_sum += float(loss.item()) * n_tok
            tok_total += n_tok

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
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        assert seen_pairs == len(train_rows), seen_pairs

        val_metrics, val_caps = evaluate_captioner(
            models, val_loader, refs, tokenize, encode, vocab, inverse, skip_ids, args.device, args.label_smoothing, amp_dtype=amp_dtype
        )
        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        epoch_sec = time.time() - t0
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        )

        score = float(val_metrics["val_bleu4"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        row = {
            "epoch": epoch,
            "train_ce_loss": tok_loss_sum / max(tok_total, 1),
            **val_metrics,
            "lr": lr_now,
            "pairs_consumed": seen_pairs,
            "epoch_sec": round(epoch_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "best_so_far": best_score,
            "improved": improved,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[{args.run_id}] ep={epoch:02d}/{args.epochs} "
            f"tr_ce={row['train_ce_loss']:.4f} val_ce={val_metrics['val_ce_loss']:.4f} "
            f"BLEU4={score:.4f} ROUGEL={val_metrics['val_rouge_l']:.4f} "
            f"chgBLEU4={val_metrics['val_changed_bleu4']:.4f} "
            f"ncFA={val_metrics['val_unchanged_fa_count']}/{val_metrics['val_unchanged_pairs']} "
            f"t={epoch_sec:.1f}s vram={peak_vram_mb:.0f}MB {'*BEST*' if improved else ''}",
            flush=True,
        )

        def _half_sd(m):
            return {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k, v in m.state_dict().items()}

        ckpt_common = {
            "encoder_dict": _half_sd(encoder),
            "encoder_trans_dict": _half_sd(fusion),
            "decoder_dict": _half_sd(decoder),
            "vocab": vocab,
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "val_metrics": val_metrics,
            "config": vars(args),
        }
        if args.save_last:
            torch.save({**ckpt_common, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, last_ckpt)
        if improved:
            torch.save(ckpt_common, best_ckpt)
            with (out_dir / "val_best_captions.jsonl").open("w", encoding="utf-8") as f:
                for rec in val_caps:
                    f.write(json.dumps(rec) + "\n")

        if (epoch + 1) >= args.min_epochs and stale >= args.patience:
            stopped_early = True
            print(f"[{args.run_id}] Early stopping at epoch {epoch} (best BLEU-4={best_score:.4f} @ ep {best_epoch})", flush=True)
            break

    final_receipt = dict(run_meta)
    final_receipt.update({
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_wall_sec": round(time.time() - t_run_start, 2),
        "epochs_completed": epoch + 1 if args.epochs > start_epoch else start_epoch,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_bleu4": best_score,
        "best_checkpoint": str(best_ckpt.resolve()),
        "best_checkpoint_sha256": sha256_file(best_ckpt),
        "last_checkpoint": str(last_ckpt.resolve()) if last_ckpt.exists() else None,
        "last_checkpoint_sha256": sha256_file(last_ckpt) if last_ckpt.exists() else None,
        "history_sha256": sha256_file(history_path),
    })
    (out_dir / "run.json").write_text(json.dumps(final_receipt, indent=2), encoding="utf-8")
    print(f"[{args.run_id}] DONE. Best val_bleu4={best_score:.4f} @ ep {best_epoch}, sha256={final_receipt['best_checkpoint_sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
