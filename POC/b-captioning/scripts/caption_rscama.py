"""B0 captions from RSCaMa (C02), author checkpoint, on Apple MPS/CPU.

Runs the unmodified upstream code (commit aa68e4c, in cache/rscama_src) with:
  1. a pure-PyTorch shim for the two CUDA kernels it calls
     (selective_scan_fn = mamba_ssm's reference scan, causal_conv1d_fn =
     grouped conv1d + SiLU); weights and state_dict keys are unchanged;
  2. the README-required CLIP patch (VisionTransformer returns patch tokens);
  3. .cuda() redirected to the local device.
Native CLIP preprocessing (224 px), author vocab rebuilt at threshold 5, greedy decoding
(decoder.sample k=1), max_length 42 -- as in the author's test.py.

  .venv-rscama/bin/python scripts/caption_rscama.py --roles development_smoke
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS, EXEC_ROOT, Timer, load_manifest, new_run_record, now, pair_paths, percentile  # noqa: E402

SRC = EXEC_ROOT / "cache" / "rscama_src"
CKPT = EXEC_ROOT / "weights" / "rscama" / "RSCaMa.pth"
UPSTREAM_COMMIT = "aa68e4c6933c7dd0f2e2f9cbd7642d61929c86a5"
CKPT_REVISION = "3e527e6656b62cc615eab1d2f3fb73777cbbf62c"


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """mamba_ssm.ops.selective_scan_interface.selective_scan_ref (real A, B/C: b n l)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B, C = B.float(), C.float()
    x = A.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.float()[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype_in)
    return (out, x) if return_last_state else out


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    L = x.shape[-1]
    y = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[1] - 1, groups=x.shape[1])[..., :L]
    return F.silu(y) if activation in ("silu", "swish") else y


def build_author_vocab(captions_json, threshold=5):
    """Rebuild the vocabulary with the author's own tokenize/build_vocab (preprocess_data.py).

    The shipped data/LEVIR_CC/vocab.json has 501 tokens (threshold 4); the released
    checkpoint's decoder has 468, which threshold 5 (the README's setting) reproduces.
    Word counts use caption text from all splits, as upstream does; no images.
    """
    import ast
    src = (SRC / "preprocess_data.py").read_text()
    keep = [n for n in ast.parse(src).body
            if (isinstance(n, ast.FunctionDef) and n.name in ("tokenize", "build_vocab"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SPECIAL_TOKENS" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "preprocess_data", "exec"), ns)
    data = json.load(open(captions_json))
    seqs = [(img["filename"], [ns["tokenize"](s["raw"], add_start_token=True, add_end_token=True,
                                              punct_to_keep=[";", ","], punct_to_remove=["?", "."])
                               for s in img["sentences"]]) for img in data["images"]]
    return ns["build_vocab"](seqs, threshold)


def install_patches(device, shim=True):
    """shim=False keeps mamba_ssm/causal_conv1d kernels (CUDA parity run)."""
    import transformers.models.mamba.modeling_mamba as mm
    if shim:
        mm.selective_scan_fn = selective_scan_ref
        mm.causal_conv1d_fn = causal_conv1d_ref
    elif mm.selective_scan_fn is None or mm.causal_conv1d_fn is None:
        raise RuntimeError("mamba_ssm / causal_conv1d kernels not importable")
    for name in ("causal_conv1d_update", "selective_state_update", "mamba_inner_fn"):
        if not hasattr(mm, name) or getattr(mm, name) is None:
            setattr(mm, name, None)

    import clip.model as cm

    def vit_forward(self, x):  # README / PromptCC issue #3: also return patch tokens before ln_post
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="development_smoke")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--ckpt", default=str(CKPT), help="released RSCaMa.pth, or best.pth from colab/rscama_colab.py")
    args = ap.parse_args()
    args.ckpt = str(Path(args.ckpt).resolve())  # main() chdirs into the upstream source tree
    retrained = Path(args.ckpt).name == "best.pth"
    mid = "C02-r20" if retrained else "C02"
    device = torch.device(args.device)
    install_patches(device)

    sys.path.insert(0, str(SRC))
    os.chdir(SRC)  # upstream modules use relative paths
    import clip
    from PIL import Image
    from model.mamba_block import MambaMixer
    from model.model_decoder import DecoderTransformer
    from model.model_encoder_attMamba import AttentiveEncoder, Encoder

    MambaMixer.forward = lambda self, h, h2, cache_params=None, cache_params_2=None: self.cuda_kernels_forward(h, h2)

    roles = args.roles.split(",")
    rows = load_manifest(roles)
    run_id = f"B0_C02r20_rscama_retrained20_{'+'.join(roles)}" if retrained else f"B0_C02_rscama_{'+'.join(roles)}"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    rec = new_run_record(
        run_id=run_id, role_model_id=mid,
        model_identity={"repo": "https://github.com/Chen-Yang-Liu/RSCaMa", "upstream_commit": UPSTREAM_COMMIT,
                        "hf_checkpoint": "retrained best.pth (Kaggle v6)" if retrained else "lcybuaa/RSCaMa/RSCaMa.pth", "backbone": "CLIP-ViT-B/32",
                        "kernel_path": "pure-PyTorch reference shim (no mamba_ssm/causal_conv1d CUDA kernels)"},
        checkpoint_revision=None if retrained else CKPT_REVISION,
        training_data_exposure=("Retrained by us on 1,313 LEVIR-MCI train pairs (colab/rscama_colab.py)." if retrained else
                                "Trained on the LEVIR-CC official train split (author paper). Development pairs "
                                "here come from that split (seen in training); screening pairs from val."),
        split="+".join(sorted({r["official_split"] for r in rows})), sample_count=len(rows), condition="B0",
        processor_config="CLIP preprocess: bicubic resize 224, centre crop 224, CLIP mean/std",
        precision_and_quantisation=f"float32, device={args.device}",
        prompt_revision="n/a (specialist captioner, no prompt)", seed=None,
        limitations="Kernel shim not parity-checked against CUDA kernels yet; greedy decoding as upstream test.py.",
    )
    rec["decoding"] = {"method": "decoder.sample(k=1) greedy", "max_length": 42}

    with Timer() as load_t:
        ckpt = torch.load(args.ckpt, map_location="cpu", mmap=not retrained, weights_only=not retrained)
        # the retrained checkpoint carries its own vocabulary (built from its training subset)
        word_vocab = ckpt["vocab"] if retrained else build_author_vocab(
            EXEC_ROOT / "data" / "levir_mci_meta" / "LevirCCcaptions.json", 5)
        assert len(word_vocab) == ckpt["decoder_dict"]["wdc.weight"].shape[0], len(word_vocab)
        _, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
        encoder = Encoder("CLIP-ViT-B/32")
        encoder_trans = AttentiveEncoder(n_layers=3, feature_size=[7, 7, 768], heads=8, dropout=0.1)
        decoder = DecoderTransformer(decoder_type="transformer_decoder", embed_dim=768,
                                     vocab_size=len(word_vocab), max_lengths=42, word_vocab=word_vocab,
                                     n_head=8, n_layers=1, dropout=0.1)
        load_report = {"vocab": f"{'from checkpoint' if retrained else 'rebuilt with author preprocess_data.py, threshold 5'}, {len(word_vocab)} tokens"}
        for name, mod, key in (("encoder", encoder, "encoder_dict"), ("encoder_trans", encoder_trans, "encoder_trans_dict"),
                               ("decoder", decoder, "decoder_dict")):
            mod.load_state_dict(ckpt[key], strict=True)
            load_report[name] = "strict ok"
            mod.eval().to(device)
        del ckpt
    rec["cold_load_s"] = round(load_t.s, 2)
    rec["load_report"] = load_report
    idx2word = {v: k for k, v in word_vocab.items()}
    skip = {word_vocab["<START>"], word_vocab["<END>"], word_vocab["<NULL>"]}

    times, done = [], 0
    with open(out / "captions.jsonl", "w") as f:
        for i, row in enumerate(rows):
            p1, p2 = pair_paths(row)
            item = {"pair_id": row["pair_id"], "run_id": run_id, "model": mid, "condition": "B0",
                    "poc_role": row["poc_role"]}
            try:
                a = preprocess(Image.open(p1).convert("RGB")).unsqueeze(0).to(device)
                b = preprocess(Image.open(p2).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad(), Timer() as t:
                    feat1, feat2 = encoder(a, b)
                    feat = encoder_trans(feat1, feat2)
                    seq = decoder.sample(feat, k=1)
                    if device.type == "mps":
                        torch.mps.synchronize()
                words = [idx2word[w] for w in seq if w not in skip]
                item.update(raw_text=" ".join(words), status="ok", output_tokens=len(words),
                            hit_token_limit=len(seq) >= 43, gen_s=round(t.s, 3))
                if i > 0:
                    times.append(t.s)
                done += 1
            except Exception as e:
                item.update(raw_text=None, status="error", error=repr(e))
                rec["errors"].append({"pair_id": row["pair_id"], "error": traceback.format_exc()[-800:]})
            f.write(json.dumps(item) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(rows)}] {row['pair_id']} {item.get('gen_s')}s :: {item.get('raw_text')}", flush=True)

    rec.update(status="completed" if done == len(rows) else "completed_with_errors", end_utc=now(),
               completion_rate=round(done / len(rows), 4), warm_latency_median_s=percentile(times, 0.5),
               warm_latency_p95_s=percentile(times, 0.95),
               peak_gpu_memory_mb=round(torch.mps.driver_allocated_memory() / 2**20) if device.type == "mps" else None,
               raw_output_path=str((out / "captions.jsonl").relative_to(RUNS.parent)))
    json.dump(rec, open(out / "run_record.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("run_id", "status", "cold_load_s", "warm_latency_median_s", "peak_gpu_memory_mb", "completion_rate")}))


if __name__ == "__main__":
    main()
