"""RSCaMa retrained under the B protocol, plus the CUDA parity run (Colab T4 or any CUDA box).

Why retrain instead of fine-tuning the released checkpoint further: that checkpoint
already saw the whole LEVIR-CC train split and was selected on val BLEU-4, so the
screening (val) pairs are not unseen for it. This run starts again from the public
CLIP ViT-B/32 weights with the upstream recipe, trains on a seeded 20% of the train
split, selects the checkpoint on the 50 development pairs (held out of training) and
never loads a val image until the final captioning pass.

  fetch   download only the needed pairs from LEVIR-MCI (HTTP range reads)
  train   upstream recipe: Adam 1e-4, ~64 captions per update (12 pairs x 5), last 6 CLIP blocks tuned
  caption greedy captions for development + screening (our captions.jsonl format)
  parity  released checkpoint on screening, shim or real kernels (--kernels)

Dataset: LEVIR-MCI, https://huggingface.co/datasets/lcybuaa/LEVIR-MCI (rev dff09c7).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))  # local checkout; on Colab all files sit together
from caption_rscama import SRC, build_author_vocab, install_patches  # noqa: E402
from remote_zip import LEVIR_MCI_URL, fetch_members, open_remote_zip  # noqa: E402

ROOT = "LEVIR-MCI-dataset"
SEED = 42
N_DEV, N_SCREEN = 50, 100
CAP = 0.20
NOCHANGE = ["<START>", "the", "scene", "is", "the", "same", "as", "before", "<END>"]
# sha256 of the newline-joined file names, computed from the frozen local manifest (sha c5dde811...)
EXPECTED = {"development": "36959f028fb958c523391cb79b78c0dfa8879314a6fae69c1b4d520d8c6d8176",
            "screening": "ce994d9e4f234dd6e2d0537c453eaff7c131a92b745ee9e05aa5a2bc4822969e",
            "finetune": "0aa6974f154df1c9d225fae71d1cec5c942a4b592b2e83ac933a975aa4456416"}


def ids_sha(recs):
    return hashlib.sha256("\n".join(r["filename"] for r in recs).encode()).hexdigest()


def select(caps):
    """Same rule as scripts/build_manifest.py, plus the fine-tune pool.

    finetune = Random('42-train-finetune').sample(train minus development, n) with
    n chosen so development + finetune = 20% of train (the POC cap).
    """
    pool = {s: sorted([r for r in caps if r["filepath"] == s], key=lambda r: r["filename"]) for s in ("train", "val")}
    dev = random.Random(f"{SEED}-train").sample(pool["train"], N_DEV)
    scr = random.Random(f"{SEED}-val").sample(pool["val"], N_SCREEN)
    dev_names = {r["filename"] for r in dev}
    n_ft = int(CAP * len(pool["train"])) - N_DEV
    rest = [r for r in pool["train"] if r["filename"] not in dev_names]
    ft = random.Random(f"{SEED}-train-finetune").sample(rest, n_ft)
    sel = {"development": dev, "screening": scr, "finetune": ft}
    for k, v in sel.items():
        if EXPECTED[k] is not None:
            assert ids_sha(v) == EXPECTED[k], f"{k} selection differs from the frozen manifest"
    return sel


def cmd_fetch(a):
    out = Path(a.data)
    z, raw = open_remote_zip()
    caps = json.loads(z.read(f"{ROOT}/LevirCCcaptions.json"))["images"]
    (out / "LevirCCcaptions.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "LevirCCcaptions.json").write_text(json.dumps({"images": caps}))
    sel = select(caps)
    wanted = [f"{ROOT}/images/{r['filepath']}/{sub}/{r['filename']}"
              for recs in sel.values() for r in recs for sub in ("A", "B")]
    t = time.time()
    for i in range(0, len(wanted), 400):
        for name, b in fetch_members(z, raw, wanted[i:i + 400]).items():
            dest = out / name[len(ROOT) + 1:]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b)
        print(f"{min(i + 400, len(wanted))}/{len(wanted)} files", flush=True)
    info = {"source": LEVIR_MCI_URL, "bytes_fetched": raw.bytes_fetched, "fetch_s": round(time.time() - t, 1),
            **{f"{k}_pairs": len(v) for k, v in sel.items()}, **{f"{k}_ids_sha256": ids_sha(v) for k, v in sel.items()},
            "train_split_pairs": sum(r["filepath"] == "train" for r in caps)}
    json.dump(info, open(out / "fetch.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def load_upstream(device, shim=True):
    install_patches(device, shim=shim)
    sys.path.insert(0, str(SRC))
    cwd = os.getcwd()
    os.chdir(SRC)
    import clip
    from model.mamba_block import MambaMixer
    from model.model_decoder import DecoderTransformer
    from model.model_encoder_attMamba import AttentiveEncoder, Encoder
    MambaMixer.forward = lambda self, h, h2, cache_params=None, cache_params_2=None: self.cuda_kernels_forward(h, h2)
    _, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    os.chdir(cwd)

    def build(vocab):
        enc = Encoder("CLIP-ViT-B/32")
        et = AttentiveEncoder(n_layers=3, feature_size=[7, 7, 768], heads=8, dropout=0.1)
        de = DecoderTransformer(decoder_type="transformer_decoder", embed_dim=768, vocab_size=len(vocab),
                                max_lengths=42, word_vocab=vocab, n_head=8, n_layers=1, dropout=0.1)
        return enc, et, de
    return build, preprocess


def tokenizer():
    import ast
    src = (SRC / "preprocess_data.py").read_text()
    keep = [n for n in ast.parse(src).body
            if (isinstance(n, ast.FunctionDef) and n.name in ("tokenize", "build_vocab", "encode"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SPECIAL_TOKENS" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "preprocess_data", "exec"), ns)
    tok = lambda s: ns["tokenize"](s, add_start_token=True, add_end_token=True,  # noqa: E731
                                   punct_to_keep=[";", ","], punct_to_remove=["?", "."])
    return tok, ns["build_vocab"], ns["encode"]


def load_images(data, recs, preprocess):
    from PIL import Image
    a = torch.stack([preprocess(Image.open(Path(data) / "images" / r["filepath"] / "A" / r["filename"]).convert("RGB")) for r in recs])
    b = torch.stack([preprocess(Image.open(Path(data) / "images" / r["filepath"] / "B" / r["filename"]).convert("RGB")) for r in recs])
    return a, b


@torch.no_grad()
def greedy(models, a, b, device, idx2word, skip):
    enc, et, de = models
    for m in models:
        m.eval()
    out, times = [], []
    for i in range(len(a)):
        t = time.perf_counter()
        f1, f2 = enc(a[i:i + 1].to(device), b[i:i + 1].to(device))
        seq = de.sample(et(f1, f2), k=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        out.append((" ".join(idx2word[w] for w in seq if w not in skip), len(seq) >= 43))
    return out, times


def bleu4(hyps, refs):
    from nltk.translate.bleu_score import corpus_bleu
    return corpus_bleu([[r.split() for r in rs] for rs in refs], [h.split() for h in hyps])


def cmd_train(a):
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    from torch.nn.utils.rnn import pack_padded_sequence
    caps = json.load(open(Path(a.data) / "LevirCCcaptions.json"))["images"]
    sel = select(caps)
    if a.limit:
        sel = {k: v[:a.limit] for k, v in sel.items()}
    tok, build_vocab, encode = tokenizer()
    ft_tokens = [(r["filename"], [tok(s["raw"]) for s in r["sentences"]]) for r in sel["finetune"]]
    vocab = build_vocab(ft_tokens, 5)  # train-subset words only (upstream: all splits)
    idx2word = {v: k for k, v in vocab.items()}
    skip = {vocab["<START>"], vocab["<END>"], vocab["<NULL>"]}

    pair_caps = []  # per pair: its (padded tokens, length) captions
    for _, seqs in ft_tokens:
        if NOCHANGE in seqs:  # upstream LEVIRCC.py collapses no-change pairs in train
            seqs = [NOCHANGE] * len(seqs)
        enc_seqs = [encode(s, vocab, allow_unk=True)[:42] for s in seqs]
        pair_caps.append([(e + [0] * (42 - len(e)), len(e)) for e in enc_seqs])
    n_samples = sum(len(c) for c in pair_caps)

    build, preprocess = load_upstream(device)
    imgA, imgB = load_images(a.data, sel["finetune"], preprocess)
    devA, devB = load_images(a.data, sel["development"], preprocess)
    dev_refs = [[" ".join(tok(s["raw"])[1:-1]) for s in r["sentences"]] for r in sel["development"]]
    enc, et, de = models = build(vocab)
    enc.fine_tune(True)
    for m in models:
        m.to(device)
    params = [p for m in models for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    crit = torch.nn.CrossEntropyLoss()
    accum = a.accum

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log, best, best_ep, start = [], -1.0, -1, 0
    if (out / "last.pth").exists():  # resume after a Colab disconnect
        st = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
        for m, k in zip(models, ("encoder_dict", "encoder_trans_dict", "decoder_dict")):
            m.load_state_dict(st[k])
        opt.load_state_dict(st["opt"])
        log, best, best_ep, start = st["log"], st["best"], st["best_ep"], st["epoch"] + 1
        torch.set_rng_state(st["rng"])
        print(f"resumed at epoch {start}")
    gen = torch.Generator().manual_seed(a.seed)
    for _ in range(start):
        torch.randperm(len(pair_caps), generator=gen)
    t_start = time.time()

    for ep in range(start, a.epochs):
        for m in models:
            m.train()
        t0, tot, n = time.time(), 0.0, 0
        order = torch.randperm(len(pair_caps), generator=gen).tolist()
        n_steps = (len(order) + a.batch - 1) // a.batch
        opt.zero_grad()
        for step, i0 in enumerate(range(0, len(order), a.batch)):
            # Encode each pair once and decode all its captions against that encoding;
            # upstream re-encodes the pair for every caption (same loss, 5x the encoder work).
            pidx = order[i0:i0 + a.batch]
            caps_b = [c for j in pidx for c in pair_caps[j]]
            idx = torch.tensor(pidx)
            reps = torch.tensor([len(pair_caps[j]) for j in pidx], device=device)
            token = torch.tensor([c[0] for c in caps_b], device=device)
            tlen = torch.tensor([c[1] for c in caps_b], device=device)
            f1, f2 = enc(imgA[idx].to(device), imgB[idx].to(device))
            feat = et(f1, f2).repeat_interleave(reps, dim=0)
            scores, caps_sorted, dl, _ = de(feat, token, tlen)
            s = pack_padded_sequence(scores, dl, batch_first=True).data
            tg = pack_padded_sequence(caps_sorted[:, 1:], dl, batch_first=True).data
            loss = crit(s, tg.long())
            (loss / accum).backward()
            tot, n = tot + loss.item(), n + 1
            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            if step % 25 == 0:
                el = time.time() - t0
                print(f"epoch {ep} step {step + 1}/{n_steps} loss {loss.item():.3f} "
                      f"{el / (step + 1):.2f} s/step, epoch eta {el / (step + 1) * (n_steps - step - 1) / 60:.1f} min",
                      flush=True)
        hyps, _ = greedy(models, devA, devB, device, idx2word, skip)
        b4 = bleu4([h for h, _ in hyps], dev_refs)
        log.append({"epoch": ep, "train_loss": round(tot / max(n, 1), 4), "dev_bleu4": round(b4, 4),
                    "epoch_s": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        state = {"encoder_dict": enc.state_dict(), "encoder_trans_dict": et.state_dict(),
                 "decoder_dict": de.state_dict(), "vocab": vocab}
        if b4 > best:
            best, best_ep = b4, ep
            torch.save(state, out / "best.pth")
        torch.save({**state, "opt": opt.state_dict(), "log": log, "best": best, "best_ep": best_ep,
                    "epoch": ep, "rng": torch.get_rng_state()}, out / "last.pth")
        if ep - best_ep >= a.patience:
            break
        if (time.time() - t_start) / 3600 + (log[-1]["epoch_s"] / 3600) > a.max_hours:
            print(f"stopping: next epoch would pass the {a.max_hours} h budget", flush=True)
            break
    json.dump({"log": log, "best_dev_bleu4": best, "best_epoch": best_ep, "vocab_size": len(vocab),
               "finetune_pairs": len(sel["finetune"]), "train_samples": n_samples,
               "config": {k: v for k, v in vars(a).items() if k != "func"}},
              open(out / "train_log.json", "w"), indent=1)
    print("best epoch", best_ep, "dev BLEU-4", round(best, 4))


def write_captions(path, run_id, model_id, rows, results, times):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for (role, r), (text, hit), t in zip(rows, results, times):
            f.write(json.dumps({"pair_id": f"{r['filepath']}/{r['filename'][:-4]}", "run_id": run_id,
                                "model": model_id, "condition": "B0", "poc_role": role, "raw_text": text,
                                "status": "ok", "output_tokens": len(text.split()), "hit_token_limit": hit,
                                "gen_s": round(t, 3)}) + "\n")


def roles(sel):
    rows = [("development_smoke" if i < 10 else "development", r) for i, r in enumerate(sel["development"])]
    return rows + [("screening", r) for r in sel["screening"]]


def cmd_caption(a):
    """Best retrained checkpoint on the same development + screening pairs as every B0 run."""
    device = torch.device(a.device)
    caps = json.load(open(Path(a.data) / "LevirCCcaptions.json"))["images"]
    rows = roles(select(caps))
    build, preprocess = load_upstream(device)
    st = torch.load(Path(a.out) / "best.pth", map_location="cpu", weights_only=False)
    vocab = st["vocab"]
    models = build(vocab)
    for m, k in zip(models, ("encoder_dict", "encoder_trans_dict", "decoder_dict")):
        m.load_state_dict(st[k], strict=True)
        m.to(device)
    A, B = load_images(a.data, [r for _, r in rows], preprocess)
    skip = {vocab["<START>"], vocab["<END>"], vocab["<NULL>"]}
    res, times = greedy(models, A, B, device, {v: k for k, v in vocab.items()}, skip)
    write_captions(Path(a.out) / "captions_retrained.jsonl", "B0_C02r20_rscama_retrained20_development_smoke+development+screening",
                   "C02-r20", rows, res, times)
    print("wrote", len(res))


def cmd_parity(a):
    """Released checkpoint, screening + development, with the shim or the real CUDA kernels."""
    device = torch.device(a.device)
    caps = json.load(open(Path(a.data) / "LevirCCcaptions.json"))["images"]
    rows = roles(select(caps))
    build, preprocess = load_upstream(device, shim=not a.kernels)
    vocab = build_author_vocab(Path(a.data) / "LevirCCcaptions.json", 5)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    models = build(vocab)
    for m, k in zip(models, ("encoder_dict", "encoder_trans_dict", "decoder_dict")):
        m.load_state_dict(st[k], strict=True)
        m.to(device)
    A, B = load_images(a.data, [r for _, r in rows], preprocess)
    skip = {vocab["<START>"], vocab["<END>"], vocab["<NULL>"]}
    res, times = greedy(models, A, B, device, {v: k for k, v in vocab.items()}, skip)
    tag = "kernels" if a.kernels else "shim"
    write_captions(Path(a.out) / f"captions_released_{tag}_{device.type}.jsonl", f"parity_C02_{tag}_{device.type}",
                   "C02", rows, res, times)
    print("wrote", len(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="levir")
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    sub = ap.add_subparsers(required=True)
    sub.add_parser("fetch").set_defaults(func=cmd_fetch)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--batch", type=int, default=12, help="image pairs per step (x5 captions)")
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--max-hours", type=float, default=8.0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--limit", type=int, default=0, help="smoke test only")
    t.set_defaults(func=cmd_train)
    sub.add_parser("caption").set_defaults(func=cmd_caption)
    p = sub.add_parser("parity")
    p.add_argument("--ckpt", default="RSCaMa.pth")
    p.add_argument("--kernels", action="store_true")
    p.set_defaults(func=cmd_parity)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
