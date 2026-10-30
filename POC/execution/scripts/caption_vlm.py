"""B0 captions from a general multi-image VLM (C04 Qwen3-VL-4B, C07 SmolVLM2-2.2B).

Condition B0: original T1/T2 pair only, fixed prompt, publisher processor and
chat template, greedy decoding, batch size 1. Writes
runs/<run_id>/captions.jsonl and runs/<run_id>/run_record.json.

  python scripts/caption_vlm.py --model smolvlm2 --roles development_smoke
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Weights are pre-fetched at pinned revisions; never send the (stale) cached HF token.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    PROMPT_REVISION, PROMPT_V1, RUNS, Timer, load_manifest, new_run_record,
    now, pair_paths, percentile,
)

MODELS = {
    "qwen3vl": ("C04", "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "smolvlm2": ("C07", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4"),
}


# SmolVLM2's default upsamples 256 px inputs to 1536 px and splits them into 17 tiles
# (~1.4k tokens per image), which exhausted 16 GB unified memory on the M4 (swap-bound,
# 60-240 s/pair). One 384 px view per image (81 tokens) is used instead.
PROCESSOR_OVERRIDES = {
    "smolvlm2": {"do_image_splitting": False, "size": {"longest_edge": 384}},
}


def messages(t1, t2):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image T1:"},
            {"type": "image", "image": t1},
            {"type": "text", "text": "Image T2:"},
            {"type": "image", "image": t2},
            {"type": "text", "text": PROMPT_V1},
        ],
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--roles", default="development_smoke")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    role_id, repo, rev = MODELS[args.model]
    roles = args.roles.split(",")
    rows = load_manifest(roles)
    run_id = f"B0_{role_id}_{args.model}_{'+'.join(roles)}{args.tag}"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    rec = new_run_record(
        run_id=run_id, role_model_id=role_id,
        model_identity={"hf_repo": repo, "class": "AutoModelForImageTextToText"},
        checkpoint_revision=rev,
        training_data_exposure="General VLM; no LEVIR-specific training disclosed by publisher. Not verified.",
        split="+".join(sorted({r["official_split"] for r in rows})),
        sample_count=len(rows), condition="B0",
        processor_config={"base": "publisher AutoProcessor", "overrides": PROCESSOR_OVERRIDES.get(args.model, {}),
                          "inputs": "original 256x256 PNGs, T1 then T2, labelled 'Image T1:' / 'Image T2:'"},
        precision_and_quantisation=f"{args.dtype}, no quantisation, device={device}",
        prompt_revision=PROMPT_REVISION, seed=None,
        limitations="Greedy decoding (do_sample=False); batch size 1; timings on Apple M4 MPS, not Colab GPU.",
    )
    rec["decoding"] = {"do_sample": False, "max_new_tokens": args.max_new_tokens}

    with Timer() as load_t:
        processor = AutoProcessor.from_pretrained(repo, revision=rev, **PROCESSOR_OVERRIDES.get(args.model, {}))
        model = AutoModelForImageTextToText.from_pretrained(
            repo, revision=rev, dtype=dtype, device_map=device
        ).eval()
    rec["cold_load_s"] = round(load_t.s, 2)

    times, done = [], 0
    with open(out / "captions.jsonl", "w") as f:
        for i, row in enumerate(rows):
            p1, p2 = pair_paths(row)
            t1, t2 = Image.open(p1).convert("RGB"), Image.open(p2).convert("RGB")
            item = {"pair_id": row["pair_id"], "run_id": run_id, "model": role_id,
                    "condition": "B0", "poc_role": row["poc_role"]}
            try:
                inputs = processor.apply_chat_template(
                    messages(t1, t2), add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(device)
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                n_in = int(inputs["input_ids"].shape[1])
                with torch.inference_mode(), Timer() as gen_t:
                    gen = model.generate(**inputs, do_sample=False,
                                         max_new_tokens=args.max_new_tokens)
                    if device == "mps":
                        torch.mps.synchronize()
                new = gen[0, n_in:]
                text = processor.decode(new, skip_special_tokens=True).strip()
                item.update(raw_text=text, status="ok", input_tokens=n_in,
                            output_tokens=int(new.shape[0]),
                            hit_token_limit=int(new.shape[0]) >= args.max_new_tokens,
                            gen_s=round(gen_t.s, 3))
                if i > 0:  # first pair includes graph warm-up
                    times.append(gen_t.s)
                done += 1
            except Exception as e:
                item.update(raw_text=None, status="error", error=repr(e))
                rec["errors"].append({"pair_id": row["pair_id"], "error": traceback.format_exc()[-800:]})
            f.write(json.dumps(item) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(rows)}] {row['pair_id']} {item.get('gen_s')}s :: {item.get('raw_text')}", flush=True)

    rec.update(
        status="completed" if done == len(rows) else "completed_with_errors",
        end_utc=now(), completion_rate=round(done / len(rows), 4),
        warm_latency_median_s=percentile(times, 0.5),
        warm_latency_p95_s=percentile(times, 0.95),
        peak_gpu_memory_mb=round(torch.mps.driver_allocated_memory() / 2**20) if device == "mps" else None,
        raw_output_path=str((out / "captions.jsonl").relative_to(RUNS.parent)),
    )
    json.dump(rec, open(out / "run_record.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("run_id", "status", "cold_load_s", "warm_latency_median_s", "warm_latency_p95_s", "peak_gpu_memory_mb", "completion_rate")}))


if __name__ == "__main__":
    main()
