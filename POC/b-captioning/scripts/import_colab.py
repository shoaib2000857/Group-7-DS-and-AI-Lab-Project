"""Bring the Colab outputs (colab/rscama_finetune.ipynb) into runs/ for the standard evaluator.

  python scripts/import_colab.py <unzipped results dir>

  runs/B0_C02r20_rscama_retrained20_<roles>/   captions.jsonl + run_record.json
  runs/parity/parity_report.json               released checkpoint: CUDA kernels / CUDA shim vs this Mac
Then: evaluate.py <run> --roles screening --tag _screening (and development), summarise.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RUNS, new_run_record  # noqa: E402

MAC_RUN = RUNS / "B0_C02_rscama_development_smoke+development+screening"


def jsonl(p):
    return [json.loads(line) for line in open(p)]


def main(src):
    src = Path(src)
    env = json.load(open(src / "colab_env.json"))
    tl = json.load(open(src / "train_log.json"))
    fetch = json.load(open(src / "fetch.json"))
    caps = jsonl(src / "captions_retrained.jsonl")
    run_id = caps[0]["run_id"]
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "captions.jsonl").write_text("".join(json.dumps(c) + "\n" for c in caps))
    times = sorted(c["gen_s"] for c in caps[1:])
    rec = new_run_record(
        run_id=run_id, role_model_id="C02-r20",
        model_identity={"repo": "https://github.com/Chen-Yang-Liu/RSCaMa", "upstream_commit": "aa68e4c",
                        "init": "public CLIP ViT-B/32 (openai), not the released RSCaMa checkpoint",
                        "training_script": "colab/rscama_colab.py train", "best_epoch": tl["best_epoch"]},
        checkpoint_sha256=env.get("best_pth_sha256"),
        training_data_exposure=(f"Trained on {tl['finetune_pairs']} LEVIR-MCI train pairs (seeded 20% of train minus the "
                                "50 development pairs); checkpoint chosen on development BLEU-4. No val or test "
                                "image was used before the final captioning pass, so screening is unseen."),
        split="train+val", sample_count=len(caps), condition="B0",
        processor_config="CLIP preprocess: bicubic resize 224, centre crop 224, CLIP mean/std",
        precision_and_quantisation=f"float32, {env.get('gpu')}",
        prompt_revision="n/a (specialist captioner, no prompt)", seed=tl["config"]["seed"],
        environment_record=env, status="completed", completion_rate=1.0,
        warm_latency_median_s=times[len(times) // 2], warm_latency_p95_s=times[int(0.95 * (len(times) - 1))],
        limitations=["Latency is a Colab GPU, not this laptop",
                     f"Vocabulary from the training subset only ({tl['vocab_size']} tokens)",
                     "Kernel path: pure-PyTorch shim (same as the Mac run)"],
        raw_output_path=str((out / "captions.jsonl").relative_to(RUNS.parent)),
    )
    rec["training"] = {"log": tl["log"], "best_dev_bleu4": tl["best_dev_bleu4"], "config": tl["config"],
                       "train_samples": tl["train_samples"], "data": fetch}
    json.dump(rec, open(out / "run_record.json", "w"), indent=1)
    print("wrote", out)

    mac = {c["pair_id"]: c["raw_text"] for c in jsonl(MAC_RUN / "captions.jsonl")}
    report = {}
    for p in sorted(src.glob("captions_released_*.jsonl")):
        other = {c["pair_id"]: c["raw_text"] for c in jsonl(p)}
        same = [k for k in mac if other.get(k) == mac[k]]
        report[p.stem] = {"pairs": len(mac), "identical_to_mac": len(same),
                          "differences": [{"pair_id": k, "mac": mac[k], "other": other.get(k)}
                                          for k in mac if other.get(k) != mac[k]][:20]}
    (RUNS / "parity").mkdir(exist_ok=True)
    report["note"] = "Released checkpoint captions, greedy decoding. Mac = MPS + pure-PyTorch shim."
    json.dump(report, open(RUNS / "parity" / "parity_report.json", "w"), indent=1)
    print(json.dumps({k: v.get("identical_to_mac") if isinstance(v, dict) else v for k, v in report.items()}))


if __name__ == "__main__":
    main(sys.argv[1])
