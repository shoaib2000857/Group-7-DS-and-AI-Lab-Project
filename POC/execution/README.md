# Work package B: captions and claims

**Owner:** Sujal Raj Pradhan · **Status:** B0 screening, RSCaMa retrain and out-of-domain check done; human audit pending.

This folder is the second pipeline stage: it turns an image pair into a caption, then into claims the later stages can check. It carries out [B_execution_plan.md](../benchmarks/B_execution_plan.md) for [work package B](../benchmarks/B_captioning.md) under the [shared protocol](../benchmarks/SHARED_PROTOCOL.md).

## Results

Full tables: [LEVIR](outputs/b0_results.md) · [WHU-CDC](outputs/b0_results_ood_whu_cdc.md) · [SECOND-CC](outputs/b0_results_ood_second_cc.md).

Share of building (and, on LEVIR, road) claims that check out, 100 seeded val pairs per dataset:

| System | LEVIR-MCI | WHU-CDC | SECOND-CC |
|---|---|---|---|
| RSCaMa, released | 88.9% | 66.7% | 50.0% |
| RSCaMa, retrained on 20% of train | 88.0% | 50.6% | 48.4% |
| Qwen3-VL-4B | 72.9% | 51.2% | 61.8% |
| SmolVLM2-2.2B | 48.4% | 46.2% | 45.3% |
| Pixel-difference template | 69.0% | 58.0% | 64.0% |
| Always "no difference" | 40.0% | 60.0% | 27.0% |

- RSCaMa leads on LEVIR, and a copy that never saw the screening pairs ties it there.
- Off LEVIR every captioner drops, and none clearly beats the pixel template. Evidence and claim checking matter more than the choice of captioner.
- Intervals are about ±10 points. These are automatic checks, not the human audit.

## Layout

```
scripts/   remote_zip.py, build_manifest.py, build_ood.py   data subsets (range-read the zips)
           caption_vlm.py, caption_rscama.py, controls.py   systems
           claims.py, evaluate.py, summarise.py             parser (rules-v3) and scoring
           export_handoff.py, make_audit_packet.py          outputs for stages C/D and the audit
           make_colab_notebook.py, import_colab.py          retrain notebook in and out
colab/     rscama_colab.py (fetch/train/caption/parity), rscama_finetune.ipynb (generated)
tests/     parser, evaluator and RSCaMa shim tests
outputs/   aggregate metrics only (committed)
data/ data_ood/ runs/ weights/ cache/   local only (git-ignored)
```

## Setup

```bash
uv venv --python 3.11 .venv-vlm && VIRTUAL_ENV=.venv-vlm uv pip install torch==2.8.0 torchvision==0.23.0 \
  transformers==4.57.1 accelerate==1.10.1 num2words pillow "numpy<2.3" pycocoevalcap==1.2 huggingface_hub \
  pytest scipy scikit-image
uv venv --python 3.11 .venv-rscama && VIRTUAL_ENV=.venv-rscama uv pip install torch==2.8.0 torchvision==0.23.0 \
  transformers==4.39.3 einops==0.6.1 ftfy regex "numpy<2" pillow tqdm "setuptools<70" wheel
VIRTUAL_ENV=.venv-rscama uv pip install --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
git clone https://github.com/Chen-Yang-Liu/RSCaMa cache/rscama_src && git -C cache/rscama_src checkout aa68e4c
./scripts/fetch_weights.sh                               # pinned revisions, ~22 GB
.venv-vlm/bin/python -m pytest -q tests
```

## Reproduce

**LEVIR screening**

```bash
.venv-vlm/bin/python scripts/build_manifest.py           # downloads only selected pairs
R=development_smoke,development,screening
.venv-vlm/bin/python scripts/caption_vlm.py --model qwen3vl --roles $R
.venv-vlm/bin/python scripts/caption_vlm.py --model smolvlm2 --roles $R
.venv-rscama/bin/python scripts/caption_rscama.py --roles $R
.venv-vlm/bin/python scripts/controls.py --roles screening
.venv-vlm/bin/python scripts/evaluate.py runs/<run> --roles screening --tag _screening
.venv-vlm/bin/python scripts/summarise.py
```

**RSCaMa retrain.** The released checkpoint saw all of LEVIR-CC train and was selected on val, so fine-tuning it would not give a fair number. The notebook retrains from public CLIP ViT-B/32 weights on 1,313 train pairs (with the 50 development pairs used for checkpoint selection, 20.0% of 6,815). It takes about 4 hours on a free T4. It needs an NVIDIA GPU: training on MPS fails in an upstream conv backward.

```bash
.venv-vlm/bin/python scripts/make_colab_notebook.py      # rebuilds the notebook from the sources
# run colab/rscama_finetune.ipynb on Kaggle or Colab, unzip b_rscama_results.zip to cache/colab_results/
.venv-vlm/bin/python scripts/import_colab.py cache/colab_results
.venv-vlm/bin/python scripts/evaluate.py runs/B0_C02r20_* --roles screening --tag _screening
```

The released checkpoint gives identical captions on the Mac shim and on CUDA, with both the shim and the real `mamba_ssm` kernels (150 of 150 pairs).

**Out-of-domain sets.** `B_DATA=data_ood/<name>` points every script at that set and writes to `runs/ood/<name>/`. Neither dataset marks roads, so only building claims are mask-checked. Sources: [WHU-CDC](https://huggingface.co/datasets/hygge10111/RS-CDC) (third-party upload, no licence stated; treat as academic use), [SECOND-CC](https://doi.org/10.5281/zenodo.16937571) (CC-BY-4.0, augmented copies excluded).

```bash
.venv-vlm/bin/python scripts/build_ood.py second_cc      # or whu_cdc
B_DATA=data_ood/second_cc .venv-rscama/bin/python scripts/caption_rscama.py --roles screening [--ckpt weights/rscama_r20/out/best.pth]
B_DATA=data_ood/second_cc .venv-vlm/bin/python scripts/evaluate.py runs/ood/second_cc/<run> --roles screening --tag _screening
B_DATA=data_ood/second_cc .venv-vlm/bin/python scripts/summarise.py
```

## Decisions worth knowing

- **RSCaMa runs without CUDA.** Its Mamba layers call two CUDA kernels. `scripts/caption_rscama.py` swaps in the reference PyTorch versions (`selective_scan_ref`, a grouped causal conv1d). Weights load strictly and unchanged; `tests/test_rscama_shim.py` checks the shim against an independent loop. Not yet parity-checked against the real kernels on Colab.
- **RSCaMa vocabulary.** The shipped `vocab.json` has 501 tokens (threshold 4); the released checkpoint has 468, which the README's threshold 5 reproduces. The wrapper rebuilds it with the author's own `tokenize`/`build_vocab`. That counts words across all splits' caption text, as upstream does; no images, and every word RSCaMa emitted is also in the train-only vocabulary.
- **SmolVLM2 image size.** The publisher processor upsamples each 256 px image to 1536 px and tiles it (about 1.4k tokens per image), which swapped a 16 GB laptop to 60–240 s per pair. Runs use one 384 px view per image (`do_image_splitting=False`), recorded in each run record.
- **Pixel-difference control.** Thresholds (1.75 standard deviations, more than 5% of the image changed) were tuned on the 50 development pairs only, then frozen.

## How to read the numbers

- **Automatic, not human.** A claim "checks out" when the oracle mask shows that object type changed (at least 50 px) and the reference captions do not contradict its direction. This mostly tests whether the right kind of change happened; the stricter *located* precision also requires a stated image location to overlap the mask. Claims about vehicles or vegetation cannot be checked against a building/road mask; they stay in the strict denominator (protocol rule), and *in-scope* precision leaves them out.
- **Reference conflicts are excluded from rates.** In 8 of the 150 inspected pairs, the mask and the five captions disagree (e.g. a highway marked changed after a slight misregistration while every caption says "no difference").
- **RSCaMa's released checkpoint is on home ground.** It was trained on LEVIR-CC train and selected on val BLEU-4, so the screening pairs are not unseen for it. The retrained copy has seen no val or test image.
- **Screening is slightly used.** Five parser rules were prompted by reading 12 SmolVLM2 screening outputs (disclosed in `claims.py`). The sealed split is the clean final test.
- **CIDEr/BLEU here use 100 pairs** and are not comparable with published full-test-set scores.

## Handoff and next steps

- **Shoaib (D) and Priyanshu (C):** cached captions and contract-shaped claims for the baseline (RSCaMa) and challenger (Qwen3-VL), development and screening, are in `runs/handoff/` (shared through the team Drive, not Git). Evidence and decision fields are left empty for C/D.
- **Human audit pilot:** `runs/audit_pilot/` has a blinded sheet (team `claim_annotations.csv` format), an image viewer and a separate key. Two raters; write the reference facts before opening the outputs.
- **Open:** B1 once Sarthak's predicted regions exist; P2 candidates (Chg2Cap, MCINet, Gemma-3, InternVL3.5); sealed-test run of the frozen finalists.
