# Work package B: first execution (captions and claims)

**Owner:** Sujal Raj Pradhan · **Started:** 26–27 October 2026 · **Status:** first-wave B0 screening done; human audit pending.

This folder carries out [B_execution_plan.md](../benchmarks/B_execution_plan.md) for [work package B](../benchmarks/B_captioning.md) under the [shared protocol](../benchmarks/SHARED_PROTOCOL.md). It answers one question: **what visibly changed between two images, and how accurately do different systems describe it?**

Results: [outputs/b0_results.md](outputs/b0_results.md) (generated). Data story for non-specialists: `story/index.html` (published privately; ask Sujal for the link).

## What was run

| Step | Result |
|---|---|
| Data | LEVIR-MCI from the author's Hugging Face copy (`lcybuaa/LEVIR-MCI@dff09c7`). The team Drive folder needs sign-in, so the same archive was read by HTTP range requests: only the selected pairs were downloaded (47 MB of 2.77 GB). |
| Subset | Seed 42, uniform within official splits: 50 train (development, first 10 = smoke fixture), 100 val (screening), 100 test (sealed: IDs only, never downloaded). 250 of 10,077 pairs (2.5%), inside the 20% cap per split and overall. Parent scenes are not recoverable from patch names, so grouping is by pair. |
| Systems (B0: image pair only) | C02 RSCaMa (author checkpoint), C04 Qwen3-VL-4B-Instruct, C07 SmolVLM2-2.2B-Instruct, plus two non-learned controls: a pixel-difference evidence template and a constant "there is no difference". |
| Claims | Rule-based parser (`scripts/claims.py`, rules-v2) turns each caption into claims in the team's [claim_decision.json](../benchmarks/templates/claim_decision.json) shape. |
| Scoring | One pinned pycocoevalcap 1.2 recipe (PTB tokeniser, Java 17) with all five references; plus an automatic reference check of every claim against the oracle change mask and reference captions. |
| Hardware | Apple M4, 16 GB, MPS. Latencies are this laptop, batch size 1. |

## Decisions worth knowing

- **RSCaMa runs without CUDA.** Its Mamba layers call two CUDA kernels. `scripts/caption_rscama.py` swaps in the reference PyTorch versions (`selective_scan_ref`, a grouped causal conv1d). Weights load strictly and unchanged; `tests/test_rscama_shim.py` checks the shim against an independent loop. Not yet parity-checked against the real kernels on Colab.
- **RSCaMa vocabulary.** The shipped `vocab.json` has 501 tokens (threshold 4); the released checkpoint has 468, which the README's threshold 5 reproduces. The wrapper rebuilds it with the author's own `tokenize`/`build_vocab`. That counts words across all splits' caption text, as upstream does; no images, and every word RSCaMa emitted is also in the train-only vocabulary.
- **SmolVLM2 image size.** The publisher processor upsamples each 256 px image to 1536 px and tiles it (about 1.4k tokens per image), which swapped a 16 GB laptop to 60–240 s per pair. Runs use one 384 px view per image (`do_image_splitting=False`), recorded in each run record.
- **Pixel-difference control.** Thresholds (1.75 standard deviations, more than 5% of the image changed) were tuned on the 50 development pairs only, then frozen.
- **Examples in the data story** are the first changed, first unchanged and first disputed screening pair in seeded order, not hand-picked.

## How to read the numbers

- **Automatic, not human.** A claim "checks out" when the oracle mask shows that object type changed (at least 50 px) and the reference captions do not contradict its direction. This mostly tests whether the right kind of change happened; the stricter *located* precision also requires a stated image location to overlap the mask. Claims about vehicles or vegetation cannot be checked against a building/road mask; they stay in the strict denominator (protocol rule), and *in-scope* precision leaves them out.
- **Reference conflicts are excluded from rates.** In 8 of the 150 inspected pairs, the mask and the five captions disagree (e.g. a highway marked changed after a slight misregistration while every caption says "no difference").
- **RSCaMa is on home ground.** It was trained on LEVIR-CC train and its released checkpoint was selected on val BLEU-4, so the screening pairs are not unseen for it. A fair specialist-vs-general comparison needs the sealed test split or another region.
- **Screening is slightly used.** Five parser rules were prompted by reading 12 SmolVLM2 screening outputs (disclosed in `claims.py`). The sealed split is the clean final test.
- **CIDEr/BLEU here use 100 pairs** and are not comparable with published full-test-set scores.

## Layout

```
scripts/   remote_zip.py, build_manifest.py   data subset (range-reads the zip)
           caption_vlm.py, caption_rscama.py   B0 captioners; controls.py
           claims.py, evaluate.py, summarise.py, finalise_records.py
           make_audit_packet.py, export_story.py
           make_colab_notebook.py, import_colab.py   Colab retrain in and out
colab/     rscama_colab.py (fetch/train/caption/parity), rscama_finetune.ipynb (generated)
tests/     parser, evaluator and RSCaMa shim tests (pytest)
outputs/   b0_results.{md,json}: aggregate metrics only (committed)
data/ runs/ weights/ cache/   local only (git-ignored): imagery, references, captions, claims
```

## Reproduce

```bash
uv venv --python 3.11 .venv-vlm && VIRTUAL_ENV=.venv-vlm uv pip install torch==2.8.0 torchvision==0.23.0 \
  transformers==4.57.1 accelerate==1.10.1 num2words pillow "numpy<2.3" pycocoevalcap==1.2 huggingface_hub \
  pytest scipy scikit-image
uv venv --python 3.11 .venv-rscama && VIRTUAL_ENV=.venv-rscama uv pip install torch==2.8.0 torchvision==0.23.0 \
  transformers==4.39.3 einops==0.6.1 ftfy regex "numpy<2" pillow tqdm "setuptools<70" wheel
VIRTUAL_ENV=.venv-rscama uv pip install --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
git clone https://github.com/Chen-Yang-Liu/RSCaMa cache/rscama_src && git -C cache/rscama_src checkout aa68e4c

.venv-vlm/bin/python scripts/build_manifest.py          # downloads only selected pairs
./scripts/fetch_weights.sh                               # pinned revisions, ~22 GB
R=development_smoke,development,screening
.venv-vlm/bin/python scripts/caption_vlm.py --model qwen3vl --roles $R
.venv-vlm/bin/python scripts/caption_vlm.py --model smolvlm2 --roles $R
.venv-rscama/bin/python scripts/caption_rscama.py --roles $R
.venv-vlm/bin/python scripts/controls.py --roles screening
.venv-vlm/bin/python scripts/evaluate.py runs/<run> --roles screening --tag _screening
.venv-vlm/bin/python scripts/summarise.py
.venv-vlm/bin/python -m pytest -q tests
```

## RSCaMa retrain (run on Kaggle T4, 2026-10-27)

The released RSCaMa checkpoint has seen all of LEVIR-CC train and was selected on val, so fine-tuning it further on LEVIR would not give a fair number. `colab/rscama_finetune.ipynb` instead retrains it from the public CLIP ViT-B/32 weights with the upstream recipe (Adam 1e-4, 12 pairs x 5 captions per step (about 60, upstream 64), last 6 CLIP blocks tuned, up to 40 epochs, early stop after 8):

- **Data:** [LEVIR-MCI on Hugging Face](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI) (rev `dff09c7`; [dataset page](https://github.com/Chen-Yang-Liu/LEVIR-MCI)). Training uses 1,313 train pairs, `Random('42-train-finetune').sample(train minus development, 1313)`. With the 50 development pairs used for checkpoint selection (BLEU-4), that is 1,363 = 20.0% of the 6,815 train pairs. The notebook fetches only these pairs plus the 100 screening pairs (about 400 MB), and asserts the ID hashes match the frozen manifest.
- **Benchmark:** the same 50 development and 100 screening pairs, parser and evaluator as every B0 run. Screening images are loaded only for the final captioning pass.
- **Parity:** the notebook also runs the released checkpoint on CUDA through the shim and, if the kernels build, through the real `mamba_ssm` kernels; `scripts/import_colab.py` compares both against the Mac captions.
- **Result (screening, 100 pairs):** claim precision 79.2% (72.3–86.2) vs 78.7% for the released checkpoint; fact recall 74.4% vs 76.8%; false alarms on unchanged 3 of 40 vs 0; missed changes 9.3% vs 14.8%; BLEU-4 0.579 vs 0.622, CIDEr-D 1.21 vs 1.32. A tie on claims within the intervals. Best epoch 14 of 23 (early stop), dev BLEU-4 0.618; about 3 h on a Kaggle T4. Development numbers are optimistic (checkpoint chosen on them).
- **Parity:** released checkpoint on CUDA, shim and real `mamba_ssm` kernels, both 150/150 captions identical to the Mac run (`runs/parity/parity_report.json`).
- **Tested locally:** fetch (image hashes match the manifest), one CPU training epoch, captioning, and the released-checkpoint parity path (150/150 captions identical to the Mac run, also under Python 3.12 + NumPy 2). Training on MPS fails in an upstream conv backward, which is why it runs on a GPU.

```bash
.venv-vlm/bin/python scripts/make_colab_notebook.py        # rebuilds the notebook from the tested sources
# run colab/rscama_finetune.ipynb on a T4, download b_rscama_results.zip, unzip to cache/colab_results/
.venv-vlm/bin/python scripts/import_colab.py cache/colab_results
.venv-vlm/bin/python scripts/evaluate.py runs/B0_C02r20_* --roles screening --tag _screening
.venv-vlm/bin/python scripts/summarise.py
```

## Out-of-domain check (WHU-CDC, SECOND-CC), 2026-10-27

`scripts/build_ood.py whu_cdc|second_cc` builds 100 seeded val pairs per dataset in the `data/` layout under `data_ood/<name>/` (git-ignored); set `B_DATA=data_ood/<name>` and every script reads that set and writes to `runs/ood/<name>/`. Same models, Mac, parser and evaluator as LEVIR. Neither dataset marks roads, so only building claims are mask-checked (road claims count as unverifiable). Sources: [WHU-CDC](https://huggingface.co/datasets/hygge10111/RS-CDC) (third-party upload, no licence stated; WHU-CD imagery is academic use), [SECOND-CC](https://doi.org/10.5281/zenodo.16937571) (CC-BY-4.0, originals only, augmented copies excluded). Results: `outputs/b0_results_ood_*.md`.

Every captioner drops sharply off LEVIR; none clearly beats the pixel-difference template. RSCaMa hallucinates buildings on WHU-CDC (32% of unchanged pairs) and misses 54% of SECOND-CC building changes; our 20% retrain generalises worse (55% false alarms on both); Qwen3-VL has the best building-claim precision on SECOND-CC (62%).

```bash
.venv-vlm/bin/python scripts/build_ood.py second_cc
B_DATA=data_ood/second_cc .venv-rscama/bin/python scripts/caption_rscama.py --roles screening [--ckpt weights/rscama_r20/out/best.pth]
B_DATA=data_ood/second_cc .venv-vlm/bin/python scripts/evaluate.py runs/ood/second_cc/<run> --roles screening --tag _screening
B_DATA=data_ood/second_cc .venv-vlm/bin/python scripts/summarise.py
```

## Handoff and next steps

- **Shoaib (D) and Priyanshu (C):** cached captions and contract-shaped claims for the baseline (RSCaMa) and challenger (Qwen3-VL), development and screening, are in `runs/handoff/` (shared through the team Drive, not Git). Evidence and decision fields are left empty for C/D.
- **Human audit pilot:** `runs/audit_pilot/` has a blinded sheet (team `claim_annotations.csv` format), an image viewer and a separate key. Two raters; write the reference facts before opening the outputs.
- **Open:** B1 once Sarthak's predicted regions exist; P2 candidates (Chg2Cap, MCINet, Gemma-3, InternVL3.5); sealed-test run of the frozen finalists.
