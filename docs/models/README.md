# TerraWatch model pipeline

The main model implementation uses the same data contract as the executed EDA and preparation notebooks. POC originals and experiment variants are preserved. Source-to-main provenance is recorded in [promotion_sources.json](promotion_sources.json).

## Run order

1. Execute [EDA](../../notebooks/01_EDA.ipynb) and [preprocessing](../../notebooks/02_Preprocessing.ipynb), producing `data/processed/{train,val,test}.jsonl`, targets and `references.json`.
2. Open [model experiments](../../notebooks/03_Model_Experiments.ipynb). Its default CPU route shows the recorded comparison and exercises the coordinator with a labelled integration control on one real prepared pair, or an explicit synthetic fixture in a fresh clone.
3. Open [evaluation](../../notebooks/04_Evaluation.ipynb). It replays recorded benchmark results and calculates new control metrics on the selected prepared validation sample. Its scopes and denominators are printed with the results.
4. Install model workers separately, acquire source/weights, configure `configs/pipeline.json`, and enable learned execution or run the CLI below.

The CPU control uses Otsu/template outputs and a defer-only decision. It is a routing demonstration, not a learned BIT/RSCaMa/Qwen result. Stored learned-model measurements retain their original populations and run provenance.

## Environments

Use Python 3.11 or 3.12 for model environments. Run commands from the repository root. Commands below are PowerShell; on Linux/Colab use `bin/python` instead of `Scripts/python`.

```powershell
python -m venv .venv-pipeline
.venv-pipeline/Scripts/python -m pip install -r requirements-pipeline.txt
python -m venv .venv-detector
.venv-detector/Scripts/python -m pip install -r requirements-detector.txt
python -m venv .venv-captioner
.venv-captioner/Scripts/python -m pip install -r requirements-captioner.txt
.venv-captioner/Scripts/python -m pip install --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
python -m venv .venv-referee
.venv-referee/Scripts/python -m pip install -r requirements-referee.txt
python scripts/setup_model_sources.py
```

Install the PyTorch build appropriate to the target GPU if the default wheel does not match its CUDA runtime. Keep Transformers 4.39.3 in the captioner environment separate from the referee's recorded 5.17.0 profile. `requirements-model-evaluation.txt` adds optional COCO-caption metrics. Those metrics use Java for PTB tokenisation. The reference-kernel RSCaMa path avoids compiling Mamba CUDA extensions. Each subprocess isolates compatibility patches and upstream module names.

## Weights and configuration

Set `detector.python`, `captioner.python` and `referee.python` in [pipeline.json](../../configs/pipeline.json) to their installed interpreters. Paths are resolved from the repository root. BIT takes a three-class task checkpoint, not an unadapted binary public checkpoint.

Released RSCaMa can be downloaded with its pinned revision and checksum:

```powershell
.venv-captioner/Scripts/python scripts/fetch_model_assets.py --rscama --captions PATH/LevirCCcaptions.json
```

That optional vocabulary reconstruction reproduces the upstream official-split vocabulary, including evaluation-caption vocabulary exposure. For controlled training, the main captioner builds a vocabulary using training captions alone and embeds it in the checkpoint; set `captioner.vocab` to `null` for that checkpoint. Never substitute a vocabulary with the same size but different token indices.

Qwen downloads its pinned configuration revision through Hugging Face on first inference and then uses the local cache. Its conditional three-code probability is an operating score, not calibrated factual correctness. `device_map: auto` with `max_gpu_memory`/`max_cpu_memory` permits explicit CPU offloading on smaller GPUs.

```powershell
.venv-pipeline/Scripts/python scripts/check_model_setup.py --config configs/pipeline.json --environments --strict
```

The preflight checks configured paths/interpreters; a learned smoke run validates checkpoint execution. The recorded friend-side raw run files and weights are listed by hash in the evidence proof and are separate from the committed source. Re-execution generates new auditable run artifacts.

## Training

```powershell
python scripts/train_models.py detector --config configs/detector_training.json --python .venv-detector/Scripts/python
python scripts/train_models.py captioner --config configs/captioner_training.json --python .venv-captioner/Scripts/python
```

Detector: fixed MCI source partitions, staged ResNet training, focal/Dice loss, AdamW, cosine decay, paired flips, accumulation and validation macro-IoU checkpoint selection. Captioner: training-only vocabulary, CLIP transfer, trainable fusion/decoder, cross-entropy over all reference captions, Adam and validation corpus BLEU-4 selection. Images are streamed through the existing archive-aware loader. Training creates a new output directory and writes configuration, history and selected checkpoint pointers. Set the selected paths from `best.json` in the inference config. CLI flags allow short fit diagnostics without altering the full profiles.

## End-to-end inference

```powershell
.venv-pipeline/Scripts/python scripts/run_pipeline.py --t1 earlier.png --t2 later.png --config configs/pipeline.json --pair-id scene_01
```

Outputs under `outputs/pipeline_runs/<run>/` include requests, responses, logs, predicted mask, crop evidence and `report.json`. Input hashes and model metadata accompany the output. The coordinator checks class masks, temporal dimensions, claim localisation, exact-count evidence and the acceptance gate. Errors remain in the report and never become fabricated predictions. Existing run directories are preserved.

## Batch captions and evaluation

```powershell
.venv-captioner/Scripts/python scripts/caption_pairs.py --split val --limit 100 --output outputs/model_evaluation/captions.jsonl
.venv-pipeline/Scripts/python scripts/evaluate_models.py captions --split val --predictions outputs/model_evaluation/captions.jsonl --output outputs/model_evaluation/caption_metrics
.venv-pipeline/Scripts/python scripts/evaluate_models.py detector --split val --predictions outputs/model_evaluation/masks.jsonl --output outputs/model_evaluation/detector_metrics
```

Detector prediction JSONL: `{"pair_id":"levir_mci/val/val_000001","mask_path":"masks/val_000001.png"}`. Paths are relative to the prediction JSONL or absolute. Masks must match the prepared target dimensions and class indices. Caption JSONL contains `pair_id`, `raw_text`, `run_id`, `status`. Duplicate and out-of-split IDs are rejected. Caption evaluation uses automatic reference proxies, with pair-bootstrap intervals; `--text-metrics` adds BLEU/ROUGE/CIDEr using the separate evaluator dependencies. Reported historical native-resolution metrics retain their original protocols; new prepared-resolution evaluation prints its own scope. Road/building class-mean IoU and binary any-change IoU are separate fields.

## Reproducibility and handover

```powershell
.venv-pipeline/Scripts/python -m pytest -q
python scripts/build_model_notebooks.py
python scripts/run_model_notebooks.py --python .venv-pipeline/Scripts/python
python scripts/package_model_run.py --run outputs/pipeline_runs/RUN_ID --output outputs/predictions/RUN_ID_results.zip
```

The export includes files and SHA-256 hashes, Python version and Git revision. Run directories and datasets stay out of Git. Commit the small run receipt, metrics and selected example outputs for assessed experiments, or provide a versioned retrievable artifact link alongside their hashes. A hash alone does not make the underlying file downloadable.

The report-to-code index is [Milestone 3 implementation map](../milestone3/Implementation.md). The original protocol commands remain under [captioning experiments](../../POC/execution/README.md) and [local benchmark evidence](../../POC/milestone3/sarthak/Benchmark_Report.md).
