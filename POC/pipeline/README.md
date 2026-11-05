# Selected pipeline: model workers and training

The coordinator implements the M3 contracts: ordered images, BIT class mask, RSCaMa caption, Sujal's claim parser, Priyanshu's crop/evidence module, Qwen referee and gated report JSON. Every run gets its own directory. Existing run directories are rejected rather than cleared.

## Environments and assets

Use the root `requirements-data.txt` and `requirements-evidence.txt` for the coordinator and CPU tests. Model workers have separate Python interpreters, configured in `pipeline.json`:

- Detector environment: PyTorch, torchvision, einops, NumPy and Pillow. `train_detector.py` uses the official BIT code and creates a three-class MCI checkpoint.
- Captioner environment: reuse Sujal's RSCaMa environment in `POC/execution/README.md` (Transformers 4.39.3 and the patched CLIP/reference Mamba kernels). Use a matching released vocabulary or the vocabulary embedded in the subset-trained checkpoint.
- Referee environment: reuse the Colab judge runtime/environment from `POC/benchmarks/colab/`. The returned sessions record Torch 2.11.0, Transformers 5.17.0 and BF16 Qwen inference on an L4. Keep this environment separate from RSCaMa's.

Set each `python` field to its actual environment interpreter. Set the detector/captioner checkpoint fields to existing trained weights. The default paths name the intended artifacts; a missing artifact produces a review/error record, never a fabricated prediction. Source masks and reference captions are not accepted as inference payload fields.

Acquire the pinned public source code:

```powershell
python POC/pipeline/setup_sources.py
```

This checks existing revisions and preserves differing checkouts. It does not reset repositories or download model weights. Model weights use the normal author/Hugging Face mechanisms in the existing model setup instructions.

## Detector training

Run after the main EDA/preprocessing notebooks produce MCI manifests and targets:

```powershell
python POC/pipeline/train_detector.py --source-dir POC/pipeline/cache/BIT_CD --processed data/processed --device cuda
```

The implementation uses three class indices, shared BIT features, focal/Dice loss, staged backbone training, AdamW, cosine decay, gradient accumulation, paired flips and validation macro-IoU checkpoint selection. Defaults are recorded in the report and training config. The script writes an epoch-indexed checkpoint for each improvement and a history JSONL; it never clears an older run. Put the selected path from `best.json` into `pipeline.json`.

## Inference

From the repository root:

```powershell
python POC/pipeline/run_pipeline.py --t1 earlier.png --t2 later.png --config POC/pipeline/pipeline.json --pair-id example_pair
```

The generated `outputs/pipeline_runs/<timestamp>/` contains worker requests/responses/logs, detector mask, paired crops and `report.json`. The final record retains original caption text, source spans, evidence, raw decisions, operating-gate reasons and an accepted-description field. Count/legality checks and missing evidence can require review even when a referee suggests acceptance.

The root test suite exercises routing using explicitly labelled test doubles; those tests do not run GPU model inference. Actual learned-model benchmark evidence remains the original per-module exports. This is stated in the internal consolidation audit so run provenance remains unambiguous.
