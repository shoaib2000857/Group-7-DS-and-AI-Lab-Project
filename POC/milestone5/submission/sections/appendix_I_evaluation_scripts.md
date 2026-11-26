## Appendix I. Evaluation and Profiling Scripts

All commands run from the repository root. `<manifests>` is the portable manifest directory (`portable_data_receipt.json` and the `mci_*.jsonl`, `second_cc_*.jsonl` files) and `<dataset_root>` is the extracted dataset directory. Checkpoints and raw data are never committed; verify SHA-256 digests (Section 2.3) before use. Detector arguments mirror the committed receipts in `docs/milestone4/evidence/full_data_m4/`.

### I.1 Script and command index

| Purpose | Script | Command |
|---|---|---|
| Environment and asset preflight | `scripts/check_model_setup.py` | `python scripts/check_model_setup.py --config configs/pipeline.json --environments --strict` |
| Acquire pinned model sources (BIT_CD, RSCaMa) | `scripts/setup_model_sources.py` | `python scripts/setup_model_sources.py` |
| Fetch released RSCaMa checkpoint (C00) | `scripts/fetch_model_assets.py` | `python scripts/fetch_model_assets.py --rscama --output models/checkpoints/rscama.pth` |
| Data audit, preparation and verification | `src/data/cli.py` | `python -m src.data.cli audit`, `python -m src.data.cli prepare`, `python -m src.data.cli verify` |
| Train detector (selected D03) | `scripts/train_detector_full.py` | `python scripts/train_detector_full.py --run-id D03_bit_lr3e4 --arch bit_resnet18 --bit-source-dir models/external/BIT_CD --manifests-dir <manifests> --extracted-root <dataset_root> --output-dir runs/detector --head-lr 3e-4 --backbone-lr 3e-5 --epochs 35 --min-epochs 15 --patience 8 --batch-size 16 --accum-steps 2 --loss focal_dice --scheduler cosine --freeze-epochs 5 --aug flip --amp bf16 --seed 42 --resume` |
| Train captioner (selected C06) | `scripts/train_captioner_full.py` | `python scripts/train_captioner_full.py --run-id C06_rscama_frozen_clip --source-dir models/external/RSCaMa --manifests-dir <manifests> --extracted-root <dataset_root> --output-dir runs/captioner --optimizer adamw --lr 1e-4 --weight-decay 0.01 --scheduler cosine --freeze-encoder --epochs 8 --min-epochs 6 --patience 3 --batch-size 12 --amp fp16 --seed 42` |
| Evaluate a trained detector or captioner on Test | `scripts/train_detector_full.py`, `scripts/train_captioner_full.py` | append `--eval-only --checkpoint <best.pt or best.pth> --eval-split test` to the training command (captioner additionally `--vocab <vocab.json>`) |
| Corruption ladder, Test, controls and SECOND-CC transfer | `scripts/eval_robustness_transfer.py` | `python scripts/eval_robustness_transfer.py --arch bit_resnet18 --bit-source-dir models/external/BIT_CD --checkpoint <best.pt> --manifests-dir <manifests> --extracted-root <dataset_root> --output <report.json> --device cuda` |
| 120-pair verification study and E2E traces | `scripts/run_verification_audit.py` | `python scripts/run_verification_audit.py --detector-arch bit_resnet18 --bit-source-dir models/external/BIT_CD --detector-ckpt <best.pt> --pred-captions-jsonl <val_best_captions.jsonl> --captioner-run-json <run.json> --manifests-dir <manifests> --extracted-root <dataset_root> --output-dir <out_dir> --accept-threshold 0.70 --num-audit-pairs 120` |
| Batch captioning of prepared pairs | `scripts/caption_pairs.py` | `python scripts/caption_pairs.py --split val --config configs/pipeline.json --output <captions.jsonl>` |
| Metric computation from predictions | `scripts/evaluate_models.py` | `python scripts/evaluate_models.py detector --split test --predictions <preds.jsonl> --output <metrics.json>` and `python scripts/evaluate_models.py captions --split test --predictions <preds.jsonl> --output <metrics.json> [--text-metrics]` |
| Single-pair end-to-end inference | `scripts/run_pipeline.py` | `python scripts/run_pipeline.py --t1 <t1.png> --t2 <t2.png> --config configs/pipeline.json --output <run_dir>` |
| Aggregate Milestone 4 receipts | `scripts/aggregate_m4_results.py` | `python scripts/aggregate_m4_results.py --evidence-dir docs/milestone4/evidence/full_data_m4` |
| Rebuild Milestone 4 figures | `scripts/build_m4_figures.py` | `python scripts/build_m4_figures.py` |
| CPU replay of Milestone 4 traces | `scripts/replay_milestone4.py` | `python scripts/replay_milestone4.py --device cpu` |
| Package a run with hashes | `scripts/package_model_run.py` | `python scripts/package_model_run.py --run <run_dir> --output <run.zip>` |
| Evidence Review UI | `src/app/server.py` | `python -m src.app.server docs/milestone4/evidence/full_data_m4/verification/e2e_traces --port 8765` |
| Regression tests | `tests/` | `python -m pytest -q` |
| Training cost table (Section 12.1) | `POC/milestone5/scripts/m5_training_cost_table.py` | `python POC/milestone5/scripts/m5_training_cost_table.py` |
| Compute profile (Section 12.2 to 12.6) | `POC/milestone5/scripts/m5_benchmark_compute.py` | `python POC/milestone5/scripts/m5_benchmark_compute.py --device {cpu,mps,cuda} --hardware-label LABEL --ckpt-dir DIR --stages detectors,captioner,witness[,referee] --out POC/milestone5/evidence/compute_profile/` |
| Human-audit aggregation (Appendix J) | `POC/milestone5/scripts/m5_compute_human_audit.py` | `python POC/milestone5/scripts/m5_compute_human_audit.py --raters-dir DIR [--adjudication CSV] [--packet JSONL] [--policy name=col ...] --out DIR` |

### I.2 Profiling invocations

Hardware labels follow Table 2.1. The CUDA profile on the Kaggle T4 includes the referee; the Apple M4 profile runs on `mps` and `cpu`.

```
python POC/milestone5/scripts/m5_benchmark_compute.py --device cuda --hardware-label kaggle_t4 --ckpt-dir <ckpt_dir> --stages detectors,captioner,witness,referee --out POC/milestone5/evidence/compute_profile/
python POC/milestone5/scripts/m5_benchmark_compute.py --device mps  --hardware-label apple_m4_mps --ckpt-dir <ckpt_dir> --stages detectors,captioner,witness --out POC/milestone5/evidence/compute_profile/
python POC/milestone5/scripts/m5_benchmark_compute.py --device cpu  --hardware-label apple_m4_cpu --ckpt-dir <ckpt_dir> --stages detectors,captioner,witness --out POC/milestone5/evidence/compute_profile/
```

### I.3 Output locations

| Output | Path |
|---|---|
| Training cost (JSON, Markdown) | `POC/milestone5/evidence/compute_profile/training_cost.json`, `training_cost.md` |
| Compute profile | `POC/milestone5/evidence/compute_profile/` (one result set per hardware label) |
| Human-audit summary | `--out` directory of `m5_compute_human_audit.py` (planned location `POC/milestone5/evidence/human_audit/`) |
| Milestone 4 receipts | `docs/milestone4/evidence/full_data_m4/` |

Test modules: `tests/test_data.py`, `test_data_preparation.py`, `test_milestone4_pipeline.py`, `test_model.py`, `test_model_metrics.py`, `test_model_pipeline.py`, `test_pipeline.py` and `test_visual_evidence.py`. The data-preparation tests additionally require `pandas` (`requirements-data.txt`).
