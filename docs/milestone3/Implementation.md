# Milestone 3 implementation index

| Report area | Main implementation | Execution instructions |
|---|---|---|
| System workflow and error handling | `src/inference/pipeline.py`, `src/models/workers.py` | `scripts/run_pipeline.py --help` |
| BIT detector and training | `src/training/detector.py`, `configs/detector_training.json` | `scripts/train_models.py detector --config configs/detector_training.json` |
| RSCaMa captioning and training | `src/models/captioner.py`, `src/training/captioner.py` | `scripts/caption_pairs.py --help`, `scripts/train_captioner.py --help` |
| Claim extraction | `src/models/claims.py` | Model experiments notebook |
| Crop and structural evidence | `src/evidence/` | Model experiments notebook |
| Qwen referee and decision gate | `src/models/referee.py`, `src/inference/pipeline.py` | Pipeline command and selected worker config |
| Automatic evaluation | `src/evaluation/` | `scripts/evaluate_models.py --help` |
| Exploratory model/evaluation notebooks | `notebooks/03_Model_Experiments.ipynb`, `notebooks/04_Evaluation.ipynb` | Run top-to-bottom; saved CPU outputs included |
| Recorded benchmark evidence | `docs/milestone3/evidence/` | Original run provenance and benchmark proof |
| Setup and reproducibility | `docs/models/README.md`, `requirements-*.txt` | Separate model environments and pinned upstream sources |
| Run export | `scripts/package_model_run.py` | Downloadable ZIP with hashes and Git revision |

See [the complete model run guide](../models/README.md). POC originals remain preserved. CPU notebook controls are labelled separately from historical learned benchmarks. The host validation runs the reusable CPU path; learned training/inference uses the configured GPU environments and task checkpoints.
