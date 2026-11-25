# TerraWatch Notebooks

Run the data-preparation notebooks first, followed by the Milestone 4 full-data training, captioning, and verification notebooks:

## Data Preparation & Baseline Analysis
1. `01_EDA.ipynb` — coverage, native dimensions, label/pixel distributions, small changes, missing joins, image variation, spatial density, duplicate/leakage evidence, outliers, correlations, semantic transitions, and caption/instruction variants.
2. `02_Preprocessing.ipynb` — deterministic exclusions, target conversion, fixed source partitions, task-specific manifests, complete loader checks, and real input batches.
3. `03_Model_Experiments.ipynb` — baseline model experiment replay and fresh CPU controls.
4. `04_Evaluation.ipynb` — baseline evaluation summaries and visual inspections.

## Milestone 4 Full-Data Training, Optimization & Evidence Review
5. `TerraWatch_M4_FullData_Detector_Training.ipynb` — reproduces the 24-run full-data LEVIR-MCI detector study (`6,444` train / `1,242` val / `1,858` held-out test), 5-architecture comparison (`BIT-ResNet18`, `Siam-ResNet18-UNet`, `EF-UNet`, `FC-Siam-Diff`, `ChangeFormer-B2`), 3-seed stability, 10-condition corruption robustness, and zero-shot SECOND-CC transfer (`scripts/train_detector_full.py`, `scripts/eval_robustness_transfer.py`).
6. `TerraWatch_M4_FullData_RSCaMa_Captioner.ipynb` — reproduces the full-data RSCaMa paired captioner experiments (`32,220` train captions, strict `388`-word train-only vocabulary, exact Triton fused selective-scan kernel), `C01`–`C07` ablations, 3-seed stability on selected **`C06_rscama_frozen_clip`**, and 1,858-pair held-out Test evaluation (`scripts/train_captioner_full.py`).
7. `TerraWatch_M4_Verification_and_Evidence_Review.ipynb` — replays the 120-pair (147-claim) `Qwen3-VL-4B-Instruct` verification comparison, visualizes all 8 authentic end-to-end traces from `docs/milestone4/evidence/full_data_m4/verification/e2e_traces/`, and launches the interactive Evidence Review UI (`python -m src.app.server docs/milestone4/evidence/full_data_m4/verification/e2e_traces --port 8765`).

> Earlier bounded Colab prototypes (`TerraWatch_M4_BIT_Training.ipynb`, `TerraWatch_M4_Learned_Demo.ipynb`, `drive_members.py`) are preserved for continuity alongside the full-data Milestone 4 notebooks above.

## Google Colab & Local Execution

- **Replay mode (CPU, zero extra setup):** All three Milestone 4 notebooks read the committed run receipts under `docs/milestone4/evidence/full_data_m4/` directly.
- **Full GPU training & audit:** Set `RUN_TRAINING = True` (or `RUN_EVAL = True` / `RUN_AUDIT = True`) in the notebook cells on a CUDA GPU (RTX 4060, Colab T4, or Colab L4) with `TERRAWATCH_MCI_DIR` pointing to the extracted LEVIR-MCI dataset.
- **Interactive Evidence Review UI (local):**
  ```powershell
  python -m src.app.server docs/milestone4/evidence/full_data_m4/verification/e2e_traces --port 8765
  ```
- **Re-aggregate receipts, rebuild figures, or rebuild PDF deliverables:**
  ```powershell
  python scripts/aggregate_m4_results.py
  python scripts/build_m4_figures.py
  python scripts/build_m4_deliverables.py
  ```

See [the training-data plan](../docs/data/Training_Data_Plan.md) and [the pipeline guide](../docs/data/Pipeline.md) for dataset contracts, schemas, and script entry points.
