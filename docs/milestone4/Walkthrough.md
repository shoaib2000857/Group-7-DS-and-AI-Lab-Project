# TerraWatch — Milestone 4 Code Walkthrough & Execution Guide

This walkthrough follows the logical execution flow of TerraWatch from governed data verification through full-data model training, robustness and transfer testing, factual verification, and the interactive Evidence Review interface.

---

## 1. System Architecture & End-to-End Flow

```text
Ordered Satellite Pair (T1, T2)
   ├─► Stage 1: BIT-ResNet18 Change Detector (3-class mask: background / road / building)
   ├─► Stage 2: RSCaMa Paired Captioner (CLIP ViT-B/32 + Mamba AttentiveEncoder + Transformer Decoder)
   └─► Stage 3: Claim Parser → Contextual 32px+ Crop Extractor → OpenCV Structural Witness
                  └─► Stage 4: Frozen Qwen3-VL-4B-Instruct Referee + Deterministic Safety Gate (τ = 0.70)
                                 └─► Auditable report.json → Interactive Evidence Review UI (SQLite Audit Trail)
```

| Stage | Selected Artifact / Model | Core Code | Measured Result |
|---|---|---|---|
| Data Governance | Strict LEVIR-MCI (6,444 / 1,242 / 1,858) + SECOND-CC (530 / 1,142) | `src/data/`, `notebooks/01_EDA.ipynb`, `02_Preprocessing.ipynb` | 100% SHA-256 verified, zero split leakage |
| Change Detection | `D03_bit_lr3e4` (BIT-ResNet18, SHA-256 `05345df154b4`) | `src/models/detector_architectures.py`, `scripts/train_detector_full.py` | 74.54% Val / **75.93% Test** Chg mIoU (3-seed: 74.23% ± 0.28%), **86.83% Test Bin F1** |
| Paired Captioning | `C06_rscama_frozen_clip` (RSCaMa, 388-word train-only vocab, SHA-256 `c5b80d6bf6b6`) | `scripts/train_captioner_full.py` (exact Triton SSM kernel) | 0.6358 Val / **0.6128 Test BLEU-4** (3-seed: 0.6297 ± 0.0183), **3.57% Test NC FA** |
| Robustness & Transfer | Frozen `D03` vs `A03` (Siam-R18-UNet) + Otsu / Always-BG controls | `scripts/eval_robustness_transfer.py` | BIT beats Siam-UNet by +12.38% (Blur r=1.0), +30.36% (Blur r=2.0), and +5.46% to +8.50% on SECOND-CC |
| Verification & Traces | Frozen `D03` + `C06` + `Qwen/Qwen3-VL-4B-Instruct` | `src/inference/verification_pipeline.py`, `scripts/run_verification_audit.py` | 120 stratified val pairs (147 claims): **91.38%** BIT mask precision, 8 authentic E2E traces |
| Evidence Review UI | Standalone HTTP + SQLite Review Application | `src/app/server.py`, `src/app/review.html` | Immutable model decisions, append-only SQLite overrides, `/api/export` JSON audit |

---

## 2. Logical Walkthrough Sequence

1. **Data Contracts & Split Verification:** Inspect `configs/datasets.yaml`, `notebooks/01_EDA.ipynb`, and `notebooks/02_Preprocessing.ipynb` for strict 6,444 / 1,242 / 1,858 LEVIR-MCI partitioning and explicit 3-class mask encoding (`0 → 0`, `128 → 1`, `255 → 2`).
2. **Multi-Architecture Detector Training (`scripts/train_detector_full.py`, `src/models/detector_architectures.py`):** Walks through the 5 architectures (`BIT-ResNet18`, `Siam-ResNet18-UNet`, `EF-UNet`, `FC-Siam-Diff`, `ChangeFormer`), Focal + Soft Dice loss, 5-epoch warmup unfreezing, AMP mixed precision, and the 24 completed run receipts in `docs/milestone4/evidence/full_data_m4/detector_runs/`.
3. **Full-Data RSCaMa Captioner & Triton Selective-Scan Kernel (`scripts/train_captioner_full.py`):** Walks through the 388-word train-only vocabulary, `_TritonSelectiveScan` (10× T4 speedup with `< 5e-7` gradient match), and the 9 controlled runs + 1,858-pair Test evaluation in `docs/milestone4/evidence/full_data_m4/captioner_runs/`.
4. **Robustness, Non-Learning Controls & SECOND-CC Transfer (`scripts/eval_robustness_transfer.py`):** Demonstrates the 10-condition corruption ladder, Otsu/Always-Background controls, and why `D03_bit_lr3e4` is deployed over `A03_siam_r18_unet`.
5. **Factual Verification & 8 Authentic E2E Traces (`src/inference/verification_pipeline.py`, `scripts/run_verification_audit.py`):** Walks through contextual 32px+ crop extraction (`src/evidence/crop_extractor.py`), singular-article vs. explicit-numeric-count gating (`src/inference/pipeline.py`), Qwen3-VL restricted-token scoring, and the 8 real traces in `docs/milestone4/evidence/full_data_m4/verification/e2e_traces/`.
6. **Live Evidence Review Interface Demo (`src/app/server.py`):**

```bash
python -m src.app.server docs/milestone4/evidence/full_data_m4/verification/e2e_traces --port 8765
```

Open `http://127.0.0.1:8765` to inspect genuine model-accepted (`val_000149`, `val_000353`, `val_000118`), model-rejected (`val_000022`, `val_000081`), and model-deferred (`val_000237`, `val_000040`) satellite traces, record a human review override into SQLite, and download the audit JSON from `/api/export`.

---

## 3. Reproducible Notebooks & Automated Tests

- **Notebooks (`notebooks/`):**
  - `notebooks/TerraWatch_M4_FullData_Detector_Training.ipynb`
  - `notebooks/TerraWatch_M4_FullData_RSCaMa_Captioner.ipynb`
  - `notebooks/TerraWatch_M4_Verification_and_Evidence_Review.ipynb`
- **Regression Suite:** Run `python -m pytest -q` to verify data contracts, mask encoding, contextual crop sizing, count-gating policy, and Evidence Review UI persistence.
