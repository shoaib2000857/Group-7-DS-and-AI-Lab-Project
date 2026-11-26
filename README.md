# TerraWatch — Evidence-Grounded Satellite Change Analysis

**IIT Madras BS Data Science & AI Lab (BSDA4001) — Group 7**  
*Priyanshu Agarwal · Sarthak Saini · Sujal Raj Pradhan · Shoaib*

TerraWatch compares co-registered satellite image pairs captured at two timestamps ($T_1$ earlier, $T_2$ later), detects pixel-level road and building changes, generates natural-language change descriptions, verifies each atomic claim against visual and structural evidence, and exposes full visual and cryptographic provenance inside an interactive **Evidence Review** web interface.

---

## System Architecture

| Stage | Model / Component | Role | Primary Implementation |
|---|---|---|---|
| **1. Change Detection** | **BIT-ResNet18** (`D03_bit_lr3e4`) | 3-class pixel change segmentation (`background`, `road`, `building`) | `src/models/detector_architectures.py` · `scripts/train_detector_full.py` |
| **2. Paired Captioning** | **RSCaMa** (`C06_rscama_frozen_clip`) | CLIP ViT-B/32 + bidirectional Mamba + Transformer decoder (10× accelerated via exact Triton selective-scan kernel) | `src/models/captioner.py` · `scripts/train_captioner_full.py` |
| **3. Claim & Witness** | Claim Parser + OpenCV Structural Witness | Extracts `(object, action, location, quantity)` claims and contextual ≥32px T1/T2 crops with structural diagnostics | `src/models/claims.py` · `src/evidence/crop_extractor.py` · `src/evidence/visual_inspector.py` |
| **4. Factual Verification** | Frozen **Qwen3-VL-4B-Instruct** + Safety Gate | Restricted-token `{accept, reject, defer}` scoring ($\tau = 0.70$) + deterministic count/evidence safeguards | `src/inference/verification_pipeline.py` · `src/inference/pipeline.py` · `scripts/run_verification_audit.py` |
| **5. Human Evidence Review** | Standalone HTTP + SQLite Review App | Inspects T1/T2, masks, overlays, crops, probabilities, and SHA-256 provenance; appends immutable human overrides & JSON export | `src/app/server.py` · `src/app/review.html` |

---

## Milestone 4 Headline Results (Full-Data LEVIR-MCI: 6,444 Train / 1,242 Val / 1,858 Test)

- **Change Detection (24 full-data runs across 5 architectures + 3-seed stability):**
  - Selected **BIT-ResNet18 (`D03_bit_lr3e4`, SHA-256 `05345df154b4`)**: **74.54%** Val Change-Class mIoU (3-seed mean **74.23% ± 0.28%**), **75.93%** Held-Out Test Change mIoU (**86.83%** Binary F1 vs. 14.20% Otsu and 0.00% Always-Background).
  - Outperforms top CNN baseline `Siam-ResNet18-UNet` (`75.02%` val / `76.73%` clean test) decisively under degradation and domain shift: **+12.38%** at Blur $r=1.0$ (`70.94%` vs. `58.56%`), **+30.36%** at Blur $r=2.0$ (`38.22%` vs. `7.86%`), and **+5.46% to +8.50%** on zero-shot **SECOND-CC** transfer (`23.17%` Any-F1 / `30.55%` Bldg-F1 on 1,142 test pairs).
- **Paired Change Captioning (9 full-data controlled runs, 32,220 train captions, 388-word train-only vocab):**
  - Selected **RSCaMa Frozen CLIP (`C06_rscama_frozen_clip`, SHA-256 `c5b80d6bf6b6`)**: **0.6358** Val BLEU-4 (3-seed mean **0.6297 ± 0.0183**), **3.32%** Val Unchanged FA, and **0.6128 BLEU-4 / 0.7326 ROUGE-L / 3.57% Unchanged FA / 11.78% Missed Changes** on Held-Out Test (1,858 pairs).
- **Factual Verification & Authentic End-to-End Traces (120 stratified val pairs / 147 claims):**
  - Full-data BIT Mask-Only Checker achieves **91.38%** precision, **99.07%** recall, and **62.96%** contradiction detection at zero VLM calls, compared against Direct Qwen3-VL (`82.43%`) and 4-View Reinspection (`72.41%`).
  - **8 authentic end-to-end satellite traces** covering genuine model-`accept`, model-`reject`, and model-`defer` outcomes are archived under `docs/milestone4/evidence/full_data_m4/verification/e2e_traces/`.

---

## Quick Start

### 1. Launch the Interactive Evidence Review UI (Real Full-Data Traces)

```bash
python -m pip install -r requirements-pipeline.txt
python -m src.app.server docs/milestone4/evidence/full_data_m4/verification/e2e_traces --port 8765
```

Open `http://127.0.0.1:8765` in your browser to inspect all 8 authentic full-data satellite traces, review masks/overlays/crops/probabilities, submit human flags or overrides to SQLite, and export the audit trail via `/api/export`.

### 2. Run Automated Tests

```bash
python -m pytest -q
```

### 3. Train, Evaluate & Rebuild Deliverables

```bash
# Full-data detector training (selected D03 BIT-ResNet18 configuration)
python scripts/train_detector_full.py --run-id D03_bit_lr3e4 --arch bit_resnet18 --head-lr 3e-4 --backbone-lr 3e-5 --epochs 35 --batch-size 16 --accum-steps 2 --loss focal_dice --scheduler cosine --freeze-epochs 5 --aug flip

# Full-data RSCaMa captioner training (selected C06 Frozen CLIP configuration, Triton SSM kernel)
python scripts/train_captioner_full.py --run-id C06_rscama_frozen_clip --optimizer adamw --lr 1e-4 --weight-decay 0.01 --scheduler cosine --freeze-encoder --epochs 8 --batch-size 12 --amp fp16

# Corruption robustness ladder, held-out Test (1,858), Otsu/BG controls, and SECOND-CC transfer
python scripts/eval_robustness_transfer.py --arch bit_resnet18 --checkpoint <path/to/best.pt> --manifests-dir <manifests> --extracted-root <dataset_root> --output <report.json>

# 120-pair Qwen3-VL verification study & authentic E2E trace exporter
python scripts/run_verification_audit.py --detector-arch bit_resnet18 --detector-ckpt <best.pt> --pred-captions-jsonl <val_best_captions.jsonl> --captioner-run-json <run.json> --manifests-dir <manifests> --extracted-root <dataset_root> --output-dir <out_dir>
```

---

## Repository Structure

| Path | Description |
|---|---|
| `reports/` | Official Milestone 1–4 reports (`Milestone4_Report.md`, `DSAI Lab_Group 7_Milestone 4.pdf`) and publication figures (`reports/figures/m4_*.png`) |
| `presentation/` | Milestone review slide decks (`DSAI Lab_Group 7_Milestone 4_Slides.pdf`) |
| `src/` | Core Python package: detector architectures, captioner/claim parser, structural witness, verification pipeline, and Evidence Review web server (`src/app/`) |
| `scripts/` | Full-data training, Triton SSM captioning, robustness/transfer evaluation, verification audit, and figure/PDF build CLIs |
| `notebooks/` | Reproducible EDA, preprocessing, full-data detector training, full-data RSCaMa captioning, and verification/review notebooks |
| `configs/` | Verified full-data training and inference configuration profiles (`detector_training.json`, `captioner_training.json`, `pipeline.json`) |
| `docs/` | Data governance specs, model contracts, Milestone 4 walkthrough (`docs/milestone4/Walkthrough.md`), and auditable experiment receipts + 8 E2E traces |
| `tests/` | Automated unit and integration tests covering data contracts, mask encoding, contextual crop sizing, count gating, and Review UI persistence |
| `POC/` | Historical milestone proofs-of-concept and raw experimental archives |

---

## Milestone Reports & Presentations

- [Milestone 1 Report](reports/Milestone1_Report.md) · [Milestone 1 Slides](presentation/DSAI%20Lab_Group%207_Milestone%201_Slides.pdf)
- [Milestone 2 Data Preparation Report](reports/Milestone2_Report.md)
- [Milestone 3 Model Selection & Architecture Report](reports/Milestone3_Report.md)
- **[Milestone 4 Model Training, Optimization & Evidence Review Report](reports/Milestone4_Report.md)** · **[Milestone 4 Report PDF](reports/DSAI%20Lab_Group%207_Milestone%204.pdf)** · **[Milestone 4 Slides PDF](presentation/DSAI%20Lab_Group%207_Milestone%204_Slides.pdf)** · **[Milestone 4 Walkthrough](docs/milestone4/Walkthrough.md)**
