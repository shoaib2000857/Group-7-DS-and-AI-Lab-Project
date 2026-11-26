# TerraWatch — Milestone 5 Master Execution Plan

**Course:** IIT Madras BS Data Science & AI Lab (`BSDA4001`) — Group 7  
**Project:** TerraWatch — Evidence-Grounded Satellite Change Detection, Captioning & Verification  
**Scope:** Complete Evaluation, Calibration, Human Audit, Error Analysis, Robustness Stress Testing, Systems Benchmarking, Deployment Packaging & Final Report/Slides

---

## 1. Strategic Goal: What Makes This Milestone 5 Stand Out

Because Milestone 4 already completed full-data training on **100% of the LEVIR-MCI cohort** (`6,444` train / `1,242` val / `1,858` held-out test pairs across 24 detector runs, 9 controlled RSCaMa runs + released benchmark, and a 120-pair `Qwen3-VL-4B-Instruct` audit), Milestone 5 does **not** need to spend time catching up on training. Instead, we execute **eight high-impact evaluation, calibration, human-audit, and systems-engineering workstreams** that directly answer every question TAs and faculty ask during final evaluation:

1. **Closes our #1 Milestone 4 discovery scientifically:** In M4, we discovered that our full-data BIT Mask-Only Checker (`91.38%` precision, `99.07%` recall) outperformed naive `Qwen3-VL` override (`72.41%`–`82.43%`) on in-ontology claims. In M5, we build, calibrate on `val`, and blindly evaluate on **held-out LEVIR-MCI `test`** our **Calibrated Mask-First Hybrid Verifier** — combining BIT's spatial precision with selective `Qwen3-VL-4B-Instruct` reasoning on out-of-ontology and borderline claims.
2. **Delivers genuine blind dual-rater human evaluation:** All 4 team members annotate stratified claims blindly inside our Evidence Review UI, yielding real **Inter-Annotator Agreement (Cohen's $\kappa$)**, **human-verified hallucination rate**, and **claim-level groundedness** — directly fulfilling Generative AI / LLM rubric criteria that almost no student team quantifies rigorously.
3. **Provides statistical rigor & full classification/segmentation/generation metrics:** Pixel-level **ROC-AUC & PR-AUC**, **3×3 normalized confusion matrices**, **BERTScore / METEOR / CIDEr / BLEU-1..4 / ROUGE-L**, **paired 1,000-sample bootstrap 95% confidence intervals**, and **McNemar / Wilcoxon significance tests** across all 1,858 held-out Test pairs.
4. **Delivers deep visual root-cause error analysis & edge-case stress testing:** Quantified failure taxonomy with TP/FP/FN visual strips plus synthetic cloud occlusion, cross-sensor radiometric shifts, registration jitter, and half-resolution degradation tests.
5. **Delivers production systems profiling:** Exact P50/P95/P99 latency breakdown by module, peak VRAM, GFLOPs, model footprint, throughput, and cloud cost per 1,000 satellite pairs across RTX 4060, Colab T4, Colab L4, and CPU — proving the real-world ROI of selective VLM triggering.

---

## 2. The 8 Milestone 5 Workstreams (WS1–WS8)

### WS1 — Complete Held-Out Test Evaluation & Statistical Significance (`1,858` LEVIR-MCI Test Pairs)
**Rubric Coverage:** Sections 4, 5, 6, 7, 8 + Appendices (Metric Tables, Confusion Matrices, ROC/PR Curves)

- **Detector Full Test Benchmark (`scripts/m5_eval_detectors_test.py`):**
  - Evaluate all 5 trained architectures on all **1,858 held-out LEVIR-MCI Test pairs** (same frozen checkpoints, zero re-tuning):
    1. `D03_bit_lr3e4` (Selected BIT-ResNet18)
    2. `A03_siam_r18_unet` (Strong CNN baseline)
    3. `A01_ef_unet` (Early-fusion UNet)
    4. `A02_fc_siam_diff` (Lightweight Siamese CNN)
    5. `A04_changeformer` (Pure Transformer baseline)
    6. Non-learning baselines: `Otsu Change Threshold` and `Always-Background`
  - Compute for every model:
    - Pixel Accuracy, Macro IoU, Change-Class mIoU, Road IoU, Building IoU, Per-Class Dice / Precision / Recall / F1
    - **Pixel-level ROC-AUC and PR-AUC** (from exact softmax change probability $1 - p_{\text{bg}}$) and save downsampled curve arrays for plotting
    - **3×3 Pixel Confusion Matrix** (`Background`, `Road`, `Building`) — raw counts and row-normalized percentages
    - **Area-Stratified Test Breakdown:** `Unchanged (0 px)`, `Tiny (1–250 px)`, `Small (251–1,000 px)`, `Medium (1,001–4,000 px)`, `Large (>4,000 px)`
    - **Unchanged-Scene False Alarm Operating Curve** across thresholds `{1, 16, 50, 100, 200, 500}` px
- **Captioner Full Test & Semantic Metrics (`scripts/m5_eval_captioners_test.py`):**
  - Evaluate on all **1,858 held-out Test pairs**:
    1. `C06_rscama_frozen_clip` (Selected, plus 3-seed `s0`/`s42`/`s2026` Test spread!)
    2. `C00_rscama_released` (Upstream released benchmark)
    3. `C01_rscama_ref` (Unfrozen Adam reference arm)
    4. `C04_rscama_adamw_cos` (Unfrozen AdamW cosine arm)
    5. Naive baselines: `Always-"no change"` baseline & `Nearest-Neighbor / Template-from-Mask` baseline
  - Compute: **BLEU-1, BLEU-2, BLEU-3, BLEU-4**, **ROUGE-L**, **METEOR**, **CIDEr**, **BERTScore (Precision / Recall / F1 using `roberta-large` or `microsoft/deberta-xlarge-mnli`)**, **Unchanged Hallucination Rate (NC FA%)**, **Changed Miss Rate (%)**, and **Object-Class Hallucination Rate** (caption mentions road/building when GT mask has 0 pixels of that class).
- **Statistical Significance (`scripts/m5_statistical_tests.py`):**
  - 1,000-resample paired bootstrap **95% confidence intervals** on Test Change mIoU, Binary F1, BLEU-4, and BERTScore.
  - Paired **Wilcoxon signed-rank test** (per-pair F1 / BLEU-4) and **McNemar's test** (per-pair unchanged false alarms & claim verification accuracy) with exact $p$-values.
- **Done Criteria:** All JSON/CSV receipts archived under `POC/milestone5/evidence/test_full_metrics/`; zero train/val leakage verified.

---

### WS2 — Calibrated Mask-First Hybrid Verifier & Held-Out Test Verification Audit
**Rubric Coverage:** Sections 5, 6, 7, 9 (Ablations), 15 + Appendices (Risk-Coverage & Calibration Curves)

- **Architectural Upgrade (`src/inference/hybrid_verifier.py` or extension of `src/inference/verification_pipeline.py`):**
  - Implement **Mask-First Hybrid Routing**:
    - **Tier 1 (In-Ontology `road`/`building`, high-confidence mask agreement):** Accept/Reject directly via calibrated BIT spatial support + OpenCV structural witness (zero VLM call → fast, high precision).
    - **Tier 2 (Out-of-ontology objects like `trees`/`vegetation`/`parking`, hedged claims, exact numeric counts, or borderline mask-witness conflict):** Route selectively to frozen `Qwen3-VL-4B-Instruct` with calibrated decision thresholds $(\tau_{\text{accept}}, \tau_{\text{reject}})$ tuned strictly on the **120-pair Validation audit**.
  - Add **Calibration Analysis**: Expected Calibration Error (**ECE**), Brier score, reliability diagram, and **Risk-Coverage AUC (AURC)** across thresholds $\tau \in [0.50, 0.90]$.
- **Strict Ablation Ladder (Same Fixed Masks & Captions — Zero Confounding):**
  1. `Unverified Captioner` (no checking)
  2. `BIT Mask-Only Checker` (no VLM, no witness)
  3. `OpenCV Witness-Only Checker` (no mask, no VLM)
  4. `Direct Qwen3-VL-4B` (full images only, no mask/crops)
  5. `4-View Reinspection (All Claims)` (unconditional VLM)
  6. `Disagreement-Triggered Qwen3-VL` (M4 baseline)
  7. **`Calibrated Mask-First Hybrid Verifier` (Proposed M5 System)**
- **Blind Held-Out Test Verification Audit (`scripts/m5_run_test_verification.py`):**
  - Lock all routing rules and thresholds from `val`, then evaluate **once** on **150 stratified held-out LEVIR-MCI *Test* pairs** (`50` unchanged, `35` building-only, `20` road-only, `25` mixed, `20` small-change → ~180–200 atomic claims).
  - Export **10 authentic held-out Test E2E traces** for the Evidence Review UI & Appendices.
- **Done Criteria:** Hybrid verifier strictly improves or matches `91%+` supported precision while catching out-of-ontology hallucinations and cutting VLM invocations by $\ge 60\%$ vs unconditional reinspection, verified on held-out Test.

---

### WS3 — Blind Dual-Rater Independent Human Evaluation & Groundedness Audit
**Rubric Coverage:** Sections 4, 5 (`Human evaluation`, `Hallucination rate`, `Groundedness`), 6, 10, 15

- **Audit Cohort:** `120` claims stratified across `accept`, `reject`, `defer`, and all 5 scene strata (drawn from the held-out Test verification cohort, plus cross-check on the 147-claim Val template).
- **Blind Dual-Rater Design (Zero Label Leakage):**
  - Every claim is reviewed independently by **2 teammates** using the Evidence Review UI (or clean blind annotation sheet showing only `T1`, `T2`, zoomed contextual crops, and the atomic claim text — **hiding** the model's accept/reject verdict and the other rater's label).
  - 3-class rubric: `supported`, `contradicted`, `unverifiable` + optional error tag (`hallucinated_object`, `wrong_count`, `wrong_location`, `missed_change`, `shadow_or_seasonal_confusion`).
  - Disagreements are adjudicated by a 3rd teammate to form the **Gold Human Consensus Set**.
- **Metrics Computed (`scripts/m5_compute_human_audit.py`):**
  - **Inter-Annotator Agreement:** Cohen's $\kappa$, percent exact agreement, and 3×3 rater confusion matrix.
  - **Human-Verified Groundedness Rate (%)** & **Human-Verified Hallucination Rate (%)** of `Unverified Captioner` vs `Mask-Only` vs `Mask-First Hybrid Verifier`.
  - **Proxy-vs-Human Alignment:** Exact agreement between our automated GT-mask proxy label and human consensus (validates our automated benchmark scientifically!).
- **Done Criteria:** ≥ 100 dual-rated claims with recorded Cohen's $\kappa$, adjudicated gold consensus CSV, and JSON summary under `POC/milestone5/evidence/human_audit/`.

---

### WS4 — Deep-Dive Error Analysis & Visual Failure Taxonomy
**Rubric Coverage:** Section 10 (Error Analysis) + Appendices (Failure Case Examples, Sample Predictions)

- **Automated Error Mining (`scripts/m5_error_analysis.py`):**
  - Extract from the 1,858 held-out Test pairs:
    - Top 20 worst Detector False Positives (unchanged pairs with highest predicted changed pixels)
    - Top 20 worst Detector False Negatives (changed pairs with lowest recall / IoU)
    - Top 15 Road-vs-Building class-confusion pairs
    - Top 20 Captioner hallucination & missed-change cases
    - Every False Accept and False Reject from the Held-Out Test Verification Audit
- **Quantified Failure Taxonomy:** Categorize failure cases by root cause with exact counts and % share:
  1. *Sub-100px tiny/occluded structures*
  2. *Solar angle / cast shadow & seasonal illumination shifts*
  3. *Road-vs-driveway / paved-lot semantic boundary ambiguity*
  4. *Vegetation clearing / bare soil transitions outside 2-class mask ontology*
  5. *Exact numeric count hallucinations (`two` vs `three` buildings)*
  6. *Parallax / minor co-registration edge halos around tall roofs*
- **Visual Galleries:** Generate high-DPI visual strips (`T1 | T2 | GT Mask | Pred Mask | Color-Coded TP/FP/FN Error Map | Caption + Verifier Verdict`) for representative success cases AND each failure mode.
- **Done Criteria:** Quantified taxonomy CSV + 2 publication-ready visual galleries (`m5_success_gallery.png` & `m5_error_taxonomy_gallery.png`).

---

### WS5 — Extended Robustness, Edge-Case Stress Testing & Cross-Dataset Domain-Gap Analysis
**Rubric Coverage:** Section 11 (Model Robustness) + Section 13 (Limitations)

- **Extended Stress Suite on Held-Out Test (`scripts/m5_stress_tests.py`):**
  - Run `D03_bit_lr3e4` vs `A03_siam_r18_unet` across **14 stress conditions** on held-out Test (`1,858` pairs):
    - Core M4 10-condition ladder on **Test** (clean, brightness ±25%, JPEG q30/q15, blur r1.0/r2.0, noise $\sigma=15$, rot 2°, shift 4px)
    - **4 New M5 Edge-Case / Adversarial Stress Tests:**
      1. *Synthetic Cloud/Haze Occlusion* (soft translucent white cloud patch covering 15% of T2)
      2. *Severe Co-Registration Drift* (`shift_8px` and `rot_5deg`)
      3. *Cross-Sensor Radiometric Mismatch* (gamma 1.4 + color-temperature shift on T2 only)
      4. *Low-Resolution Sensor Degradation* (`0.5×` bilinear downscale + upscale simulating lower-GSD satellites)
- **Captioner Robustness Probe (`200` stratified Test pairs):** Measure how `C06_rscama_frozen_clip` BLEU-4 & NC FA% degrade under blur, brightness shift, and synthetic haze.
- **SECOND-CC Domain-Gap Dissection:** Quantify why zero-shot `D03` achieves `30.55%` Building-Proxy F1 vs `23.17%` Any-Change F1 on SECOND-CC Test (`1,142` pairs) by breaking down recall across SECOND-CC's native semantic classes (`building`, `non-vegetated ground`, `low veg`, `tree`, `water`, `playground`) — proving the gap is ontology mismatch rather than detector collapse.
- **Done Criteria:** Full stress-test JSON/tables + radar/bar comparison figure (`m5_extended_robustness.png`) + SECOND-CC class breakdown plot.

---

### WS6 — Computational Performance, Memory, Throughput & Cost Profiling
**Rubric Coverage:** Section 12 (Computational Performance)

- **Hardware Profiling Script (`scripts/m5_benchmark_compute.py`):**
  - Benchmark all 5 detectors, `C06_rscama_frozen_clip` (comparing Triton fused kernel vs Python reference loop!), OpenCV witness, and `Qwen3-VL-4B-Instruct` across:
    - **Local Laptop GPU:** NVIDIA RTX 4060 Laptop (8 GB VRAM)
    - **Cloud GPUs:** Google Colab T4 (16 GB) and L4 (24 GB)
    - **CPU Fallback:** Multi-core CPU (for detector + witness + UI review)
  - Record exact measurements (50 warmup + 200 timed passes where feasible):
    - Total & trainable parameters (M), checkpoint size on disk (MB), GFLOPs (via `thop`/`fvcore` or torch flop counter)
    - Latency per satellite pair: **Mean, P50, P95, P99 (ms)**
    - End-to-end throughput (**pairs/sec** at batch=1 and batch=16)
    - Peak GPU VRAM (`torch.cuda.max_memory_allocated()`, MB) & average GPU utilization (%)
    - Total training wall-clock time across all 33 full-data runs + Colab Compute Unit (CU) & estimated cloud dollar cost per 1,000 processed pairs (comparing **Unconditional VLM** vs **Mask-First Selective Hybrid**!).
- **Done Criteria:** Verified `compute_profile.json` + stacked latency/cost chart (`m5_compute_profile.png`).

---

### WS7 — Deployment Packaging, Single-Command Inference CLI & Reproducibility Audit
**Rubric Coverage:** Sections 2, 16 (Readiness for Deployment) + Appendix (Model Config & Eval Scripts)

- **Production Single-Pair / Batch Inference CLI (`scripts/run_terrawatch_inference.py`):**
  - Clean end-to-end command taking any `--t1 <img1> --t2 <img2> --out <dir>` (or manifest batch), running `D03` → `C06` → `OpenCV Witness` → `Calibrated Mask-First Hybrid Verifier`, writing an Evidence Review UI-ready bundle (`report.json`, `mask_overlay.png`, `crops/`, SHA-256 provenance), and opening directly in `python -m src.app.server <out_dir>`.
- **Checkpoint Registry & Verification Script (`configs/model_registry.json`):**
  - Exact filenames, SHA-256 digests, download/Drive pointers, and integrity verifier (`--verify-checksums`) so any reviewer or TA can reproduce every number without guessing weights.
- **Full Pytest & Smoke Verification:** Expand automated tests to cover the hybrid verifier routing, calibration metrics, human-audit parser, and inference CLI contract.

---

### WS8 — Comprehensive 16-Section Milestone 5 Report, Appendices & Final Defense Slide Deck
**Rubric Coverage:** ALL 16 Sections + All 10 Suggested Appendices

- **Report Authoring & PDF Pipeline:**
  - Write `POC/milestone5/submission/Milestone5_Report.md` covering every single bullet of Sections 1–16 plus Appendices A–J (see [REPORT_OUTLINE_AND_EVIDENCE_MAP.md](REPORT_OUTLINE_AND_EVIDENCE_MAP.md)).
  - Style & tone rules (strictly enforced):
    - **Formal, confident capstone presentation tone** — present completed verified results cleanly with zero internal TODOs, scratch notes, or pending review disclaimers inside the submitted PDF/slides.
    - **Crisp typography:** `Segoe UI` body/headers + subtly scaled `Consolas` inline tags + proportional non-wrapping dark-navy/teal tables + high-DPI (180 DPI) publication figures matching Group 7's Navy/Teal/Lime palette.
  - Build `DSAI Lab_Group 7_Milestone 5.pdf` (target: ~18–22 tight, high-signal pages including key Appendices) and `DSAI Lab_Group 7_Milestone 5_Slides.pdf` (14–16 executive defense slides) via `scripts/build_m5_deliverables.py`, visually verifying every rendered page.

---

## 3. Recommended Execution Schedule (4 Clean Phases)

| Phase | Target Window | Core Deliverables | Owners |
| :--- | :--- | :--- | :--- |
| **Phase 1: Sync & Setup** | Day 1 | Merge PR #22 & PR #23 into `main`; share frozen checkpoints (`D03`, `A03`, `C06`, `A01`, `A02`, `A04`) via Group 7 Google Drive; every teammate runs `pytest` + Evidence Review UI locally. | All 4 members |
| **Phase 2: Core Test Eval, Hybrid Verifier & Human Audit** | Days 2–4 | Run WS1 (full 1,858-pair Test detector + BERTScore/CIDEr captioner + significance tests) & WS2 (Calibrated Mask-First Hybrid Verifier on val → blind Test); generate blind human-audit packets and complete dual-rater annotations (WS3). | Shoaib + Sujal + Priyanshu + Sarthak (see [TEAM_HANDOFF.md](TEAM_HANDOFF.md)) |
| **Phase 3: Error Analysis, Stress Tests & Compute Profiling** | Days 5–6 | Run WS4 (error mining + failure taxonomy + galleries), WS5 (14-condition Test stress suite + SECOND-CC breakdown), WS6 (RTX 4060 / T4 / L4 / CPU profiling), and WS7 (CLI inference + unit tests). | All 4 members |
| **Phase 4: Figures, 16-Section Report, Slides & Final PRs** | Days 7–8 | Generate all M5 figures, draft & cross-review all 16 sections + Appendices, render & visually inspect PDF report & defense slide deck, and submit clean promotion PRs to `main`. | All 4 members (final typography & QA by Shoaib) |

---

## 4. TA / Professor "Top-Marks" Quality Checklist

Before freezing Milestone 5 deliverables, verify every single box below is checked:
- [ ] **Zero data leakage:** Validation used only for selection & verifier threshold calibration; all final Milestone 5 numbers reported on the **1,858 held-out LEVIR-MCI Test pairs** and **1,142 SECOND-CC Test pairs**.
- [ ] **All problem-domain metrics covered:** Segmentation (`IoU`, `Dice`, `Pixel Accuracy`, `3×3 Confusion Matrix`), Classification/Detection (`Precision`, `Recall`, `F1`, `ROC-AUC`, `PR-AUC`, `FA@50px`), Generative Captioning (`BLEU-1..4`, `ROUGE-L`, `METEOR`, `CIDEr`, `BERTScore`), and Verification/LLM (`Hallucination Rate`, `Groundedness`, `ECE`, `AURC`, `Cohen's κ` human agreement).
- [ ] **Statistical rigor:** 95% bootstrap CIs and paired significance tests ($p$-values) on every headline comparison.
- [ ] **True ablation table (Section 9):** Clear step-by-step contribution of every module (backbone unfreezing, focal+dice loss, augmentation, frozen CLIP encoder, Triton scan, contextual 32px crop, OpenCV witness, BIT mask checker, selective Qwen3-VL hybrid routing).
- [ ] **Honest, quantitative error analysis (Section 10):** Real failure taxonomy with counts, visual error maps, and actionable root causes (no hand-waving).
- [ ] **Real systems numbers (Section 12):** Actual measured ms/pair, VRAM MB, GFLOPs, disk MB, and cost savings from selective VLM triggering.
- [ ] **Polished formal presentation:** Crisp typography, zero table text clipping/wrapping, zero internal process notes in submitted PDFs, and verified GitHub reproducibility.
