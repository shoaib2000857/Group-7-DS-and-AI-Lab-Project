# TerraWatch — Team Handoff & Milestone 5 Collaboration Guide

**Repository:** `https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project`  
**Team (IIT Madras BS DSAI Lab — Group 7):**
- **Shoaib** (`@shoaib2000857`, initials `SSSM`)
- **Priyanshu Agarwal** (`@23f2003488`)
- **Sarthak Saini** (`@Sarthak-Saini265`)
- **Sujal Raj Pradhan** (`@SujalPradhan`)

---

## 1. Where We Stand Right Now (Complete Milestone 4 Baseline)

Two clean, conflict-free Milestone 4 PRs are open against `main` for team review:
1. **PR #22 (Reports, Slides, Figures, Walkthrough & README):** `https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/22`
2. **PR #23 (Full-Data Code, Scripts, Notebooks, Configs, Evidence Receipts & Tests):** `https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/23`

**First action for teammates:** Review and approve **PR #22** and **PR #23** so `main` has the complete Milestone 4 baseline before cutting Milestone 5 branches!

### Frozen Selected Models from Milestone 4 (100% Full-Data LEVIR-MCI)

| Component | Selected Run / Model | Primary Validation Score | Held-Out Test Score (1,858 pairs) | Disk Size & SHA-256 (first 16 chars) |
| :--- | :--- | :--- | :--- | :--- |
| **Change Detector** | `D03_bit_lr3e4` (`bit_resnet18`, 32/35 ep, best ep 24) | **74.54%** Change mIoU (3-seed: `74.23% ± 0.28%`) | **75.93%** Change mIoU / **86.83%** Bin F1 / `10.28%` FA@50px | `45.68 MB` (`05345df154b4d748`) |
| **Strong Detector Baseline** | `A03_siam_r18_unet` (25/25 ep, best ep 22) | **75.02%** Change mIoU (3-seed: `74.73% ± 0.27%`) | **76.73%** Change mIoU (but `-12.4%` to `-30.4%` worse on blur & `-5.5%` worse on SECOND-CC) | `59.72 MB` (`c9d2c9c994c63411`) |
| **Change Captioner** | `C06_rscama_frozen_clip` (8/8 ep, best ep 6, 388-word train vocab) | **0.6358** Val BLEU-4 (3-seed: `0.6297 ± 0.0183`) | **0.6128** BLEU-4 / **0.7326** ROUGE-L / **3.57%** NC FA | `1,714.63 MB` (`c5b80d6bf6b6b0b2`) |
| **Visual Referee** | `Qwen/Qwen3-VL-4B-Instruct` (frozen BF16) | 120 val pairs / 147 claims audited | To be audited blindly on 150 Test pairs in M5 | HF rev `ebb281ec70b05090` |

---

## 2. 10-Minute Onboarding: Pull, Verify & Explore Locally

Once PR #22 and PR #23 are merged into `main`, run these commands on your machine:

```bash
# 1. Clone or pull latest main
git clone https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project.git
cd Group-7-DS-and-AI-Lab-Project
git checkout main && git pull origin main

# 2. Create a Python 3.11+ virtual environment and install core dependencies
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS / WSL: source .venv/bin/activate
pip install -r requirements-data.txt
pip install opencv-python-headless pytest matplotlib pillow numpy

# 3. Verify all 18 core tests pass cleanly (takes ~2 seconds, CPU-only!)
pytest tests/test_milestone4_pipeline.py tests/test_model_pipeline.py tests/test_pipeline.py tests/test_visual_evidence.py -v

# 4. Launch the live Interactive Evidence Review UI on the 8 committed real E2E traces
python -m src.app.server docs/milestone4/evidence/full_data_m4/verification/e2e_traces --port 8765
# Open http://127.0.0.1:8765 in your browser!
```

### Accessing Datasets & Model Checkpoints (NEVER Commit Weights to Git!)

- **Committed in Git (CPU replay ready immediately):**
  - All 24 detector receipts, 10 captioner receipts, robustness/transfer JSONs, 120-pair verification audit, and 8 E2E traces live in `docs/milestone4/evidence/full_data_m4/`.
  - All 3 reproducible Milestone 4 notebooks live in `notebooks/TerraWatch_M4_*.ipynb` (they run in **CPU replay mode** out of the box!).
- **Large Binaries (Shared Group 7 Google Drive — outside Git):**
  - **Datasets:** Extracted LEVIR-MCI (`9,544` pairs) and SECOND-CC (`530` val / `1,142` test) in `/MyDrive/TerraWatch_Extracted/`.
  - **Frozen M4 Checkpoints (verified disk sizes & SHA-256 digests, outside Git):**
    - `D03_bit_lr3e4/best.pt` — `47,898,192 B` (`45.68 MB`), SHA-256 `05345df154b4d748432bccd72a439d3eddbda0fa2e10795ecdac29cd2cd38feb`
    - `A01_ef_unet/best.pt` — `31,114,207 B` (`29.67 MB`), SHA-256 `5831a24249abf330f65a0b2c278040b2908b9137a2e95472544ccf34f91a437a`
    - `A02_fc_siam_diff/best.pt` — `31,110,751 B` (`29.67 MB`), SHA-256 `32329cf708a7fed397c96689922eca2a2f29556108c35c214a1c332326b23105`
    - `A03_siam_r18_unet/best.pt` — `62,621,663 B` (`59.72 MB`), SHA-256 `c9d2c9c994c63411096703397ee1842cde5ef917bee43d7b974a3f47c511e586`
    - `A04_changeformer/best.pt` — `164,282,148 B` (`156.67 MB`), SHA-256 `fe137902d7dc4ca4816ab1e777f93d1f383cd132ca4542bc23a3b96c26ea6bc7`
    - `C06_rscama_frozen_clip/best.pth` — `1,797,923,010 B` (`1,714.63 MB`), SHA-256 `c5b80d6bf6b6b0b2628c48d47df07ad77bc086ff0004bdd7160d0b0ff5545796` (plus `s0`/`s2026`, `C01_rscama_ref/best.pth`, `C04_rscama_adamw_cos/best.pth`)
  - **Rule:** Always verify SHA-256 digests before running evaluation scripts, and **never** `git add` any `.pt`, `.pth`, `.bin`, `.safetensors`, `.zip`, or raw dataset files.

---

## 3. Repository & Git Workflow Rules (Please Follow Strictly)

1. **Branch naming:** Use plain descriptive branch names (e.g., `m5-error-analysis`, `m5-human-audit`, `m5-compute-benchmark`, `m5-hybrid-verifier`). Do **not** use `feat/` or `fix/` prefixes.
2. **Where to keep work-in-progress:** Keep all Milestone 5 experimental scripts, raw CSV/JSON evidence receipts, and draft reports under **`POC/milestone5/`** until the team is ready for promotion PRs into `src/`, `scripts/`, `notebooks/`, `reports/`, and `presentation/`.
3. **Never push directly to `main`:** Always open a PR against `main` and request review from the other 3 teammates.
4. **Formal report writing standard:** All text going into final reports/slides (`reports/`, `presentation/`) must use polished, confident, formal academic-project prose — never include internal TODOs, scratch process comments, or "pending review" notes in submitted PDFs (keep coordination notes in GitHub PRs or chat).

---

## 4. Parallel Role & Task Division Across All 4 Teammates

Every teammate has a **high-ownership, independent technical track** plus an equal share of the **blind dual-rater human audit** so nobody is blocked waiting on anyone else:

### Track 1 — Shoaib (`@shoaib2000857`): Hybrid Verifier, Held-Out Test Verification, Compute Orchestration & Final Deliverable Assembly
- **Owns:** WS2 (Calibrated Mask-First Hybrid Verifier + 150-pair Held-Out Test Verification Audit + ECE/AURC calibration), WS7 (Single-command inference CLI + checkpoint registry), GPU compute coordination (Omen RTX 4060 / Colab T4 & L4), and WS8 final report/slide typesetting & visual QA.
- **Concrete Deliverables:**
  - `POC/milestone5/scripts/m5_run_test_verification.py` & calibrated hybrid verifier in `src/inference/verification_pipeline.py`
  - `POC/milestone5/evidence/hybrid_verifier/` (Val calibration + blind Test 150-pair receipts + 10 Test E2E traces)
  - Blind human-audit claim packets distributed to all 4 teammates (Day 2)
  - Final polished `DSAI Lab_Group 7_Milestone 5.pdf` & `DSAI Lab_Group 7_Milestone 5_Slides.pdf`
- **Report Sections Owned:** Sections 1, 3, 9 (Verification Ablations), 15, 16 + Final Integration.

### Track 2 — Priyanshu Agarwal (`@23f2003488`): Full Held-Out Test Detector Metrics, ROC/PR Curves, Confusion Matrices & Extended Stress Testing
- **Owns:** WS1 (Detector half: 5 architectures + 2 non-learning baselines on all 1,858 Test pairs, pixel ROC-AUC/PR-AUC, 3×3 confusion matrices, change-area stratified breakdown, bootstrap CIs & Wilcoxon/McNemar tests) + WS5 (14-condition Test stress suite including synthetic cloud occlusion, severe jitter, radiometric mismatch, 0.5× resolution + SECOND-CC per-class domain-gap analysis).
- **Concrete Deliverables:**
  - `POC/milestone5/scripts/m5_eval_detectors_test.py` & `POC/milestone5/scripts/m5_stress_tests.py`
  - `POC/milestone5/evidence/test_full_metrics/detector_test_full.json` & `POC/milestone5/evidence/stress_tests/stress_test_results.json`
  - Figures: `m5_roc_pr_curves.png`, `m5_confusion_matrices.png`, `m5_stratified_area_performance.png`, `m5_extended_robustness.png`
- **Report Sections Owned:** Sections 4, 5 (Segmentation/Detection metrics), 6 (Detector tables), 7 (Detector baselines), 11 (Model Robustness) + Appendices B, C.

### Track 3 — Sarthak Saini (`@Sarthak-Saini265`): Captioner Semantic Metrics (`BERTScore`/`METEOR`/`CIDEr`), Deep-Dive Error Analysis & Visual Failure Taxonomy
- **Owns:** WS1 (Captioner half: `C06` 3-seed Test spread vs `C00`/`C01`/`C04` vs Naive/Template baselines on all 1,858 Test pairs with `BLEU-1..4`, `ROUGE-L`, `METEOR`, `CIDEr`, `BERTScore`, Hallucination Rate & Groundedness) + WS4 (Automated mining of worst Test detector FPs/FNs, captioner hallucinations, verifier errors, quantified 6-category Failure Taxonomy, and visual success/failure gallery strips).
- **Concrete Deliverables:**
  - `POC/milestone5/scripts/m5_eval_captioners_test.py` & `POC/milestone5/scripts/m5_error_analysis.py`
  - `POC/milestone5/evidence/test_full_metrics/captioner_test_full.json` & `POC/milestone5/evidence/error_analysis/failure_taxonomy.csv`
  - Figures: `m5_captioner_semantic_comparison.png`, `m5_success_gallery.png`, `m5_error_taxonomy_gallery.png`
- **Report Sections Owned:** Sections 5 (Generative AI/LLM metrics), 6 (Captioner tables), 8 (Hyperparameter Analysis), 10 (Error Analysis) + Appendices F, G (canonical lettering in `REPORT_OUTLINE_AND_EVIDENCE_MAP.md`).

### Track 4 — Sujal Raj Pradhan (`@SujalPradhan`): Systems & Computational Performance Profiling, Dual-Rater Audit Aggregation, Limitations & Deployment Packaging
- **Owns:** WS6 (Cross-hardware profiling on Colab T4, L4, Laptop GPU & CPU: exact parameter counts, model disk sizes, GFLOPs, P50/P95/P99 per-stage latency, peak VRAM, batch throughput, Triton vs Python speedup chart, and cost per 1,000 pairs comparing unconditional vs selective VLM) + WS3 Aggregation (collecting all 4 blind human-audit CSVs, computing Cohen's $\kappa$, adjudicating disagreements, and producing the human-verified groundedness/hallucination tables) + Evidence Review UI end-to-end QA.
- **Concrete Deliverables:**
  - `POC/milestone5/scripts/m5_benchmark_compute.py` & `POC/milestone5/scripts/m5_compute_human_audit.py`
  - `POC/milestone5/evidence/compute_profile/compute_profile.json` & `POC/milestone5/evidence/human_audit/human_audit_summary.json`
  - Figures: `m5_compute_profile.png`, `m5_human_audit_agreement.png`
- **Report Sections Owned:** Sections 2 (Experimental Setup), 12 (Computational Performance), 13 (Limitations & Ethics), 14 (Possible Improvements) + Appendices I, J (canonical lettering in `REPORT_OUTLINE_AND_EVIDENCE_MAP.md`: `Appendix I` = Evaluation & Profiling Scripts, `Appendix J` = Human Annotation Protocol & Ethics).
- **GPU Hourly Pricing Reference for Section 12:** Report both (a) **actual team Colab Pro/PAYG rate** (`$9.99 / 100 CU ≈ $0.10/CU`, observed M4 burn rates: T4 `1.15 CU/hr ≈ $0.115/hr`, L4 `4.82 CU/hr ≈ $0.482/hr`, local RTX 4060 `$0` marginal cloud cost) and (b) **public on-demand cloud reference** (GCP T4 `$0.35/hr` GPU-only or `~$0.54/hr` with `n1-standard-4`, or AWS `g4dn.xlarge` `$0.526/hr`), explicitly labeled.
- **Submission Format:** Deliver clean Markdown (`.md`) files under `POC/milestone5/submission/sections/` plus JSON/CSV receipts under `POC/milestone5/evidence/` and 180-DPI PNG figures; the final ReportLab build compiles Markdown sections directly into the unified PDF.

---

## 5. Blind Dual-Rater Human Annotation Protocol (All 4 Members — ~45 Mins Each)

To give Milestone 5 an ironclad **Human Evaluation & Groundedness** section, we annotate **120 stratified held-out Test claims** (`60` claims per teammate, paired so every single claim is independently rated by **2 teammates**):

| Claim Slice | Rater A | Rater B | Adjudicator (if A ≠ B) |
| :--- | :--- | :--- | :--- |
| Claims `001–030` | Shoaib | Priyanshu | Sujal |
| Claims `031–060` | Priyanshu | Sarthak | Shoaib |
| Claims `061–090` | Sarthak | Sujal | Priyanshu |
| Claims `091–120` | Sujal | Shoaib | Sarthak |

### Rules for Strictly Valid Blind Annotation
1. **Stay blind:** Do **not** look at the automated model verdict (`accept`/`reject`/`defer`) or discuss labels with your paired rater before submitting your CSV.
2. **Inspect the visual evidence:** Look at `T1`, `T2`, and the zoomed 32px+ contextual crop pair for each claim.
3. **Assign one primary `human_label`:**
   - `supported`: The claimed object and change type (appearance/disappearance/no-change) are clearly visible in `T1` → `T2`.
   - `contradicted`: The claim is factually wrong (e.g., claims new building on unchanged ground, wrong object class, or clear hallucination).
   - `unverifiable`: Resolution, cloud/shadow occlusion, crop ambiguity, or exact numeric count uncertainty makes a confident human verdict impossible.
4. **Assign one `error_category` (only when `contradicted` or `unverifiable`):** `none`, `hallucinated_object`, `missed_change`, `wrong_class_road_vs_bldg`, `wrong_count`, `wrong_location`, `shadow_or_seasonal`, `tiny_or_occluded`.
5. **Locked Packet Schema (`audit_id` & Hybrid Verifier Column):**
   - Every claim row carries a unique `audit_id` (`M5T_001` … `M5T_120`).
   - **Blind rater CSV (`rater_<INITIALS>.csv`):** `audit_id, pair_id, stratum, claim_id, caption, claim_span, object, action, location, quantity, human_label, error_category, notes`
   - **Unblinded master key (`test_audit_master_key.csv`, kept separate from blind packets):** `audit_id, pair_id, stratum, claim_id, caption, claim_span, object, action, location, quantity, proxy_label, unverified_decision, mask_only_decision, direct_vlm_decision, reinsp_all_decision, hybrid_verifier_decision, hybrid_verifier_prob, vlm_triggered`
6. **Save your completed sheet** as `POC/milestone5/evidence/human_audit/rater_<initials>.csv` (`SSSM`, `PA`, `SS`, `SRP`) and push on your branch. Sujal runs `m5_compute_human_audit.py` to compute exact Cohen's $\kappa$, agreement matrix, and human-verified groundedness!

---

## 6. Quick Troubleshooting Reference

- **Windows `pytest` temp directory permission error (`WinError 5`):** Run `pytest --basetemp=./_tmp_pytest -v` and remove `_tmp_pytest` afterward.
- **`No module named 'cv2'`:** Install `pip install opencv-python-headless`.
- **RSCaMa `mamba_ssm` wheel error on Colab/Windows:** Do **not** compile `mamba_ssm` from source — `scripts/train_captioner_full.py` automatically injects our exact Triton fused selective-scan kernel (`_TritonSelectiveScan`), giving 10× speedup with `< 5e-7` numerical agreement on any CUDA GPU.
- **Need help or checkpoint links?** Ping Shoaib (`SSSM`) in the Group 7 chat or open a GitHub issue/PR comment.
