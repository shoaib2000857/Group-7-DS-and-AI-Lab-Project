# TerraWatch — Milestone 5 Live Task Tracker

Status updated after full held-out Test evaluation, cross-platform compute profiling, PR #24 merge, and 240/240 blind dual-rater human annotations.

---

## Phase 1 — Repo Sync & Checkpoint Sharing
- [x] Review & merge **PR #22** (`milestone4-reports` → `main`)
- [x] Review & merge **PR #23** (`milestone4-code-and-notebooks` → `main`)
- [x] Upload all 12 frozen M4 checkpoints + `SHA256SUMS.txt` to shared Group 7 Google Drive folder (`TerraWatch_Checkpoints_M4`)
- [x] Merge **PR #24** (`m5-compute-and-audit` → `main`) with Sujal's compute profiling, human-audit aggregation, and corrected Sections 2, 12–14, App. I–J

## Phase 2 — Held-Out Test Metrics, Hybrid Verifier & Blind Dual-Rater Human Audit
- [x] **WS1-Det:** All 5 detectors + Otsu + Always-BG on 1,858 held-out Test pairs (`detector_test_full.json`: ROC-AUC, PR-AUC, 3×3 CM, area-stratified breakdown, 95% CIs, Wilcoxon/McNemar)
- [x] **WS1-Cap:** All 6 captioners (`C06`, `C00`, `C04`, `C01`, `bit_mask_template`, `always_no_change`) on 1,858 held-out Test pairs (`captioner_test_full.json`: BLEU-1..4, ROUGE-L, METEOR, CIDEr-D, BERTScore-F1, NC FA%, Miss%, ObjHall%, 95% CIs, Wilcoxon)
- [x] **WS2:** Val-locked **Calibrated Mask-First Hybrid Verifier** evaluated blindly on 150 stratified Test pairs / 185 claims (`test_verification_summary.json`: `93.94%` proxy precision, `0.0710` ECE, `90.27%` VLM savings) + 10 authentic Test E2E traces
- [x] **WS3-Packet & Web App:** Exported 120-claim zero-leak blind packet (`M5T_001`–`M5T_120`), uploaded strips zip to Drive, deployed `blind_audit_web_app.py` on Omen
- [x] **WS3-Annotate:** All 4 teammates completed 60/60 blind ratings (240/240 total, 0 blanks, 0 warnings):
  - [x] `rater_SSSM.csv` (60/60 — Shoaib)
  - [x] `rater_PA.csv` (60/60 — Priyanshu Agarwal)
  - [x] `rater_SS.csv` (60/60 — Sarthak Saini)
  - [x] `rater_SRP.csv` (60/60 — Sujal Raj Pradhan)
- [x] **WS3-Aggregate:** Ran `m5_compute_human_audit.py` (`human_audit_summary.json`, `gold_labels.csv`, `disagreements.csv`, `m5_human_audit_agreement.png`): `85.0%` exact agreement (`102/120`), `100.0%` error-category agreement, **`100.0%` human-verified groundedness (`62/62`, `0.0%` hallucination rate)** for M5 Hybrid

## Phase 3 — Error Analysis, Stress Testing, Systems Profiling & CLI Packaging
- [x] **WS4:** Mined top Test detector FPs/FNs, road/bldg swaps, caption hallucinations & residual verifier errors; built quantified 6-category `failure_taxonomy.csv` + visual gallery
- [x] **WS5:** Executed 17-condition held-out Test stress suite (`D03` vs `A03`, D03 wins 12/17), 200-pair `C06` corruption probe, and SECOND-CC 6-class semantic domain-gap dissection (`70.5%` non-building ontology proven)
- [x] **WS6:** Profiled all models across RTX 4060 (`C06` Triton `3.83×` speedup), Colab T4 (`D03`, `A01`–`A04`, witness, `Qwen3-VL-4B`), Apple M4 MPS, and Apple M4 CPU + 33-run training cost (`16.63` GPU-hrs)
- [x] **WS7:** Shipped `run_terrawatch_inference.py`, `model_registry.json`, and 23/23 passing automated Milestone 5 tests (`test_m5_hybrid_and_cli.py` + `test_m5_compute_human_audit.py`)

## Phase 4 — Figures, 16-Section Report, Appendices & Defense Slides
- [x] Generated all 11 high-DPI Milestone 5 publication figures (`m5_*.png`)
- [x] Assembled all 16 sections + Appendices A–J in `POC/milestone5/submission/Milestone5_Report.md` (with Sujal's corrected pure-measured Sections 2 & 12 and real human-audit Section 6.4; zero unfilled markers)
- [x] Built & visually verified `DSAI Lab_Group 7_Milestone 5.pdf` and `DSAI Lab_Group 7_Milestone 5_Slides.pdf` (15 slides)
- [ ] Optional: Promote final report/slides into `docs/milestone5/` and merge `milestone5-evaluation` → `main` when Shoaib requests a PR
