# TerraWatch — Milestone 5 Live Task Tracker

Check off each task as its evidence JSON/CSV, figure, or PR lands under `POC/milestone5/`.

---

## Phase 1 — Repo Sync & Checkpoint Sharing (Day 1)
- [ ] Review & merge **PR #22** (`milestone4-reports` → `main`) — *Priyanshu, Sarthak, Sujal*
- [ ] Review & merge **PR #23** (`milestone4-code-and-notebooks` → `main`) — *Priyanshu, Sarthak, Sujal*
- [ ] Upload frozen M4 checkpoints (`D03`, `A03`, `A01`, `A02`, `A04`, `C06` 3-seeds, `C01`, `C04`) to shared Group 7 Google Drive folder `/MyDrive/TerraWatch_Checkpoints_M4/` — *Shoaib*
- [ ] All 4 teammates pull `main`, run `pytest` (18/18 passing), and verify Evidence Review UI opens on `http://127.0.0.1:8765` — *All*

## Phase 2 — Held-Out Test Metrics, Hybrid Verifier & Blind Dual-Rater Human Audit (Days 2–4)
- [ ] **WS1-Det:** Run all 5 detectors + Otsu + Always-BG on 1,858 held-out Test pairs (`m5_eval_detectors_test.py`): ROC-AUC, PR-AUC, 3×3 confusion matrix, area-stratified breakdown, 95% bootstrap CIs & Wilcoxon/McNemar tests — *Priyanshu*
- [ ] **WS1-Cap:** Run `C06` (3 seeds) + `C00` + `C01` + `C04` + No-Change/Template baselines on 1,858 held-out Test pairs (`m5_eval_captioners_test.py`): BLEU-1..4, ROUGE-L, METEOR, CIDEr, BERTScore, NC FA% & object hallucination rate — *Sarthak*
- [ ] **WS2:** Implement & calibrate **Mask-First Hybrid Verifier** on 120-pair Val audit (ECE, Brier, AURC); run locked blind evaluation on **150 stratified held-out Test pairs** + export 10 Test E2E traces (`m5_run_test_verification.py`) — *Shoaib*
- [ ] **WS3-Packet:** Distribute 120-claim blind annotation packets (`60` claims/person, 2 independent raters/claim) — *Shoaib*
- [ ] **WS3-Annotate:** Complete blind ratings:
  - [ ] Claims `001–030` + `091–120` (`rater_SSSM.csv`) — *Shoaib*
  - [ ] Claims `001–060` (`rater_PA.csv`) — *Priyanshu*
  - [ ] Claims `031–090` (`rater_SS.csv`) — *Sarthak*
  - [ ] Claims `061–120` (`rater_SRP.csv`) — *Sujal*
- [ ] **WS3-Aggregate:** Adjudicate disagreements, compute Cohen's $\kappa$, agreement matrix, human-verified groundedness & hallucination rate (`m5_compute_human_audit.py`) — *Sujal*

## Phase 3 — Error Analysis, Stress Testing, Systems Profiling & CLI Packaging (Days 5–6)
- [ ] **WS4:** Mine top Test detector FPs/FNs, road/building confusions, caption hallucinations & verifier errors (`m5_error_analysis.py`); build 6-category Failure Taxonomy CSV + visual success & failure galleries — *Sarthak*
- [ ] **WS5:** Run 14-condition held-out Test stress suite (`D03` vs `A03`: core 10 + synthetic cloud occlusion, severe jitter, radiometric mismatch, 0.5× downscale), captioner blur/haze probe, and SECOND-CC per-class transfer dissection (`m5_stress_tests.py`) — *Priyanshu*
- [ ] **WS6:** Benchmark all models across RTX 4060, Colab T4, Colab L4 & CPU (`m5_benchmark_compute.py`): params, disk MB, GFLOPs, P50/P95/P99 latency, peak VRAM, GPU util, Triton speedup, throughput & \$/1,000 pairs cost — *Sujal*
- [ ] **WS7:** Ship single-command inference CLI (`scripts/run_terrawatch_inference.py`), `configs/model_registry.json` SHA-256 verifier, and expanded pytest coverage — *Shoaib + Sujal*

## Phase 4 — Figures, 16-Section Report, Appendices & Defense Slides (Days 7–8)
- [ ] Generate all 10 high-DPI Milestone 5 publication figures (`m5_*.png`, 180 DPI) — *All owners*
- [ ] Draft all 16 Milestone 5 report sections + Appendices A–J in `POC/milestone5/submission/Milestone5_Report.md` (formal confident tone, zero internal caveats) — *All per ownership map*
- [ ] Build & visually verify `DSAI Lab_Group 7_Milestone 5.pdf` (Segoe UI + Consolas typography, proportional non-wrapping tables) & `DSAI Lab_Group 7_Milestone 5_Slides.pdf` (15 defense slides) — *Shoaib*
- [ ] Cross-review final PDFs + slides as a team, then open clean promotion PRs into `main` when authorized — *All*
