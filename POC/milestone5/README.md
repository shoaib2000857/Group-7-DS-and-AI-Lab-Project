# TerraWatch — Milestone 5 Hub (Model Evaluation, Error Analysis, Robustness & Final Deployment Readiness)

Welcome to the **Milestone 5** workspace for **TerraWatch** (IIT Madras BS Data Science & AI Lab — BSDA4001, Group 7).

Milestone 4 delivered our complete, full-data trained model suite on 100% of the governed LEVIR-MCI cohort (`6,444` train / `1,242` val / `1,858` held-out test pairs):
- **Selected Change Detector:** `D03_bit_lr3e4` (BIT-ResNet18) — **74.54%** Val Change mIoU (`74.23% ± 0.28%` 3-seed), **75.93%** Held-Out Test Change mIoU (**86.83%** Bin F1), decisively stronger than `A03_siam_r18_unet` under optical blur (`+12.38%` to `+30.36%`) and zero-shot SECOND-CC transfer (`+5.46%` Test F1).
- **Selected Change Captioner:** `C06_rscama_frozen_clip` (RSCaMa with exact Triton fused selective-scan kernel & 388-word train-only vocabulary) — **0.6358** Val BLEU-4 (`0.6297 ± 0.0183` 3-seed), **0.6128** Held-Out Test BLEU-4 / **0.7326** ROUGE-L / **3.57%** Unchanged False-Alarm Rate.
- **Verification & Evidence Review Pipeline:** OpenCV structural witness (32px border-clamped contextual crops), explicit article vs numeric count gating, frozen `Qwen3-VL-4B-Instruct` visual referee, and interactive Evidence Review UI (`src/app/server.py`) with append-only SQLite audit logging.

**Milestone 5** turns this strong full-data foundation into a **publication-grade final evaluation, error analysis, robustness, systems-benchmarking, human-audit, and deployment-readiness package** covering all **16 required report sections + full appendices**.

---

## Documents in This Folder

| Document | Purpose | Audience |
| :--- | :--- | :--- |
| [M5_MASTER_PLAN.md](M5_MASTER_PLAN.md) | Complete end-to-end Milestone 5 execution plan: 8 workstreams (WS1–WS8), scripts to build, statistical tests, timeline, and TA/Professor quality bar. | All team members & execution lead |
| [TEAM_HANDOFF.md](TEAM_HANDOFF.md) | Step-by-step repo onboarding after PR #22 & #23 merge, environment & checkpoint setup, Git rules, parallel task split across all 4 teammates, and blind dual-rater human annotation protocol. | Priyanshu, Sarthak, Sujal, Shoaib |
| [REPORT_OUTLINE_AND_EVIDENCE_MAP.md](REPORT_OUTLINE_AND_EVIDENCE_MAP.md) | Exact blueprint mapping every bullet of the 16-section + Appendices Milestone 5 rubric to concrete tables, figures, scripts, and committed JSON/CSV receipts. | Report authors & reviewers |

---

## Planned Directory Structure under `POC/milestone5/`

```text
POC/milestone5/
├── README.md                              # This index
├── M5_MASTER_PLAN.md                      # 8-workstream master plan & schedule
├── TEAM_HANDOFF.md                        # Teammate onboarding, role split & annotation guide
├── REPORT_OUTLINE_AND_EVIDENCE_MAP.md     # 16-section + Appendices exact evidence map
├── scripts/                               # M5 evaluation, profiling, error-mining & figure scripts
├── evidence/                              # JSON/CSV receipts (NEVER commit .pt/.pth weights!)
│   ├── test_full_metrics/                 # 1,858-pair Test ROC/PR, confusion matrices, BERTScore/CIDEr
│   ├── hybrid_verifier/                   # Calibrated Mask-First Hybrid Verifier val + Test audit
│   ├── human_audit/                       # Blind dual-rater annotations + Cohen's kappa summary
│   ├── error_analysis/                    # Failure taxonomy CSV + worst-case gallery metadata
│   ├── stress_tests/                      # Extended Test corruptions + synthetic cloud/jitter/downscale
│   └── compute_profile/                   # Latency (P50/P95/P99), VRAM, GFLOPs, throughput & cost
└── submission/                            # Polished Milestone 5 Markdown, PDF report, slides & figures
```

---

## Prerequisites Before Starting Milestone 5 Work

1. Review and merge the two open Milestone 4 PRs into `main`:
   - **PR #22 (Reports, Slides, Figures, Walkthrough & README):** `https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/22`
   - **PR #23 (Code, Training/Eval Scripts, Notebooks, Evidence Receipts & Tests):** `https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/23`
2. Pull the updated `main` branch and verify the 18-test baseline suite passes (`pytest -v`).
3. Follow [TEAM_HANDOFF.md](TEAM_HANDOFF.md) to pick up your assigned workstream and blind human-audit packet.
