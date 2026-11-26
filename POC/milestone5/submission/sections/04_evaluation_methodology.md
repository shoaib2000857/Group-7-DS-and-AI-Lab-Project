## 4. Evaluation Methodology

### 4.1 Evaluation Protocol and Anti-Leakage Discipline

Milestone 5 enforces a strict **single-pass, frozen-artifact evaluation protocol**:

1. **SHA-256 Provenance Gate:** Every evaluation script recomputes the SHA-256 digest of each loaded checkpoint and verifies it against `POC/milestone5/configs/model_registry.json` and the Milestone 4 receipts before running inference.
2. **Strict Train / Validation / Test Separation:** All model hyperparameters, architecture selections, captioner vocabularies (388 train-only tokens), and hybrid verifier routing rules (`tau_accept = 0.60`, `tau_reject = 0.75`, `mask_alarm_px = 50`, escalation predicates) were **locked exclusively on Validation** (`1,242` pairs / `147` Val claims) prior to evaluating on the held-out Test split (`1,858` pairs / `150` stratified blind Test pairs).
3. **Statistical Rigor & Split Justification (Cross-Validation Strategy):** Rather than $k$-fold cross-validation—which risks geographic tile leakage across adjacent urban patches and breaks comparability with published LEVIR-MCI baselines—evaluation uses the official non-overlapping spatial Train/Validation/Test partition combined with **3-seed training replication** (`seeds 42, 0, 2026` on both `D03` and `C06`), **1,000-resample paired bootstrap 95% confidence intervals** (fixed seed `42`), paired **Wilcoxon signed-rank tests** over per-pair scores, and exact **McNemar's test** on unchanged-scene false-alarm discordance.
4. **No Ground-Truth Leakage into Inference Workers:** Ground-truth segmentation masks and reference captions are read exclusively by the offline evaluator after predictions are serialized; inference workers receive only $(T_1, T_2)$ imagery.

### 4.2 Test Datasets and Ground-Truth Preparation

| Benchmark | Split | Pairs | Pixel / Caption Scale | Ground-Truth Format & Preparation |
|---|---|---:|---|---|
| **LEVIR-MCI Test** | Locked Held-Out Test | **1,858** (934 changed, 924 unchanged) | 121.77M pixels; 9,290 human captions (5 per pair) | Official 3-class masks (`0=background`, `128=road`, `255=building`) mapped deterministically to `{0, 1, 2}`; 5 lowercase reference captions per pair tokenized identically across all captioners. |
| **LEVIR-MCI Test Stratified Audit** | Blind Verification Subset | **150** (50 unchanged, 35 bldg-only, 20 road-only, 25 mixed, 20 small $<1\%$) | Extracted atomic claims from `C06` captions | Deterministic `seed=2026` stratified sampling across 5 change-type strata; dual evaluation via automated mask-caption proxy labels AND blind human dual-rater consensus (`M5T_001`–`M5T_120`). |
| **SECOND-CC-AUG** | Zero-Shot Transfer (`val` + `test`) | **530** Val + **1,142** Test | 6 semantic land-cover classes (`building`, `non_vegetated_ground`, `low_vegetation`, `tree`, `water`, `playground`) | Zenodo 16937571 RGB semantic masks decoded into exact 6-class change indicators to measure both aggregate any-change F1 and per-class zero-shot recall without fine-tuning. |

### 4.3 Blind Human-Annotation Protocol (120 Held-Out Test Claims)

Automated proxy labels derived from segmentation masks cannot reliably judge spatial relations (*"along the road"*), partial visibility, or out-of-ontology concepts (*"trees"*, *"parking lot"*). To establish an unbiased human ground truth on held-out Test data, **120 claims (`M5T_001`–`M5T_120`)** were sampled from the 150-pair blind Test verification audit — retaining all non-accepted and proxy-contradicted claims plus a balanced random sample of accepted claims, then deterministically shuffled (`seed=2026`) so raters cannot infer verdict from row order:

- **Strict Blinding:** Every rater receives a 4-panel visual inspection strip (`T1`, `T2 + candidate bbox`, `Zoom Crop T1`, `Zoom Crop T2`) and the bare claim span with **zero model verdicts, probabilities, trigger flags, or proxy labels** present (`m5_compute_human_audit.py` programmatically fails on any column leak).
- **Overlapping Dual-Rater + Adjudicator Design:** The 120 claims are partitioned into four 30-claim slices (`001–030`, `031–060`, `061–090`, `091–120`), each independently labeled (`supported`, `contradicted`, or `unverifiable`, plus an 8-class `error_category`) by two project raters, with disagreements resolved blindly by a designated third adjudicator (full annotation instructions and rubric in **Appendix J**).
- **Inter-Rater Reliability:** Pooled and per-slice **Cohen's $\kappa$**, exact percent agreement, $3 \times 3$ rater confusion matrix, proxy-vs-human alignment $\kappa$, and Wilson 95% confidence intervals on human-verified groundedness and hallucination rate are computed via `POC/milestone5/scripts/m5_compute_human_audit.py` (verified by 13 automated unit tests).

### 4.4 Baseline Models and Verification Policies Compared

To contextualize every architectural and algorithmic decision, TerraWatch is benchmarked against three layers of baselines:

1. **Detector Baselines:** Four neural architectures spanning early fusion (`A01_ef_unet`), pure Siamese differencing (`A02_fc_siam_diff`), Siamese ResNet18-UNet (`A03_siam_r18_unet`), and hierarchical vision transformer (`A04_changeformer`), plus two non-learning controls (`otsu_diff` pixel intensity differencing and `always_background`).
2. **Captioner Baselines:** Upstream released checkpoint (`C00_rscama_released`), unfrozen-CLIP controlled variants (`C04_rscama_adamw_cos`, `C01_rscama_ref`), a deterministic `bit_mask_template` generator translating `D03` masks into structured sentences, and the `always_no_change` lower bound.
3. **Verification Policies:** Seven regimes spanning no verification (`1_unverified_captioner`), deterministic spatial checks (`2_mask_only_checker`, `3_witness_only_checker`), unconditional 4B VLM inference (`4_direct_qwen3vl_gated`, `5_reinspection_all_gated`), Milestone 4's high-trigger policy (`6_m4_triggered_reinspection`), and Milestone 5's Validation-locked selective policy (`7_m5_calibrated_mask_first_hybrid`).

### 4.5 Project Success Criteria

| Dimension | Target Success Criterion | Verification Source |
|---|---|---|
| **Detector Accuracy** | Held-out Test Change-Class mIoU $\ge 74.0\%$, Binary F1 $\ge 85.0\%$, ROC-AUC $\ge 0.98$ | Section 6.1 (`detector_test_full.json`) |
| **Detector Robustness** | Clear robustness advantage for selected `D03` over clean CNN baseline `A03` on blur/downscale/noise and SECOND-CC zero-shot transfer | Section 11 (`stress_test_levir_mci_test.json`, `second_cc_6class_dissection.json`) |
| **Captioner Quality** | Held-out Test Corpus BLEU-4 $\ge 0.60$, BERTScore-F1 $\ge 0.94$, unchanged-scene hallucination rate $\le 6.0\%$ | Section 6.2 (`captioner_test_full.json`) |
| **Verifier Groundedness & Efficiency** | Accepted-claim precision $\ge 90\%$ with selective risk $< 10\%$, beating both unverified captions and standalone Qwen3-VL while cutting VLM calls by $> 70\%$ | Section 6.3 (`test_verification_summary.json`) |
| **Operational Readiness** | Interactive inference latency $< 200\text{ ms}$/pair (mask-first path) on consumer GPU, verifiable SHA-256 provenance, passing end-to-end tests | Section 12 & `test_m5_hybrid_and_cli.py` |
