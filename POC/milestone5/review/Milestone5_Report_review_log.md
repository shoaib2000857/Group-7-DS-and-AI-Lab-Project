# Milestone 5 report review log

Line-by-line evidence check of `reports/Milestone5_Report.md`. Each row gives the original text, the problem, the evidence file and the fix. Categories: ERROR (wrong value), UNSUPPORTED (no evidence), OVERCLAIM, GAP, STYLE. Evidence paths refer to the `milestone5-evaluation` branch unless noted.

## Sections 1-4


Paths below are relative to the evidence checkout unless stated. M4 = `docs/milestone4/evidence/full_data_m4/`; M5E = `POC/milestone5/evidence/`.

| # | Original (short quote) | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | Table 3.1, A04 "Effective Batch Size 32 (`16 × 2`)" | ERROR. A04 ran batch 8 x 4 accumulation | M4 detector_runs/A04_changeformer/run.json: config.batch_size 8, accum_steps 4 | `8 × 4` |
| 2 | Table 3.1, A04 "`bf16` AMP (RTX 4060)" | ERROR. A04 used fp16 | run.json amp_actual torch.float16 (all other RTX 4060 runs are bf16) | `fp16` AMP; Table 2.1 local row notes A04 used fp16 |
| 3 | Table 3.1, D03 "`patience=8`" | ERROR. Receipt and command line say patience 7; best epoch 24, last epoch 31 (zero-based), i.e. 7 non-improving epochs | detector_runs/D03_bit_lr3e4/run.json and config.json (`--patience 7`), history.jsonl. `configs/detector_training.json` says 8 (repo config disagrees with receipt) | `patience=7`; sentence in 2.3 notes the discrepancy |
| 4 | "Ep 24", "Ep 6" etc. | ERROR/clarity. Epoch numbers are zero-based in history.jsonl (D03 history has epochs 0..31), so Ep 24 is the 25th epoch | history.jsonl epoch field | Added a note under Table 3.1 |
| 5 | 3.3 "Seed 42 achieved the best combined balance ... lowest unchanged-scene hallucination rate (3.32% vs 5.70%-7.75%)" | ERROR/OVERCLAIM. s2026 has higher BLEU-4 (0.6442); C04 has higher ROUGE-L (0.7498) and changed BLEU-4 (0.438); C02 (0.79%) and C03 (0.47%) have lower hallucination; s0 7.44% and s2026 4.43% | M4 captioner_runs/*/history.jsonl at best_epoch | Rewritten: seed 42 is the primary seed and the selected run; hallucination advantage not stable across seeds; 5.70-7.75% stated only for C01, C04, C05, C07 |
| 6 | 3.3 "dramatically superior resilience to blur (+30.36 pts at r=2.0)" | ERROR (rounding) and tone. 0.382248-0.078570 = 0.303678 -> 30.37 (30.36 came from subtracting rounded values). +12.38 correct | M4 robustness_and_transfer_bit.json / _siam.json levir_mci_val_corruptions blur_both_r1.0 / r2.0 change_miou | +30.37; "far more resilient". Check Section 11 uses the same figure |
| 7 | 3.3 "stronger cross-dataset zero-shot generalization on SECOND-CC" | OVERCLAIM (one-sided). D03 any-change F1 on SECOND-CC Val 22.44% vs A03 13.94%, but precision 60.1% vs 85.3% and unchanged FA@50 22.1% vs 4.5% | M4 robustness_and_transfer_*.json second_cc_transfer.val | Stated as higher any-change F1, with lower precision and more false alarms |
| 8 | 3.3 "`A03` edged `D03` by +0.50" then selection rationale | OVERCLAIM. 0.50 is about 1.8 x seed std (0.275). Also seed runs ran on different hardware (D03 s0 RTX 4060, D03 s2026 and A03 s0/s2026 on L4), so spread mixes hardware and seed | run.json environment.gpu_name for the six seed runs | Added ratio and hardware caveat (2.3, 3.3). 74.54/74.00/74.15 and 75.02/74.49/74.68 verified; means 74.23/74.73 and sample std 0.28/0.27, 0.0183 verified (statistics.stdev, ddof=1) |
| 9 | 4.3 "13 automated unit tests" | ERROR. 16 `def test_` functions in the audit test file (one is parametrized over 4 columns, so collected cases likely exceed 16; pytest not installed here, not collected) | POC/milestone5/tests/test_m5_compute_human_audit.py | "16 test functions" |
| 10 | 2.4 step 7 / 4.3 "a third adjudicates disagreements"; "dual evaluation ... blind human dual-rater consensus" | OVERCLAIM / UNSUPPORTED. Adjudication not performed: gold_from_adjudication 0, 18 unresolved claims, 102 agreed | M5E/human_audit/human_audit_summary.json counts; gold_labels.csv (102 rows, source=agreed); disagreements.csv (18 rows) | State design vs status; metrics over 102 agreed claims. Added agreement 85.0% and pooled kappa 0.384 (from summary.pooled) |
| 11 | 4.3 "retaining all non-accepted and proxy-contradicted claims plus a balanced random sample of accepted claims" | ERROR. Script keeps every claim that is not (hybrid accept and proxy supported), including proxy-unverifiable; tops up with an unbalanced random sample (rng seed 2026) | POC/milestone5/scripts/m5_eval_captioners_and_verifier.py lines ~1068-1080 | Rewritten; added caveat that packet over-represents hard claims so groundedness describes the packet |
| 12 | 4.3 "unbiased human ground truth" | OVERCLAIM. Raters are team members (SRP wrote the packet schema and rated 60 claims); blind to verdicts only | human_audit_summary.json coverage; m5_compute_human_audit.py SLICES | Removed "unbiased"; stated raters are blind but not independent |
| 13 | 4.3 "programmatically fails on any column leak" | Imprecise. Check rejects columns ending `_decision` or in {proxy_label, hybrid_verifier_prob, vlm_triggered} | m5_compute_human_audit.py lines 20, 66-68 | Named the columns |
| 14 | 1.2 "sub-pixel/multi-pixel misregistration shifts" | ERROR. Shifts are 2, 4, 8 px, none sub-pixel. 17 conditions includes `clean` | stress_test_levir_mci_test.json conditions | "clean plus 16 corruptions", shifts 2, 4, 8 px |
| 15 | 1.4 obj 2 "confirming simultaneous gains in ... groundedness, hallucination rejection, VLM savings" | OVERCLAIM (stated as foregone conclusion). Test proxy precision: hybrid 0.9394 vs mask-only 0.9366, 0.6 pts apart; human packet: mask-only 100% (n=71) vs hybrid 100% (n=62) | M5E/test_captioner_and_verifier/test_verification_summary.json test_settings_comparison; human_audit_summary.json policies | Reworded as a question being tested |
| 16 | 1.4 obj 1 "across every detector, captioner, and verification policy" | OVERCLAIM. Wilcoxon/McNemar exist for detectors and captioners; no per-policy paired test found in the scripts | m5_eval_detectors_and_stress.py L464-529; m5_eval_captioners_and_verifier.py L728-760 | "where the comparison is paired" |
| 17 | 1.4 obj 4 "deployment-grade", "authentic held-out Test ... UI traces" | OVERCLAIM (marketing). Traces exist in test_e2e_traces | M5E/test_captioner_and_verifier/test_e2e_traces | Neutral wording |
| 18 | Intro "immutable Evidence Review Web UI" | UNSUPPORTED. No evidence of immutability | none found | "Evidence Review web interface" |
| 19 | Table 2.1 local row "einops, timm, OpenCV, NLTK"; "Omen"; "Colab Pro" | UNSUPPORTED. Receipts record only python, torch, GPU name, hostname | run.json environment | Replaced with n/a statement; dropped "Omen" and "Pro" |
| 20 | Table 2.1 T4 row "Triton 3.8.0; Transformers 4.39.3" | UNSUPPORTED for training time. Triton 3.8.0 comes from the later profiling session; 4.39.3 is the requirements pin | compute_profile_colab_t4.json env; requirements-captioner.txt | Reworded with sources |
| 21 | 2.1 "Milestone 4 verification receipt records bfloat16" | Incomplete. Test verification summary records float16 | test_verification_summary.json models.referee.dtype | Added |
| 22 | 2.1 "every captioner receipt" kernel_mode | Imprecise. 9 training receipts have it; C00 is eval-only | M4 captioner_runs/*/run.json | "all nine captioner training receipts" |
| 23 | 2.1 "exact fused Triton" / "do not exist for Python 3.13" | Soft overclaim. M4 report says "unavailable" | reports/Milestone4_Report.md line 70 | "were not available" |
| 24 | 2.2 "Every ... receipt repeats these digests, and the trainers assert exact pair counts and zero ... overlap" | Imprecise. Receipts repeat train and val digests only; assertion confirmed only for detector trainer (6444/1242/1858 and zero overlap) | scripts/train_detector_full.py L377-391; run.json data_scope | Narrowed; added SECOND-CC train split (3,241 pairs) unused (portable_data_receipt.json) |
| 25 | 2.3 "identical across resumed and uninterrupted runs"; "empirical reproducibility bound" | OVERCLAIM. Only shuffle seed formula verified; 3.11e-4 delta is loss delta from M4 report | scripts/train_detector_full.py L559; reports/Milestone4_Report.md L76 | Softened; cited source |
| 26 | 2.3 "digests of all 12 frozen checkpoints" (Section 3.3) | Imprecise. model_registry.json has 12 entries but C00 is not among them (its sha is in captioner_runs/C00_rscama_released/eval_val.json); table shows 7 | POC/milestone5/configs/model_registry.json | Registry statement in 2.3; removed "all 12" in 3.3 |
| 27 | 2.4 step 3 "fixes the referee operating threshold (tau = 0.70) and the routing policy" | Incomplete. tau 0.70 is M4 receipt; M5 hybrid tau_accept 0.60, tau_reject 0.75, mask_alarm 50 were locked on the same 147 claims, so Val verifier numbers are calibration-set numbers | M4 verification_comparison_summary.json operating_threshold; test_verification_summary.json validation_calibration_locked; model_registry.json calibration_split | Rewritten |
| 28 | 3.1 captioner "39.81 GFLOPs/pair" | Caveat missing | compute_profile_rtx4060_cap.json flops_note | Added "excludes elementwise scan ops"; D03 17.45 counts conv/matmul/attention |
| 29 | 3.1 "245.53M" implied active | Clarified as trainable (231.90 + 13.63) | C06 run.json trainability | Added arithmetic |
| 30 | 4.1 item 3 "risks geographic tile leakage ... breaks comparability" | UNSUPPORTED as stated fact | none found | Reworded as rationale, not measured |
| 31 | 4.1 item 3 "3-seed replication on both D03 and C06" | Incomplete (A03 also) | aggregated_summary.json detector_multiseed | Added A03 |
| 32 | 4.5 "Project Success Criteria" | GAP/UNSUPPORTED provenance. No dated record of when thresholds were fixed; several sit just under Test results (mIoU 74.0 vs 75.93; BLEU-4 0.60 vs 0.6128) | grep M5_MASTER_PLAN.md, README.md, REPORT_OUTLINE_AND_EVIDENCE_MAP.md found no thresholds | Added neutral disclosure sentence; verifier targets marked as proxy-label based; latency row marked as target |
| 33 | 1.3 "Selected Production Detector/Captioner/Verifier" | OVERCLAIM (no production deployment evidence) | none | "Selected" only |
| 34 | 1.3 `C00` "External Published Benchmark" | Imprecise. C00 beats C06 on Test BLEU-4 (0.6551 vs 0.6126) with lower unchanged FA (1.19% vs 3.57%); selected C06 on Val/other reasons | captioner_test_full.json models | "External published reference". Intro does not call C06 best. Flag for Sections 6.2 and 14 |
| 35 | Many | STYLE. Em dashes removed (title, bullets), bold phrases reduced, "strictly", "exact", "Crucially"-type intensifiers, "resolves this tension", "dramatically", "ironclad"-type language removed, sentences varied | n/a | Summary row |

## Verified without change (spot list)
Params D03 11,944,043 (head 254,531), A01 7,763,971, A02 7,763,107, A03 15,630,115, A04 41,038,391; C06 total 898,805,894, encoder 151,277,313, fusion 733,899,265 (frozen 501,995,520, trainable 231,903,745), decoder 13,629,316; wall times and s/epoch (1874.25/32=58.6; 2106.31/25=84.25; 2332.71/25=93.3; 4140.95/25=165.6; 1818.58/8=227.3); peak VRAM 2287.3/1409.5/2289.1/6570.4/4820.4; 681 total epochs (sum of epochs_completed over 33 receipts); hardware split 14/7/3 detectors, 9 captioners on T4; checkpoint sizes 47,898,192 B = 45.68 MiB, 1,797,923,010 B = 1,714.63 MiB; all SHA-256 in the table match receipts/registry/C00 eval_val.json; Test 1,858 (924 unchanged), 121.77M px = 1858*256*256, 9,290 = 1858*5; Val 610/632; audit strata Val 40/30/15/20/15 (147 claims) and Test 50/35/20/25/20 (185 claims); tau 0.60/0.75/50; seed 2026 sampling; bootstrap n=1000 seed 42; exact McNemar via binomtest; D03 Test mIoU 0.7593, F1 0.8683, ROC-AUC 0.9901; C06 Test BLEU-4 0.6126 (config JSON says 0.6128; captioner_test_full shows 0.612583), BERTScore-F1 0.949, FA 3.57%; hybrid VLM call rate 9.73% (90.27% saving).

## Numbers in the fixed chunk that are not in the original chunk
- 185 Test claims: test_verification_summary.json audit_scope.total_claims.
- 16 corruptions (17 conditions minus clean): stress_test_levir_mci_test.json conditions list; shifts 2/4/8 px from names.
- 3,241 SECOND-CC train pairs: portable_data_receipt.json second_cc.train.count.
- 12 checkpoints in registry: model_registry.json (len of checkpoints = 12).
- patience 7: D03 run.json config.patience; 8 in configs/detector_training.json.
- Batch 8 x 4 for A04 (8*4=32): A04 run.json.
- +30.37 pts: 0.382248 - 0.078570 = 0.303678.
- 22.44% vs 13.94% any-change F1 SECOND-CC Val: 0.22439 and 0.13945 in robustness_and_transfer_bit/siam.json.
- 1.8 times: 0.50 / mean(0.28, 0.27) = 1.82.
- 7.44% (C06 s0) and 4.43% (C06 s2026) unchanged FA; 5.70% to 7.75% restricted to C01, C04, C05, C07: val_unchanged_fa_rate at best epoch in history.jsonl (0.0744, 0.0443, 0.0570, 0.068, 0.0649, 0.0775).
- 5,469 MB vs 4,820 MB (diff 649 MB): peak_vram_mb 5468.9 and 4820.3.
- 102 agreed of 120, 18 disagreements, 85.0% agreement, pooled kappa 0.384: human_audit_summary.json counts and pooled.
- 60 claims per rater: coverage block / rater CSVs (60 rows each).
- 16 test functions: grep `def test_`.

## Outline gaps (not filled)
- No gap in chunk outline items for Sections 1-4 except: success criteria provenance (row 32); Section 2.1 lacks driver/OpenCV versions (not recorded, already stated); GPU utilisation intentionally absent.
- Adjudication of 18 disagreements remains pending (row 10); Sections 7 to 10 and Appendix J should be checked for the same overclaim.

## UNVERIFIED (kept with neutral wording)
- T4 "15 GB" and L4 "24 GB" VRAM (public specs; not in receipts beyond the T4 profile).
- Referee "Colab GPU (L4 or T4)" (receipt silent).
- Triton kernel gradient error 1.2e-7 to 4.4e-7 and resume delta 3.11e-4: only in reports/Milestone4_Report.md, no raw file located.
- Detector architectural description (L=4 tokens, ImageNet pretrained, 8-layer decoder): from M4 report prose, not re-derived from source.
- Upstream commit pins for ChangeFormer (`afd1b7ed640a`): found only in M5 SOURCES.md and report, not in an independent receipt.
- "Evidence Review UI" behaviour, 200 ms latency target relation to Section 12 (mask-first path latency not recomputed here).

## Sections 5-9


E = evidence root `POC/milestone5/evidence` (M5), M4 = `docs/milestone4/evidence/full_data_m4`.

| # | Original (short quote) | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | "adjudication ... logged cleanly for third-party adjudication" / table col "3rd Rater ... Panel" | ERROR/OVERCLAIM: adjudication never run; 18 disputes excluded | E/human_audit/human_audit_summary.json `counts.gold_from_adjudication=0`, `unresolved_excluded=18`; gold_labels.csv all source=agreed | Stated plainly; column renamed "Assigned Adjudicator" |
| 2 | "Zero Human-Verified Hallucinations ... eliminated 100%" | OVERCLAIM: only 62/62 agreed accepted; 5 hybrid-accepted claims disputed (ids 46,64,69,73,85), SS rated 4 contradicted; worst case 62/67 = 92.54% | test_audit_master_key.csv + rater_*.csv (my computation) | Reworded, caveat added |
| 3 | "120 stratified held-out Test claims" | ERROR: packet = all non-accepted or non-proxy-supported claims + random draw of accepted-supported (not representative) | scripts/m5_eval_captioners_and_verifier.py lines ~1066-1075 | Sampling described; coverage flagged as packet-only |
| 4 | "93.94% proxy precision vs 100% human" | ERROR: different claim sets (185 vs 62). On the packet proxy precision of hybrid = 59/67 = 88.06% | master key (my computation) | Both numbers stated, not compared |
| 5 | "This proves ... conservative lower bound" | OVERCLAIM: proxy also called 1 claim supported that humans rated contradicted (M5T_010); 12 proxy-contradicted -> human supported, 7 agree | proxy_gt_vs_gold confusion [[65,1,0],[12,7,0],[17,0,0]] | Rewritten with counts |
| 6 | "Independent Confirmation 4-View hurts" | OVERCLAIM: 48/54 vs 94/102, CIs overlap | human_audit_summary.json policies | "agrees in direction" |
| 7 | "independent" human evaluation | OVERCLAIM: raters are project team | AGENTS/brief | Wording changed |
| 8 | "Test BLEU/ everything from a single locked pass ... locked on Validation before opening Test" | OVERCLAIM: M4 report already reported Test (75.93, C06 0.6128) | reports/Milestone4_Report.md lines 189-215 | Reworded; M4 0.6128 vs M5 0.6126 distinguished |
| 9 | "12 of 17 stress conditions" | ERROR: 17 includes clean; D03 beats A03 in 12 of 16 perturbed, loses 4 (cloud haze -6.48); only D03 vs A03 tested; most wins <1.5 pt | E/test_detector_and_stress/stress_test_levir_mci_test.json (my count) | Corrected |
| 10 | "confirming zero Validation overfitting" | OVERCLAIM | 75.93 vs 74.54 | Softened |
| 11 | "C00 473-word vocabulary ... longer schedule" | ERROR 473 -> 468; "longer schedule" UNSUPPORTED | M4/captioner_runs/C00_rscama_released/eval_test.json `vocab_size 468` | 468; schedule claim removed |
| 12 | "ObjHall% fraction of pairs" | ERROR: denominator is object claims (1,096 for C06) | scripts/m5_eval_captioners_and_verifier.py lines 229-271, captioner_test_full.json `object_claims_total` | Definition fixed |
| 13 | "+0.1305 ROUGE-L", "+0.3247 CIDEr" vs always_no_change | ERROR (rounding): 0.130409, 0.32461 | captioner_test_full.json paired_significance | +0.1304, +0.3246 |
| 14 | C06 "Statistically significant dominance", "recovers 99.75% of published BERTScore" | OVERCLAIM: C06 significantly below C00 on ROUGE-L/CIDEr/BERT (p 1.7e-7, 6.6e-7, 0.011); vs C01 ROUGE p=0.060, CIDEr p=0.098; vs C04 BERT p=7.3e-8 favours C04; BERTScore floor 0.9291 | paired_significance | Rewritten, p-values added |
| 15 | C06 "best groundedness-to-fluency balance" | OVERCLAIM/trade-off hidden: miss rate 11.78 vs 6.53 (C04), 5.14 (C01) | Table 6.2 | Trade-off stated |
| 16 | Coordinator note: C06 vs C00 / selection | C06 not best on Test BLEU-4 (C00 .6551, C01 .6135, C06 .6126, C04 .6101); selected on Validation (Val B4 .6358, NC FA 3.32 among C01-C07) | captioner_test_full.json; M4 report line 215 | Added "Selection versus Test ranking" paragraph; Table 8.2 verdict qualified (C00 val B4 0.6668 is higher) |
| 17 | "complete generalization from Validation to blind Test" | OVERCLAIM; hybrid rules written on Val claims | val 94.50 vs test 93.94 | Softened, in-sample caveat |
| 18 | "blind verification audit" for 185 claims | MISLEADING: labels are proxy (mask/reference captions), not human | test_verification_summary.json audit_scope | Stated |
| 19 | "Lowest Selective Risk & Highest Precision"; "vastly outperforms"; Table 7.2 | OVERCLAIM: 0.28 pt over mask-only = ~1 claim (8/132 vs 9/142 contradicted accepted); mask-only recall 97.79 vs 91.18 | test_verification_summary.json test_settings_comparison | Rewritten; policies 5=6 identical noted |
| 20 | "Dramatic Calibration Breakthrough" | OVERCLAIM: hybrid confidences are hand-set constants (0.94, 0.84-0.96, 0.05, 0.50); bins 0.5-0.6 (n=25, acc .24) and 0-0.1 (n=27, acc .22) miscalibrated | scripts/m5_eval_captioners_and_verifier.py lines 410-425; test_calibration.m5_hybrid.reliability_bins | Caveat added |
| 21 | "< 15 ms ... zero VLM cost" | ERROR: witness P50 14.68 ms, max 18.1 | E/compute_profile/compute_profile_rtx4060.json stages.witness_opencv | Reworded to ~15 ms (P50 14.68) CPU witness |
| 22 | Table 6.1 Otsu FA@50 "-", Table 7.1 Otsu "<1.0 ms" | Otsu FA=100% (1.0 at all thresholds) available; latency UNSUPPORTED | detector_test_full.json non_learning_controls; no Otsu profile in compute_profile | FA 100.00 filled; latency "-" |
| 23 | 8.3 "swept tau in [0.50,0.90] ... calibrated routing rule" | UNSUPPORTED for hybrid on Val. Val sweep exists only for direct/reinspection (M4); hybrid sweep only on Test | M4/aggregated_summary.json threshold_sweep; script line ~994 | Rewritten accurately; Test hybrid range 93.85-94.03 |
| 24 | "Direct Qwen deferring 40.8% due to small-object visual ambiguity"; "crops remove spatial context and bias VLM" | UNSUPPORTED mechanisms | none found | Marked hypotheses/removed |
| 25 | 9.3 item 5 abstention rules "forces ... false accepts" | UNSUPPORTED: no ablation without these rules | settings list in test_verification_summary.json | Described as design choice, not run |
| 26 | 9.3 "strictly Pareto-dominating every ablation" | ERROR: mask-only has higher recall/coverage (Val .9907/78.91% vs .9626/74.15%; Test 97.79 vs 91.18) | verification summaries | Replaced by trade-off statement |
| 27 | Table 9.1 "Random H/V flips (D01)" | ERROR: `flip` = horizontal flip only (p=0.5); D15 adds vertical flip + jitter, so jitter not isolated | scripts/train_detector_full.py lines 130-155 | Corrected |
| 28 | Table 8.1 / 9.1 deltas vs D01 | CONFOUND unreported: D01, D03 max 35 epochs; D02, D04-D16 25 epochs | M4/detector_runs/*/run.json config.epochs | Caveat paragraph added |
| 29 | Table 8.1 D13 "Frz 5" etc; "Permanently frozen" | OK; D13 freeze_epochs=999 | run.json | kept |
| 30 | 8.1 finding 4 "WD 0.01 outperformed both" | WEAK: D04 and D05 identical mIoU 69.94, budget confound | EXPERIMENT_TABLES.md | Hedged |
| 31 | 8.2 "razor-sharp", "collapsed immediately", "mode collapse" | OVERCLAIM: C02 ran 2 epochs (stopped early), C03 best at epoch 0 with low changed B4; val B4 noisy by epoch | M4/captioner_runs/C02,C03/history.jsonl | Reworded to what the histories show |
| 32 | 8.2 / 9.2 "2.3x reduction", "halving hallucinations" | OVERCLAIM: seed repeats NC FA 3.32 / 4.43 / 7.44 (mean 5.06); B4 std .0183 > .0094 margin | M4/aggregated_summary.json captioner_multiseed; EXPERIMENT_TABLES.md | Seed caveat added |
| 33 | 9.2 "AdamW+Cosine (C06) vs Adam+Constant (C01)" | ERROR: confounded with CLIP freezing; clean pair is C04 vs C01 | run configs | Row uses C04 vs C01 (NC FA -2.05) |
| 34 | 9.2 "templates provide zero object-class hallucination" | ERROR: template ObjHall 11.66% (highest) | Table 6.2 | Corrected with numbers |
| 35 | 9.2 Triton "Zero accuracy loss", "37-hour sweep" | OVERCLAIM/derived: rel. grad err <=4.4e-7; 37 h is derived 58 ep x 38.5 min (naive epoch time from M4 report text, not a receipt); 3.88 GPU-h measured | M4 report line 70; SOURCES.md line 36; training_cost.json per_kind_gpu_hours.captioner 3.883 | Labelled as estimate |
| 36 | 7.1 "D03 outperforms A04 by +1.55" | OVERCLAIM: Wilcoxon p=0.22, CIs overlap | detector_test_full.json paired_significance | Hedged |
| 37 | 7.1 "indispensable", "strictly safer", "massive real-world robustness" | OVERCLAIM; absolute SECOND-CC F1 only 23.17 | stress/M4 report | Toned down; ensemble marked untested |
| 38 | 8.1 "Selected (best mIoU & lowest FA!)" | Qualified: best among D runs only; A03 Val 75.02 and A02 FA 4.11 are higher/lower | EXPERIMENT_TABLES.md | Qualified |
| 39 | Table 6.3 / 6.2 bolding | STYLE/accuracy: bold marked C06 best on metrics where C00 is higher | table 6.2 | Bold removed |
| 40 | Table 6.4a: rater strictness | ADDED finding: SS marked 17/60 contradicted (SRP 2, PA 7, SSSM 8); no rater used "unverifiable" | rater_*.csv; disagreements.csv | Added |
| 41 | "zero verdict leakage verified" | UNVERIFIED: only that rater files lack decision columns | full_blind_120.csv header | Reworded |
| 42 | "Figure 6.5" reference | UNVERIFIED numbering | none | Replaced by "the figure below" |
| 43 | Whole chunk | STYLE: removed em dashes (table "-" empty cells kept), marketing words (Dramatic, Breakthrough, Authentic, Massive, Decisively, Crucially, !), excessive bold | n/a | Single summary row |

## Verified without change
Table 6.1 (all cells, CIs, FA@50) vs detector_test_full.json; Table 6.2 vs captioner_test_full.json; Table 6.3 and all ratios (4.37, 3.50, 4.30, 5.01, 4.2, 6.3, 3.0, 2.1) vs test_verification_summary.json; Table 6.4a/b, kappa 0.384, 102/120, 94/102, proxy-vs-gold 70.59% / 0.244 vs human_audit_summary.json; Tables 8.1, 8.2 vs EXPERIMENT_TABLES.md and run.json; Table 7.1 P50/GFLOPs/params vs compute_profile_rtx4060.json; SECOND-CC 23.17/17.71 vs robustness_and_transfer_*.json; VRAM 4.82/5.47 GB vs history peak_vram_mb 4820.4/5469.0; Wilcoxon/McNemar p-values all computed in scripts/m5_eval_detectors_and_stress.py and m5_eval_captioners_and_verifier.py (none UNSUPPORTED).

## Outline gaps (not filled)
No qualitative success/failure examples or error-analysis prose in this chunk (E/error_analysis/failure_taxonomy.csv exists; presumably in another chunk). No Val-sweep for hybrid thresholds. No hybrid ablation without abstention rules or without witness.

## Numbers in fixed chunk not in original
- 468 (C00 vocab): M4/captioner_runs/C00_rscama_released/eval_test.json.
- +0.1304, +0.3246: 0.732356-0.601947 = 0.130409; 1.337656-1.013046 = 0.32461.
- 100.00 (Otsu FA@50): detector_test_full.json unchanged_fa_rates.
- 1,096 object claims: captioner_test_full.json.
- p-values 1.7e-7, 6.6e-7, 0.011 (C06 vs C00), 0.060, 0.098 (vs C01), 7.3e-8 (vs C04 BERT), 1.8e-19, 2.8e-16, 0.088, 0.22 (detector Wilcoxon): paired_significance blocks.
- 0.9291 BERTScore floor: always_no_change f1.
- 12 of 16 / 4 losses, -6.48: stress_test json, my count of 16 perturbed conditions.
- 655 px / 10,000 bins / [-15,15]: scripts line 286-337.
- 167 non-VLM claims = 185-18; 14.68 ms witness P50.
- 5 disputed accepted claims, 62/67 = 92.54%; packet proxy precision 59/67 = 88.06%; 62/25/7 hybrid on 94 supported; 8 contradicted all rejected: master key + gold computation.
- 14 vegetation / 2 parking / 1 bareland among 17 proxy-unverifiable; 7 "scene same as before" among 12; M5T_010: master key.
- 17 of 18 disputes supported-vs-contradicted; SS stricter in 13 of 14; SS 17/60, SRP 2, PA 7, SSSM 8 contradicted: rater CSVs, disagreements.csv.
- Hybrid Test sweep 93.85-94.03%: test_threshold_sweep.
- Direct Qwen Test 78.30% (tau .50) to 82.76% (tau .90), accept 31.35%: test_threshold_sweep.
- Val direct tau .90 93.62%, 32.0% accept: aggregated_summary threshold_sweep (0.9362; 0.3197).
- 10.3x = 100/9.73; 3.12 = 94.50-91.38; 7.41 = 70.37-62.96; Val recall .9626 vs .9907, accept 74.15 vs 78.91.
- Road IoU drop -9.38 (69.72-60.34), building -6.52 (73.92-67.40); FA +3.64: EXPERIMENT_TABLES.md.
- C06 seeds NC FA 4.43, 7.44, mean 5.06; B4 .6442, .6092; D03 3-seed 74.23 +- 0.28: aggregated_summary.
- 35 vs 25 epoch budgets, D03 stopped at 32: run.json.
- 2.05 pts = 7.75-5.70; 2.88x unchanged (kept).
- Val C00 B4 0.6668 (not in text, only used to qualify).

## Sections 10-12


Evidence root: `.../scratchpad/evidence/POC/milestone5/evidence/` (abbreviated EV).

| # | Original (short quote) | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | E2 "89 tiny+small pairs (<=1,000 px) ... Small (<1%) 48.88% F1" | ERROR (scope mismatch). 48.88% is `small_change_f1`, pooled over pairs with 0 < GT px < 655 (<1% of tile), which is 64 pairs, not 89. The 251-1000 bucket F1 is 62.92%. | EV/test_detector_and_stress/detector_test_full.json `small_change_f1`, `area_stratified`; scripts/m5_eval_detectors_and_stress.py lines 336-340; count of 64 from detector_per_pair_d03_a03.jsonl | Table cell now states 89 pairs (1-1,000 px), 64 pairs below 655 px, tiny 20 pairs 12.99%, pooled 48.88% |
| 2 | "Over 97% of detector pixel errors are foreground-vs-background" | Correct but loose. Off-diagonal sum 2,878,928, swaps 31,042 = 1.08%, so 98.9%. | detector_test_full.json `confusion_matrix` | Replaced with computed figures |
| 3 | E1/E2/E3/E4/E5 "Primary Root Cause" column (seasonal foliage, cast shadows, glare, canopy occlusion, stride-8 pooling, vocabulary ceiling) | UNSUPPORTED as causes. The CSV only asserts them; no counted analysis (no brightness/shadow/season statistic per pair). | failure_taxonomy.csv (assertion only); no per-pair radiometric data | Column renamed "Likely Cause (hypothesis)", wording hedged; 10.2 text says causes were not tested |
| 4 | 10.2 item 1 "identical residential neighborhoods ... deeper cast shadows and darker winter canopy" for 000977 and 002083 | UNSUPPORTED/ERROR. Viewing the images: 000977 is residential and darker in T2; 002083 is a wooded scene with a few houses (not residential) and D03 also predicts road-class blobs. "Winter" not evidenced. | detector_visual_cases/false_positive_unchanged__*; per-pair jsonl (2,593 and 2,300 FP px) | Rewritten as observation plus hedged hypothesis; added counted context: 120/924 pairs with any FP, 95 with >=50, top-20 pairs = 56.4% of FP pixels |
| 5 | 10.2 item 2 "overexposed ... specular brightness saturation breaks ResNet-18 local contrast features"; 001393 "subtle highway lane widening under muted grey asphalt" | UNSUPPORTED mechanism. Images: 001393 is a road with new lane markings plus a small building (both missed), 6,927 GT px. | gallery PNG; detector_error_mining.json | Rewritten; mechanism hedged; added counted fact: 22 of 934 changed pairs have zero predicted px (47,666 GT px), 3 in the largest bucket |
| 6 | 10.2 item 4 "single 141-pixel shed ... ResNet-18 stride-8 feature maps pool away"; 11.4 "ResNet18 stride-4 receptive field" | UNSUPPORTED and mutually inconsistent (stride-8 vs stride-4). BIT source not in repo; "shed" not evidenced. | grep of src/models/detector_architectures.py shows BIT built from official BIT_CD, stride not stated | Both stride claims removed; "shed" -> "building"; added 5 of 20 tiny pairs have zero predicted px |
| 7 | 10.3 "`test_001307` where ... `D03` and `C06` are simultaneously blinded" | ERROR for 001307: D03 predicts 7,910 px (F1 0.449); only 001303 is missed by both. | detector_per_pair_d03_a03.jsonl | Corrected |
| 8 | 10.3 false-accept attribution "D03 false positive on unchanged scene (001422, 001787)" | ERROR. 001422 is a small_change pair (claim road, mask 6,539 road px, proxy contradicted), 001787 is road_only (mask 4,625 building px). 000381 (unchanged, 499 px) omitted from list. | test_verification_claims.jsonl fields `stratum`, `mask_summary`, `dec_m5_hybrid`, `proxy_gt`, `hybrid_route_reason` | Rewritten pair by pair with mask counts; 6/8 and 3/4 mask-decided, 2/8 and 1/4 referee-decided confirmed |
| 9 | 10.3 001729 "with residual mask pixels" | ERROR. Mask shows 0 changed px (escalated because claim is hedged). | claims jsonl, 001729 any_change_pixels 0 | Corrected |
| 10 | 10.3 001038 rejected "due to seasonal color drift" | UNSUPPORTED. Image shows a field with different tone/texture; no referee reasoning recorded. | test_e2e_traces/test_rejected_second__..._001038/T1.png,T2.png | "reason not known" |
| 11 | "zero unverifiable claims erroneously accepted (0/19), confirming ... `100%` reliability" | OVERCLAIM. 0 of 19 on 185 claims from 150 pairs; 23 deferrals = 16 vegetation + 1 bareland + 6 exact count, covering the 19 unverifiable plus 4 supported. | test_verification_summary.json threshold sweep "0.60"; claims jsonl counts | Reworded to the count; no reliability claim |
| 12 | Intro "all 185 blind Test verification claims" | Imprecise: 185 claims come from 150 stratified pairs. | test_verification_summary.json `audit_scope` | Reworded |
| 13 | E5 "388-word train-only vocabulary ceiling & conservative no-change prior" | UNSUPPORTED as cause (vocab 388 is a fact, causal link untested). | captioner_test_full.json has rates only | Marked as untested hypothesis |
| 14 | E6 cause list ("woods", "bareland" mentions) | Partly wrong: of 23 defers, 16 vegetation, 1 bareland, 6 exact counts; "bareland" claims such as 001319 were not deferred. | claims jsonl | Replaced with the counted breakdown |
| 15 | 11 intro/11.1 "seed=2026" for all corruptions | ERROR. Detector corruptions use per-pair seed 10000+idx; captioner probe corruptions 20000+idx; 2026 only selects the 150 stratified blind pairs. | scripts/m5_eval_detectors_and_stress.py line 148; m5_eval_captioners_and_verifier.py line 348, 547 | Corrected |
| 16 | "Validation Selection Generalizes 100% to Held-Out Test ... confirming ... holds strongly" | OVERCLAIM. D03 wins 12/17 by mIoU, but A03 is better on clean Test (-0.80), brightness 1.25, JPEG q40, haze, cross-sensor; A03 has lower FA in most photometric/blur/JPEG rows; several margins <1 pt with no CIs. | stress_test_levir_mci_test.json (all 17 rows recomputed, all table cells match) | Rewritten as a mixed ranking; sub-1-point margins called ties |
| 17 | "Global Semantic Tokens Prevent Optical & Resolution Collapse" | UNSUPPORTED mechanism (never tested). | none found | Hypothesis wording |
| 18 | "Geometric Alignment Resilience" | OVERCLAIM: both models lose >25 mIoU points at 8 px shift and 5 deg. | stress json: D03 75.93 to 50.51 / 50.83 | Renamed "Misregistration", loss stated |
| 19 | "Honest Vulnerability ... BIT's global token attention pools bright cloud patches ... leaks FP ... validating the OpenCV witness's `overexposed_or_cloudy` flag" | UNSUPPORTED. No `overexposed_or_cloudy` flag exists anywhere in src/scripts/evidence (witness flags: low_contrast, underexposed, saturated, small_crop, illumination_corrected_*, subpixel_jitter_*). Witness never evaluated on haze. | src/evidence/visual_inspector.py lines 48-60, 255; grep "overexposed" finds only the report | Flag name removed; "not tested" stated; mechanism hedged |
| 20 | "motivating explicit cloud-mask pre-filtering (Section 14.2)", "adversarial ... (Section 14.7)" | GAP/cross-ref: submitted 14.2 and 14.7 do not mention cloud-mask pre-filtering or adversarial hardening. | POC/milestone5/submission/Milestone5_Report.md lines 876, 896 | References removed from my chunk; reviewer of Section 14 may add them |
| 21 | 11.2 "200 stratified held-out Test pairs (seed=2026)" | ERROR. Probe = the 150 stratified blind pairs + first 50 other Test pairs (87 unchanged, 113 changed). | m5_eval_captioners_and_verifier.py lines 772-774; probe json counts | Corrected |
| 22 | 11.2 "remarkably invariant", "raises ... false alarms" | OVERCLAIM. FA change 5 vs 7 of 87 pairs; miss change 26 vs 30 of 113; no CI. BLEU-4 haze delta is +0.00056 (report said +0.0005). | captioner_test_full.json `robustness_probe_200` (0.057471*87=5, 0.08046*87=7; 0.230088*113=26, 0.265487*113=30) | Counts added, hedged; +0.0006 |
| 23 | "Cross-Sensor" row | Overstated: it is a gamma + channel-gain change on T2, synthetic. | m5_eval_detectors_and_stress.py `corrupt_cross_sensor` | Labelled synthetic |
| 24 | 11.3 "root cause is predominantly semantic ontology mismatch rather than random detector failure" | OVERCLAIM. Even building recall is 22.80% and precision 58.29%, so domain/annotation differences also matter; not separable here. | second_cc_6class_dissection.json | Reworded: class mismatch is one reason, other factors not separated |
| 25 | 11.3 "correctly suppressing out-of-ontology tree, water, playground" | ERROR in framing: these are ground-truth changes that D03 misses (low recall), not correct suppressions. | same json | Removed |
| 26 | 11.3 "cross-sensor/cross-city domain shift" | UNSUPPORTED description of SECOND-CC. | none found | "another dataset" |
| 27 | Table 11.2 "Partial (bare earth/road)", "Direct match" | Our interpretation, not data. | none | "Possibly partly (our reading)"; "same name" |
| 28 | 11.4 "mitigated downstream by C06 ... hybrid mask-caption agreement" and "mitigated by the OpenCV witness quality check" | UNSUPPORTED (no measurement of FP removal; witness check not evaluated on haze). | none found | Removed "mitigated" |
| 29 | 11.4 "signed provider archives rather than untrusted user uploads" | UNSUPPORTED rationale. | none found | Removed; stated attacks not run, resistance unknown |
| 30 | Table 12.7 hybrid cost "$0.0122" | ERROR. 1000 x (0.01748 + 0.0236 + 1.2333 x 0.01078 + 0.12 x 0.24584) x 0.526/3600 = 0.012255 (0.012262 with the report's rounded inputs), i.e. 0.0123. Seconds per pair 0.0839 (0.084) and 11.92 pairs/s agree. Other rows recheck: 0.0519, 0.0470, 0.0079. | compute_profile_colab_t4.json (mean 17.48 ms, witness mean 10.78), rtx4060.json (triton mean 23.6), colab_t4_referee.json (mean 245.84); test_vlm_efficiency | Changed to $0.0123 |
| 31 | 12.5 "compute saturation is evidenced by ... launch-bound ... approaching compute saturation" | OVERCLAIM from throughput only. | batch scaling 2.08x, 1.11x recomputed from json (134.74 ms and 110.14 ms batch-16 P50) | Hedged as inference; GPU utilisation not measured |
| 32 | Section 12 spot checks | Verified OK: training totals (33 runs, 681 epochs, 997.8 min, 16.63 h; per-GPU rows), latency P50/P95/P99 for D03 on T4/4060/MPS/CPU, captioner 23.30/89.35, witness 14.30, referee 243.7/336.0, peaks 551.3/551.5/1,371.7/8,906.2/3,518.4/4,403.8/8,589.9/8,652.2, 12,660 sum, 3.83x, 35.4%, passes/warm-ups, commits f568a1d and 4be3e22, 35 and 23 distinct pairs. Note: Mac captioner rows labelled "full-length decode" carry the `ref_scan` values (193.86 MPS, 438.37 CPU), not `ref_scan_full_len` (208.0, 436.75); the report explains random weights decode 42 steps, left as is. | compute_profile_*.json | Prose only |
| 33 | STYLE | Em dashes removed (none in section 12), bold removed from table cells and most prose, marketing phrasing ("Honest Vulnerability Exposed", "Decisive", "Strong", "Crucially"-style) removed, Winner column simplified, Section 12 intro sentences smoothed. Heading numbers, table numbers, figure links unchanged. | n/a | n/a |

## Verified unchanged (spot list)
Table 11.1 all 17 rows x 8 numeric columns recomputed from stress_test_levir_mci_test.json: all match. Table 11.1b all cells match. Table 11.2 and all prose numbers (22.80, 21.66, 6.16, 88%, 22.44/13.94, 23.17/17.71, 5.46/8.50, 70.46) match second_cc_6class_dissection.json. Taxonomy numbers (95/924, 12.99, 15.80, 12.53, 31,042, 0.29%, 33/924, 110/934, 4.93%, 8/132, 6.06%, 4, 23, 0/19) match.

## Outline gaps
- Error analysis: good coverage of patterns and where the model does well/poorly; no quantitative proof of any cause (stated as hypotheses). Success cases (top10 high-F1 in detector_error_mining.json, images in detector_visual_cases/success_high_f1_*) are not described in Section 10; could be added by someone with figure access.
- Robustness: no confidence intervals on the stress suite or probe; no real second-sensor data.
- Other section: submitted 14.2 says 63.88% mIoU at a 4-pixel shift, but Table 11.1 gives 64.56 (D03, shift_t2_4px, Test). Not my chunk, flag to the Section 14 reviewer.

## Numbers in the fixed chunk that are NOT in the original chunk
- 54 object-hallucination claims of 1,096: 0.04927 x 1096 = 54.0 (captioner_test_full.json `object_claims_total` 1096).
- 64 pairs with 1-654 GT px: count in detector_per_pair_d03_a03.jsonl; F1 recomputed 0.48880 equals `small_change_f1`.
- 5 of 20 tiny pairs with zero predicted px; 22 of 934 changed pairs with zero predicted px, 47,666 GT px, 3 in the large bucket: per-pair jsonl.
- 2,878,928 misclassified pixels: 499,938 + 849,805 + 650,809 + 9,511 + 847,334 + 21,531; 31,042/2,878,928 = 1.08%; 98.9% = 100 - 1.08. Source: confusion_matrix.
- 120 unchanged pairs with at least one FP pixel (`unchanged_fa_counts["1"]` = 120); top-20 FP-pair share 56.4% (per-pair jsonl).
- 2,300 FP px on test_002083; 7,910 predicted px and F1 0.449 on test_001307; 6,927 GT px on test_001393 (per-pair jsonl).
- Mask counts in 10.3: 499 (000381), 4,625 (001787), 6,539 (001422), 0/0/8 (000092/001250/001760), 226 (000299), 0 (000840), 436 (001417), 145 (001319): test_verification_claims.jsonl `mask_summary`.
- 23 defers = 16 vegetation + 1 bareland + 6 exact-count (3 building, 2 parking, 1 road) and cover all 19 unverifiable plus 4 supported: claims jsonl.
- 2,878,928 aside, 62.92% not used in the final text (only mentioned here).
- 87 unchanged and 113 changed pairs in the probe; 5 vs 7 of 87 and 26 vs 30 of 113 (rates x counts); BLEU-4 haze +0.0006 (0.548747 - 0.548183 = 0.000564).
- Corruption seeds 10000+idx and 20000+idx; shift offsets 8 px horizontal, 4 px vertical (`corrupt_shift(b, 8, 4)`).
- "more than 25 mIoU points" lost at 8 px / 5 deg: 75.93-50.51 = 25.42, 75.93-50.83 = 25.10, A03 76.73-49.16 = 27.57, 76.73-48.55 = 28.18.
- Combined 11.3 range "between 2.39% and 13.48%" (other classes' D03 recall).
- Table 12.7 hybrid cost $0.0123 (see row 30). Throughput sentence: 2.08x = 119.0/57.2, 1.11x = 144.6/129.8 (already in original).

## Sections 13-16


Paths: EV = evidence checkout `.../scratchpad/evidence/`, M5E = `EV/POC/milestone5/evidence/`, M4E = `EV/docs/milestone4/evidence/full_data_m4/`.

| # | Original (short quote) | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | "corpus BLEU-4 is 0.6128 ... changed-pair BLEU-4 is 0.3681" (13.2, Table 14.1) | ERROR. Test values are 0.6126 and 0.3686; the report elsewhere uses 0.6126 | M5E/test_captioner_and_verifier/captioner_test_full.json models.C06.bleu4 0.612583, changed_bleu4 0.368582 | Corrected in 13.2 and Table 14.1 |
| 2 | "15.81% of road pixels and 12.57% of building pixels ... background" (13.2, 14.3) | ERROR. Row-normalised confusion gives 15.80% and 12.53% (Section 10 of the report already says 15.80 / 12.53) | M5E/test_detector_and_stress/detector_test_full.json D03.confusion_matrix_row_norm [0.158029 ..., 0.125273]; raw counts 650,809/4,118,292 and 847,334/6,763,905 | Corrected |
| 3 | "confused ... for less than 0.35% of pixels" (13.2) and "<0.29% road-vs-building confusion across 121.77M Test pixels" (16.1) | ERROR / mismatch of denominator. Row-normalised confusion is 0.23% (road to building) and 0.32% (building to road). The 0.29% figure is 31,042 px over 10.88M foreground pixels, not over 121.77M pixels | detector_test_full.json D03.confusion_matrix_row_norm [0.002309, 0.003183]; report Table 10.1 E4 | Replaced with 0.23% and 0.32% of respective class pixels |
| 4 | "small-change subset ... 48.87%" (13.2, 13.6, Table 14.1) | ERROR (rounding). Test value 0.488796 rounds to 48.88; 48.87 comes from the M4 re-evaluation (0.488746) | detector_test_full.json D03.small_change_f1 0.488796; report Table 6.1/10.1 use 48.88 | Changed to 48.88 everywhere. Definition added: fewer than 655 changed pixels (`scripts/train_detector_full.py` line ~302, `POC/milestone5/scripts/m5_eval_detectors_and_stress.py` line 336) |
| 5 | "+12.38 and +30.36 mIoU points at radius 1.0 and 2.0" (13.2, 14.5) | ERROR + ambiguity. These are Validation deltas (A03 vs D03) mixed into a paragraph of Test numbers; exact Val delta at radius 2.0 is 30.37 (0.382248 - 0.078570). Test deltas are +11.98 and +33.63 | M4E/robustness_and_transfer_bit.json and _siam.json levir_mci_val_corruptions blur_both_r1.0/r2.0; stress_test_levir_mci_test.json | Labelled as Validation, 30.37, and Test +11.98 / +33.63 added |
| 6 | "tau = 0.70" as the verifier operating point (14.4) | ERROR / incomplete. 0.70 is the M4 referee gate; the M5 hybrid locks tau_accept 0.60, tau_reject 0.75 | M5E/test_captioner_and_verifier/test_verification_summary.json validation_calibration_locked.locked_params; M4E/verification/verification_comparison_summary.json operating_threshold 0.7 | Both stated |
| 7 | "validated against ... full 14-condition stress matrix" (14.5) | ERROR. Test suite has 17 conditions | stress_test_levir_mci_test.json conditions (len 17) | Changed to 17-condition Test matrix |
| 8 | "its latency and memory on consumer hardware are reported in Section 12" (13.5) | ERROR. The referee was profiled only on the Colab Tesla T4; no consumer-hardware referee run exists. Peak 8,589.9 MB exceeds the 8 GB RTX 4060 laptop GPU | M5E/compute_profile/compute_profile_colab_t4_referee.json (peak_mem_mb 8589.95/8652.23); no referee entry in compute_profile_rtx4060.json | Rewritten: not measured on consumer hardware; peak exceeds 8 GB |
| 9 | Table 14.1 "4B bf16 model" | ERROR. M5 Test referee run and profile used float16 (M4 Validation audit used bfloat16) | test_verification_summary.json models.referee.dtype torch.float16; compute_profile_colab_t4_referee.json dtype float16; M4E verification_comparison_summary.json dtype bfloat16 | Changed to "4B-parameter referee ... float16 on the T4"; Section 13.5 states float16 |
| 10 | "so throughput scales linearly with devices" (13.4) | UNSUPPORTED. No multi-device run; end-to-end throughput was not measured at all (report Section 12.5) | none found | Reworded to "can in principle be spread over devices; no multi-device run was made" |
| 11 | "Training also set cudnn.benchmark = True" (13.2) | ERROR (scope). Only the detector trainer sets it | EV/scripts/train_detector_full.py:366; no match in captioner scripts (grep) | Changed to "detector trainer" (matches Section 2.3 of the report) |
| 12 | "checks object class, change type and explicit counts against evidence" (13.2) | Self-contradiction: elsewhere explicit counts are deferred, not verified | M4E/verification/verification_claims.jsonl val_000237 forced_reasons exact_count_without_independent_instance_evidence; Test hybrid route counts deterministic_defer_ontology_or_exact_count 23 | Reworded: checks class and type, defers counts |
| 13 | "(iv) ... deterministically deferred, as in trace val_000237" (13.6) | OVERCLAIM. In that trace dec_mask_only is accept; deferral comes from the referee-based policies | M4E/verification/verification_claims.jsonl (val_000237) | Qualified |
| 14 | "Failures (iv) to (vi) are converted into deferrals or flagged items by the safety gate" (13.6) | OVERCLAIM. (v) is a residual error: hybrid still accepts 8 of 132 proxy-contradicted Test claims | test_verification_summary.json test_settings_comparison.7_m5_calibrated_mask_first_hybrid breakdown accepted_contradicted 8, accepted 132 | Limited to (iv) and (vi); (v) stated as residual with 8/132 |
| 15 | "Proxy labels ... The blind dual-rater study addresses this only for its own claim set" (13.1) | GAP / OVERCLAIM context. Human audit has 18 unresolved disagreements, no adjudication (gold_from_adjudication 0); raters are team members; pooled kappa 0.384; proxy vs human agreement 70.59%, kappa 0.244 | M5E/human_audit/human_audit_summary.json counts, pooled, proxy_gt_vs_gold | Added to 13.1 and Table 14.1 (adjudicate the 18) |
| 16 | "13 claims routed for being out of the mask ontology" (13.1) | Missing qualifier: these are Validation audit claims (Test hybrid has 23 deterministic defers for ontology or exact count) | M4E verification_comparison_summary.json trigger_reason_counts T_out_of_mask_ontology 13; test_verification_summary.json hybrid_route_counts | Labelled Validation |
| 17 | "D03 was selected for ... SECOND-CC transfer (+5.46 any-change F1 points), so deployments ... give up 0.80 mIoU points" (13.2) | INCOMPLETE. D03's SECOND-CC advantage is recall only; A03 has precision 83.80% vs 58.29% and FA@50 4.62% vs 25.08% | M4E/robustness_and_transfer_siam.json second_cc_transfer.test (any_change_precision 0.838006, unchanged_fa_rates 50 = 0.046204); bit.json (0.582896, 0.250825) | Added the trade-off sentence |
| 18 | "falls to ... 63.88% under a 4-pixel registration shift. Imagery ... is outside the validated operating range" (13.2) | Weak: haze loss for D03 (62.66 vs A03 69.13, FA@50 26.95%) was not in the limitations | stress_test_levir_mci_test.json cloud_haze_t2_15pct | Added Test ladder values and haze; "validated operating range" softened |
| 19 | (not in original) captioner trade-off | GAP. C06 vs C01 and vs released C00 trade-offs missing; C00 beats C06 on all text metrics and FA (1.19%) | captioner_test_full.json models.C00 (bleu4 0.655098, unchanged_fa_rate 0.011905), C01 (0.094156, miss 0.051392) | New paragraph "Captioner trade-off" in 13.2 |
| 20 | (not in original) hybrid vs mask-only on Test | GAP. Hybrid precision 93.94% vs mask-only 93.66% (accepts 132 vs 142; contradicted accepts 8 vs 9) | test_verification_summary.json test_settings_comparison 2_mask_only_checker, 7_m5_calibrated_mask_first_hybrid | Added to 13.2 and 15.1/15.2 |
| 21 | "Higher-resolution sources, which the improvement paths in Section 14 invite" (13.7) | Slightly inaccurate: Section 14 proposes higher-resolution decoders, not imagery | Section 14 text | Reworded |
| 22 | 13.1/13.7 "Evidence Review ... checkpoint digests" | UNVERIFIED part. Immutable model output and append-only SQLite audit log verified; "checkpoint digests" in the store not checked | EV/src/app/server.py lines 5, 99 | Dropped "checkpoint digests" |
| 23 | Table 14.1 "Review corrections are not yet used for learning" | OK but sharpened: no reviewer-override data exists to use | no override dataset in evidence | Reworded to "no reviewer data has been collected or used" |
| 24 | 14 intro: improvements | OVERCLAIM risk. Added statement that none is implemented or tested; benefits are expectations | n/a | Added |
| 25 | "(63.88% mIoU at a 4-pixel shift)" called "the dominant sensitivity" (14.2) | UNSUPPORTED ranking. Blur r2.0 (38.22 Val / 40.54 Test) and 8 px shift (50.51 Test) are larger drops | stress_test_levir_mci_test.json | Reworded to "one of the clearest sensitivities", added Test values |
| 26 | "D15 68.34%, D16 68.55% against 71.82% for D01" (14.2) | Confound not stated: D01 ran up to 35 epochs, D14-D16 25 | M4E/aggregated_summary.json detector_ranking epochs_run | Caveat added, matched-budget test suggested |
| 27 | "head learning rate ... and backbone unfreezing, which interact" (14.4) | UNSUPPORTED (factors never varied jointly) | M4E aggregated_summary detector_ranking | "may interact, since not varied together" |
| 28 | "oversampling road-bearing tiles is a direct response to the lower road IoU" (14.1) | UNSUPPORTED causal link | none | Reworded as candidate; IoU gap does not show class frequency is cause |
| 29 | "A03 wins ... BIT wins under blur and transfer; feature fusion targets both" (14.3) | Speculative; marked untried; "transfer" narrowed to SECOND-CC recall | see #17 | Reworded |
| 30 | "Seed ensembling of the three D03 and three C06 runs ... lower-cost variance reduction" (14.5) | ERROR/OVERCLAIM. Ensembling multiplies inference cost; the registry pins three D03 checkpoints but only one C06 checkpoint | EV/POC/milestone5/configs/model_registry.json checkpoints (12 entries, one C06) | Reworded |
| 31 | "Distilling ... or quantising to 4-bit reduces memory and per-call latency" (14.6) | Speculative gain stated as fact | none | "would be expected to"; "neither was tried" |
| 32 | (not in original) checkpoint pruning in 14.6 / 16.2 | Moved the Section 16.2 claim "halve C06's disk footprint without changing outputs" to computed expectation: 501,995,520/898,805,894 = 55.85%; 1,714.6 x (1 - 0.5585) = 757.0 MB. Original "halve" and "without changing outputs" unsupported (no run) | M5E/compute_profile/compute_profile_rtx4060_cap.json checkpoint_size_mb 1714.633, params_total 898,805,894; unused tables 501,995,520 (brief); checkpoint size = 898.8M x 2 B / 2^20 = 1,714.4 MiB (fp16 storage) | Phrased as calculation, not measurement; regression run needed |
| 33 | "so raising its coverage through better detector calibration shrinks the referee workload directly" (14.6) | Speculative | none | "any gain in coverage at equal precision would reduce ..." |
| 34 | "Human-in-the-loop learning is the most promising direction" (14.7) | UNSUPPORTED ranking; "corrections are labelled hard cases targeted at exactly the claims the system gets wrong" is overclaim | none | "natural next step"; "would supply labelled examples" |
| 35 | "Every Milestone 5 objective was met or exceeded" (15.1) | OVERCLAIM. Robustness target met against A03 only, not on all conditions (D03 loses 5 of 17); verifier objective met numerically but hybrid is only +0.28 pt over mask-only | stress_test_levir_mci_test.json; test_verification_summary.json | Rewritten against Section 4.5 targets with caveats |
| 36 | "exceeded Validation performance (74.54%), confirming zero overfitting" (15.1) | OVERCLAIM. Test 75.93 vs Val 74.54 is +1.39 pt; this does not establish "zero overfitting" | detector_test_full.json change_miou 0.759296; M4E robustness_and_transfer_bit.json clean 0.745394 | "no generalisation gap is visible on this split" |
| 37 | "met all fluency targets while holding unchanged-scene hallucinations to 3.57% (2.64x lower than C01)" (15.1) | Incomplete. Misses on changed pairs 11.78% vs C01 5.14%; released C00 better on all metrics | captioner_test_full.json | Added; 2.64 = 0.094156/0.035714 = 2.636 |
| 38 | "Hybrid ... 100.00% human-verified groundedness across 62/62 consensus-agreed blind Test claims ... eliminated all 8 human-verified caption hallucinations" (15.1) | OVERCLAIM. Gold set is 102 agreed claims of 120; 18 unresolved/unadjudicated; 5 of the hybrid's 67 accepts are unscored; mask-only also 71/71; interval upper bound for hallucination 5.8% | human_audit_summary.json policies.hybrid (n_accepted 67, with_gold 62, groundedness_ci95 [0.9417, 1.0], hallucination_ci95 upper 0.0583), policies.mask_only (71/71), counts.unresolved_excluded 18 | Reworded with these caveats; "none of the 8 ... was accepted" |
| 39 | "one-tenth the VLM compute cost" (15.1) | Imprecise: it is the referee call rate (9.73%), not compute cost | test_verification_summary.json test_vlm_efficiency m5_hybrid_vlm_call_rate 0.0973, vlm_savings 90.27 | Reworded to "referee calls" |
| 40 | "Specialized Small Vision Models Beat General 4B VLMs" (15.2 #1) | OVERCLAIM. Based on a proxy-label audit; "simultaneously maximized precision, calibration and throughput" unsupported (mask-only 93.66% vs hybrid 93.94%; throughput not measured end to end); causal "due to loss of global scene context" untested | test_verification_summary.json test_settings_comparison, test_calibration | Rewritten; cause marked untested |
| 41 | "(`9.7%` of claims)" "BIT's structural blind spots" | Wording: 9.73% is referee-escalation rate | see #39 | Reworded |
| 42 | "A03 collapsed on blur ... Multi-condition stress evaluation proved essential" (15.2 #2) | Numbers verified; "proved" overclaim and D03 weaknesses omitted | stress_test_levir_mci_test.json (A03 6.91 vs 40.54; 59.94 vs 71.98) | Softened; added D03 wins 12/17 incl. margins under 1 pt and its 5 losses |
| 43 | "`D03` retained meaningful precision (`58.3%`)" on building (15.2 #3) | ERROR (misattribution). 58.29% is any-change precision over all pairs, not building precision (building-change precision is 63.09%) | M4E/robustness_and_transfer_bit.json second_cc_transfer.test any_change_precision 0.582896, building_change_precision 0.630866 | Stated as any-change precision; building-class figures replaced with recall-based statement (22.80% recall, 88% label-correct = 20.16/22.80) |
| 44 | "Dissecting ... proved that 70.46%" (15.2 #3) | OVERCLAIM verb; class list incomplete (trees omitted) | M5E/stress_tests/second_cc_6class_dissection.json D03 test building gt_share 29.54 -> 100 - 29.54 = 70.46 | "In the dissection ..."; added trees |
| 45 | "10.2x (227 s/ep vs 2,310 s/ep)", "gradient parity within 4.4e-7" (15.2 #4) | Source is Milestone 4 report prose (38.5 min/epoch on T4, 1.2e-7 to 4.4e-7 relative error), not a machine receipt; 227 s is C06 run receipt (227.3) | EV/reports/Milestone4_Report.md line 70; M5E/compute_profile/training_cost.md C06 227.3 s/epoch | Marked "reported in Milestone 4"; range given |
| 46 | "enabling a full 33-run controlled experiment matrix in just 16.63 GPU-hours ($8.75 T4-equivalent)" | OVERCLAIM. Only the 9 captioner runs (3.88 GPU-h) depend on the kernel; $8.75 is a pricing convention | training_cost.json per_kind_gpu_hours captioner 3.883, grand_total 16.631; 16.631 x 0.526 = 8.748; 58 x 38.5/60 = 37.2 | Reworded; 3.88 and ~37 GPU-h added |
| 47 | "directly applicable ... workbench" (15.3) | OVERCLAIM for readiness. Never deployed. "41.3 ms/pair" sums P50 of D03 (7.64), P50 of C06 Triton (23.30) and the MEAN of the witness (10.39; P50 14.68), per claim not per pair, with no end-to-end run; report Section 12.5/12.6 states end-to-end not measured and gives 54.3 ms mixed-platform estimate. Referee peak 8,589.9 MB > 8 GB laptop | compute_profile_rtx4060.json D03 batch1 p50 7.6396; witness mean_ms 10.3873, p50 14.678; compute_profile_rtx4060_cap.json triton_scan p50 23.3019; 7.64+23.30+10.39 = 41.33; 1000/41.33 = 24.2 | Rewritten as a summed estimate with caveats, referee memory caveat added |
| 48 | "1,672 SECOND-CC-AUG zero-shot transfer pairs" (16.1) | ERROR. Test split is 1,142 pairs (1,672 = 530 val + 1,142 test); all reported transfer numbers use the 1,142-pair Test split | portable_data_receipt.json second_cc val 530 / test 1142; second_cc_6class_dissection.json test num_pairs 1142 | Changed to 1,142 |
| 49 | "four hardware profiling platforms" (16.1) | ERROR. Profiled platforms: Colab T4, RTX 4060, Apple M4 (CPU and MPS backends). L4 was never profiled for inference | M5E/compute_profile/*.json (six files, three devices); report Section 12 intro | "three profiled platforms (four backends)" |
| 50 | "leakage-free held-out evaluation", "185 blind Test verification claims" | OVERCLAIM wording | n/a | "offline", "185 Test verification claims from 150 pairs"; Test held out from training and threshold selection (locked on Validation per test_verification_summary.json) |
| 51 | "Reliable, Stress-Resilient Change Detection" ... "wins 12 of 17" (16.1) | OVERCLAIM. D03 falls to 40.54% under blur r2.0 and 62.66% under haze; "12 of 17" is only against A03 | stress_test_levir_mci_test.json | Heading neutral; "compared with A03" and losses listed |
| 52 | "Low-Hallucination, Accelerated Change Captioning" | OVERCLAIM/incomplete (see #37) | captioner_test_full.json | Neutral heading, 11.78% miss rate included |
| 53 | "High-Precision, Low-Cost Hybrid ... Blind Human Groundedness ... generalizes zero-shot ... zero unverifiable leaks ... Independently confirmed by 240 blind dual-rater human annotations (85.0% exact agreement, 100% error-category agreement)" (16.1) | OVERCLAIM. Raters are team members (not independent); 100% error-category agreement is 8 of 8 claims; kappa 0.384 omitted; 18 disagreements unadjudicated; mask-only 71/71 as well | human_audit_summary.json pooled (percent_agreement 0.85, kappa 0.3836, error_category_agreement n 8), counts | Rewritten with all caveats; "generalizes zero-shot" removed; "zero unverifiable leaks" -> "no proxy-unverifiable claim accepted" (accepted_unverifiable 0) |
| 54 | "10 authentic held-out Test Evidence Review bundles and a zero-leak 120-claim blind human-audit packet" (16.1) | "authentic" and "zero-leak" are unverified labels. 10 trace dirs and 12 registry checkpoints confirmed; packet has no decision columns and a blindness-violation test exists | M5E/test_captioner_and_verifier/test_e2e_traces (10 dirs); model_registry.json (12); full_blind_120.csv header; tests/test_m5_compute_human_audit.py test_blindness_violation | Reworded: "withholds model decisions from the rater files". "verified by automated regression tests" reduced to "regression tests exist" (tests were not re-run in this review) |
| 55 | "ready for controlled human-in-the-loop pilot deployment" (16.2) | OVERCLAIM. System evaluated offline on LEVIR-MCI Test and SECOND-CC only; no live deployment | n/a | "results support a controlled pilot as the next step ... not been deployed ... should not be used for unattended decisions" |
| 56 | "(2) ... close both the clean +0.80 pt and localized cloud-haze -6.48 pt gaps" (16.2) | UNSUPPORTED gains stated as outcomes. Haze gap -6.48 ✓ (62.6567 - 69.1343 = -6.4776); the "automated cloud-haze pre-filtering" has no evidence | stress_test_levir_mci_test.json | "aimed at ... with no result yet"; pre-filter dropped from the claim, ensemble kept |
| 57 | "(3) leveraging logged Evidence Review analyst overrides" (16.2) | Presented as available data; none collected | none | Stated as a plan (adjudication plus overrides) |
| 58 | "~15% deferred claim queue" (16.2) | Verified (28/185 = 15.14%) but Validation defer rate is 12.24%; now stated as Test | test_verification_summary.json 7_m5 defer_rate 0.1514 | "about 15% of Test claims" |
| 59 | "0.9490 BERTScore-F1", "1.3377 CIDEr-D" etc | Verified, unchanged | captioner_test_full.json | None |
| 60 | Pure style edits (all of 13-16) | STYLE. Removed "Crucially/Yes/..." style fillers, bold over-use, em dashes, marketing phrasing ("comprehensive", "cryptographically pinned", "return on investment", "multiplied research velocity"); varied sentence length; 15.1 bullets and 16.1 items kept in structure | n/a | Done |

## Verified unchanged (no edit needed)
All of the following were checked against evidence and left as in the original: D03/A03 Test mIoU 75.93/76.73 and FA@50 10.28/8.23; binary F1 86.83; ROC-AUC 0.9901, PR-AUC 0.9361; CI [74.79, 77.04]; SECOND-CC 23.17/30.55/58.29/14.46/25.08, +5.46; 17.71 (A03); Val ladder 70.94/38.22/67.99/63.88, FA 6.80 -> 19.30; Val small-change 42.72 and binary F1 86.25 (aggregated_summary val_binary_f1 86.25); C06 seed BLEU-4 values, 0.6297 +/- 0.0183, C01/C07/C04 values; 11.78% (110/934); 3.57% (33/924); Val verifier numbers (91.38/82.43/72.22/72.41, 116 accepted, 10 contradicted, 17 of 27 = 62.96%, 60/147 = 40.8%, 13 rejected, 3 of 52 untriggered); class weights 0.10/0.45/0.45; Test pixel counts 110.88/4.12/6.76/121.77M; road IoU 74.53, building 77.33; 1.225 / 0.792 referee calls per pair; 38.5 min vs 227 s; 33 runs, 16.63 GPU-hours; 1.9x T4 vs 4060; referee P50 243.7/336.0 ms and peaks 8,589.9/8,652.2 MB; 23.58 ms (16.93 + 6.65); 51 trigger events / 95 triggered claims; 3,241 SECOND-CC train pairs; D02/D01/D03/D13/D15/D16/ChangeFormer Val mIoU; 93.94 / 6.06 / 91.18 / 70.00 / 0.0710 / 90.27 / 9.73; 78.82, 3.3 (3.33), 69.61, ECE 0.446; A01 77.36; 6.91 vs 40.54; 59.94 vs 71.98; 70.46%; 22.80 vs 16.64; 7.84% and 11.11%; 85.0% agreement, 100% on 8; 62/62 and [94.2, 100.0]; 12 checkpoints; 10 traces; 3.83x (89.352/23.302 = 3.834); 502.0M; 227 s/ep; 10.2x; +12.03 (downscale, 12.035 unrounded), +33.63.

## Numbers in the fixed chunk that are NOT in the original chunk
| Number | Evidence |
|---|---|
| 0.6126, 0.3686 (replace 0.6128, 0.3681) | captioner_test_full.json C06 bleu4 0.612583, changed_bleu4 0.368582 |
| 15.80%, 12.53% (replace 15.81, 12.57) | detector_test_full.json D03 confusion_matrix_row_norm |
| 0.23%, 0.32% | same, entries [1][2] 0.002309 and [2][1] 0.003183 |
| 48.88% (replace 48.87) | detector_test_full.json small_change_f1 0.488796 |
| 655 pixels (small-change definition) | POC/milestone5/scripts/m5_eval_detectors_and_stress.py line 336 (`gt_chg < 655`, 1% of 256x256 = 655.36) |
| +30.37 Val (replace 30.36) | M4E robustness_and_transfer_bit/siam levir_mci_val_corruptions blur_both_r2.0: 0.382248 - 0.078570 = 0.303678 |
| +11.98, +33.63 (Test blur deltas) | stress_test_levir_mci_test.json: 72.0969 - 60.1185 = 11.978; 40.541 - 6.9144 = 33.627 |
| Test D03: 40.54, 68.66, 64.56, 50.51, cloud haze 62.66 vs 69.13, FA@50 26.95 | stress_test_levir_mci_test.json |
| A03 SECOND-CC Test precision 83.80%, FA@50 4.62% | M4E/robustness_and_transfer_siam.json second_cc_transfer.test (0.838006; 0.046204) |
| 1.19%, 9.42%, 5.14% (C00 / C01 rates) | captioner_test_full.json (0.011905; 0.094156; 0.051392) |
| 70.59%, kappa 0.244 (proxy vs human) | human_audit_summary.json proxy_gt_vs_gold |
| 0.384 kappa, 18 disagreements, 102 agreed, 8 of 8 | human_audit_summary.json pooled, counts |
| 93.66% mask-only Test precision, 142 accepted, 9 contradicted | test_verification_summary.json 2_mask_only_checker |
| 132 accepted, 8 contradicted, 28 deferred, 15.14% | test_verification_summary.json 7_m5_calibrated_mask_first_hybrid |
| 71 of 71 mask-only human groundedness; 67 accepted / 62 with gold / 5 unscored | human_audit_summary.json policies.mask_only, policies.hybrid (67 - 62 = 5) |
| 0.60 / 0.75 locked tau | test_verification_summary.json locked_params |
| 69.61% vs 78.82% (Test, 4-view vs direct) in 14.3 | test_verification_summary.json 5_reinspection_all_gated, 4_direct_qwen3vl_gated |
| 3.88 GPU-hours, ~37 GPU-hours (58 x 38.5 / 60 = 37.2), 8.75 USD (16.631 x 0.526 = 8.748), 2,310 s | training_cost.json per_kind_gpu_hours; M4 report line 70; Section 12.1 of full report |
| 1.2e-7 to 4.4e-7 | EV/reports/Milestone4_Report.md line 70 (prose, not a receipt) |
| 501,995,520; 898,805,894; 55.85%; 1,714.6 MB; about 757 MB; about 56% | compute_profile_rtx4060_cap.json (params_total, checkpoint_size_mb); unused-table count from brief; 501,995,520 / 898,805,894 = 0.55851; 1,714.633 x (1 - 0.55851) = 756.99 |
| 1.39 points (Test minus Val mIoU) | 0.759296 - 0.745394 = 0.0139 (75.93 - 74.54 = 1.39) |
| 54.3 ms (mask-only mixed-platform estimate) | full report Section 12.3 / 12.6 (S_det 0.0175 + S_cap 0.0236 + 1.225 x 0.0108); not re-derived from raw JSON beyond those means |
| 41.3 ms, 24 pairs/s, 10.39 ms witness mean | 7.64 + 23.30 + 10.39 = 41.33; 1000 / 41.33 = 24.2; compute_profile_rtx4060.json witness mean_ms 10.3873 (original had 10.4) |
| 8 GB RTX 4060 laptop | report Table 2.1 / brief |
| 1,142 SECOND-CC test pairs (replace 1,672) | portable_data_receipt.json |
| 88% (20.16 / 22.80 = 88.4%), 20.16 | second_cc_6class_dissection.json D03 test building recall_by_pred_bldg_pct 20.16, recall_by_any_pred_pct 22.8 |
| 12 of 17 wins, 5 losses | counted from stress_test_levir_mci_test.json (A03 wins: clean, brightness_t2_1.25, jpeg_both_q40, cloud_haze_t2_15pct, cross_sensor_radiometric) |
| 3 profiled platforms / 4 backends | compute_profile_*.json |
| 95% interval wording on hybrid and 5.8% | human_audit_summary.json policies.hybrid |

## UNVERIFIED items kept with neutral wording
- "Texas, USA" source of imagery (design/data description): supported by POC/milestone2/Milestone2_Report.md line 34 (cites the dataset papers), not by raw data.
- Regression test names exist; they were not re-run here.
- Gradient parity 1.2e-7 to 4.4e-7: only in the Milestone 4 report text, no receipt JSON.
- 38.5 min per epoch sequential scan: only in Milestone 4 report text (the 227 s/epoch is a receipt).
- "Seasonal/radiometric" cause claims in 14.2 and the benefit of any proposed remedy (all Section 14 proposals) are expectations, labelled as such.
- 13.3 "template-like captions" is qualitative; the 388-word vocabulary is verified.

## Outline coverage (16-section outline, Sections 13-16)
- 13 Limitations: dataset, model, bias, scalability, compute, failure scenarios, ethics all present. GAP noted: no explicit "anomalies / gap between expected and actual" paragraph in 13; covered in 15.2 and 13.2 (C06 vs C00, hybrid vs mask-only).
- 14 Improvements: all seven subsections present, each tied to evidence in Table 14.1.
- 15 Discussion: 15.1-15.3 present. GAP: no stated limitation of the discussion itself regarding the human-audit sample size; added in 15.1.
- 16 Conclusion: summary of findings and next steps present.

## Appendices A-H


Checks: chunkE_check.py, chunkE_check2.py, chunkE_check3.py in rv/. Verified programmatically with zero mismatches: all of A.1 (incl. CIs, FA@16/50/200), A.2, A.3, A.4 (incl. CIs), A.5, B.1 counts, B.2, C.2, E.1 (24 runs, every cell except the aug label), E.2 (9 runs), H.1 sizes and all 12 SHA-256 vs run.json best_checkpoint_sha256 and registry.

| # | Original (short quote) | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | G.1 test_002083 "1,648 (bldg+road)" | ERROR: pred changed px is 2,300 (1,558 bldg, 742 road) | detector_error_mining.json top20_false_positives_unchanged | corrected |
| 2 | G.1 test_001393 GT "5,840" | ERROR: GT is 6,927 (6,354 road, 573 bldg) | detector_error_mining.json top20_false_negatives | corrected |
| 3 | D "+8-point jump ... at epoch 5" | ERROR: 60.70 -> 65.74 = +5.04 | D03 history.jsonl | corrected |
| 4 | D "patience 8 triggered" | ERROR: D03 config patience 7 (seed runs used 8) | D03 run.json config | 7 |
| 5 | D "Train/Val loss gap stayed tight with zero late-epoch overfitting" | OVERCLAIM: train 0.046 vs val 0.1375 at end (gap ~0.09); val loss flat | D03 history.jsonl | rewritten with numbers |
| 6 | D figure alt text "BIT-ResNet18 and architecture-baseline sweeps"; "reproduced directly below" | ERROR: detector figure shows only 6 BIT runs; epochs numbered from 1 | image m4_detector_learning_curves.png | caption and intro fixed |
| 7 | D A04 "165.6 s/epoch" | INCONSISTENT: history epoch_sec averages 364 s (sum 9,100 s > wall 4,141 s); resume=true | A04 run.json, history.jsonl; training_cost.md has 165.6 | kept 69.0 min / 165.6 as wall/25, flagged caveat |
| 8 | D "C02 diverged after epoch 0 ... early-stopped at epoch 2" | UNSUPPORTED "diverged": BLEU-4 0.4956 -> 0.4853, stopped after 2 of 8 epochs | C02 history | reworded |
| 9 | D C06 "Smooth cross-entropy descent" | train CE smooth, val CE minimum at epoch 5 then rises; BLEU-4 not monotone | C06 history | stated |
| 10 | C.1 "All five detectors and Otsu ... log-odds ... 10,000 bins" | ERROR: Otsu uses abs pixel difference, 1,000 bins | m5_eval_detectors_and_stress.py lines 286-320, 405-408 | fixed |
| 11 | B.1 "confirming that spatial resolution ... dominate"; "remarkably low"; "Over 97% ... boundary/extent misses" | OVERCLAIM: matrix cannot separate boundary from whole-object misses; 97.97% are BG misses | arithmetic: 1,498,143/1,529,185 | softened |
| 12 | F test_deferred_out_of_ontology "Villa accepted via building mask" | OMISSION/ERROR: pair is unchanged stratum, proxy GT contradicted: a false accept | test_verification_claims.jsonl | stated |
| 13 | F test_rejected_second "flagged for analyst check" | proxy GT supported: it is a false reject | claims jsonl | stated |
| 14 | F test_rejected_hallucination "C06 missed a new cul-de-sac house" | UNSUPPORTED detail (not in traces); route is mask_first_reject, not VLM | report.json | replaced by mask pixels (732) |
| 15 | F accepted_unchanged "Zero predicted change pixels confirm" | route actually vlm_escalate->vlm_accept | report.json | stated |
| 16 | F vlm_escalated "Direct Qwen3-VL-4B" | route does not say which VLM mode | report.json | "Qwen3-VL-4B" |
| 17 | F intro "authentic", "Evidence Review UI bundles", "spanning all outcomes" | UNVERIFIED UI claim; cases are picked per category | trace_index.json | reworded |
| 18 | G.2 rows 000010, 000064, 001307 "Caught & rejected by D03 mask" | UNSUPPORTED: pairs not in the 150-pair verifier set; for 000064 D03 predicts 1,586 road px (contradicts "class-specific check catches it") | claims jsonl; detector_per_pair_d03_a03.jsonl | rewritten as not run, with D03 pixel facts |
| 19 | G.1/G.2 root causes (glare, shadows, canopy thinning, n-gram loop, seasonal shift) | UNSUPPORTED per pair (only category-level causes in failure_taxonomy.csv); viewed images for 977, 2083, 1303, 1393, 1217, 226 | failure_taxonomy.csv | column renamed "likely cause (untested)", wording hedged, 001393 "widening" -> resurfacing/lane marking |
| 20 | G.2 001729 "VLM missed small subtle building edit" | partly supported: GT 937 bldg px, D03 predicted 0 | per_pair jsonl | stated as facts |
| 21 | H intro "TerraWatch_Checkpoints_M4 repository"; "enforces full SHA-256 verification before inference" | UNSUPPORTED name (only a Drive folder URL in registry); --verify-checksums is a separate mode | model_registry.json; run_terrawatch_inference.py l.307-324 | rewritten |
| 22 | H.1 arch names/roles (rscama_vit_b32_frozen, Clean Challenger, Transformer Baseline) | ERROR vs registry | model_registry.json | aligned; C00 revision added; Qwen dtype mismatch noted (registry bfloat16 vs traces float16) |
| 23 | A.1/A.4 bold on D03/C06 cells that are not best | style/accuracy (implied best) | table | bold removed from those cells |
| 24 | E notes | GAP: epochs/wall time not in tables; seed replicates used 30-epoch budget; C06_s2026 val BLEU-4 0.6442 > selected | run.json | notes added |
| 25 | B.2 closing | proxy labels vs human labels | brief | caveat added |
| 26 | Style | removed "Diagnostic", "vastly", "cleanly", em-dash-free prose, fewer bolds | | STYLE |

UNVERIFIED: cross-references to Sections 6.3, 8 and 10.2 (other chunks); visual root causes; why the 001038 VLM rejected.

New numbers in fixed text (evidence): 10,882,197 (=4,118,292+6,763,905); 1,498,143 (=650,809+847,334); 1,529,185 (+9,511+21,531); 97.97% (1,498,143/1,529,185); 1,349,743 (=499,938+849,805); 0.0050, 0.0073 (AP differences); 0.189/0.113/0.046 train loss, 0.137, 0.0 gap ~0.09 (history); 54.91->60.70->65.74 (+5.04); 2.605, 1.052, 1.307 (C06 history); 0.4853 (C02 epoch 1); 4,140.95 s, 364 s mean (A04); 1,818.6 s; 6,927/6,354/573; 2,300/1,558/742; 732, 1,772, 7,119, 499, 316 (~0.48% = 316/65,536), 272, 5,636, 9,717 (trace reports); 6,539, 937, 1,586, 7,910, 26,132 (=3,320+22,812... per_pair gt 26,132); 0.6092/0.6442 and ~0.035 spread (0.6442-0.6092=0.0350); revision 3e527e66...

## Appendices I-J


Paths: EV = scratchpad/evidence/ (branch milestone5-evaluation), MAIN = scratchpad/review/ (m5-report-review checkout; `git ls-tree main POC/milestone5` is empty, so main has none of the M5 scripts or evidence, only M1-M4 scripts).

| # | Original (short quote) | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | "The CUDA profile on the Kaggle T4 includes the referee" | ERROR. There was no Kaggle run; the platform is Colab Tesla T4, and the referee was a separate session (`colab_t4_referee`) | EV/POC/milestone5/evidence/compute_profile/compute_profile_colab_t4.json `env`/`settings.argv`; compute_profile_colab_t4_referee.json | Rewrote I.2 |
| 2 | I.2 three commands with labels `kaggle_t4`, `apple_m4_mps`, `apple_m4_cpu`, "Hardware labels follow Table 2.1" | ERROR. Real labels are `rtx4060`, `rtx4060_cap`, `colab_t4`, `colab_t4_referee`, `mac_m4_mps`, `mac_m4_cpu`; the RTX 4060 runs were missing; Mac runs had no `--ckpt-dir` (random-init weights) and used `--stage-overrides`; colab_t4 used detectors,witness only | `settings.argv` in the six compute_profile_*.json | Replaced with the six real invocations (paths placeholdered), added random-init note |
| 3 | "--patience 8" in D03 command | ERROR. Receipt has patience 7 | EV/docs/milestone4/evidence/full_data_m4/detector_runs/D03_bit_lr3e4/config.json `config.patience`=7 and `command` | Changed to 7 |
| 4 | "Detector arguments mirror the committed receipts" | Receipts record `scripts/train_detector.py` / `train_captioner.py` (7-line wrappers of `src/training/*`, with different argparse), while the flags match `train_*_full.py`. Also the `command` in the receipts omits `--bit-source-dir`. | receipt `command` fields; scripts/train_detector.py; argparse of train_detector_full.py matches receipt `config` keys | Added a sentence stating this |
| 5 | Test-evaluation row: only `--eval-only` on training scripts | GAP/ERROR. The M5 Test JSONs (detector_test_full.json, captioner_test_full.json, test_verification_summary.json, stress_test_levir_mci_test.json, second_cc_6class_dissection.json) were produced by `m5_eval_detectors_and_stress.py`, `m5_eval_captioners_and_verifier.py`, `m5_second_cc_dissection.py`, which were not listed. `--eval-only` itself exists (train_detector_full.py:357, 411; train_captioner_full.py:660, 690) | EV/POC/milestone5/scripts/*.py docstrings and argparse | Kept eval-only row (relabelled "single model"), added 3 rows with argparse-checked flags |
| 6 | Scripts location | The three M5 scripts in the report exist in MAIN checkout (m5-report-review) but not on `main`; the other M5 scripts exist only on milestone5-evaluation. All M1-M4 scripts exist on main and EV | `ls MAIN/POC/milestone5/scripts` = 3 files; `git ls-tree main POC/milestone5` empty | Added a branch note to intro paragraph; flagged evaluation-branch rows. NOTE: statement "all other scripts are present on main" covers M1-M4 scripts only; the three M5 scripts (cost table, benchmark, human-audit) are on the review/POC branch, not main. Please adjust wording to your branch plan if they are merged by submission |
| 7 | `m5_compute_human_audit.py` flags | Missing `--master-key`, `--id-map` (the actual run used the master key CSV) | m5_compute_human_audit.py:425-431 | Added to command |
| 8 | "Corruption ladder, Test, controls..." | Ladder is on Validation (`levir_mci_val_corruptions`); "120-pair verification study" is the Validation study | robustness_and_transfer_bit.json keys; verification_comparison_summary.json | Added "Validation" |
| 9 | I.3 "planned location `POC/milestone5/evidence/human_audit/`" | Stale; outputs exist there. "one result set per hardware label" imprecise | ls EV/POC/milestone5/evidence/human_audit | Gave actual paths and file names; added Test evaluation receipt folders |
| 10 | J.1/J.5 "a third member adjudicates every disagreement to form the gold consensus set" | ERROR/UNSUPPORTED. No adjudication file; 18 disagreements unresolved; gold = 102 agreed claims | human_audit_summary.json `counts` {gold_labels 102, gold_from_agreement 102, gold_from_adjudication 0, unresolved_excluded 18}; gold_labels.csv (94 supported/8 contradicted, all source "agreed"); disagreements.csv 18 rows | Added Status paragraph; reframed adjudication as design/pending in J.1, J.4(4), J.5, J.6 |
| 11 | "The claim set is stratified over accept, reject, defer outcomes and over the scene strata" | ERROR. Sampling rule in code: all claims not (hybrid-accepted and proxy-supported) plus random accepted proxy-supported claims to reach 120 (seed 2026), then shuffled; no stratified allocation. Scene strata counts are unequal | m5_eval_captioners_and_verifier.py:1066-1079; master key: accept&supported=59, others=61 (computed); strata counts 34/26/24/20/16 | Rewrote J.2 paragraph with the real rule and counts; added caveat that sample is enriched and not representative |
| 12 | Scene strata "unchanged, building-only, road-only, mixed, small-change" | Names/counts differ: `mixed_both` etc. | test_audit_master_key.csv `stratum` | Used actual strata with counts |
| 13 | J.3 rubric wording | Definitions differ from what raters saw (app text): "no change asserted when real change", "wrong count/location" count as contradicted; "supported" covers lack of change; unverifiable = too tiny/blurry/occluded, shadow/seasonal, out of scope. Report said "exact numeric count" makes unverifiable | blind_audit_web_app.py:176-181, 235-243 | Rewrote 3 label definitions to the tool wording; error_category table already matches `CATS` in script and tool unchanged |
| 14 | J.1 raters see "the zoomed contextual crop pair" | Mostly right (strip: T1, T2 with box, crop T1, crop T2). Also shows claim span and parsed object/action/location/quantity. Crop min 32 px, pad 0.25 confirmed | m5_eval_captioners_and_verifier.py:890; configs/pipeline.json witness; app lines 322-325 | Added the box and parsed fields |
| 15 | J.5 "the number of claims on which the adjudicator overruled both raters is reported" | UNSUPPORTED (cannot be, no adjudication) | counts.gold_from_adjudication=0 | Marked as not reportable until adjudication |
| 16 | J.5 policies list | Script also computes `reinsp_all` | policies keys in summary json | Added; also stated that claims without gold are excluded |
| 17 | J.6 "Enforced by repository rules ... archives" | Overclaim: .gitignore ignores data/raw, data/processed, models/checkpoints, *.pt, *.pth; no *.zip rule | review/.gitignore lines 52-61 | Listed the actual patterns, dropped "archives" |
| 18 | J.6 data licence "others are research-only" | Imprecise: datasets.yaml has research-only (LEVIR family, S2Looking, SYSU-CD, SECOND), CC BY 4.0 for SECOND-CC, a CC BY 4.0 annotation entry, and one excluded caption mirror | configs/datasets.yaml lines 16-151 | Narrowed to SECOND-CC and LEVIR family |
| 19 | J.6 annotator welfare "informed ... Imagery is 0.5 m aerial tiles without identifiable individuals" | UNVERIFIED: 0.5 m ground sampling distance not found in repo files searched (M3/M4 reports, datasets.yaml); "no identifiable individuals" not checked tile by tile | none found | Reworded to what is true (project members, no external annotators, overhead imagery); dropped the 0.5 m number and the identifiability claim here (Section 13.7 owner should verify) |
| 20 | J.6 rater independence/label provenance "Rater and adjudication CSVs retained" | adjudication CSV does not exist | evidence/human_audit listing | Status column says adjudication pending |
| 21 | J.6 model transparency "recorded with every decision" | UNSUPPORTED as stated. `configs/pipeline.json` records D03 and C06 SHA-256, Qwen3-VL revision ebb281ec..., accept_threshold 0.70; the server verifies source-image SHA-256 on load (docstring) | configs/pipeline.json; src/app/server.py:1-6 | Reworded to what is recorded where |
| 22 | J.6 human in loop, purpose limitation, privacy, scope | Requirements, not measured facts | server.py docstring (append-only SQLite, immutable outputs); `/api/export` at server.py:182 | Labelled as requirements where applicable; auditability now says records "can be exported" |
| 23 | Appendix I intro "Checkpoints and raw data are never committed" | Overclaim | .gitignore | "are not committed" |
| 24 | Command verification | All flags in the other rows checked against argparse: check_model_setup, fetch_model_assets (`--rscama` required, `--output`), src/data/cli (audit/prepare/verify subcommands), train_detector_full and train_captioner_full flags, eval_robustness_transfer, run_verification_audit (all flags exist; `--detector-arch`, `--bit-source-dir`, `--detector-ckpt` required), caption_pairs, evaluate_models (task positional, `--text-metrics`), run_pipeline (`--t1 --t2 --config --output`), aggregate_m4_results, build_m4_figures, replay_milestone4 (`--device cpu`), package_model_run, src.app.server (`runs` positional, `--port`), m5_training_cost_table, m5_benchmark_compute. setup_model_sources takes no arguments. Test module list (8 files) matches tests/. Only `python -m pytest`, `--help` and the scripts were not executed (no GPU/data) | argparse blocks as above | No change; added a sentence that commands were checked against argument definitions and not re-executed |
| 25 | All remaining prose | STYLE: no em dashes found; sentences merged, bold reduced, "never" claims toned down | n/a | one summary row |

Not changed but worth noting: the D03 "--resume" and `--num-workers 6` flags in receipts are omitted (default or optional). The M4 receipts show D03 trained with `--patience 7`; the earlier run.json `config.patience` is 7. Also `train_detector_full.py` defaults `--bit-source-dir` to the author's absolute path, so the command passes it explicitly (as before).

## Numbers in the fixed chunk that are not in the original chunk

| Number | Evidence |
|---|---|
| patience 7 (replaces 8) | D03 config.json |
| 120 claims `M5T_001` to `M5T_120` | blind_audit_web_app.py docstring; rater CSV audit_id |
| 18 disagreements; 102 gold (94 supported, 8 contradicted) | human_audit_summary.json counts; gold_labels.csv |
| 85.0% agreement (102/120), pooled kappa 0.38 (0.3836) | summary pooled; 102/120=0.85 |
| per slice agreement 100.0, 83.3, 70.0, 86.7%; kappa 1.00, 0.47, 0.08, 0.00 | summary per_slice (1.0/1.0; 0.8333/0.4681; 0.7/0.0816; 0.8667/0.0) |
| proxy vs gold 70.6%, kappa 0.24; 17 unverifiable, 12 contradicted, 1 supported cells | summary proxy_gt_vs_gold: 0.7059, 0.2441, matrix [[65,1,0],[12,7,0],[17,0,0]]. Row orientation (rows = proxy) inferred from proxy label totals (71/30/19 in master key vs row sums 66/19/17 over gold claims); mild uncertainty |
| 61 non-accept-or-non-supported claims, 59 random accepted claims, seed 2026 | master key counts computed (accept & proxy supported = 59; 120-59=61); script lines 1066-1079 |
| strata 34/26/24/20/16 (building_only, unchanged, mixed_both, small_change, road_only) | master key `stratum` counts |
| hybrid verdicts 67 accept, 28 defer, 25 reject | master key `hybrid_verifier_decision` counts |
| 60 claims per rater, no blank rows | summary unrated_blank_rows all 0; rater CSVs 60 rows |
| reduced pass counts: referee warmup=3,iters=20; captioner warmup=5,iters=30 (RTX), 10/50 (MPS), 2/10 (CPU); detectors CPU 5/30, batch 1/5; port 8770 | compute_profile settings.argv; blind_audit_web_app.py:553 |
| six result files | ls compute_profile |
| 0.70 threshold, 32 px, 25% | configs/pipeline.json |

Not verified: whether the Test evaluation scripts' commands run end to end (not executed); 0.5 m GSD claim (removed); whether `main` will contain the M5 scripts at submission.
