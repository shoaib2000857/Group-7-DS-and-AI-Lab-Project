# Sources and Measurement Slots for Sections 2, 12, 13, 14 and Appendices I, J

Paths are relative to the repository root. M4 evidence directory: `docs/milestone4/evidence/full_data_m4/` (abbreviated `EV/`). Every number in the sections traces to one of the files below; values written as slots are measured by `POC/milestone5/scripts/m5_benchmark_compute.py` (or noted otherwise) and replace the slot text verbatim.

## 1. Fact sources by section

### Section 2 (`02_experimental_setup.md`)
| Fact | Source |
|---|---|
| Python, torch, GPU names, AMP dtype per run; run-to-hardware assignment (14 laptop, 7 T4, 3 L4 detector runs) | `EV/detector_runs/*/run.json` (`environment`, `amp_actual`); `POC/milestone5/evidence/compute_profile/training_cost.json` |
| Captioner environment (T4, torch 2.14.1+cu130, Python 3.13.16), `kernel_mode`, seed, AMP | `EV/captioner_runs/*/run.json` |
| Triton 3.8.0; Transformers 4.39.3 / 5.22.1; laptop VRAM 8 GB, T4 15 GB, L4 24 GB; 1.2e-7 to 4.4e-7 kernel error; loss delta 3.11e-4 | `reports/Milestone4_Report.md` sections 4, 5, 12 |
| Referee dtype, revision, Transformers 5.22.1, torch 2.14.1+cu130; device not recorded | `EV/verification/verification_comparison_summary.json` (`models.referee`) |
| Pinned upstream commits (BIT_CD `adcd7aea6f23`, ChangeFormer `afd1b7ed640a`, RSCaMa `aa68e4c6933c`, CLIP `dcba3cb2e282`) | `reports/Milestone4_Report.md` section 3; `scripts/setup_model_sources.py` |
| Install pins (torch 2.8.0, torchvision 0.23.0, transformers 4.39.3 and 5.17.0) | `requirements-detector.txt`, `requirements-captioner.txt`, `requirements-referee.txt` |
| Split counts, manifest and pair-id SHA-256, SECOND-CC counts (530, 1,142; train 3,241) | `EV/portable_data_receipt.json` |
| Changed/unchanged counts (610/632, 934/924), captions (32,220; 6,210; 9,290), mask encoding, normalisation, 388 vocab | `reports/Milestone4_Report.md` section 2; `EV/captioner_runs/C06_rscama_frozen_clip/run.json` |
| LEVIR-MCI HF revision, SECOND-CC Zenodo 16937571, licences | `configs/datasets.yaml` |
| Seeds 42, 0, 2026; 3-seed statistics | `EV/*_runs/*/run.json` (`config.seed`); `reports/Milestone4_Report.md` section 8 |
| Seeding code, per-epoch generator `seed + epoch * 1009`, `cudnn.benchmark = True`, no deterministic-algorithms flag | `scripts/train_detector_full.py` (lines 60 to 66, 366, 559); `scripts/train_captioner_full.py` (lines 56 to 61, 843) |
| Checkpoint SHA-256 values | `EV/detector_runs/*/run.json` (`best_checkpoint_sha256`), `EV/captioner_runs/C06_rscama_frozen_clip/run.json`, `EV/captioner_runs/C00_rscama_released/eval_test.json`, `configs/pipeline.json` |
| Hyperparameters of D03 and C06 | `configs/detector_training.json`, `configs/captioner_training.json`, `EV/detector_runs/D03_bit_lr3e4/run.json` |
| 120 pairs, strata, 147 claims, tau 0.70 | `EV/verification/verification_comparison_summary.json` |
| Table 2.1 RTX 4060 profiling row (Python 3.13.9, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, Triton 3.6.0, Transformers 4.57.6 detector and 4.39.3 captioner, i7-14650HX, 24 CPUs, Linux 6.14) | `POC/milestone5/evidence/compute_profile/compute_profile_rtx4060.json` (`env`, including `env.transformers_captioner`); `POC/milestone5/evidence/compute_profile/compute_profile_rtx4060_cap.json` (`env.transformers`) |
| Table 2.1 Colab T4 profiling row (Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA 13.0, Triton 3.8.0, Transformers 5.22.1, Xeon 2.00 GHz, 2 CPUs, Linux 6.6) | `POC/milestone5/evidence/compute_profile/compute_profile_colab_t4.json` (`env`); `POC/milestone5/evidence/compute_profile/compute_profile_colab_t4_referee.json` (`env`; referee `dtype` float16 in `stages.Qwen3-VL-4B-Instruct.dtype`) |
| Table 2.1 Apple M4 profiling row (Python 3.11.9, torch 2.15.0, torchvision 0.29.2, Transformers 4.39.3, macOS 26.5.1, 10 CPUs, no CUDA or Triton; random-init weights, full-length 42-step decode) | `POC/milestone5/evidence/compute_profile/compute_profile_mac_m4_mps.json`, `POC/milestone5/evidence/compute_profile/compute_profile_mac_m4_cpu.json` (`env`; `stages.*.weights` = `random_init`; `stages.C06_rscama_frozen_clip.ref_scan.decode_steps_mean` = 42) |
| OpenCV version | not recorded in any profile JSON; stated as not recorded; install spec from `requirements-pipeline.txt` |
| Driver version | not recorded in any profile JSON; stated as not recorded |

### Section 12 (`12_computational_performance.md`)
| Fact | Source |
|---|---|
| All training-cost numbers (Tables 12.1, 12.2; per-run params, epochs, wall time, sec/epoch, peak VRAM; totals 997.8 min, 16.63 GPU-hours, 681 epochs) | `POC/milestone5/scripts/m5_training_cost_table.py` over `EV/detector_runs/*/run.json`, `EV/captioner_runs/*/run.json` and `history.jsonl`; output `POC/milestone5/evidence/compute_profile/training_cost.json`, `training_cost.md` |
| Detector samples/s range (103.7 to 141.8; 74.0 to 85.5) | `EV/detector_runs/D03_bit_lr3e4/history.jsonl`, `EV/detector_runs/A03_siam_r18_unet/history.jsonl` |
| 38.5 min/epoch to 227 s/epoch, 10x, 5.4 GB, frozen embedding tables, 231.9 M / 13.63 M / 42.53 M parameters | `reports/Milestone4_Report.md` sections 3.2, 4, 12.1; `EV/captioner_runs/C06_rscama_frozen_clip/run.json` (`trainability`) |
| Estimated 37 GPU-hours without kernel | derived: 58 epochs x 38.5 min |
| Referee calls per claim 1.0 / 0.6463 / 0; 95 of 147 triggered; claims 147, pairs 120 | `EV/verification/verification_comparison_summary.json` (`trigger_statistics`, `audit_scope`) |
| Policy precisions 82.43%, 72.22%, 91.38% | `EV/verification/verification_comparison_summary.json` (`settings_comparison`) |
| Cost formula and derived 1.225 / 0.792 calls per pair, 35.4% saving | derived from the two rows above |
| Table 12.3 params, trainable (detectors), checkpoint MiB, GFLOPs (C06 GFLOPs from the RTX 4060 run; FlopCounterMode excludes selective-scan elementwise ops via `flops_note`) | `POC/milestone5/evidence/compute_profile/compute_profile_colab_t4.json` (`stages.<run>.params_total`, `checkpoint_size_mb`, `gflops_per_pair`); `POC/milestone5/evidence/compute_profile/compute_profile_rtx4060.json` (`stages.C06_rscama_frozen_clip`); trainable 245.53 M for C06 from `POC/milestone5/evidence/compute_profile/training_cost.json` (`runs[].params_trainable_max`) |
| Table 12.3 Qwen3-VL params 4,437.82 M, load peak VRAM 8,464.5 MB; disk size not recorded | `POC/milestone5/evidence/compute_profile/compute_profile_colab_t4_referee.json` (`stages.Qwen3-VL-4B-Instruct.params_total`, `load_peak_vram_mb`) |
| Table 12.4 detector latency T4 / RTX 4060 / M4 MPS / M4 CPU | `compute_profile_colab_t4.json`, `compute_profile_rtx4060.json`, `compute_profile_mac_m4_mps.json`, `compute_profile_mac_m4_cpu.json` (`stages.D03_bit_lr3e4.batch1.p50_ms/p95_ms/p99_ms`) |
| Table 12.4 captioner Triton and reference scan on RTX 4060 | `compute_profile_rtx4060_cap.json` (identical to `compute_profile_rtx4060.json`) `stages.C06_rscama_frozen_clip.triton_scan` and `.ref_scan` |
| Table 12.4 captioner on Apple M4 (reference scan, 42 decode steps) | `compute_profile_mac_m4_mps.json`, `compute_profile_mac_m4_cpu.json` `stages.C06_rscama_frozen_clip.ref_scan` |
| Table 12.4 witness | `compute_profile_colab_t4.json`, `compute_profile_mac_m4_cpu.json` `stages.witness_opencv.p50_ms/p95_ms/p99_ms` |
| Table 12.4 referee 2-image and 4-image | `compute_profile_colab_t4_referee.json` `stages.Qwen3-VL-4B-Instruct.full_pair_2img` and `.pair_plus_crops_4img` |
| Baseline detector P50 (A01 to A04 on T4) | `compute_profile_colab_t4.json` `stages.<arch>.batch1.p50_ms` |
| Table 12.5 memory | `peak_mem_mb` of the same stage entries (`batch1.peak_mem_mb` for detectors; `triton_scan.peak_mem_mb` 3,518.4 and `ref_scan.peak_mem_mb` 4,403.75 for C06; `full_pair_2img` 8,589.9 and `pair_plus_crops_4img` 8,652.2; `witness_opencv.peak_mem_mb` as process RSS) |
| Table 12.6 throughput | `batch1.throughput_pairs_per_s`, `batch16.throughput_pairs_per_s` (detectors); 1000 / `triton_scan.mean_ms` and 1000 / `ref_scan.mean_ms` (captioner) |
| Table 12.7 stage means S_det 0.0175, S_cap 0.0236, S_wit 0.0108, S_ref 0.2458 / 0.3380 | `compute_profile_colab_t4.json` (D03 `batch1.mean_ms`, `witness_opencv.mean_ms`); `compute_profile_rtx4060.json` (C06 `triton_scan.mean_ms`); `compute_profile_colab_t4_referee.json` (`full_pair_2img.mean_ms`, `pair_plus_crops_4img.mean_ms`) |
| Table 12.7 seconds per pair, cost, 12.5 end-to-end throughput | derived: s = S_det + S_cap + c x S_wit + c x v x S_ref; cost = 1000 x s x 0.526 / 3600; c = 147/120 (policies 1 to 3) and 185/150 (hybrid); v = 1.0, 0.6463, 0, 0.0973 (hybrid, 18 calls) |
| Hybrid call rate 0.0973 (18 calls, 185 claims, 150 pairs), Test precision 93.94%, Val precision 94.50% | `POC/milestone5/evidence/test_captioner_and_verifier/test_verification_summary.json` (`test_vlm_efficiency`, `test_settings_comparison.7_m5_calibrated_mask_first_hybrid`, `validation_calibration_locked`); present on `origin/milestone5-evaluation` only, not on this branch |
| Referee decision is a three-logit forward pass | `src/inference/verification_pipeline.py` `Qwen3VLReferee.decide` (`logits_to_keep=1`, softmax over three label tokens); Milestone 3 7.605 s: `reports/Milestone3_Report_LocalBenchmarked.md` |
| Not measured (stated as such in the section) | captioner on the Colab T4 (latency, throughput, VRAM, Triton speedup); any inference profile on the NVIDIA L4; Qwen3-VL on the Apple M4 or RTX 4060; GPU utilisation (not recorded by any profile); end-to-end pipeline latency (estimated only); OpenCV and driver versions |
| Plot reference `figures/m5_compute_profile.png` (exists on `origin/milestone5-evaluation` only; its bars agree with the JSON values above) | `POC/milestone5/REPORT_OUTLINE_AND_EVIDENCE_MAP.md` on `origin/milestone5-plan` |

### Section 13 (`13_limitations.md`)
| Fact | Source |
|---|---|
| SECOND-CC any-change F1 23.17%, recall 14.46%, precision 58.29%, building F1 30.55%, FA@50 25.08% | `EV/robustness_and_transfer_bit.json` (`second_cc_transfer.test`) |
| LEVIR-MCI Test binary F1 86.83%, small-change F1 48.87%, FA@50 10.28%, confusion-matrix percentages and pixel counts | `EV/robustness_and_transfer_bit.json` (`levir_mci_test`) |
| Val small-change F1 42.72%, 86.25% F1; corruption results (blur 70.94/38.22, JPEG q20 67.99 and FA@50 19.30%, shift 63.88, clean 74.54) | `EV/robustness_and_transfer_bit.json` (`levir_mci_val_corruptions`); `EV/detector_runs/D03_bit_lr3e4/run.json` |
| A03 Test mIoU 76.73%, FA@50 8.23%; D03-A03 advantages | `reports/Milestone4_Report.md` section 10; `EV/robustness_and_transfer_siam.json` |
| Captioner Test BLEU-4 0.6128, changed BLEU-4 0.3681, missed 11.78% (110 of 934), NC FA 3.57% | `EV/captioner_runs/C06_rscama_frozen_clip/eval_test.json` |
| Seed values C06 (0.6092, 0.6442, 0.6358; best epoch 1, 1, 6), arm values C01, C07, C04 | `EV/EXPERIMENT_TABLES.md`; `EV/captioner_runs/*/run.json` |
| Proxy-label caveat, 13 unverifiable, 10 contradicted accepted, 17 of 27 caught, 60 deferred, 3 untriggered errors, 13 out-of-ontology triggers, 4-view 72.41% | `EV/verification/verification_comparison_summary.json` |
| Texas, 0.5 m, 256 x 256, no per-tile dates or coordinates | `reports/Milestone2_Report.md` (data section, line 34) |
| Loss class weights 0.10, 0.45, 0.45 | `configs/detector_training.json` and `EV/detector_runs/D03_bit_lr3e4/config.json`, `run.json` |
| Hardware speed ratio 1.9x | derived from `training_cost.json` (D04/D07/D09 vs D05/D06/D08) |
| Ethics statements | design commitments; no numeric source |

### Section 14 (`14_possible_improvements.md`)
| Fact | Source |
|---|---|
| Weakness figures | as Section 13 sources |
| Ablation values D01 71.82, D02 64.55, D03 74.54, D13 62.45, D15 68.34, D16 68.55, A04 70.46 | `EV/EXPERIMENT_TABLES.md` |
| Road IoU 74.53% and building IoU 77.33% on Test | `reports/Milestone4_Report.md` section 10.1 |
| Road pixel share 3.4% (4.12 of 121.77 million) | derived from Test confusion matrix in `EV/robustness_and_transfer_bit.json` |
| Candidate datasets (S2Looking, SYSU-CD, LEVIR-CD+), SECOND-CC train 3,241 | `configs/datasets.yaml`; `EV/portable_data_receipt.json` |
| 51 `T_witness_disagrees_no_visual_change` trigger events, 95 triggered claims | `EV/verification/verification_comparison_summary.json` |

### Appendix I (`appendix_I_evaluation_scripts.md`)
| Fact | Source |
|---|---|
| Existing script arguments | `scripts/*.py` argparse blocks; `src/evaluation/cli.py`; `src/inference/pipeline.py`; `src/data/cli.py`; `src/app/server.py`; `README.md` |
| Hyperparameters in commands | `configs/detector_training.json`, `configs/captioner_training.json`, `EV/detector_runs/D03_bit_lr3e4/run.json` |
| `m5_benchmark_compute.py`, `m5_compute_human_audit.py` CLIs | specified by the Milestone 5 coordination message (scripts under construction by teammates) |
| Test module list | `tests/` |

### Appendix J (`appendix_J_annotation_protocol.md`)
| Fact | Source |
|---|---|
| Slice, rater and adjudicator table, rubric labels, error categories, blinding rules, file naming | `POC/milestone5/TEAM_HANDOFF.md` section 5 on `origin/milestone5-plan` |
| Metric definitions | `POC/milestone5/M5_MASTER_PLAN.md` WS3 on `origin/milestone5-plan`; standard Cohen's kappa definition |
| Crop padding 25%, minimum 32 px | `configs/pipeline.json` (`witness`) |

## 2. Discrepancies between documents and how they were handled

| Item | Documents | Handling |
|---|---|---|
| D03 epochs | Plan: 30 epochs. Config and receipt: cap 35, 32 completed, early stop, best epoch 24 | Receipt values used |
| C06 epochs and duration | Plan: 15 epochs, about 27 min. Config and receipt: cap 8, 8 completed, best epoch 6, 30.3 min on T4 | Receipt values used |
| A03 checkpoint SHA | Handoff: `d8beea3945f267aa`. Receipt and `EXPERIMENT_TABLES.md`: `c9d2c9c994c6` | Receipt value used |
| A03 parameters | M4 report: 15.65 M. Receipt: 15,630,115 (15.63 M) | Receipt value used |
| D03 patience | Config: 8. Receipt command: 7 | Not stated in sections beyond the config cap |
| Triggered call rate | M4 report: 0.65. Summary JSON: 0.6463 | JSON value used |
| Direct referee precision | Raw 82.35%, gated 82.43% | Gated (operating policy) used |
| Python/torch pins | `requirements-*.txt`: torch 2.8.0, transformers 5.17.0. Receipts: torch 2.11.0 and 2.14.1, Transformers 5.22.1 | Section 2 reports versions used and states the pins are the install specification |
| Referee GPU | Not recorded in the verification summary | Stated as L4 or T4 |
| Captioner parameter total | Receipt total 898.81 M includes 502.0 M frozen fusion-block parameters = 13 x nn.Embedding(50280, 768), never called in CaMambaModel.forward (models/external/RSCaMa/model/mamba_block.py:217 vs forward uses inputs_embeds) | Stated as unused, source-verified 2026-11-26 |

## 3. Measurement slots

Slots are replaced by the exact value printed by the benchmark (a value such as `n/a (out of memory)` is acceptable where a stage cannot run on a platform). Slots for sections 2 and 12 are all filled and removed; remaining slots belong to other sections (a `12, 13` slot is shared with section 13).

| Slot key | Appears in | Filled by |
|---|---|---|
| `m4_chip_ram` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `m4_python` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `m4_torch` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `m4_macos` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `m4_cv2` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `m4_transformers` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_gpu_name` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_vram_gb` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_python` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_torch` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_cuda` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_triton` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_transformers` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `kt4_driver` | 02 | Environment string recorded by the benchmark at run time (platform, library or driver version / hardware name / VRAM GB) |
| `bench_warmup_n` | 12 | Benchmark protocol constant (warm-up passes, timed passes, pairs, git revision) from the profile JSON |
| `bench_timed_n` | 12 | Benchmark protocol constant (warm-up passes, timed passes, pairs, git revision) from the profile JSON |
| `bench_num_pairs` | 12 | Benchmark protocol constant (warm-up passes, timed passes, pairs, git revision) from the profile JSON |
| `bench_git_sha` | 12 | Benchmark protocol constant (warm-up passes, timed passes, pairs, git revision) from the profile JSON |
| `t4_hourly_usd` | 12 | FILLED 0.526: AWS g4dn.xlarge on-demand, us-east-1 Linux, checked 2026-11-26 (instances.vantage.sh/aws/ec2/g4dn.xlarge, calculator.holori.com/aws/ec2/g4dn.xlarge/us-east-1) |
| `train_equiv_usd` | 12 | FILLED 8.75 = 16.63 x 0.526 |
| `flop_method` | 12 | FLOP counter name and version used (e.g. torch flop counter) |
| `a01_disk_mb` | 12 | FILLED from on-disk byte sizes reported by Shoaib 2026-11-26 (SHA-256 matched to run.json); MiB. Re-confirm from benchmark JSON |
| `a01_gflops` | 12 | GFLOPs per forward pair from the FLOP counter |
| `a02_disk_mb` | 12 | FILLED from on-disk byte sizes reported by Shoaib 2026-11-26 (SHA-256 matched to run.json); MiB. Re-confirm from benchmark JSON |
| `a02_gflops` | 12 | GFLOPs per forward pair from the FLOP counter |
| `a03_disk_mb` | 12 | FILLED from on-disk byte sizes reported by Shoaib 2026-11-26 (SHA-256 matched to run.json); MiB. Re-confirm from benchmark JSON |
| `a03_gflops` | 12 | GFLOPs per forward pair from the FLOP counter |
| `a04_disk_mb` | 12 | FILLED from on-disk byte sizes reported by Shoaib 2026-11-26 (SHA-256 matched to run.json); MiB. Re-confirm from benchmark JSON |
| `a04_gflops` | 12 | GFLOPs per forward pair from the FLOP counter |
| `d03_disk_mb` | 12 | FILLED from on-disk byte sizes reported by Shoaib 2026-11-26 (SHA-256 matched to run.json); MiB. Re-confirm from benchmark JSON |
| `d03_gflops` | 12 | GFLOPs per forward pair from the FLOP counter |
| `c06_disk_mb` | 12 | FILLED from on-disk byte sizes reported by Shoaib 2026-11-26 (SHA-256 matched to run.json); MiB. Re-confirm from benchmark JSON |
| `c06_gflops` | 12 | GFLOPs per forward pair from the FLOP counter |
| `ref_params_m` | 12 | Parameters (millions) counted from the loaded model |
| `ref_disk_mb` | 12 | Checkpoint file size in MB (os.stat) for the named model |
| `ref_gflops` | 12 | GFLOPs per forward pair from the FLOP counter |
| `d03_t4_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_t4_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_t4_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_m4mps_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_m4mps_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_m4mps_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_m4cpu_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_m4cpu_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_m4cpu_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_triton_t4_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_triton_t4_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_triton_t4_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_t4_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_t4_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_t4_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_m4mps_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_m4mps_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_m4mps_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_m4cpu_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_m4cpu_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_ref_m4cpu_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `wit_host_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `wit_host_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `wit_host_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `wit_m4cpu_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `wit_m4cpu_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `wit_m4cpu_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `ref_t4_p50_ms` | 12, 13 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `ref_t4_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `ref_t4_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `ref_m4mps_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `ref_m4mps_p95_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `ref_m4mps_p99_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `a01_t4_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `a02_t4_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `a03_t4_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `a04_t4_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `cap_triton_speedup_t4` | 12 | cap_ref_t4_p50_ms / cap_triton_t4_p50_ms (derived) |
| `e2e_t4_maskonly_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `e2e_t4_direct_p50_ms` | 12 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
| `d03_t4_vram_mb` | 12 | torch.cuda.max_memory_allocated (MB) |
| `d03_m4mps_mem_mb` | 12 | Peak MPS driver/process memory (MB) on Apple M4 |
| `d03_m4cpu_rss_mb` | 12 | Peak resident set size (MB) on Apple M4 CPU |
| `cap_t4_vram_mb` | 12 | torch.cuda.max_memory_allocated (MB) |
| `cap_m4mps_mem_mb` | 12 | Peak MPS driver/process memory (MB) on Apple M4 |
| `cap_m4cpu_rss_mb` | 12 | Peak resident set size (MB) on Apple M4 CPU |
| `ref_t4_vram_mb` | 12, 13 | torch.cuda.max_memory_allocated (MB) |
| `ref_m4mps_mem_mb` | 12 | Peak MPS driver/process memory (MB) on Apple M4 |
| `ref_m4cpu_rss_mb` | 12 | Peak resident set size (MB) on Apple M4 CPU |
| `wit_m4cpu_rss_mb` | 12 | Peak resident set size (MB) on Apple M4 CPU |
| `e2e_t4_vram_mb` | 12 | torch.cuda.max_memory_allocated (MB) |
| `e2e_m4mps_mem_mb` | 12 | Peak MPS driver/process memory (MB) on Apple M4 |
| `e2e_m4cpu_rss_mb` | 12 | Peak resident set size (MB) on Apple M4 CPU |
| `d03_t4_b1_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `d03_t4_b16_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `d03_m4mps_b1_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `d03_m4mps_b16_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `d03_m4cpu_b1_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `d03_m4cpu_b16_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_triton_t4_b1_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_triton_t4_b16_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_ref_t4_b1_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_ref_t4_b16_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_ref_m4mps_b1_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_ref_m4mps_b16_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_ref_m4cpu_b1_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `cap_ref_m4cpu_b16_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `e2e_t4_maskonly_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `e2e_t4_triggered_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `e2e_t4_direct_pps` | 12 | Throughput in pairs per second at the stated batch size |
| `t4_direct_s_per_pair` | 12 | Sum of non-referee stage seconds + c x v x S_ref for the policy (derived from the t4_*_mean_s slots) |
| `cost_t4_direct_usd` | 12 | 1000 x s_per_pair x t4_hourly_usd / 3600 (derived) |
| `t4_triggered_s_per_pair` | 12 | Sum of non-referee stage seconds + c x v x S_ref for the policy (derived from the t4_*_mean_s slots) |
| `cost_t4_triggered_usd` | 12 | 1000 x s_per_pair x t4_hourly_usd / 3600 (derived) |
| `t4_maskonly_s_per_pair` | 12 | Sum of non-referee stage seconds + c x v x S_ref for the policy (derived from the t4_*_mean_s slots) |
| `cost_t4_maskonly_usd` | 12 | 1000 x s_per_pair x t4_hourly_usd / 3600 (derived) |
| `hybrid_vlm_calls_per_claim` | 12 | Calibrated mask-first hybrid verifier statistic from POC/milestone5/evidence/hybrid_verifier/ |
| `hybrid_vlm_calls_per_pair` | 12 | Calibrated mask-first hybrid verifier statistic from POC/milestone5/evidence/hybrid_verifier/ |
| `hybrid_precision_pct` | 12 | Calibrated mask-first hybrid verifier statistic from POC/milestone5/evidence/hybrid_verifier/ |
| `t4_hybrid_s_per_pair` | 12 | Sum of non-referee stage seconds + c x v x S_ref for the policy (derived from the t4_*_mean_s slots) |
| `cost_t4_hybrid_usd` | 12 | 1000 x s_per_pair x t4_hourly_usd / 3600 (derived) |
| `t4_det_mean_s` | 12 | Mean seconds per pair (per call for ref) on Kaggle T4 from the benchmark |
| `t4_cap_mean_s` | 12 | Mean seconds per pair (per call for ref) on Kaggle T4 from the benchmark |
| `t4_wit_mean_s` | 12 | Mean seconds per pair (per call for ref) on Kaggle T4 from the benchmark |
| `t4_ref_mean_s` | 12 | Mean seconds per pair (per call for ref) on Kaggle T4 from the benchmark |
| `ensemble_t4_p50_ms` | 14 | Latency percentile (ms), batch 1, timed passes; stage_hardware encoded in key (t4=Kaggle T4, m4mps/m4cpu=Apple M4, host=Kaggle host CPU) |
