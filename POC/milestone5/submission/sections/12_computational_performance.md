## 12. Computational Performance

This section reports the training cost recorded in Milestone 4 receipts and the inference cost measured in Milestone 5 on the platforms of Table 2.1. Training figures are read directly from the 33 run receipts by `POC/milestone5/scripts/m5_training_cost_table.py`; inference figures are produced by `POC/milestone5/scripts/m5_benchmark_compute.py` and written under `POC/milestone5/evidence/compute_profile/`. Profiling protocol: ⟦M:bench_warmup_n⟧ warm-up passes and ⟦M:bench_timed_n⟧ timed passes per stage on ⟦M:bench_num_pairs⟧ held-out Test pairs, with CUDA synchronisation before every timestamp (benchmark revision ⟦M:bench_git_sha⟧). The summary figure is `reports/figures/m5_compute_profile.png`.

### 12.1 Training Cost

The Milestone 4 programme comprises 33 completed or terminated full-data runs (24 detector runs and 9 captioner runs), totalling 681 training epochs and 997.8 minutes (16.63 GPU-hours) of wall-clock time. Detector runs account for 623 epochs and 12.75 GPU-hours; captioner runs for 58 epochs and 3.88 GPU-hours.

**Table 12.1. Aggregate training cost by accelerator.**

| GPU | Runs | Epochs | Wall-clock (min) | GPU-hours |
|---|---:|---:|---:|---:|
| NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) | 14 | 372 | 399.6 | 6.66 |
| NVIDIA L4 (24 GB) | 3 | 77 | 53.7 | 0.90 |
| Tesla T4 (15 GB) | 16 | 232 | 544.5 | 9.07 |
| **Total** | **33** | **681** | **997.8** | **16.63** |

Valued at a uniform T4 rate, the programme corresponds to 16.63 x 0.526 = 8.75 USD.

**Selected models.** D03 completed 32 of its 35-epoch cap before early stopping (best epoch 24) in 31.2 minutes on the RTX 4060 (58.6 s per epoch, peak training VRAM 2,287 MB). C06 completed all 8 epochs (best epoch 6) in 30.3 minutes on a Tesla T4 (227.3 s per epoch, peak training VRAM 4,820 MB). Training the full BIT-ResNet18 plus RSCaMa pair therefore costs roughly 1.0 GPU-hour (61.5 minutes) when each model is trained once.

**Hardware effect on identical workloads.** BIT arms with an identical 25-epoch protocol ran at 56.7 to 58.0 s per epoch on the RTX 4060 (bf16; D05, D06, D08, D10, D11, D14, D15) and at 108.7 to 109.4 s per epoch on the Colab T4 (fp16; D04, D07, D09), a 1.9x difference; D03 seed 2026 ran at 47.7 s per epoch on the L4. D16 (photometric and scale augmentation) is the slowest BIT arm at 125.1 s per epoch on the T4.

**Triton selective scan.** The upstream sequential Mamba scan required 38.5 minutes per epoch on a Tesla T4; the exact fused Triton kernel reduces this to 227 s per epoch, a 10x reduction (38.5 min = 2,310 s; 2,310 / 227 = 10.2), at 5.4 GB peak VRAM for the unfrozen configurations (C01 5,469 MB) and 4.8 GB for the frozen-CLIP configuration (C06 4,820 MB). Without the kernel, the nine-run captioner matrix (58 epochs) would have cost an estimated 37 GPU-hours (58 epochs x 38.5 min) instead of 3.88.

**Table 12.2. Per-run training cost (all 33 receipts).** Reproduced from `evidence/compute_profile/training_cost.md`.

| Run | Family | Params (M) | Trainable (M) | GPU | Epochs run / cap | Best epoch | Wall (min) | Sec / epoch | Peak train VRAM (MB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `A01_ef_unet` | ef_unet | 7.76 | 7.76 | Tesla T4 | 25 / 25 | 23 | 38.9 | 93.3 | 2,289 |
| `A02_fc_siam_diff` | fc_siam_diff | 7.76 | 7.76 | Tesla T4 | 24 / 25 | 17 | 49.1 | 122.7 | 3,083 |
| `A03_siam_r18_unet` | siam_resnet18_unet | 15.63 | 15.63 | Tesla T4 | 25 / 25 | 22 | 35.1 | 84.3 | 1,410 |
| `A03_siam_r18_unet_s0` | siam_resnet18_unet | 15.63 | 15.63 | L4 | 25 / 25 | 24 | 15.6 | 37.5 | 1,408 |
| `A03_siam_r18_unet_s2026` | siam_resnet18_unet | 15.63 | 15.63 | L4 | 22 / 25 | 15 | 14.2 | 38.8 | 1,408 |
| `A04_changeformer` | changeformer | 41.04 | 41.04 | RTX 4060 Laptop | 25 / 25 | 22 | 69.0 | 165.6 | 6,570 |
| `D01_bit_ref` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 35 / 35 | 27 | 34.2 | 58.6 | 2,287 |
| `D02_bit_lr3e5` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 20 | 24.0 | 57.6 | 2,287 |
| `D03_bit_lr3e4` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 32 / 35 | 24 | 31.2 | 58.6 | 2,287 |
| `D03_bit_lr3e4_s0` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 30 / 30 | 29 | 29.1 | 58.3 | 2,287 |
| `D03_bit_lr3e4_s2026` | bit_resnet18 | 11.94 | 11.94 | L4 | 30 / 30 | 23 | 23.8 | 47.7 | 2,492 |
| `D04_bit_wd1e3` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 20 | 45.6 | 109.4 | 2,285 |
| `D05_bit_wd5e2` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 20 | 24.2 | 58.0 | 2,287 |
| `D06_bit_eb16` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 20 | 24.2 | 58.0 | 2,274 |
| `D07_bit_eb64` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 20 | 45.5 | 109.2 | 2,286 |
| `D08_bit_const` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 24 | 24.0 | 57.6 | 2,287 |
| `D09_bit_step` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 24 | 45.3 | 108.7 | 2,287 |
| `D10_bit_cedice` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 22 | 23.8 | 57.1 | 2,287 |
| `D11_bit_wce` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 22 | 23.6 | 56.7 | 2,287 |
| `D12_bit_unfreeze0` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 20 | 25.2 | 60.5 | 2,745 |
| `D13_bit_frozen` | bit_resnet18 | 11.94 | 0.25 | RTX 4060 Laptop | 25 / 25 | 23 | 19.0 | 45.6 | 2,174 |
| `D14_bit_noaug` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 20 | 24.1 | 57.9 | 2,287 |
| `D15_bit_photo` | bit_resnet18 | 11.94 | 11.94 | RTX 4060 Laptop | 25 / 25 | 22 | 24.0 | 57.5 | 2,287 |
| `D16_bit_photoscale` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 22 | 52.1 | 125.1 | 2,287 |
| `C01_rscama_ref` | rscama | 898.81 | 288.06 | Tesla T4 | 8 / 8 | 5 | 31.3 | 234.9 | 5,469 |
| `C02_rscama_lr3e4` (diverged_nonfinite) | rscama | 898.81 | 288.06 | Tesla T4 | 2 / 8 | 0 | 7.6 | 228.4 | 5,469 |
| `C03_rscama_lr3e5` | rscama | 898.81 | 288.06 | Tesla T4 | 6 / 8 | 0 | 24.2 | 241.7 | 5,469 |
| `C04_rscama_adamw_cos` | rscama | 898.81 | 288.06 | Tesla T4 | 8 / 8 | 6 | 34.0 | 254.7 | 5,469 |
| `C05_rscama_ls01` | rscama | 898.81 | 288.06 | Tesla T4 | 7 / 8 | 3 | 31.2 | 267.8 | 5,469 |
| `C06_rscama_frozen_clip` | rscama | 898.81 | 245.53 | Tesla T4 | 8 / 8 | 6 | 30.3 | 227.3 | 4,820 |
| `C06_rscama_frozen_clip_s0` | rscama | 898.81 | 245.53 | Tesla T4 | 6 / 8 | 1 | 24.4 | 243.8 | 4,820 |
| `C06_rscama_frozen_clip_s2026` | rscama | 898.81 | 245.53 | Tesla T4 | 6 / 8 | 1 | 22.6 | 225.7 | 4,820 |
| `C07_rscama_eb24` | rscama | 898.81 | 288.06 | Tesla T4 | 7 / 8 | 3 | 27.4 | 234.8 | 5,888 |

`C02_rscama_lr3e4` terminated with a non-finite loss after 2 epochs and is included in the totals because its compute was spent. Parameter counts for RSCaMa are the receipt totals, which include 502.0 million frozen parameters in the Mamba fusion block (fusion block: 733.9 million total, 231.9 million trainable); the trainable count is 245.53 million for C06 (frozen CLIP) and 288.06 million for the unfrozen arms (an additional 42.53 million from CLIP blocks 6 to 11).

### 12.2 Model Size and Complexity

**Table 12.3. Parameters, checkpoint size and compute per forward pass.** Parameter counts are copied from run receipts; disk size is the `best` checkpoint file size in MiB (bytes / 1,048,576), SHA-256 matched to the run receipt; GFLOPs are measured per 256 x 256 pair (detectors) or per 224 x 224 pair with greedy decoding (captioner) with the torch FLOP counter (method ⟦M:flop_method⟧).

| Model | Parameters (M, total) | Trainable (M) | Checkpoint on disk (MiB) | GFLOPs per pair |
|---|---|---:|---:|---:|
| `A01_ef_unet` | 7.76 (7,763,971) | 7.76 | 29.67 | ⟦M:a01_gflops⟧ |
| `A02_fc_siam_diff` | 7.76 (7,763,107) | 7.76 | 29.67 | ⟦M:a02_gflops⟧ |
| `A03_siam_r18_unet` | 15.63 (15,630,115) | 15.63 | 59.72 | ⟦M:a03_gflops⟧ |
| `A04_changeformer` | 41.04 (41,038,391) | 41.04 | 156.67 | ⟦M:a04_gflops⟧ |
| **`D03_bit_lr3e4` (selected)** | **11.94 (11,944,043)** | 11.94 (head 0.25) | 45.68 | ⟦M:d03_gflops⟧ |
| **`C06_rscama_frozen_clip` (selected)** | **898.81 (receipt total)** | 245.53 | 1,714.63 | ⟦M:c06_gflops⟧ |
| Qwen3-VL-4B-Instruct (frozen, bf16) | ⟦M:ref_params_m⟧ | 0 | ⟦M:ref_disk_mb⟧ | ⟦M:ref_gflops⟧ |

The selected detector is the third-smallest of the five architectures and 3.4x smaller than ChangeFormer (41.04 / 11.94). The captioner dominates the parameter budget of the trained components, but 502.0 million of its recorded parameters are frozen weights in the fusion block and only 245.53 million are trainable. The referee is the largest component of the deployed pipeline and determines the minimum GPU memory tier (Section 12.4).

### 12.3 Inference Latency per Stage

**Table 12.4. Per-stage latency in milliseconds (batch 1).**

| Stage | Platform | P50 | P95 | P99 |
|---|---|---:|---:|---:|
| Detector D03 (BIT-ResNet18) | Kaggle T4 (CUDA) | ⟦M:d03_t4_p50_ms⟧ | ⟦M:d03_t4_p95_ms⟧ | ⟦M:d03_t4_p99_ms⟧ |
| Detector D03 (BIT-ResNet18) | Apple M4 (MPS) | ⟦M:d03_m4mps_p50_ms⟧ | ⟦M:d03_m4mps_p95_ms⟧ | ⟦M:d03_m4mps_p99_ms⟧ |
| Detector D03 (BIT-ResNet18) | Apple M4 (CPU) | ⟦M:d03_m4cpu_p50_ms⟧ | ⟦M:d03_m4cpu_p95_ms⟧ | ⟦M:d03_m4cpu_p99_ms⟧ |
| Captioner C06, Triton fused scan | Kaggle T4 (CUDA) | ⟦M:cap_triton_t4_p50_ms⟧ | ⟦M:cap_triton_t4_p95_ms⟧ | ⟦M:cap_triton_t4_p99_ms⟧ |
| Captioner C06, reference scan | Kaggle T4 (CUDA) | ⟦M:cap_ref_t4_p50_ms⟧ | ⟦M:cap_ref_t4_p95_ms⟧ | ⟦M:cap_ref_t4_p99_ms⟧ |
| Captioner C06, reference scan | Apple M4 (MPS) | ⟦M:cap_ref_m4mps_p50_ms⟧ | ⟦M:cap_ref_m4mps_p95_ms⟧ | ⟦M:cap_ref_m4mps_p99_ms⟧ |
| Captioner C06, reference scan | Apple M4 (CPU) | ⟦M:cap_ref_m4cpu_p50_ms⟧ | ⟦M:cap_ref_m4cpu_p95_ms⟧ | ⟦M:cap_ref_m4cpu_p99_ms⟧ |
| OpenCV witness (per pair) | Kaggle host CPU | ⟦M:wit_host_p50_ms⟧ | ⟦M:wit_host_p95_ms⟧ | ⟦M:wit_host_p99_ms⟧ |
| OpenCV witness (per pair) | Apple M4 (CPU) | ⟦M:wit_m4cpu_p50_ms⟧ | ⟦M:wit_m4cpu_p95_ms⟧ | ⟦M:wit_m4cpu_p99_ms⟧ |
| Qwen3-VL-4B referee (per call) | Kaggle T4 (CUDA) | ⟦M:ref_t4_p50_ms⟧ | ⟦M:ref_t4_p95_ms⟧ | ⟦M:ref_t4_p99_ms⟧ |
| Qwen3-VL-4B referee (per call) | Apple M4 (MPS) | ⟦M:ref_m4mps_p50_ms⟧ | ⟦M:ref_m4mps_p95_ms⟧ | ⟦M:ref_m4mps_p99_ms⟧ |

Baseline detector P50 latency on the Kaggle T4: A01 ⟦M:a01_t4_p50_ms⟧ ms, A02 ⟦M:a02_t4_p50_ms⟧ ms, A03 ⟦M:a03_t4_p50_ms⟧ ms, A04 ⟦M:a04_t4_p50_ms⟧ ms. The inference-time speedup of the Triton fused scan over the reference scan on the T4 is ⟦M:cap_triton_speedup_t4⟧x. The end-to-end P50 latency of a pair through the mask-only policy is ⟦M:e2e_t4_maskonly_p50_ms⟧ ms and through the unconditional-referee policy is ⟦M:e2e_t4_direct_p50_ms⟧ ms.

### 12.4 Memory and GPU Utilisation

During training the detector family peaks at 1.4 to 6.6 GB of VRAM (A03 1,410 MB, D03 2,287 MB, A04 6,570 MB) and the captioner 4.8 to 5.9 GB (C06 4,820 MB, C07 5,888 MB), so every selected model trains within the 8 GB RTX 4060. Inference-time memory on the profiled platforms:

**Table 12.5. Peak memory and utilisation at inference (batch 1).**

| Stage | Peak VRAM, Kaggle T4 (MB) | Mean GPU utilisation, Kaggle T4 (%) | Peak process memory, M4 MPS (MB) | Peak RSS, M4 CPU (MB) |
|---|---:|---:|---:|---:|
| Detector D03 | ⟦M:d03_t4_vram_mb⟧ | ⟦M:d03_t4_gpu_util_pct⟧ | ⟦M:d03_m4mps_mem_mb⟧ | ⟦M:d03_m4cpu_rss_mb⟧ |
| Captioner C06 (Triton on T4, reference scan on M4) | ⟦M:cap_t4_vram_mb⟧ | ⟦M:cap_t4_gpu_util_pct⟧ | ⟦M:cap_m4mps_mem_mb⟧ | ⟦M:cap_m4cpu_rss_mb⟧ |
| Qwen3-VL-4B referee | ⟦M:ref_t4_vram_mb⟧ | ⟦M:ref_t4_gpu_util_pct⟧ | ⟦M:ref_m4mps_mem_mb⟧ | ⟦M:ref_m4cpu_rss_mb⟧ |
| OpenCV witness | n/a (CPU) | n/a (CPU) | n/a (CPU) | ⟦M:wit_m4cpu_rss_mb⟧ |
| Full pipeline, peak resident stages | ⟦M:e2e_t4_vram_mb⟧ | ⟦M:e2e_t4_gpu_util_pct⟧ | ⟦M:e2e_m4mps_mem_mb⟧ | ⟦M:e2e_m4cpu_rss_mb⟧ |

Utilisation is sampled at ⟦M:util_sample_hz⟧ Hz through ⟦M:util_sampler⟧. The final row reports the peak with all resident stages loaded simultaneously.

### 12.5 Throughput

**Table 12.6. Stage throughput in pairs per second.**

| Stage | Platform | Batch 1 | Batch 16 |
|---|---|---:|---:|
| Detector D03 | Kaggle T4 (CUDA) | ⟦M:d03_t4_b1_pps⟧ | ⟦M:d03_t4_b16_pps⟧ |
| Detector D03 | Apple M4 (MPS) | ⟦M:d03_m4mps_b1_pps⟧ | ⟦M:d03_m4mps_b16_pps⟧ |
| Detector D03 | Apple M4 (CPU) | ⟦M:d03_m4cpu_b1_pps⟧ | ⟦M:d03_m4cpu_b16_pps⟧ |
| Captioner C06 (Triton fused scan) | Kaggle T4 (CUDA) | ⟦M:cap_triton_t4_b1_pps⟧ | ⟦M:cap_triton_t4_b16_pps⟧ |
| Captioner C06 (reference scan) | Kaggle T4 (CUDA) | ⟦M:cap_ref_t4_b1_pps⟧ | ⟦M:cap_ref_t4_b16_pps⟧ |
| Captioner C06 (reference scan) | Apple M4 (MPS) | ⟦M:cap_ref_m4mps_b1_pps⟧ | ⟦M:cap_ref_m4mps_b16_pps⟧ |
| Captioner C06 (reference scan) | Apple M4 (CPU) | ⟦M:cap_ref_m4cpu_b1_pps⟧ | ⟦M:cap_ref_m4cpu_b16_pps⟧ |

End-to-end throughput on the Kaggle T4 at batch 1 is ⟦M:e2e_t4_maskonly_pps⟧ pairs per second for the mask-only policy, ⟦M:e2e_t4_triggered_pps⟧ for the triggered policy and ⟦M:e2e_t4_direct_pps⟧ for the unconditional-referee policy. Training throughput reference points from the receipts are 103.7 to 141.8 samples per second for D03 on the RTX 4060 and 74.0 to 85.5 samples per second for A03 on the T4.

### 12.6 Cost per 1,000 Pairs

Cost is dominated by the referee, whose invocation count is governed by the verification policy. The per-1,000-pair cost on a platform with hourly price `P` is

`Cost_1000 = 1000 x (S_det + S_cap + S_wit + c x v x S_ref) x P / 3600`,

where `S_det`, `S_cap` and `S_wit` are mean seconds per pair of the detector, captioner and witness (non-referee stages), `S_ref` is the mean seconds per referee call, `c = 147 / 120 = 1.225` is the audited claims-per-pair ratio (147 claims over 120 validation pairs) and `v` is the referee calls per claim. The three policies take `v` from the Milestone 4 comparison: direct 1.0, disagreement-triggered 0.6463 (95 of 147 claims triggered) and mask-only 0. The resulting referee calls per pair are 1.225, 0.792 (1.225 x 0.6463) and 0, and triggering removes 35.4% of referee calls relative to the direct policy.

**Table 12.7. Cost per 1,000 pairs on a cloud T4 (AWS g4dn.xlarge on-demand, us-east-1 Linux, 0.526 USD per hour).**

| Policy | Referee calls per claim | Referee calls per pair | Proxy supported precision (Val audit) | Seconds per pair | Cost per 1,000 pairs (USD) |
|---|---:|---:|---:|---:|---:|
| Direct Qwen3-VL (gated) | 1.0000 | 1.225 | 82.43% | ⟦M:t4_direct_s_per_pair⟧ | ⟦M:cost_t4_direct_usd⟧ |
| Disagreement-triggered reinspection | 0.6463 | 0.792 | 72.22% | ⟦M:t4_triggered_s_per_pair⟧ | ⟦M:cost_t4_triggered_usd⟧ |
| Mask-only checker | 0 | 0 | 91.38% | ⟦M:t4_maskonly_s_per_pair⟧ | ⟦M:cost_t4_maskonly_usd⟧ |
| Calibrated mask-first hybrid | ⟦M:hybrid_vlm_calls_per_claim⟧ | ⟦M:hybrid_vlm_calls_per_pair⟧ | ⟦M:hybrid_precision_pct⟧ | ⟦M:t4_hybrid_s_per_pair⟧ | ⟦M:cost_t4_hybrid_usd⟧ |

Measured stage means entering the formula: `S_det` = ⟦M:t4_det_mean_s⟧ s, `S_cap` = ⟦M:t4_cap_mean_s⟧ s, `S_wit` = ⟦M:t4_wit_mean_s⟧ s and `S_ref` = ⟦M:t4_ref_mean_s⟧ s. The mask-only policy is both the cheapest and, on the proxy-labelled Validation audit, the most precise, but it cannot adjudicate out-of-ontology claims (for example vegetation or parking), which is the case the hybrid policy routes to the referee. Precision values are scored against proxy labels derived from ground-truth masks and reference captions, not against human labels (Section 13.1).
