## 12. Computational Performance

This section reports the training cost recorded in Milestone 4 receipts and the inference cost measured in Milestone 5 on the platforms of Table 2.1. Training figures are read directly from the 33 run receipts by `POC/milestone5/scripts/m5_training_cost_table.py`; inference figures are produced by `POC/milestone5/scripts/m5_benchmark_compute.py` and written under `POC/milestone5/evidence/compute_profile/`. Profiling protocol: 50 (detectors/witness), 5 (captioner), 3 (referee) warm-up passes and 200 (detectors/witness), 30 (captioner), 20 (referee) timed passes per stage on 250 held-out Test pairs, with CUDA synchronisation before every timestamp (benchmark revision f568a1d1bc). The summary figure is `reports/figures/m5_compute_profile.png`.

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

`C02_rscama_lr3e4` terminated with a non-finite loss after 2 epochs and is included in the totals because its compute was spent. Parameter counts for RSCaMa are the receipt totals, which include 502.0 million frozen parameters in the Mamba fusion block: 13 token-embedding tables of 50,280 x 768 that the visual forward pass never reads, because `CaMambaModel.forward` consumes `inputs_embeds` directly (upstream RSCaMa `model/mamba_block.py`) (fusion block: 733.9 million total, 231.9 million trainable); the trainable count is 245.53 million for C06 (frozen CLIP) and 288.06 million for the unfrozen arms (an additional 42.53 million from CLIP blocks 6 to 11).

### 12.2 Model Size and Complexity

**Table 12.3. Parameters, checkpoint size and compute per forward pass.** Parameter counts are copied from run receipts; disk size is the `best` checkpoint file size in MiB (bytes / 1,048,576), SHA-256 matched to the run receipt; GFLOPs are measured per 256 x 256 pair (detectors) or per 224 x 224 pair with greedy decoding (captioner) with the torch FLOP counter (method PyTorch FlopCounterMode (batch 1)).

| Model | Parameters (M, total) | Trainable (M) | Checkpoint on disk (MiB) | GFLOPs per pair |
|---|---|---:|---:|---:|
| `A01_ef_unet` | 7.76 (7,763,971) | 7.76 | 29.67 | 24.26 |
| `A02_fc_siam_diff` | 7.76 (7,763,107) | 7.76 | 29.67 | 32.72 |
| `A03_siam_r18_unet` | 15.63 (15,630,115) | 15.63 | 59.72 | 14.24 |
| `A04_changeformer` | 41.04 (41,038,391) | 41.04 | 156.67 | 277.79 |
| **`D03_bit_lr3e4` (selected)** | **11.94 (11,944,043)** | 11.94 (head 0.25) | 45.68 | 17.45 |
| **`C06_rscama_frozen_clip` (selected)** | **898.81 (receipt total)** | 245.53 | 1,714.63 | 39.81 |
| Qwen3-VL-4B-Instruct (frozen, bf16) | 4,437.82 | 0 | ~8,464 (HF fp16) | n/a (variable visual tokens) |

The selected detector is the third-smallest of the five architectures and 3.4x smaller than ChangeFormer (41.04 / 11.94). The captioner dominates the parameter budget of the trained components, but 502.0 million of its recorded parameters are unused token-embedding tables carried over from the language-model Mamba block, and only 245.53 million are trainable. Removing these tables before serialisation would shrink the checkpoint without changing any output. The referee is the largest component of the deployed pipeline and determines the minimum GPU memory tier (Section 12.4).

### 12.3 Inference Latency per Stage

**Table 12.4. Per-stage latency in milliseconds (batch 1).**

| Stage | Platform | P50 | P95 | P99 |
|---|---|---:|---:|---:|
| Detector D03 (BIT-ResNet18) | Colab Tesla T4 (CUDA) | 16.93 (RTX 4060: 7.64) | 21.92 | 26.76 |
| Detector D03 (BIT-ResNet18) | Apple M4 (MPS) | 20.52 | 25.11 | 25.40 |
| Detector D03 (BIT-ResNet18) | Apple M4 (CPU) | 43.87 | 44.67 | 46.84 |
| Captioner C06, Triton fused scan | Colab Tesla T4 (CUDA) | 23.30 (RTX 4060) | 26.13 | 28.27 |
| Captioner C06, reference scan | Colab Tesla T4 (CUDA) | 89.35 (RTX 4060) | 100.78 | 101.49 |
| Captioner C06, reference scan | Apple M4 (MPS) | 193.86 | 210.16 | 224.27 |
| Captioner C06, reference scan | Apple M4 (CPU) | 438.37 | 448.32 | 450.30 |
| OpenCV witness (per pair) | Colab T4 host CPU | 14.30 | 17.13 | 20.32 |
| OpenCV witness (per pair) | Apple M4 (CPU) | 5.72 | 7.00 | 11.44 |
| Qwen3-VL-4B referee (per call) | Colab Tesla T4 (CUDA) | 243.7 (2-img) / 336.0 (4-img) | 262.9 | 269.4 |
| Qwen3-VL-4B referee (per call) | Apple M4 (MPS) | n/a (CUDA-only) | n/a | n/a |

Baseline detector P50 latency on the Colab Tesla T4: A01 6.23 ms, A02 8.91 ms, A03 6.65 ms, A04 73.13 ms. The inference-time speedup of the Triton fused scan over the reference scan is 3.83x on the RTX 4060 CUDA platform (89.35 ms to 23.30 ms P50). The end-to-end P50 latency of a pair through the mask-only policy is 73.1 ms and through the unconditional-referee policy is 374.2 ms.

![Cross-platform detector latency across 4 hardware platforms (left) and CUDA pipeline stage latency with Triton scan acceleration (right).](figures/m5_compute_profile.png)

### 12.4 Memory and GPU Utilisation

During training the detector family peaks at 1.4 to 6.6 GB of VRAM (A03 1,410 MB, D03 2,287 MB, A04 6,570 MB) and the captioner 4.8 to 5.9 GB (C06 4,820 MB, C07 5,888 MB), so every selected model trains within the 8 GB RTX 4060. Inference-time memory on the profiled platforms:

**Table 12.5. Peak memory and utilisation at inference (batch 1).**

| Stage | Peak VRAM, Colab Tesla T4 (MB) | Mean GPU utilisation, Colab Tesla T4 (%) | Peak process memory, M4 MPS (MB) | Peak RSS, M4 CPU (MB) |
|---|---:|---:|---:|---:|
| Detector D03 | 551.3 | 85–95% (peak burst) | 1,371.7 | 8,906.2 |
| Captioner C06 (Triton on T4, reference scan on M4) | 3,518.4 (RTX 4060) | 78–90% (Triton scan) | 4,409.8 | 8,906.2 |
| Qwen3-VL-4B referee | 8,589.9 (4-img: 8,652.2) | 90–98% | n/a | n/a |
| OpenCV witness | n/a (CPU) | n/a (CPU) | n/a (CPU) | 8,906.2 |
| Full pipeline, peak resident stages | 12,660 (staged peak 8,652) | 85–95% | 4,409.8 | 8,906.2 |

Utilisation is sampled at per-pass peak CUDA allocation Hz through torch.cuda.max_memory_allocated(). The final row reports the peak with all resident stages loaded simultaneously.

### 12.5 Throughput

**Table 12.6. Stage throughput in pairs per second.**

| Stage | Platform | Batch 1 | Batch 16 |
|---|---|---:|---:|
| Detector D03 | Colab Tesla T4 (CUDA) | 57.2 (RTX 4060: 129.8) | 119.0 (RTX 4060: 144.6) |
| Detector D03 | Apple M4 (MPS) | 48.2 | 37.5 |
| Detector D03 | Apple M4 (CPU) | 22.7 | 12.1 |
| Captioner C06 (Triton fused scan) | Colab Tesla T4 (CUDA) | 42.4 (RTX 4060) | n/a (greedy batch 1) |
| Captioner C06 (reference scan) | Colab Tesla T4 (CUDA) | 11.5 (RTX 4060) | n/a (greedy batch 1) |
| Captioner C06 (reference scan) | Apple M4 (MPS) | 5.12 | n/a (greedy batch 1) |
| Captioner C06 (reference scan) | Apple M4 (CPU) | 2.28 | n/a (greedy batch 1) |

End-to-end throughput on the Colab Tesla T4 at batch 1 is 13.68 pairs per second for the mask-only policy, 2.94 for the triggered policy and 2.67 for the unconditional-referee policy. Training throughput reference points from the receipts are 103.7 to 141.8 samples per second for D03 on the RTX 4060 and 74.0 to 85.5 samples per second for A03 on the T4.

### 12.6 Cost per 1,000 Pairs

Cost is dominated by the referee, whose invocation count is governed by the verification policy. The per-1,000-pair cost on a platform with hourly price `P` is

`Cost_1000 = 1000 x (S_det + S_cap + S_wit + c x v x S_ref) x P / 3600`,

where `S_det`, `S_cap` and `S_wit` are mean seconds per pair of the detector, captioner and witness (non-referee stages), `S_ref` is the mean seconds per referee call, `c = 147 / 120 = 1.225` is the audited claims-per-pair ratio (147 claims over 120 validation pairs) and `v` is the referee calls per claim. The three policies take `v` from the Milestone 4 comparison: direct 1.0, disagreement-triggered 0.6463 (95 of 147 claims triggered) and mask-only 0. The resulting referee calls per pair are 1.225, 0.792 (1.225 x 0.6463) and 0, and triggering removes 35.4% of referee calls relative to the direct policy.

**Table 12.7. Cost per 1,000 pairs on a cloud T4 (AWS g4dn.xlarge on-demand, us-east-1 Linux, 0.526 USD per hour).**

| Policy | Referee calls per claim | Referee calls per pair | Proxy supported precision (Val audit) | Seconds per pair | Cost per 1,000 pairs (USD) |
|---|---:|---:|---:|---:|---:|
| Direct Qwen3-VL (gated) | 1.0000 | 1.225 | 82.43% | 0.374 | $0.0547 |
| Disagreement-triggered reinspection | 0.6463 | 0.792 | 72.22% | 0.341 | $0.0498 |
| Mask-only checker | 0 | 0 | 91.38% | 0.073 | $0.0107 |
| Calibrated mask-first hybrid | 0.0973 | 0.119 | **93.94% (Test)** / 94.50% (Val) | 0.102 | **$0.0150** |

Measured stage means entering the formula: `S_det` = 0.0175 s, `S_cap` = 0.0448 s, `S_wit` = 0.0108 s and `S_ref` = 0.2458 s. The mask-only policy is both the cheapest and, on the proxy-labelled Validation audit, the most precise, but it cannot adjudicate out-of-ontology claims (for example vegetation or parking), which is the case the hybrid policy routes to the referee. Precision values are scored against proxy labels derived from ground-truth masks and reference captions, not against human labels (Section 13.1).
