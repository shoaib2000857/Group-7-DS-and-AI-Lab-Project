## 12. Computational Performance

This section reports the training cost recorded in Milestone 4 receipts and the inference cost measured in Milestone 5 on the three profiled platforms of Table 2.1 (RTX 4060 workstation, Colab Tesla T4 and Apple M4). The NVIDIA L4 appears only in the training-cost results; no inference profile was run on it. Training figures are read directly from the 33 run receipts by `POC/milestone5/scripts/m5_training_cost_table.py`; inference figures are produced by `POC/milestone5/scripts/m5_benchmark_compute.py` and written under `POC/milestone5/evidence/compute_profile/`. Profiling protocol on the CUDA platforms: 50 (detectors, witness), 5 (captioner) and 3 (referee) warm-up passes and 200 (detectors, witness), 30 (captioner) and 20 (referee) timed passes per stage at batch 1, with device synchronisation before every timestamp (benchmark revision f568a1d on the CUDA platforms, 4be3e22 on the Mac). Inputs are LEVIR-MCI Validation pairs from `POC/execution/data/pairs` (250 loaded on the CUDA platforms, 150 on the Mac); the captioner and referee timed passes cycle through 35 and 23 distinct pairs respectively, and the witness runs on ground-truth masks. On the Apple M4 the MPS detector and witness use the same 50 and 200 passes, the MPS captioner 10 and 50, while the CPU backend uses 5 and 30 for detectors (5 timed passes at batch 16) and 2 and 10 for the captioner. The Qwen3-VL referee was profiled on the Colab T4 only. The captioner was profiled on the RTX 4060 and the Apple M4 only; it was not measured on the Colab T4. The summary figure is `reports/figures/m5_compute_profile.png`.

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

**Table 12.3. Parameters, checkpoint size and compute per forward pass.** Parameter counts are copied from run receipts; disk size is the `best` checkpoint file size in MiB (bytes / 1,048,576), SHA-256 matched to the run receipt; GFLOPs are measured per 256 x 256 pair (detectors) or per 224 x 224 pair with greedy decoding (captioner) with the torch FLOP counter (method PyTorch FlopCounterMode (batch 1)). The counter includes only matrix multiplication, convolution and attention operations; the elementwise operations of the selective scan are not counted, so the captioner figure understates its arithmetic.

| Model | Parameters (M, total) | Trainable (M) | Checkpoint on disk (MiB) | GFLOPs per pair |
|---|---|---:|---:|---:|
| `A01_ef_unet` | 7.76 (7,763,971) | 7.76 | 29.67 | 24.26 |
| `A02_fc_siam_diff` | 7.76 (7,763,107) | 7.76 | 29.67 | 32.72 |
| `A03_siam_r18_unet` | 15.63 (15,630,115) | 15.63 | 59.72 | 14.24 |
| `A04_changeformer` | 41.04 (41,038,391) | 41.04 | 156.67 | 277.79 |
| **`D03_bit_lr3e4` (selected)** | **11.94 (11,944,043)** | 11.94 (head 0.25) | 45.68 | 17.45 |
| **`C06_rscama_frozen_clip` (selected)** | **898.81 (receipt total)** | 245.53 | 1,714.63 | 39.81 |
| Qwen3-VL-4B-Instruct (frozen) | 4,437.82 | 0 | not recorded (load peak VRAM 8,464.5 MB in float16) | n/a (variable visual tokens) |

The selected detector is the third-smallest of the five architectures and 3.4x smaller than ChangeFormer (41.04 / 11.94). The captioner dominates the parameter budget of the trained components, but 502.0 million of its recorded parameters are unused token-embedding tables carried over from the language-model Mamba block, and only 245.53 million are trainable. Removing these tables before serialisation would shrink the checkpoint without changing any output. The referee is the largest component of the deployed pipeline and determines the minimum GPU memory tier (Section 12.4).

### 12.3 Inference Latency per Stage

**Table 12.4. Per-stage latency in milliseconds (batch 1).**

| Stage | Platform | P50 | P95 | P99 |
|---|---|---:|---:|---:|
| Detector D03 (BIT-ResNet18) | Colab Tesla T4 (CUDA) | 16.93 | 21.92 | 26.76 |
| Detector D03 (BIT-ResNet18) | RTX 4060 (CUDA) | 7.64 | 8.28 | 8.58 |
| Detector D03 (BIT-ResNet18) | Apple M4 (MPS) | 20.52 | 25.11 | 25.40 |
| Detector D03 (BIT-ResNet18) | Apple M4 (CPU) | 43.87 | 44.67 | 46.84 |
| Captioner C06, Triton fused scan | RTX 4060 (CUDA) | 23.30 | 26.13 | 28.27 |
| Captioner C06, Triton fused scan | Colab Tesla T4 (CUDA) | not measured | not measured | not measured |
| Captioner C06, reference scan | RTX 4060 (CUDA) | 89.35 | 100.78 | 101.49 |
| Captioner C06, reference scan | Colab Tesla T4 (CUDA) | not measured | not measured | not measured |
| Captioner C06, reference scan, full-length decode | Apple M4 (MPS) | 193.86 | 210.16 | 224.27 |
| Captioner C06, reference scan, full-length decode | Apple M4 (CPU) | 438.37 | 448.32 | 450.30 |
| OpenCV witness (per claim) | Colab T4 host CPU | 14.30 | 17.13 | 20.32 |
| OpenCV witness (per claim) | Apple M4 (CPU) | 5.72 | 7.00 | 11.44 |
| Qwen3-VL-4B referee (per call, 2 images) | Colab Tesla T4 (CUDA) | 243.7 | 262.9 | 269.4 |
| Qwen3-VL-4B referee (per call, pair plus 2 crops, 4 images) | Colab Tesla T4 (CUDA) | 336.0 | 352.6 | 354.5 |
| Qwen3-VL-4B referee | Apple M4 | not measured | not measured | not measured |

The RTX 4060 captioner rows decode a mean of 8.69 steps with the trained checkpoint. The Apple M4 captioner rows use randomly initialised weights, which never emit the end-of-sequence token, so each pass decodes the full 42 steps and the figures are an upper bound on the latency of the trained model. The Mac detector figures also use random-initialised weights. The witness unit is one claim (candidate box, contextual crop, overlay and structural inspection).

The referee latency is a single forward pass that scores three verdict-label logits, not free-text generation: `Qwen3VLReferee.decide` in `src/inference/verification_pipeline.py` calls the model with `logits_to_keep=1` and applies a softmax over the three label tokens. This is why it is far below the 7.6 s per generated response recorded for Qwen3-VL in Milestone 3 (`reports/Milestone3_Report_LocalBenchmarked.md`).

Baseline detector P50 latency on the Colab Tesla T4: A01 6.23 ms, A02 8.91 ms, A03 6.65 ms, A04 73.13 ms. The inference-time speedup of the Triton fused scan over the reference scan is 3.83x on the RTX 4060 (89.35 ms to 23.30 ms P50); it was not measured on the T4. Section 12.6 gives mean-based end-to-end estimates per pair: 54.3 ms for the mask-only policy and 355.4 ms for the unconditional-referee policy. These are sums of separately measured stage means on mixed platforms (detector, witness and referee on the T4, captioner on the RTX 4060), not a measured end-to-end run.

![Cross-platform detector latency across 4 hardware backends (left) and per-stage latency on the RTX 4060 and Colab T4 (right); the Colab T4 has no captioner bar because the captioner was not measured there.](figures/m5_compute_profile.png)

### 12.4 Memory

During training the detector family peaks at 1.4 to 6.6 GB of VRAM (A03 1,410 MB, D03 2,287 MB, A04 6,570 MB) and the captioner 4.8 to 5.9 GB (C06 4,820 MB, C07 5,888 MB), so every selected model trains within the 8 GB RTX 4060. Inference-time memory on the profiled platforms is the peak of `torch.cuda.max_memory_allocated()` on CUDA, the polled MPS driver allocation on the Apple M4 GPU, and the cumulative peak process resident set size on CPU:

**Table 12.5. Peak memory at inference (batch 1).**

| Stage | Colab Tesla T4, peak VRAM (MB) | RTX 4060, peak VRAM (MB) | Apple M4 MPS, peak process memory (MB) | Apple M4 CPU, peak RSS (MB) |
|---|---:|---:|---:|---:|
| Detector D03 | 551.3 | 551.5 | 1,371.7 | 8,906.2 |
| Captioner C06 (Triton scan on RTX 4060; reference scan on Apple M4) | not measured | 3,518.4 | 4,403.8 | 8,906.2 |
| Qwen3-VL-4B referee (2 images; 4 images: 8,652.2) | 8,589.9 | not measured | not measured | not measured |
| OpenCV witness | 1,925.7 (host RSS) | 3,461.4 (host RSS) | 3,071.5 (host RSS) | 8,906.2 |

The Apple M4 CPU and witness figures are the cumulative peak resident set size of the whole benchmark process, which is why they are identical across stages, and the witness host RSS values on the other platforms are likewise process-level peaks. The Mac captioner figures use random-initialised weights. A conservative upper bound for holding the detector, captioner and referee resident together is 12,660 MB, the sum of the peaks above (551.3 MB T4 detector, 3,518.4 MB RTX 4060 captioner Triton scan and 8,589.9 MB T4 referee); this mixes platforms, and no co-resident run was measured. The largest single-stage peak, with stages loaded one at a time, is 8,652.2 MB (referee, 4 images).

### 12.5 Throughput

**Table 12.6. Stage throughput in pairs per second.** Batch-1 throughput is the reciprocal of the mean batch-1 latency.

| Stage | Platform | Batch 1 | Batch 16 |
|---|---|---:|---:|
| Detector D03 | Colab Tesla T4 (CUDA) | 57.2 | 119.0 |
| Detector D03 | RTX 4060 (CUDA) | 129.8 | 144.6 |
| Detector D03 | Apple M4 (MPS) | 48.2 | 37.5 |
| Detector D03 | Apple M4 (CPU) | 22.7 | 12.1 |
| Captioner C06 (Triton fused scan) | RTX 4060 (CUDA) | 42.4 | n/a (greedy batch 1) |
| Captioner C06 (Triton fused scan) | Colab Tesla T4 (CUDA) | not measured | not measured |
| Captioner C06 (reference scan) | RTX 4060 (CUDA) | 11.5 | n/a (greedy batch 1) |
| Captioner C06 (reference scan, full-length decode) | Apple M4 (MPS) | 5.12 | n/a (greedy batch 1) |
| Captioner C06 (reference scan, full-length decode) | Apple M4 (CPU) | 2.28 | n/a (greedy batch 1) |

End-to-end throughput is not measured. From the stage means of Section 12.6 (mixed-platform estimate: detector, witness and referee on the T4, captioner on the RTX 4060) it is 18.42 pairs per second for the mask-only policy, 3.11 for the disagreement-triggered policy, 2.81 for the unconditional-referee policy and 11.92 for the calibrated mask-first hybrid. Training throughput reference points from the receipts are 103.7 to 141.8 samples per second for D03 on the RTX 4060 and 74.0 to 85.5 samples per second for A03 on the T4.

### 12.6 Cost per 1,000 Pairs

Cost is dominated by the referee, whose invocation count is governed by the verification policy. The per-1,000-pair cost on a platform with hourly price `P` is

`Cost_1000 = 1000 x (S_det + S_cap + c x S_wit + c x v x S_ref) x P / 3600`,

where `S_det` and `S_cap` are mean seconds per pair of the detector and captioner, `S_wit` is the mean seconds per claim of the witness, `S_ref` is the mean seconds per referee call, `c` is the claims-per-pair ratio and `v` is the referee calls per claim. The audited ratio is `c = 147 / 120 = 1.225` (147 claims over 120 validation pairs) for the three Milestone 4 policies, and `c = 185 / 150 = 1.2333` (185 claims over 150 Test pairs) for the hybrid, whose call count of 18 referee calls over 150 Test pairs (0.0973 per claim, 0.120 per pair) was measured on the Test audit. The three Milestone 4 policies take `v` from the Milestone 4 comparison: direct 1.0, disagreement-triggered 0.6463 (95 of 147 claims triggered) and mask-only 0. The resulting referee calls per pair are 1.225, 0.792 (1.225 x 0.6463) and 0, and triggering removes 35.4% of referee calls relative to the direct policy.

**Table 12.7. Cost per 1,000 pairs, mixed-platform estimate priced at a cloud T4 (AWS g4dn.xlarge on-demand, us-east-1 Linux, 0.526 USD per hour).**

| Policy | Referee calls per claim | Referee calls per pair | Proxy supported precision | Seconds per pair | Cost per 1,000 pairs (USD) |
|---|---:|---:|---:|---:|---:|
| Direct Qwen3-VL (gated), 2-image calls | 1.0000 | 1.225 | 82.43% (Val audit) | 0.355 | $0.0519 |
| Disagreement-triggered reinspection, 4-image calls | 0.6463 | 0.792 | 72.22% (Val audit) | 0.322 | $0.0470 |
| Mask-only checker | 0 | 0 | 91.38% (Val audit) | 0.054 | $0.0079 |
| Calibrated mask-first hybrid, 2-image calls | 0.0973 | 0.120 | **93.94% (Test)** / 94.50% (Val) | 0.084 | **$0.0123** |

Measured stage means entering the formula: `S_det` = 0.0175 s (Colab T4), `S_cap` = 0.0236 s (RTX 4060, Triton scan), `S_wit` = 0.0108 s per claim (Colab T4 host CPU), `S_ref` = 0.2458 s for a 2-image call and 0.3380 s for a 4-image call (Colab T4). The captioner was not measured on the T4, so the figures in Table 12.7 are a mixed-platform estimate: the captioner term uses the RTX 4060 measurement and, because the T4 was slower than the RTX 4060 on the detector (16.93 ms against 7.64 ms P50), it probably understates the cost of a pure-T4 deployment. The direct and hybrid policies are priced with the 2-image referee call, the triggered policy with the 4-image call (pair plus crops). Mean-based seconds per pair are sums of separately measured means and were not measured end to end. The mask-only policy is both the cheapest and, on the proxy-labelled Validation audit, the most precise, but it cannot adjudicate out-of-ontology claims (for example vegetation or parking), which is the case the hybrid policy routes to the referee. Precision values are scored against proxy labels derived from ground-truth masks and reference captions, not against human labels (Section 13.1).
