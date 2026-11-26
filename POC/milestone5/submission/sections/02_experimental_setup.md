## 2. Experimental Setup

This section fixes the hardware, software, data and reproducibility contract under which every Milestone 5 number is produced. All models evaluated in this milestone are the frozen Milestone 4 artifacts; no model is retrained, re-tuned or re-selected during evaluation.

### 2.1 Hardware and Software Environment

Training in Milestone 4 used three GPU classes, and inference profiling in Milestone 5 uses three platforms: the local RTX 4060 workstation, a Colab Tesla T4 session and an Apple M4 laptop. Versions for the Milestone 4 platforms are copied from the `environment` block of each run receipt; versions for the Milestone 5 profiling platforms are copied from the `env` block of the corresponding file in `evidence/compute_profile/`, which the benchmark script writes at run time.

**Table 2.1. Compute platforms and software stack.**

| Platform | Accelerator | Python | PyTorch | Other pinned runtime | Role |
|---|---|---|---|---|---|
| Local workstation (Omen, Linux) | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM | 3.13.9 | 2.11.0+cu130 (bf16 AMP) | einops, timm, OpenCV, NLTK | 14 detector runs (12 BIT arms, D03 seed 0, A04 ChangeFormer), robustness ladder, LEVIR-MCI Test and SECOND-CC transfer evaluation |
| Colab Pro, Tesla T4 | Tesla T4, 15 GB VRAM | 3.13.16 | 2.14.1+cu130 (fp16 AMP) | Triton 3.8.0; Transformers 4.39.3 (RSCaMa) | 7 detector runs (A01, A02, A03, D04, D07, D09, D16), all 9 captioner runs, Qwen3-VL 120-pair verification |
| Colab Pro, L4 | NVIDIA L4, 24 GB VRAM | 3.13.16 | 2.14.1+cu130 (fp16 AMP) | n/a (run receipts record Python and PyTorch only) | 3 seed runs (D03 s2026, A03 s0, A03 s2026) |
| Referee environment (Colab Pro) | Colab GPU (L4 or T4; the verification receipt does not record the device) | separate environment | 2.14.1+cu130 | Transformers 5.22.1; Qwen3-VL-4B-Instruct in bfloat16 | Frozen visual referee in the 120-pair verification |
| Milestone 5 profiling: RTX 4060 workstation | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), Intel Core i7-14650HX (24 logical CPUs), Linux 6.14 | 3.13.9 | 2.11.0+cu130 (torchvision 0.26.0+cu130) | CUDA 13.0; Triton 3.6.0; Transformers 4.57.6 (detector and witness run) and 4.39.3 (captioner run); OpenCV version not recorded; driver version not recorded | Detector, captioner (trained C06 checkpoint, Triton and reference scan) and witness latency and memory |
| Milestone 5 profiling: Colab Tesla T4 | Tesla T4 (15 GB), Intel Xeon @ 2.00 GHz (2 logical CPUs), Linux 6.6 | 3.13.16 | 2.14.1+cu130 (torchvision 0.29.1+cu130) | CUDA 13.0; Triton 3.8.0; Transformers 5.22.1 (detector, witness and referee runs); OpenCV version not recorded; driver version not recorded | Detector, witness and Qwen3-VL referee latency and memory; captioner not measured on this platform |
| Milestone 5 profiling: Apple M4 | Apple M4 (10 logical CPUs, unified memory), backends `cpu` and `mps`, macOS 26.5.1 | 3.11.9 | 2.15.0 (torchvision 0.29.2) | Transformers 4.39.3; no CUDA or Triton; OpenCV version not recorded | Detector, captioner and witness latency and memory with randomly initialised weights (timing and memory only); referee not measured |

The RTX 4060 detector and witness profile used Transformers 4.57.6 (`env.transformers` in `compute_profile_rtx4060.json`), whereas its captioner profile used Transformers 4.39.3 (`env.transformers_captioner` and `compute_profile_rtx4060_cap.json`), matching the RSCaMa pin. The OpenCV version is not recorded by the benchmark; the install specification is `opencv-python-headless>=4.10,<6` (`requirements-pipeline.txt`). The Qwen3-VL referee was profiled on the T4 in float16 (`compute_profile_colab_t4_referee.json`), whereas the Milestone 4 verification receipt records bfloat16. On the Apple M4 the detector and captioner run with random-initialised weights because latency and memory do not depend on weight values, and the random-initialised captioner never emits the end-of-sequence token, so every Mac captioner figure is the full-length decode (42 steps) and an upper bound on the latency of the trained model.

**Software stack.** Detector and captioner code derive from pinned upstream sources: BIT_CD at commit `adcd7aea6f23`, ChangeFormer at `afd1b7ed640a`, RSCaMa at `aa68e4c6933c` and OpenAI CLIP ViT-B/32 at `dcba3cb2e282`. The visual referee is `Qwen/Qwen3-VL-4B-Instruct` at the pinned revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` (`configs/pipeline.json`). The deterministic structural witness is implemented with OpenCV (LAB luminance shift, Sobel edge density, block anomaly, SSIM and sub-pixel alignment jitter). Prebuilt `mamba_ssm` binaries do not exist for Python 3.13 with CUDA 13.0, so RSCaMa training and inference use the exact fused Triton selective-scan kernel (`kernel_mode: triton_fused_exact` in every captioner receipt). The captioner (Transformers 4.39.3) and the referee (Transformers 5.22.1) run in separate Python environments and exchange verified JSONL payloads with SHA-256 provenance.

**Install specification versus versions used.** The `requirements-*.txt` files record the minimum installable specification, not the environment that produced the results. They pin `torch==2.8.0` and `torchvision==0.23.0` (`requirements-detector.txt`, `requirements-captioner.txt`), `transformers==4.39.3` for the captioner (matching the version used) and `transformers==5.17.0` for the referee (`requirements-referee.txt`), whereas the receipts record torch 2.11.0+cu130 and 2.14.1+cu130 and the M4 report records Transformers 5.22.1 for Qwen3-VL. Reported results correspond to the versions in Table 2.1.

### 2.2 Dataset Versions and Splits

**Table 2.2. Data contract.**

| Partition | Source | Pairs | Captions | Use in Milestone 5 |
|---|---|---:|---:|---|
| LEVIR-MCI Train | LEVIR-MCI (HF revision `dff09c709999b5b1f1f1806024811c65a6b14dfe`) | 6,444 | 32,220 | Training only; sole source of the 388-token captioner vocabulary (minimum frequency 5) |
| LEVIR-MCI Validation | LEVIR-MCI | 1,242 (610 changed, 632 unchanged) | 6,210 | Model selection, robustness ladder, 120-pair verification audit (147 claims) and verifier calibration |
| LEVIR-MCI Test | LEVIR-MCI | 1,858 (934 changed, 924 unchanged) | 9,290 | Locked held-out evaluation of the frozen system |
| SECOND-CC-AUG Validation | SECOND-CC-AUG (Zenodo 16937571) | 530 | n/a | Zero-shot cross-dataset transfer, detector only |
| SECOND-CC-AUG Test | SECOND-CC-AUG | 1,142 | n/a | Zero-shot cross-dataset transfer, detector only |

Masks use the encoding background 0, road change 128 and building change 255, mapped to class indices 0, 1 and 2 in every trainer and evaluator. Detector inputs are 256 x 256 RGB scaled to [-1, 1]; captioner inputs follow CLIP ViT-B/32 normalisation at 224 x 224. The split is governed by a portable manifest whose receipt (`portable_data_receipt.json`) stores the SHA-256 of each manifest file and of each ordered pair-identifier list, for example the train pair-identifier digest `2e38e5e22aa27c5b8101dd2f4fc7ea5127a8378d069c5eae5c3f90832e3f137e` and the validation digest `bf165d95ac0a6a4fa52770fc8e2f9270f6926b98f53143001d883e8e7a581511`. Every detector and captioner receipt repeats these digests, and the trainers assert exact pair counts and zero train, validation and test overlap. SECOND-CC-AUG is evaluation-only: no model has seen it during training, and WHU-CDC is excluded from training.

### 2.3 Random Seeds and Reproducibility

**Seeds.** The primary seed is 42. Stability is established with seeds 42, 0 and 2026 for D03 (BIT-ResNet18), A03 (Siam-ResNet18-UNet) and C06 (RSCaMa with frozen CLIP). Every run receipt stores its seed and command line, and the verification pipeline seeds the worker processes through `configs/pipeline.json` (`seed: 42`).

**Deterministic settings actually used.** Both full-data trainers call `seed_everything(seed)`, which seeds Python `random`, NumPy, `torch.manual_seed` and `torch.cuda.manual_seed_all`. The per-epoch shuffle uses a dedicated generator seeded with `seed + epoch * 1009`, so data order is a pure function of the seed and the epoch and is identical across resumed and uninterrupted runs. Detector training sets `torch.backends.cudnn.benchmark = True` and does not enable `torch.use_deterministic_algorithms`; consequently GPU kernels may select different algorithms across hardware and bitwise reproducibility across devices is not claimed. The seed spread measured in Milestone 4 (D03 74.23% +/- 0.28% Val change mIoU, A03 74.73% +/- 0.27%, C06 0.6297 +/- 0.0183 Val BLEU-4) is the empirical reproducibility bound. Resumed versus continuous training equivalence was checked before the sweep (loss delta 3.11e-4), and the Triton kernel matches FP32 reference gradients within 1.2e-7 to 4.4e-7 relative error.

**Checkpoint SHA-256 registry.** Every `run.json` records `best_checkpoint_sha256`, and `configs/pipeline.json` pins the digests loaded at inference. Weights are never committed to Git; evaluation scripts verify the digest before loading.

| Component | Run | Checkpoint SHA-256 |
|---|---|---|
| Selected detector | `D03_bit_lr3e4` | `05345df154b4d748432bccd72a439d3eddbda0fa2e10795ecdac29cd2cd38feb` |
| Baseline `A01_ef_unet` | `A01_ef_unet` | `5831a24249abf330f65a0b2c278040b2908b9137a2e95472544ccf34f91a437a` |
| Baseline `A02_fc_siam_diff` | `A02_fc_siam_diff` | `32329cf708a7fed397c96689922eca2a2f29556108c35c214a1c332326b23105` |
| Baseline `A03_siam_r18_unet` | `A03_siam_r18_unet` | `c9d2c9c994c63411096703397ee1842cde5ef917bee43d7b974a3f47c511e586` |
| Baseline `A04_changeformer` | `A04_changeformer` | `fe137902d7dc4ca4816ab1e777f93d1f383cd132ca4542bc23a3b96c26ea6bc7` |
| Selected captioner | `C06_rscama_frozen_clip` | `c5b80d6bf6b6b0b2628c48d47df07ad77bc086ff0004bdd7160d0b0ff5545796` |
| Released RSCaMa reference | `C00_rscama_released` | `5eeba271fb7c2ff4722b725cbf5ce1a1fdc0a33ddf1b077dcf5381aebd643449` |
| Visual referee | `Qwen/Qwen3-VL-4B-Instruct` | Hugging Face revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

**Selected hyperparameters.** `configs/detector_training.json` and `configs/captioner_training.json` record the final settings: D03 trains BIT-ResNet18 with AdamW, head LR 3e-4, backbone LR 3e-5, weight decay 0.01, cosine schedule, Focal plus Soft Dice loss, 5 frozen-backbone epochs, flip augmentation, effective batch 32 (16 x 2 accumulation), gradient clipping 1.0 and bf16 AMP, with a 35-epoch cap; C06 trains RSCaMa with frozen CLIP, AdamW, LR 1e-4, weight decay 0.01, cosine schedule, batch 12, gradient clipping 5.0, fp16 AMP and an 8-epoch cap.

### 2.4 Evaluation Workflow

1. **Verify provenance.** Recompute the SHA-256 of each checkpoint and of the manifest pair-identifier lists against Section 2.3 and `portable_data_receipt.json`; abort on any mismatch.
2. **Freeze selection.** Detector and captioner selection uses Validation only (primary metrics: Change-Class mIoU and corpus BLEU-4). No threshold or checkpoint is revisited after Test is opened.
3. **Calibrate the verifier on Validation.** The 120-pair, 147-claim stratified audit (40 unchanged, 30 building-only, 15 road-only, 20 mixed, 15 small-change) fixes the referee operating threshold (tau = 0.70) and the routing policy.
4. **Evaluate on held-out Test.** Run the frozen detectors and captioner once on the 1,858 LEVIR-MCI Test pairs and the detector on SECOND-CC-AUG, with ground-truth masks and reference captions read only by the offline evaluator, never by the inference workers.
5. **Stress and transfer.** Apply the deterministic corruption ladder and cross-dataset transfer protocol to D03 and A03.
6. **Verify claims.** Extract atomic claims from generated captions, score each against the OpenCV witness and the detector mask, route selectively to the Qwen3-VL referee, and emit accept, reject or defer with the full evidence record.
7. **Blind human audit.** Two raters label each audited claim without seeing the model verdict; a third adjudicates disagreements (Appendix J); Cohen's kappa and human-verified groundedness are computed.
8. **Profile compute.** Time every stage on each profiled platform of Table 2.1 and compute memory, throughput and cost per 1,000 pairs (Section 12); stages that were not run on a platform are reported as not measured.

```
 T1,T2 --> BIT detector (D03) --> 3-class mask ------------------+
   |                                                             v
   +--> RSCaMa captioner (C06) --> caption --> claim extractor --> OpenCV witness --+
                                                                                    v
                                      mask-first routing --(ambiguous)--> Qwen3-VL referee
                                                  |                           |
                                                  +----> accept / reject / defer ----> Evidence Review UI
                                                                                    |
                                                      human audit (2 raters + adjudicator), kappa
```
