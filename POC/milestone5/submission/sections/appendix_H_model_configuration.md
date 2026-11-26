## Appendix H. Model Configuration and Provenance Registry

All frozen Milestone 4 weights are stored in the team's verified artifact repository (`TerraWatch_Checkpoints_M4`) and cryptographically pinned in `POC/milestone5/configs/model_registry.json`. Weights are never committed to Git; `POC/milestone5/scripts/run_terrawatch_inference.py --verify-checksums` enforces full SHA-256 verification before inference.

### H.1 Frozen Checkpoint Registry (12 Checkpoints + 2 Hugging Face Revisions)

| Run / Asset ID | Role | Architecture | Exact Byte Size | SHA-256 Digest (Full) |
|---|---|---|---:|---|
| `D03_bit_lr3e4` | **Selected Detector** | `bit_resnet18` | 47,898,192 | `05345df154b4d748432bccd72a439d3eddbda0fa2e10795ecdac29cd2cd38feb` |
| `D03_bit_lr3e4_s0` | Seed 0 Replication | `bit_resnet18` | 47,898,192 | `782e352207804d761be6942ed5b730c4939fab59ff0285872be3718e4da10d45` |
| `D03_bit_lr3e4_s2026` | Seed 2026 Replication | `bit_resnet18` | 47,898,000 | `0d7fbda18cc943ac0f5ff70b36319dd084993146c6ae8be3cf9e191a53b91534` |
| `A01_ef_unet` | Architecture Baseline | `ef_unet` | 31,114,207 | `5831a24249abf330f65a0b2c278040b2908b9137a2e95472544ccf34f91a437a` |
| `A02_fc_siam_diff` | Architecture Baseline | `fc_siam_diff` | 31,110,751 | `32329cf708a7fed397c96689922eca2a2f29556108c35c214a1c332326b23105` |
| `A03_siam_r18_unet` | Clean Challenger | `siam_resnet18_unet` | 62,621,663 | `c9d2c9c994c63411096703397ee1842cde5ef917bee43d7b974a3f47c511e586` |
| `A03_siam_r18_unet_s0` | Seed 0 Replication | `siam_resnet18_unet` | 62,621,663 | `1b3474c7b33f13749580210308c471ae189ce194c678a933729dfb769d678890` |
| `A03_siam_r18_unet_s2026` | Seed 2026 Replication | `siam_resnet18_unet` | 62,621,663 | `ad063b288b245554256412d95d4e5e5787f588a6e4de012bcf85acb97dc2d9a8` |
| `A04_changeformer` | Transformer Baseline | `changeformer` | 164,282,148 | `fe137902d7dc4ca4816ab1e777f93d1f383cd132ca4542bc23a3b96c26ea6bc7` |
| `C06_rscama_frozen_clip` | **Selected Captioner** | `rscama_vit_b32_frozen` | 1,797,923,010 | `c5b80d6bf6b6b0b2628c48d47df07ad77bc086ff0004bdd7160d0b0ff5545796` |
| `C01_rscama_ref` | Captioner Baseline | `rscama_vit_b32_adam` | 1,797,923,010 | `979cff69acb852f22e24ca1edf2daaa9bf295bf69fc5c6f447c6508fdd24d7e3` |
| `C04_rscama_adamw_cos` | Captioner Baseline | `rscama_vit_b32_adamw` | 1,797,923,010 | `415cadc4abac43108aba91dca3ae547c0b8d95c8f69d6f66776e669857297e46` |
| `C00_rscama_released` | Published Benchmark | `lcybuaa/RSCaMa` | — | `5eeba271fb7c2ff4722b725cbf5ce1a1fdc0a33ddf1b077dcf5381aebd643449` |
| `Qwen3-VL-4B-Instruct` | **Selective Referee** | `Qwen/Qwen3-VL-4B-Instruct` | — | HF revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

### H.2 Locked Hybrid Verifier Policy Configuration (`calibrated_mask_first_hybrid_v1`)

```json
{
  "detector": "D03_bit_lr3e4",
  "captioner": "C06_rscama_frozen_clip",
  "referee": "Qwen/Qwen3-VL-4B-Instruct",
  "verifier_policy": "calibrated_mask_first_hybrid_v1",
  "calibration_split": "levir_mci_val_120_pairs_147_claims",
  "tau_accept": 0.60,
  "tau_reject": 0.75,
  "mask_alarm_px": 50,
  "weak_mask_escalate_px": 200,
  "residual_no_change_escalate_px": 10,
  "min_crop_size": 32,
  "crop_padding_fraction": 0.25
}
```
