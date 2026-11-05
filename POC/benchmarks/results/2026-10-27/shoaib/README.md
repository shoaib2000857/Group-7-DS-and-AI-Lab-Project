# Shoaib (SSSM): received Colab runs

The three original results ZIPs were moved from Downloads into the local `archives/` subfolder on 27 October 2026. SHA-256 hashes were verified before and after the move; ZIP CRC checks passed. The originals remain unmodified. Raw archives are retained locally and excluded from Git; this inventory and diagnosis are tracked.

| Archive | Profile | Real pairs | Size (bytes) | GPU | Initial result |
|---|---|---:|---:|---|---|
| `SSSM_20261027_141130_results.zip` | smoke | 10 | 4,216,027 | NVIDIA L4 | Worker environment creation failed |
| `SSSM_20261027_141651_results.zip` | smoke | 10 | 4,216,026 | NVIDIA L4 | Worker environment creation failed |
| `SSSM_20261027_141846_results.zip` | screen | 40 | 11,792,920 | NVIDIA L4 | Worker environment creation failed |

All three used runtime `2026-10-27.1`, Python 3.13.16, CUDA PyTorch 2.11.0 and an L4 with approximately 22.49 GiB. The first two used four train/development and six validation pairs; the screen used ten and thirty. Model IDs were J05, J01, J06 and J07.

## Diagnosis

The dataset/controls/export path completed, but J06 witness setup and every selected judge failed at `python -m venv --system-site-packages`. No learned-model inference completed. These runs are setup diagnostics, not model-accuracy evidence. Their reference/proxy/rule outputs must not be reported as learned benchmark improvements.

The old runtime did not capture this subprocess's detailed stderr in its ZIP, so the exact interpreter-level cause is not recoverable from these archives. Missing/disabled pip bootstrapping is a plausible failure mode, not a proven diagnosis. The repair replaces this dependency with a private pinned virtualenv bootstrap, logs setup output and checks environment creation before loading benchmark data.

See [inventory.json](inventory.json) for exact hashes and source/run metadata. The repaired notebook and its current launch link are in [the Colab instructions](../../../colab/README.md).

## Checked findings

- Every file listed in each export manifest matches its SHA-256; all ZIP CRC checks pass. Each archived runtime matches the notebook implementation provided in commit `256a1dd`.
- The two smoke runs have identical metric CSVs and selected pair IDs. Both ten-pair sets are contained in the forty-pair screen. The union is forty distinct real pairs, not sixty independent observations.
- There are no `predictions/J*` files in any ZIP. All learned completion rates are zero because setup stopped before inference; quality metrics are missing rather than measured zeroes. The J06 witness is also unavailable.
- R02 defers on every real satellite claim because witness evidence is missing. Its perfect score on constructed logic fixtures is a fixture check, not satellite accuracy.
- The diagnostic mask flags all six smoke validation pairs and all thirty screen validation pairs as changed. Dataset annotations mark three of six and sixteen of thirty as unchanged. This is a false-alarm warning about the abs-difference/Otsu control; no pixel-mask F1 or human factuality has been measured.

| Validation-only annotation-proxy measure | Smoke (either run) | Screen |
|---|---:|---:|
| Real validation pairs | 6 | 30 |
| Constructed/proxy claims per control | 30 | 150 |
| R00 accepted precision | 40.0% | 40.0% |
| R01 accepted precision | 50.0% | 46.7% |
| R01 true-claim acceptance | 25.0% | 23.3% |
| R01 false-claim rejection | 25.0% | 23.3% |
| R01 deferral | 60.0% | 60.0% |
| R02 deferral | 100.0% | 100.0% |
| Any learned judge outputs | 0 | 0 |

R00's 40% follows from the designed two-supported/two-false/one-unverifiable claim mixture and its accept-all rule; it is not an estimate of real caption factuality. R01 only handles the two presence/absence propositions and defers on the remaining three, so its 60% deferral is also structural. Do not pool the synthetic fixtures with these satellite annotation proxies.

Next run: use the repaired notebook, a fresh run folder and `PROFILE="smoke"`; leave optional runtimes disabled. Do not resume a `.1` run into `.2` or repeat screen until the new smoke ZIP contains successful learned predictions. The old archives remain the diagnostic baseline.

## Additional receipt - 29 October 2026

Three further ZIPs were moved from Downloads into rchives/, with before/after SHA-256 verification and passing ZIP CRC checks. Both files named for the 164811 run are preserved separately; their archive SHA-256 hashes differ. The 164103 run is smoke; the two 164811 files are screen exports. Quick status: J05/J06 completed, J01 failed or timed out, and J07 was partial or failed. Receipt checks covered run status only; no detailed analysis, model runs or fixes were performed. See the additional inventory records.
