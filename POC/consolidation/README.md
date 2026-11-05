# Consolidated POC workspace

The `poc-consolidation` branch combines the current main branch, Shoaib's preserved POC/data preparation work, Sujal's latest data fixes and captioning implementation, and Priyanshu's Stage C evidence module. Source branches and their histories remain intact.

Start with [pipeline readiness](PIPELINE_STATUS.md), [experiment interpretation](EXPERIMENTS.md) and the [Milestone 3 report](../milestone3/Milestone3_Report.md).

## Canonical locations

| Work | Current location | Earlier work retained |
|---|---|---|
| Dataset preparation | `src/data/`, `configs/datasets.yaml`, `notebooks/01_EDA.ipynb`, `02_Preprocessing.ipynb` | M2 audit/scripts under `POC/milestone2/`; conflicting variants in `archive/2026-11-05/` |
| Captioning and claim extraction | `POC/execution/` | `POC/b-captioning/`, unique older files in `archive/branch-heads/`, original Git branches |
| Visual evidence | `POC/evidence/`, `POC/c-evidence/` | Original result bytes under `POC/c-evidence/results/original/` and preserved former-path copies under `results/legacy-output-paths/` |
| Decision benchmarking | `POC/benchmarks/colab/` | Original failed/successful ZIPs under `POC/benchmarks/results/2026-10-27/shoaib/archives/` |
| Dataset experiments | `POC/data_quality/` | Older executed notebook in archived branch variants |
| Submission reports | Current submissions under `reports/`; M3 review package under `POC/milestone3/` | Report sources and previous versions preserved under POC and Git history |

## Preservation

The initial inventory contains **1,787 local files / 3,229,317,824 bytes**. SHA-256 verification confirms those files were preserved unchanged. Large archives, datasets, previews and test working files stay local and Git-ignored; their inventory is tracked. No raw dataset was added to Git. A complete pre-consolidation Git bundle is retained locally at `.git/preservation/2026-11-05/all-branches.bundle`.

All 11 inspected remote branch heads are listed in `evidence/branch_inventory.json`. Both versions of 18 conflicting files were archived before selecting the newer data implementation. Nine additional unique POC blobs from older remote heads were preserved under `archive/branch-heads/`. Archive copies are historical evidence, not active modules or test targets.

## Reproduce the checks

Use Python 3.11+ in a CPU environment with `requirements-data.txt` and `POC/evidence/requirements.txt` installed, then run:

```powershell
python POC/consolidation/scripts/run_checks.py
python POC/consolidation/scripts/collect_experiments.py
python POC/consolidation/scripts/verify_preservation.py
```

The checks record 81 passing tests and one Torch-dependent collection skip in the inspected CPU environment. They do not load captioner/judge weights or establish GPU inference quality. The model packages retain their own isolated worker environments.

`inventory_branches.py` records remote-head provenance and archives unique POC content. It operates on fetched refs; fetch changes before a new inventory. Preserve the original dated inventory when taking another consolidation snapshot.

## Folder boundary - 6 November 2026

The unpromoted visual-evidence package, its tests, orchestration tests, experimental notebook and result artifacts are contained under POC. Main `src/`, `scripts/`, `configs/` and the two data notebooks match the already merged data-pipeline PR. New model/inference code remains under `POC/pipeline/`; promotion to the production code folders requires a separate authorised PR.
