# Dataset source and usage register

Source statements were inspected on 29-31 October 2026. Captured text and author metadata are in `evidence/source_snapshots/`. A public download link and a code licence do not independently grant rights to every underlying image or derivative annotation.

| Dataset | Controlling source | Recorded facts | Package disposition |
|---|---|---|---|
| LEVIR-MCI | [Author-linked release](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI), [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent) | Pinned revision `dff09c709999b5b1f1f1806024811c65a6b14dfe`; card describes dataset and links authors but gives no standalone dataset licence. Change-Agent's README discusses MIT and academic use of code; that is not substituted for an imagery grant. | Primary local academic research package. Author attribution retained. Public materials contain code, identifiers, hashes and aggregates; no raw-image redistribution or commercial-rights claim. |
| LEVIR-CC | [Author dataset](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) | Caption source for the same paired imagery; five references per pair. | Same primary record population, not another independent dataset added to its size. |
| SECOND-CC | [Author repository](https://github.com/ChangeCapsInRS/SecondCC), [Zenodo record 16937571](https://doi.org/10.5281/zenodo.16937571) | Author-linked record specifies CC BY 4.0 and `SECOND-CC-AUG.zip` (2,539,187,782 bytes). Source imagery/semantic conventions belong to [SECOND](https://captain-whu.github.io/SCD/). | Separate transfer evaluation. Preserve attribution and provenance, exclude augmented copies from the original-pair probe, do not merge its labels or scores with LEVIR training. |
| WHU-CDC mirror | [Captioned mirror](https://huggingface.co/datasets/hygge10111/RS-CDC), [WHU source imagery](https://gpcv.whu.edu.cn/data/building_dataset.html) | Inspected mirror revision `a8a0343de8b05586ada84cf423da746f7bedf508` has no dataset card/licence. Source imagery terms do not automatically establish rights to the caption derivative. | Excluded from governed training package. Earlier exploratory results remain attributed historical evidence, not a new licensed training release. |

The original [LEVIR-CD site](https://justchenhao.github.io/LEVIR/) states academic-only/non-commercial use and respect for Google Earth terms. This is recorded as parent-dataset context, not automatically transferred to every LEVIR derivative.

## Acquired primary artifact

- Archive: `LEVIR-MCI-dataset.zip`, 2,771,943,190 bytes.
- SHA-256: `8f7d52298fa3ca32aec983e84addef80f61ad7d981c0ff97a92c5880c4b83e9d`.
- Local retained original: `data/raw/LEVIR-MCI-dataset.zip`.
- Exact file members and individual hashes: frozen manifests and `evidence/full_audit/pair_profile.csv`.
- No original file is overwritten by cleaning; source disagreements are recorded in a separate exclusion ledger.

## Existing experiment evidence

`evidence/branch_audit.json` records `origin/execution@230f3dc`. Its aggregate captioning results match the previously inspected `feat/poc-b-captioning@dc8f3ce` results. The reorganisation is not another model run. Shoaib's returned decision ZIPs remain under `POC/benchmarks/results/2026-10-27/shoaib/`; they are not merged into the new cleaned-cohort performance totals. No new model leaderboard is claimed by this M2 revision.

The acquired primary archive SHA-256 and size independently match the publisher LFS object ID and size. The publisher API response is retained in `evidence/primary_release_api.json`.
