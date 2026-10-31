# Main data preparation: execution evidence

Executed locally on 31 October 2026. The saved notebook outputs and `run_summary.json` are the evidence for this run; they do not establish model performance.

| Source | Executed scope |
|---|---|
| LEVIR-MCI | All 10,077 discovered pairs; publisher SHA-256 verified |
| SECOND-CC | All 6,041 original pairs; supplied augmented copies excluded; publisher MD5 verified |
| LEVIR-CD+ | Eight convenience pairs from the supplied Drive; not a full-release estimate |
| S2Looking | Five convenience pairs from the supplied Drive; not a full-release estimate |
| SYSU-CD | Eight convenience pairs from the supplied Drive; not a full-release estimate |
| ChangeChat | Eight published JSON files; 193,042 raw rows include overlapping train variants and evaluation views |

Sources absent locally, the LEVIR-CC shared-image alias and query-dependent HLS are shown explicitly in notebook coverage. A mounted-Drive run can inspect the complete available folders using the same adapters.

## Prepared data

14,470 image pairs remain after recorded exclusions: 9,698 training, 1,772 validation and 3,000 test. Every retained pair was loaded with source/target hashes, matching dimensions and opt-in references checked. Image-only inference fields passed separation checks. These counts combine compatible task views, not a single universal target ontology. Task-specific counts and manifest hashes are in `run_summary.json`.

## Verification

- `python -m pytest tests/ -q --basetemp=outputs/data/tests-temp-03`: 16 passed. The isolated `data-pipeline` PR worktree also passed all 16 tests and notebook-generation/path checks.
- The two main notebooks executed top to bottom: 20 code cells, nine figures, zero execution errors. The optional five-cell/two-figure experiment notebook and its execution receipt are retained under `POC/data_quality/`.
- HTML exports opened in headless Edge: tables and embedded figures rendered; no document-wide horizontal overflow. Figure and page screenshots were inspected locally under ignored `outputs/notebook_previews/`.
- SECOND-CC Drive upload: 26 archive parts, source attribution and parts manifest. Remote file sizes sum to the original archive size; the remote manifest matches the local manifest. Each remote part was not independently downloaded again. The reader verifies the reconstructed publisher checksum during audit.

Raw data, detailed profiles, prepared targets and downloaded instruction files remain outside Git. The notebooks retain bounded illustrative outputs. No model training, externally validated robustness gain or new independent human factuality annotation is claimed.

Regenerate evidence with the commands in `notebooks/README.md`, followed by `python scripts/summarize_data_run.py`. When using a different source collection, retain its actual scope and approval/annotation status.
