# Milestone 2 - report and reproducible preparation

The current report is `Milestone2_Report.md`; its matching seven-page PDF is `output/pdf/TerraWatch_Milestone2_Report.pdf`. Both use the approved M1 cover and formal theme. The previous report is preserved under `sources/pre_31oct_revision/`.

## What is complete

- Full LEVIR-MCI audit: 10,077 pairs, all 40,308 image/label files, 50,385 reference captions.
- Strict joint-task handover: 9,544 pairs, with 6,444 train / 1,242 validation / 1,858 test.
- Content-hash duplicate checks, mask/caption conflict policy, exclusion ledger and canonical pair IDs.
- All retained records passed file-hash, decoding, target conversion and reference checks; inference records exclude evaluation references.
- Reproducible EDA figures, detailed EDA, dictionary, source register, review revision log and contributor record.

The source archive is not modified. Conflicting annotations are excluded from the strict cohort rather than rewritten. These checks establish dataset readiness, not trained-model performance or new human factuality labels.

## Folder layout

| Path | Contents |
|---|---|
| `data/raw/` | Local-only acquired source archive and supplementary source metadata |
| `data/prepared/` | Local-ready primary train/val/test records and separately stored references |
| `data/transfer/` | Local supplementary evaluation data, separate from primary training |
| `scripts/` | Retrieval, full audit/preparation, loading, evidence checks, cohort selection, figures |
| `evidence/full_audit/` | Source/cleaned statistics, profiles, exact groups, candidates, exclusions and verification |
| `evidence/source_snapshots/` | Captured author release metadata and source statements |
| `assets/` | Logo, vector preparation diagram and executed EDA figures |
| `output/pdf/` | Current PDF delivery |
| `sources/` | Supplied files and preserved earlier versions |
| `qa/` | Rendered pages and layout checks |

`data/` is ignored by Git. Metadata manifests under `evidence/full_audit/` are inspectable handover records; their archive paths resolve in the documented `data/prepared/` layout. They do not include image bytes.

## Reproduce

From the repository root, with Python, NumPy, Pillow, SciPy, requests, matplotlib and the report dependencies installed:

```powershell
python POC/milestone2/scripts/fetch_primary.py
python POC/milestone2/scripts/prepare_and_audit.py
python POC/milestone2/scripts/prepared_dataset.py
python POC/milestone2/scripts/verify_evidence.py
python POC/milestone2/scripts/build_poc_splits.py
python POC/milestone2/scripts/eda_figures.py
python POC/milestone2/build_m2_report.py
```

The pinned primary archive is about 2.77 GB. An existing completed archive is reused. The acquisition script retains failed downloads as `.part`; only a full archive is accepted. Full-release EDA includes test pixels for quality inspection and does not train a model.

For training, use `scripts/prepared_dataset.py` directly against `data/prepared/train.jsonl`; validation/test are separate. Refer to `DATA_DICTIONARY.md` for shapes, target values, processor expectations and smaller POC cohorts. The small cohorts are new post-cleaning selections; old benchmark scores must not be reassigned to them.

## Read alongside the report

- `EDA_Report.md`: detailed quantitative results and figures.
- `DATA_DICTIONARY.md`: actual schema and loading contract.
- `SOURCE_REGISTER.md`: dataset provenance and usage dispositions.
- `CHANGELOG.md`: document revisions, review recommendations and incorporation evidence.
- `TEAM_GUIDE.md`: meeting summary, technical explanations and four-person starting plan; internal, not submission prose.
- `QUALITY_CHECKS.md`: executed validation and visual review.

The report contains a names-and-dates approval table. Its date/signature cells are intentionally unfilled: approval of earlier versions does not establish approval of this revision. Record actual team confirmations before submission. No new human annotations or signatures are entered on anyone's behalf.

## Supplementary evaluation probe

The SECOND-CC probe is complete: 100 seeded original validation pairs, 400 acquired RGB/semantic PNGs, derived building-change masks and 500 captions. It remains separate from primary training. Reproduce with `scripts/capture_secondary_metadata.py` then `scripts/prepare_transfer_probe.py`. The source index and caption metadata are cached; individual files are CRC-checked and hashed. Zenodo rate limits are honoured and cached files reused. `evidence/second_probe_summary.json` records source split counts, the exact selection rule and narrow building supervision.
