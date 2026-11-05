# Milestone 2 document revision log

This log applies to `Milestone2_Report.md` and its generated PDF. The source versions before the 31 October revision are preserved in `sources/pre_31oct_revision/`. No entry represents a new approval or a new human annotation exercise.

## 31 October 2026 - M2 review incorporation

| Review recommendation | Incorporated change | Evidence / report location |
|---|---|---|
| Present completed preparation rather than internal plans | Rewrote the report around executed findings; removed preparation to-do lists, future-scope material, role-swapping discussion and model-selection digressions | Report sections 1-13; internal work retained in `TEAM_GUIDE.md` |
| Expand EDA beyond 150 pairs and provide the actual analysis code | Executed a full 10,077-pair audit, preserving per-pair results, distributions, outliers, correlations and image-relative location statistics | `scripts/prepare_and_audit.py`, `scripts/eda_figures.py`, `EDA_Report.md`, `evidence/full_audit/` |
| Explain images, masks, colour labels and five captions clearly | Verified RGB storage of identical grayscale channels; recorded black/yellow/red mapping and class indices; added a real record and input/target shapes | Report sections 3, 5, 6; `DATA_DICTIONARY.md` |
| Confirm duplicates and prevent train/test leakage | Used file and decoded-pixel SHA-256, unordered pair groups, a cross-split dHash screen and a deterministic exclusion policy. Checked that retained splits share no exact frames | `exact_duplicate_groups.json`, candidate/ledger files, `independent_checks.json` |
| Resolve mask/caption conflicts and inconsistent pair identifiers | Quarantined conflicting joint supervision without changing source labels; wrote canonical dataset/split/filename records with shared references | Report sections 6-10; prepared manifests and exclusion ledger |
| Explain the threshold and evaluation meaning | Identified 50 pixels as the earlier evaluator's setting, measured its sensitivity and retained small source-mask regions. Removed the caption-model metric discussion from the M2 narrative | Report section 5.1; detailed EDA threshold table |
| Clarify what the captioner sees during training and inference | Distinguished reference masks from predicted masks and isolated references from image-only inference records | Report section 7; loader reference-separation verification |
| Establish usage dispositions and do not conflate licences | Captured author release statements; kept raw imagery local; separated SECOND-CC evaluation and excluded the unlicensed WHU caption mirror from governed training | Report sections 2, 4; `SOURCE_REGISTER.md` |
| Produce a training-ready handover with final counts | Built a 9,544-pair strict package, checked every retained record through the loader, and recorded final split/class distributions | `handover_validation.json`, report sections 9-13 |
| Organise source, processed, scripts and outputs | Documented `data/raw`, `data/prepared`, `data/transfer`, `scripts`, `evidence`, `assets` and `output` | `README.md` |
| Maintain individual contributions and names/dates approvals | Added a concise contribution table and a full four-person approval table to both PDF and Markdown. Unconfirmed dates/signatures are left empty | Final report page; actual dates must be supplied by the signatories |
| Preserve a professional, concise visual presentation | Kept original cover/logo/team details and formal typography; used executed EDA charts and vector preparation diagram; removed short spill pages | Seven-page PDF, rendered review in `qa/` |

## Supplementary preparation evidence

Completed the author-linked SECOND-CC 100-original-validation-pair probe: 400 source files, generated building masks and 500 captions, with CRC, decoding, dimensions and SHA-256 checks. It remains separate from primary training. Primary archive SHA-256 and size also match the publisher LFS record. No new model or human-audit result is claimed.

## Earlier document history

| Date | Revision | Repository evidence |
|---|---|---|
| 29 October 2026 | Prepared the original polished M2 package from the supplied draft and committed captioning evidence | `2661fd8` on the POC branch |
| 29-30 October 2026 | Published PDF/Markdown through PR #14; limited its scope to exactly those two report files | `1127fb8`, `bd5d315` |
| 30 October 2026 | Clarified mask coverage, evaluator thresholds, privacy/bias, readiness scope and schema | `77f06ff`; subsequently merged in `ceb1f4e` |

The 31 October revision is a new POC delivery. Historical GitHub approvals and WhatsApp confirmation of earlier versions are not represented as approvals of these changed findings. No review was dismissed or reset during this revision.

## POC publication

Published the 31 October report, reproducible EDA/preparation scripts, metadata evidence, source records and team guidance on the existing POC working branch. Raw archives, image data and render scratch files remain local. Main and PR/review state are unchanged.
