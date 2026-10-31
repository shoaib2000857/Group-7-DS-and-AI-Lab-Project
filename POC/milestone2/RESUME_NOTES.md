# Historical pause checkpoint - superseded by current README

Resumed at Shoaib's request. The complete primary audit and handover now supersede the failed intermediate attempt described below; see `README.md` and `QUALITY_CHECKS.md`.

Historical record: paused at Shoaib's explicit request. No background analysis or dataset downloads are intended to remain running. No changes from this session were pushed, and no PR was created or updated.

## Requested outcome

- Revise the PDF and Markdown under `POC/` using the TA's Milestone 2 checklist and the latest meeting.
- Keep the approved formal cover/theme; concise, professional wording, clear technical explanations and useful figures.
- Present completed preparation and findings; remove future-scope/team-coordination prose from the submission. Complete applicable preparation rather than simply relabelling unfinished work.
- Comprehensive reproducible EDA, code, prepared manifests, leakage/conflict handling, data dictionary and training input verification.
- Document review revisions and individual contributions. Do not invent approval dates or human annotations.
- Shoaib confirmed that NEW independent human annotation will be done by the team later. Do not claim it has been completed. Use supplied dataset annotations for completed M2 preparation; put later team work in separate internal notes.
- Inspect newly pushed branches; do not merge other people's work without a reason or change their reviews.

## Review findings already read

Both `POC/DOCS/Standard Recording3.txt` and `Standard Recording3Summary.txt` are identical copies of the roughly 69-minute transcript. The complete transcript was read in condensed speaker turns. Make a separate actual summary without overwriting these originals.

Key TA points: distinguish images, categorical masks and five reference sentences; explain model training versus inference; do not call SHA-256 encryption; verify image sizes, grayscale/RGB label equivalence and caption agreement; resolve duplicates/leakage; unify pair IDs; explain the 50-pixel evaluator threshold without attributing it to the dataset; document licences; prepare evaluation references; make EDA code available with a clear folder layout. Location distribution was discussed at the recorded start; the missing opening was not reconstructed. The TA accepted Monday resubmission.

## Branch inspection

`git fetch origin` completed. PR #14 is already merged into main (`ceb1f4e`), so there is no open PR #14 to update.

New `origin/execution`, commit `230f3dc`, was inspected without merging. Its `POC/execution/` contains 31 files corresponding to the existing captioning package: 27 have identical text ignoring line endings; three Python files change directory references; the generated notebook differs accordingly. Aggregate result files are identical. Do not count this reorganisation as new experiments. Evidence inventory saved in `evidence/branch_audit.json`. Existing `origin/feat/poc-b-captioning` remains `dc8f3ce`.

## Data and scripts

- Pinned LEVIR-MCI archive fully downloaded: `data/raw/LEVIR-MCI-dataset.zip`, 2,771,943,190 bytes, source revision `dff09c709999b5b1f1f1806024811c65a6b14dfe`.
- SECOND download stopped; `data/raw/SECOND-CC-AUG.part` is only 12,582,912 bytes. Preserve it; it is not a valid ZIP. Old `build_ood.py second_cc` range-fetch process was also stopped. Do not leave/restart either process until task resumes.
- New scripts: `scripts/fetch_primary.py`, `scripts/prepare_and_audit.py`, `scripts/prepared_dataset.py`. They are work in progress, not verified deliverables.
- Audit decoded/profiled all 10,077 records but **FAILED** during aggregation. No valid final summary or training handover was generated. `evidence/full_audit/*` and `data/prepared/*` contain incomplete intermediate outputs; empty manifests MUST NOT be called prepared data.
- Failure: `IndexError` at quantile calculation because `matrix` is empty. Likely inspect actual source mask modes/shapes versus the script's assumptions first. The script currently requires 2-D `label`; verify rather than guess. Source `pair_profile.csv` contains diagnostics. The incorrectly empty exact/near-duplicate files and empty manifests should be regenerated only after this is corrected. Do not use their numbers in the report.
- Script policy as drafted: exact pair hash dedup with test > val > train priority; cross-split perceptual candidate exclusion (dHash <=4 on both dates, conservative, not proof); quarantine mask/caption flag conflicts and conflicting duplicate targets; keep raw source unchanged; model inputs separate from evaluation references. Validate these policies and results before reporting.
- Public aggregate figures/EDA report and the new main PDF/MD are not yet authored. Main report source and existing seven-page PDF remain the previous version.
- Local Python has numpy/scipy/Pillow/matplotlib/PyMuPDF/reportlab, but no torch. Framework-neutral loader returns NumPy detector batches and uint8 images for VLM processors; do not claim a torch training run.

## Sources checked

`evidence/source_snapshots/` stores current source pages/cards and Zenodo metadata.

- LEVIR-MCI pinned card has no dataset-specific licence. Change-Agent README has MIT plus academic-code wording; do not misrepresent that as an image licence. Original LEVIR-CD terms say academic-only, non-commercial and respect Google Earth terms; do not automatically transfer them to every derivative. Final governance can state a conservative research-only/no-raw-redistribution disposition without claiming a licence grant that is absent.
- SECOND-CC author-linked Zenodo record `16937571`: CC BY 4.0, `SECOND-CC-AUG.zip` 2,539,187,782 bytes. Primary metadata obtained via requests even when browser fetch failed.
- WHU captioned mirror has no card/licence; source revision `a8a0343de8b05586ada84cf423da746f7bedf508`. Separate historical exploratory results from a final governed handover; do not assert full permission based on the meeting's verbal assumption.

## Approvals and generation

GitHub PR #14 approvals: Priyanshu and Sarthak at 29 October UTC (30 October IST), both on `bd5d315`, before the later technical update. Shoaib previously confirmed all four reviewed the earlier report on WhatsApp, but no individual WhatsApp approval dates were supplied. Do not reuse these as approvals of the new report.

PDF skill read at `C:/Users/shoai/.codex/plugins/cache/openai-primary-runtime/pdf/26.1029.11837/skills/pdf/SKILL.md`. Its artifact-operation marker was run once successfully for edit/one PDF before authoring. Main PDF builder still strips everything after `## Team Review`, which must be reconsidered because TA requires names/dates in submitted documents. Render and inspect all final PDF pages before delivery. Keep support notes separate.

## Next useful steps after resume

1. Inspect actual archive modes/shapes and the recorded failures, correct the audit and run it to completion; check independent counts and hashes.
2. Verify final manifests and detector/VLM input loading, and make any supplementary-data scope/permission decision explicit.
3. Build reproducible aggregate EDA figures and an inspectable detailed EDA record from real outputs.
4. Rewrite the concise main PDF/MD, per-document revision log, contribution record and separate team action/defence notes. No invented human factuality labels or sign-offs.
5. Render/check outputs and explain completed results and any human confirmations needed outside the report. Keep all work in POC until instructed otherwise.
