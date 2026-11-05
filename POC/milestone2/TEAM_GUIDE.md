# Milestone 2 - team review and starting guide

Internal working notes. This document is separate from the submission report.

## What the meeting actually asked for

The TA asked for the M2 data preparation to be completed and explained, rather than writing that necessary work would be done before M3. He also asked for the EDA code, a clear directory structure, resolved duplicate/label decisions, dataset rights and a usable evaluation reference set. The missing start of the recording is not reconstructed. Both files named Recording3 and Recording3Summary contain the same transcript; this is the actual condensed action summary.

| Transcript point | Meaning | Current response |
|---|---|---|
| 00:00-00:05 | Check location distribution and distinguish in-domain results from generalisation | Full image-relative 3 x 3 change-density plot; transfer sources kept separate; no universal robustness claim |
| 00:09-00:20 | Demonstrate actual records; distinguish masks, sentences and tokens | Real paired record, verified colour map and five-reference schema |
| 00:20-00:24 | Confirm dimensions and clarify absent geographic metadata | Every primary image is checked; dates/CRS are null because absent, not guessed |
| 00:24-00:34 | Explain implemented preparation and training versus inference | Executed preparation scripts, target/reference separation, loader and clear input contract |
| 00:34-00:41 | Resolve duplicate/leakage candidates and inconsistent pair IDs | Full decoded-pixel hash audit, deterministic exclusions and canonical keys |
| 00:41-00:48 | Explain EDA conclusions and the 50-pixel setting | Clear class/pixel imbalance, actual small-change counts and threshold sensitivity |
| 00:49-00:50; 01:02-01:03 | Verify licences and usage statements | Source register and conservative dataset-specific dispositions |
| 00:50-00:58 | Evaluation references, mask/caption disagreement and human annotation | Existing reference package prepared; strict conflict policy. Shoaib explicitly deferred new independent human annotations to the team; no such labels are fabricated |
| 00:59-01:07 | Remove internal to-do prose, provide EDA code and sensible folders | Main report describes completed work; scripts/evidence/output split documented; team tasks stay here |

## What changed technically

1. The source was audited in full, not extrapolated from the earlier 150 records. The old 150-record result was reproduced exactly: 65 no-change masks, 28 building-only, 11 road-only, 46 both and eight mask/caption flag conflicts.
2. All 10,077 pairs decode and join. Mask PNGs use RGB storage with equal channels; semantic values are 0,128,255, and the colour visualisation is black/yellow/red. The transcript's approximate values/colours are not treated as specifications.
3. There are 16 repeated unordered image-pair groups and five train/test groups. Different masks on repeated imagery are also quarantined.
4. There are 511 coarse mask/caption change-flag disagreements. The strict joint-task package excludes incompatible supervision rather than guessing which source annotation is correct.
5. The union of exclusions is 533 records. It leaves 9,544 pairs: 6,444 train, 1,242 validation and 1,858 test. This is a curated cohort, not the untouched published benchmark. Any results must identify which cohort and which exclusion policy they use.
6. Every retained record passed loading and integrity checks. New canonical keys align the data contract; old per-role experiment samples have not magically become the same samples.

## Defend these distinctions clearly

- **SHA-256 is a hash, not encryption.** Byte hashes check downloaded files; decoded-pixel hashes can identify the same visual array despite different file encoding.
- **A mask is supervision, not a verified prediction.** At training, predicted masks are compared with supplied targets. At inference, the detector must produce evidence from the image pair; the answer mask/captions must not be fed into the checker.
- **Five captions are alternative references.** They are not five independent image pairs, and do not automatically agree on every count or detail. A check for standard no-change phrases is not a semantic factuality audit.
- **50 pixels is an evaluator choice.** It is not a physical minimum building size, not a universal Indian-condition rule and not stated by the dataset as a truth threshold. One-pixel versus 50-pixel class presence is now reported explicitly.
- **Background has a narrow label meaning.** It excludes annotated building/road change, not every possible vegetation or other visible change.
- **Components are not buildings.** One object may have fragmented components; several adjacent buildings may merge. Instance counting needs suitable references.
- **Brightness is descriptive.** A large luminance difference can exist without construction. No automatic adjustment is claimed to eliminate lighting or registration problems.
- **Curated data changes the benchmark.** Quote coverage and the exclusion policy, and keep original-source reporting distinct from clean-cohort reporting. Zero retained coarse conflicts does not certify all captions as factually correct.
- **Data readiness is not pipeline success.** No new model is trained in this audit. The dataset/loader check is complete; end-to-end reliability and independently human-rated factuality are different experiments.
- **Decision models do not guarantee truth.** A second language model can repeat or amplify a wrong premise. Its value has to be tested against independent reference facts and appropriate controls.

## Four-person starting plan

Everyone starts from `DATA_DICTIONARY.md`, the frozen cohort ledger and the same pair keys. The small cleaned cohorts contain 1,313 fitting pairs, 50 development pairs, 100 validation screening pairs and 100 test evaluation pairs. Do not fit on the development or screening set. The test pixels were already inspected for dataset quality, so do not describe them as unseen by all team processes.

| Owner | First concrete task | Handover |
|---|---|---|
| Sarthak | Connect detector experiments to the prepared loader; confirm target order background/road/building; run a small baseline on the frozen fitting/development cohorts | Predicted masks, pair IDs, preprocessing/checkpoint settings and per-class metrics; source masks stored separately |
| Sujal | Map existing caption adapters to these canonical IDs; preserve image order and the five-reference structure; separate pair-only and predicted-mask-guided captioning | Captions and parsed claims with exact cohort, checkpoint, prompt and tokenizer revisions |
| Priyanshu | Establish the crop/region contract using the original image coordinate frame; preserve context and distinguish components from instances | Region records linked to pair IDs and predicted masks, with transforms and uncertainty |
| Shoaib / SSSM | Join the outputs by canonical IDs, validate decision input isolation and run deterministic controls before learned decision models | Decision records, input/evidence audit and combined result export with exact populations |

Each adapter should load the same first four record IDs and show the same images before expensive runs. Pilot results belong to the new frozen cohort. Existing model scores remain attached to their original manifests; they are not recomputed or relabelled by copying files into another branch.

For caption training, keep all five references available. If the trainer expands a pair into five text examples, group and split at image-pair level first; average sequence losses in a documented way. Do not concatenate the five descriptions into a supposedly adjudicated ground truth.

## New independent human annotation

Shoaib confirmed this will be completed by the team separately. Establish reference facts from original images before showing model outputs, blind model identities, retain two independent judgements and record disagreements/adjudication. Existing source captions and coarse mask checks are not substitutes. The M2 report does not claim that this exercise has already happened.

## Report approval and submission

The PDF and Markdown both contain the four names, correct initials (SSSM for Shoaib) and date/signature fields. Actual approval dates have not been supplied for this changed revision. Fill them from the team's real confirmations; WhatsApp confirmation is acceptable as a source record if the member/date/version are recorded. Do not transfer earlier PR approvals onto new findings or invent dates.

PR #14 is already merged. This revision stays in POC. A new submission PR is created only when Shoaib explicitly asks; no existing review requests or approvals are reset.
