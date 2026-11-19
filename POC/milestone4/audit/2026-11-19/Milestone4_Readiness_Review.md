# Milestone 4 readiness review - 19 November 2026

This is an internal evidence audit and work plan, not the submission report. No teammate files or experiments were removed, no branch was merged into main, and the submitted reports were not rewritten during this review.

## Repository state

All 16 origin references were fetched. `main` remains at `9b595e5`; the main-project model modules/notebooks are on `model-pipeline` (`17a8d55`). The newest work is Priyanshu's `milestone-4-training-and-review` (`1f0fa29`), which includes the model-pipeline history. It adds the M4 Markdown/HTML/PDF, `src/app/` interface, static audit records, real satellite assets and the demonstration script. It adds no training-source changes, hyperparameter-run manifests or epoch histories. A local `milestone-4-review` branch contains this audit and preserves the fetched work unchanged.

## What is established

- M2 source audit: 10,077 pairs / 40,308 decoded image-label files; 9,544 strict retained pairs, split 6,444 / 1,242 / 1,858. Exclusions have ledgers; masks map 0/128/255 to 0/1/2. Real datasets and prepared manifests remain available locally.
- Raw-release background/building/road pixel percentages: 91.55 / 4.85 / 3.60. Cleaned-cohort percentages: 91.24 / 5.11 / 3.65. These populations must be labelled separately.
- M3 BIT: two epochs, 100 training / 25 validation / 100 evaluation pairs from the raw archive; frozen backbone, batch two, 50 optimizer steps per epoch. Best checkpoint epoch 1, validation all-class macro-IoU 47.1% versus epoch-2 44.6%. Evaluation binary F1 45.8%, binary IoU 29.7%, road/building mean IoU 26.8%; 31/45 unchanged-pair alarms. The original checkpoint and raw logs are identified in teammate-side records, not distributed here.
- M3 RSCaMa: released checkpoint plus a controlled 1,313-pair subset retrain, six-model local caption rerun, transfer screens and hash-backed aggregate evidence. The local benchmark report identifies a 13-epoch retrain with best epoch 4; its full training history is needed for fresh convergence plots.
- Main reusable code: independent model workers, orchestration, parsing, witness, training profiles, automatic caption/pixel evaluation, source acquisition and environment instructions. Both model notebooks have saved CPU outputs. They do not imply completed learned-model training on this laptop.
- Current main CPU tests: 43 passed on 19 November 2026. This suite tests the data/contracts/witness/metrics/coordinator, not the new browser app or GPU generalisation.
- The three app T1/T2 pairs (`train_000010`, `val_000005`, `val_000042`) match original LEVIR-MCI source pixels exactly. Image authenticity does not establish prediction authenticity.

## TA training requirements: coverage and work

| TA requirement | Current coverage | Submission work |
|---|---|---|
| 1. Selected models and training objectives | BIT, RSCaMa, deterministic witness and frozen Qwen already specified | Describe which modules learn and which stay fixed |
| 2. Training/validation datasets, splits, transforms | Governed M2 data and separate raw-archive M3 experiments | Bind every new run to exact manifests and counts; retain fixed validation/test separation; report only transforms actually used |
| 3. Architecture and pretrained/custom configuration | Main model builders and pinned source code | Store trainable/frozen parameters and exact component config with each run |
| 4. Hardware/software | Historical metadata exists | Capture each M4 execution environment. Current draft's 16-GB RTX/PyTorch 2.4 claims conflict with recorded local 8-GB RTX/PyTorch 2.9+cu128 |
| 5. Training methodology | Executable detector/captioner training code exists | Export actual epoch histories and commands; separate completed two-epoch pilot from full 50-epoch profile |
| 6. Hyperparameter experiments | M3 model comparison and one BIT pilot; no committed A1/A2/A4 run evidence | Run a small matched comparison, save configs/history/validation scores, then construct the results table from those outputs |
| 7. Optimisation | BIT AdamW/cosine; captioner Adam; patience/checkpoints | Make experiment parameters configurable and log them. Claim AMP/clipping/warmup only for runs that implement and enable them |
| 8. Regularisation | Detector horizontal paired flips, weighted focal/Dice, weight decay; caption modules use dropout | Describe implemented methods. Keep geometry-sensitive captions unchanged unless their directions are transformed |
| 9. Training progress/convergence | Historical checkpoint-selection summaries; training code writes train loss and validation selection metric | Export real train and validation loss curves plus task-metric curves; no invented 50-epoch trajectory |
| 10. Checkpoint selection | Pilot epoch-1 selection and historical captioner choices known | Produce best-checkpoint pointer, validation criterion, chosen epoch, hash and comparison to other actual runs. Reserve test performance for evaluation |
| 11. Challenges/resolutions | Imbalance, false alarms, lighting and environment isolation documented | Tie each resolution to measured examples; avoid claiming complete hallucination prevention |
| 12. Summary/M5 handover | Inference/evaluation interfaces available | Package selected weights or retrievable links, processors/vocabulary, manifests, environment, run receipts and exact reproduction commands |

The TA's optimizer/regularisation examples are alternatives: there is no requirement to implement every optimizer, every scheduler or cross-validation. A concise measured experiment set is sufficient; a fabricated large search is not.

## Mandatory interface enhancement

| Requirement | Existing app | Required integration / evidence |
|---|---|---|
| Original T1/T2 per claim | Present, genuine source pixels | Import paths and hashes from actual pipeline runs |
| Predicted mask | Mask viewer exists | Construction mask is pixel-identical to the source annotation; label it ground truth, and supply an actual detector output separately |
| Relevant region/crops | Present, hand-authored boxes and supplied crops | Reuse `candidate_region_ids` / `evidence.crop_paths` from coordinator records |
| Accept/reject/defer and reason | Present in static app JSON | Import raw and gated decisions with model revision, source kind and reasons from recorded execution |
| Flag for human review | Checkbox and client state present | Persist a review event against immutable claim/run IDs |
| Record corrections | Notes and decision override present | Preserve corrected claim/count separately from the generated claim; show original and corrected values |
| Audit trail | Current record snapshot, browser localStorage and whole-file POST | Preserve earlier decisions with append-only events; validate/recompute summaries; do not overwrite source evidence |
| Reuse existing architecture | UI uses a distinct hand-authored schema | Add a normalising importer for main `report.json` records (`claim`, `evidence`, `decision`, `candidate_region_ids`) |
| Real accepted plus rejected/deferred demonstration | Three real pairs, four curated claim records | Demonstrate actual recorded referee decisions or clearly identify human-authored demo decisions; retain raw model outputs for any model-attributed outcome |

The interface is a valuable foundation and should be retained. It is not yet evidence of a complete learned A/B/C/D run.

## Specific draft corrections

1. Section 2 calls 533 pairs corrupt/duplicate, while M2 decoded every source pair successfully and excludes annotation conflicts plus duplicate/inconsistent groups. Also, the draft attributes raw pixel shares and the 511 raw conflicts to the cleaned cohort. Use the M2 source/prepared tables and exclusion ledger.
2. The draft says all runs used the strict M2 package. The documented two-epoch detector and repeated caption screen use the original archive/subsets; keep those scopes distinct. M2 test imagery has already been inspected for quality, so do not call it unopened or newly sealed.
3. A1/A2/A4 numbers and the asserted 50-epoch loss trajectory / 82.5% convergence have no committed run histories. Do not include them as measured results until their logs are supplied or runs are executed.
4. A3 reuses the pilot's test change-class mean IoU (26.8%) as validation IoU and gives effective batch 32. The pilot used batch two without that accumulation. Otsu's 9.1% is binary IoU, not road/building class-mean IoU.
5. The main detector implements cosine decay with no linear warmup, plain focal/Dice training with no AMP or clipping, and horizontal flips. The captioner uses ordinary cross-entropy with no label smoothing and its own CLIP fine-tuning policy. The draft's shared five-epoch freeze for both models, rotations/jitter/70:30 sampler, clipping, label smoothing and classifier dropout 0.2 need implementation/run evidence or removal. The displayed squared-denominator Dice equation also differs from the code's linear denominator.
6. `scripts/demo_milestone4_end_to_end.py` calls the structural witness but hard-codes the caption, bounding box, all referee decisions/probabilities and several printed metrics. Its third case prints decisions/flags without running the witness. There is no model invocation or generated audit-file write in that script. Replaying it executes image processing, not Qwen/BIT/RSCaMa inference.
7. Its `cv2.imread` arrays are BGR while the witness contract expects RGB. Correct conversion is needed when rerunning witness evidence.
8. The construction app mask exactly equals the source label. The app's detector IoU 0.884 and model-attributed decisions have no corresponding run output in this commit. Ground-truth labels and model outputs must stay distinct.
9. The source caption says 'several'; manually assigning quantity 3 and finding three one-dimensional brightness peaks is not a validated building-instance estimator. Connected-component counts are not building counts. The existing coordinator correctly defers exact counts without independent instance evidence; preserve that safeguard.
10. The app audit snapshot reports zero human-reviewed claims but two records contain `reviewed: true`. Browser save overwrites the current review, the API overwrites both source JSON snapshots, and a success toast appears without checking HTTP status. Tests need to cover persistence and earlier review retention. User-entered corrections should not alter original evidence/model decisions.
11. Absolute guarantees ('cannot produce hallucinations', 'completely eliminating', 'calibrated referee') exceed the measured evidence. The Qwen score is a conditional label-token softmax, not calibrated factual probability. Use measured screen rates and the documented acceptance policy.
12. The M4 sign-off table has all names/dates prefilled. This audit does not infer new team approval from the earlier M3 approval; confirm M4 sign-offs separately. Keep revision/contribution logs in their separate administrative files.

## Practical work packages

1. **Sarthak - detector training:** supply/checkpoint the actual pilot, execute a compact matched hyperparameter comparison on a fixed training/validation contract, and return configs, epoch train/val losses, class IoUs, selected weights and environment receipt.
2. **Sujal - caption training:** supply the retrain's history/config/checkpoint and training-only vocabulary; provide validation BLEU-4 curves and describe checkpoint selection. Any extra fine-tuning comparison uses the same data/validation budget.
3. **Priyanshu - review UI:** retain the current viewer; add importer + immutable source records + review events/corrections. Replace sample model attribution with real output provenance, and preserve exact-count deferral until independent instance evidence exists.
4. **Shoaib - integration and evidence:** connect real worker outputs to the UI, execute an actual accepted and rejected/deferred claim demonstration, capture logs/receipts, and rebuild a short M4 report from verified experiment tables/curves.

These work packages can proceed independently until the final model/interface demonstration. The code-training profiles are specifications; successful CPU checks do not substitute for actual training histories.

## Evidence and reproduction

- Remote-branch file inventory: `branch_training_inventory.json`.
- Direct comparison of six app T1/T2 assets against the original archive: `asset_provenance.json`.
- Source M2 audit: `POC/milestone2/evidence/full_audit/summary.json` and `reports/Milestone2_Report.md`.
- Historical M3 evidence: `POC/milestone3/sarthak/Benchmark_Report.md` and `evaluation_proof.json`.
- Local demo replay copied inputs to `demo_replay/` to avoid overwriting teammate crops. It reproduces authored output; it establishes neither learned inference nor instance-count accuracy.
- Current main CPU command: `outputs/data/consolidation-env/Scripts/python.exe -m pytest -q --basetemp outputs/data/m4_review_tests_20261119_01` (43 passed). Use a fresh basetemp path for another run.
- Model preflight reports missing detector/captioner source checkouts/checkpoints/vocabulary on this host. Colab/teammate environments can supply these for the actual GPU runs.
