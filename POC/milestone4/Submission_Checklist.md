# Milestone 4 submission and review checklist

Checked against `milestone-4-review` and the fetched remote branches. The latest teammate implementation is `origin/milestone-4-training-and-review` (`1f0fa29`). Use the [readiness audit](audit/2026-11-19/Milestone4_Readiness_Review.md) for evidence and the specific discrepancies in the draft. Mark a task complete when its output is available and traceable to the code or run that produced it.

## Colab review status

- **Training:** L4, 192 train / 64 validation, four epochs per arm. Selected 0.0003 at epoch 3. [Run configuration, histories and checkpoint hashes](evidence/colab_comparison/).
- **Learned pipeline:** two real pairs completed all workers and both final decisions deferred. [Portable records and verification receipt](evidence/colab_learned_demo/).
- **Review demonstration:** the app records an accepted reviewer override and a deferral in the scripted control walkthrough. A live review on learned evidence, or a genuine learned `accept` if the examiner requires an automatic acceptance, is the remaining demo item.
- **Submission administration:** confirm M4 sign-offs, grant examiner access to the training ZIP or selected checkpoint, and approve the repository branch used for review.

## Training and model selection

- [ ] Recover existing BIT and RSCaMa configurations, epoch histories, validation logs, selected weights and vocabulary from the machines or hosted sessions that ran them. Record artifact locations, hashes and access instructions.
- [x] Record the exact dataset revision, manifests, seeds and counts for every run. Keep the original 10,077-pair release, strict 9,544-pair cohort (6,444 train / 1,242 validation / 1,858 test) and historical subsets distinct. Record SECOND-CC and WHU-CDC as separate evaluation sources where used.
- [x] Use the correct pixel shares (91.55% background in the raw release, 91.24% in the cleaned cohort). Explain the 533 exclusions using the M2 ledger. Document the split boundaries, mask mapping and preprocessing. Reserve test scores for evaluation and disclose prior quality inspection of test images.
- [x] Record architecture settings, pretrained weights, trainable and frozen layers, input sizes, mask classes, caption tokenisation and parameter counts if measured. Explain which stages are trained and which use fixed rules or frozen weights.
- [x] Capture actual Python, PyTorch, CUDA, library and GPU versions and available memory for each run. Resolve the draft's 16 GB claim against the recorded 8 GB local run.
- [x] Make the settings being compared configurable and save the resolved configs. Run a focused comparison on common train and validation manifests with matched seed and budget. Vary a meaningful setting such as learning rate, loss balance or effective batch size. Record results and choose using validation scores.
- [x] Document actual batch size, accumulation, epochs, optimiser, learning rates, scheduler, loss, early stopping and regularisation. Include augmentation, clipping, AMP, warmup, dropout or label smoothing only where the run used them. Preserve spatial meaning in captions when applying transforms. Match reported equations to the code.
- [x] Export per-epoch training and validation loss, task metric, learning rate and freeze state where available. Plot real histories and explain any plateau, overfitting or underfitting within the observed epochs.
- [x] Correct the reported BIT pilot to two epochs, batch two, and 100 train / 25 validation / 100 evaluation pairs from the raw archive. Distinguish validation checkpoint selection from evaluation metrics and binary IoU from road/building mean IoU. Replace the A1/A2/A4 rows and the 50-epoch/82.5% trajectory with supported results.
- [ ] Record the selected checkpoint, epoch, validation criterion, competing configurations and rationale. Package weights or retrievable links, checksums, config, vocabulary/processors and loading commands for M5.

## Pipeline, review interface and demo

- [x] Set up the separate model worker environments and run preflight with actual source checkouts, weights and vocabulary. Execute the learned pipeline on real satellite pairs and retain the logs.
- [x] Capture T1/T2 hashes, predicted mask, generated caption, parsed claims, regions, crops, witness diagnostics, raw referee responses, gated decisions and final report. Keep reference annotations out of inference.
- [ ] Connect the viewer to actual pipeline `report.json` records using an importer or adapter. Reuse the coordinator's regions, crops and decisions and expose their provenance in the UI.
- [x] Display the original T1/T2 images, predicted mask, region overlay, matching crops, claim, referee outcome and available decision reasons. Label source annotations separately. Explain relevant quality and visibility flags.
- [x] Preserve the exact-count deferral safeguard. Record human corrections to counts or wording separately from the model's original claim. Brightness peaks and mask components provide region cues and require independent evidence for instance counts.
- [x] Let reviewers flag claims, record a decision and explain corrections. Save successive review events with claim/run IDs, reviewer, timestamp, previous value and correction. Keep original evidence and model decisions intact.
- [x] Validate review writes, check HTTP responses before showing success, and derive summaries from the stored records. Verify that flags, corrections and earlier review events survive reload and appear in the export.
- [x] Replace hard-coded model-attributed captions and referee outcomes with recorded execution. Label illustrative cases as authored and convert OpenCV BGR inputs to RGB wherever the witness expects RGB.
- [ ] Demonstrate at least one accepted and one rejected or deferred claim on real satellite imagery. Show an actual review action, reload the viewer and recover the audit record. Test missing artifacts, invalid pairs and unsupported counts.
- [x] Package reproducible demo outputs and validate the package against the walkthrough. Keep notebook outputs clear about whether they are fresh execution, CPU controls or historical benchmarks.

## Report, presentation and code walkthrough

- [x] Revise the M4 report to cover the TA's twelve sections: introduction, data, configuration, environment, methodology, hyperparameter experiments, optimisation, regularisation, training progress, selection, challenges and M5 readiness.
- [x] Tie each results table and curve to a specific run and command. Link model, dataset, config, checkpoint, code and demo evidence. Use measured claims and describe the referee score as a conditional label-token score.
- [ ] Add the review enhancement, its user value, and the demo with traceable outputs. Rebuild the report deliverables and check consistency across report, slides, code and outputs. Update revision and contribution records and collect actual M4 sign-offs.
- [x] Prepare a brief slide deck covering the architecture, training setup, comparison, learning curves, selected checkpoints and evidence-review demo.
- [x] Prepare a walkthrough that follows one input through validation and preprocessing, BIT, RSCaMa, claims, witness, referee, final JSON and human review. Show intermediate outputs at each step. Keep recorded outputs available for GPU-heavy steps.
- [x] Explain the model choices and alternatives, tensor/text/record contracts, actual training settings and validation policy, model integration, external sources and worker dependencies. Show the referee prompt, T1/T2 ordering, how evidence reaches it and how the output is parsed and gated.
- [x] Cover hardware, library profiles and memory constraints, design trade-offs, known failure cases and limits. Explain how run-linked evidence files are retrieved and why the current design uses those files for context.

## Examiner-facing GitHub repository

- [ ] Reconcile the teammate, pipeline and submission branches. Put the reviewed deliverables on the examiner-facing branch through the repository's review process. Preserve POC history and original work.
- [x] Rewrite the root README with objectives, architecture, selected models, current progress, repository map and a quick start that produces a visible result. Link data setup, train, inference, evaluation, review UI, report, slides and evidence. Update stale paths and claims.
- [ ] Organise code, configs, dependencies, notebooks, documentation and supporting artifacts. Provide pinned source revisions, working install/run commands, expected outputs, asset locations and hashes. Keep links accessible to examiners.
- [ ] Verify the documented route from a clean checkout, including tests, notebook outputs, demo replay and audit persistence. Check that the exact code and evidence shown at the review are committed and pushed and match the report.
