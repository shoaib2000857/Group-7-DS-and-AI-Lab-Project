# Execution roadmap

Prepared 24 October 2026. This research pass creates plans only. Implementation and experiments below are subsequent work, confined to `POC/` until explicitly promoted.

## POC sequence and checkpoints

| Step | Work | Deliverable | Exit criterion / fallback |
|---|---|---|---|
| 1. Research synthesis | Read close methods fully; resolve factuality/verification overlap | Updated matrix, mechanism comparison, provisional research statement | Can explain a specific contrast without claiming unsupported novelty |
| 2. Compute/access reconnaissance | Inventory GPU/VRAM/disk; inspect dataset release and rights; locate checkpoints | Access register and environment proposal | One usable data source and one feasible baseline, or documented fallback |
| 3. Subset and EDA | Build bounded, deterministic manifests; audit pairing, labels, duplicates and scenes | Dataset card, subset counts/hashes, visual examples and exclusion log | Cap respected, no known split conflict hidden, label ontology understood |
| 4. Baseline plumbing | Run 5–10 development pairs through detector, event extraction, baseline caption and JSON report | Saved overlays, raw outputs, provenance and timings | Valid unchanged case, changed case and uncertain case; no invented metadata |
| 5. Checker baseline | Implement mask-only rules and audited claim extraction | V0 outputs and error examples | Clearly distinguishes mask agreement from factual correctness |
| 6. Candidate checker | Add one feasible image-backed witness and abstention rules | V1 adapter and evidence-linked outcomes | Automated capability demonstrated; human-oracle diagnostics clearly separated |
| 7. Small audit | Freeze prompts/config, compare identical captions, run blinded annotation and wrong-evidence challenges | Paired metric table, retention/omission rates, disagreement examples | Results reflect both reliability and useful coverage; negative findings retained |
| 8. Optional learning pilot | Only if needed and feasible, short compact-detector runs | Curves, measured runtime/VRAM, fixed configs | Bounded subset/step budget respected; no full training |
| 9. POC decision | Summarize results, gaps and main-project cost | Go/simplify/pivot decision memo | A concrete, evidence-based architecture and evaluation plan ready for team review |

Do not build the dashboard before step 4 works. Do not wait for a large VLM to validate the data and simple baseline. When a dataset or model cannot run, capture the blocker and move to the documented fallback.

## Suggested near-term scheduling

- **October 24–25:** finish close-prior-art reading and prepare a concise M1 revision draft, especially the comparison matrix and qualified novelty statement. Check the TA's actual resubmission deadline.
- **October 25–29:** prioritize the M2 data audit, subset, provenance, labels, split integrity and preprocessing evidence. Avoid optional semantic datasets.
- **October 30–November 5:** complete a small end-to-end POC and checker feasibility demonstration to inform M3.

These are planning suggestions tied to course dates, not measured effort estimates. The review deadline may require M1 revision before the POC concludes. A report can state what is still proposed instead of waiting to manufacture positive results.

## Main-project roadmap after the POC

| Stage | Promote only after | Work to perform | Evidence for course review |
|---|---|---|---|
| Scope freeze / M1 revision | Close-prior-art comparison and team decision | Select primary axis: factual verification or detector robustness; qualify novelty | Short problem statement, analytical matrix, explicit gap and experimental design |
| Data / M2 | Verified sources and audited labels | Full-project data plan, clean manifests and reproducible preprocessing | Data card, split/overlap audit, paired examples, task-label mapping |
| Architecture / M3 | POC end-to-end compatibility | Stable tensor/JSON contracts, model adapters, training-ready losses | Real subset outputs, architecture diagram, justified alternatives |
| Training / M4 | Working loss/gradient pilot and compute estimate | Train chosen detector baselines; limited systematic loss/augmentation comparisons; caption tuning only if justified | Configs, logs, checkpoints, curves, training and validation observations |
| Evaluation / M5 | Frozen protocol and trained candidates | Held-out tests, primary checker ablation, blinded audit, limited robustness and optionally transfer | Paired effects, uncertainty, successful/failed examples and limitations |
| Deployment / M6 | Stable useful inference | Local demo or approved deployment; clean-install reproduction and report | Demo, source instructions, model/data provenance and final report |

The main project may later use more data. That does not relax the POC cap or make POC-inspected test samples untouched. Preserve manifests and distinguish POC-trained from full-project-trained models in all reports.

## Four-person ownership proposal

Assign names as a team; this is a role proposal, not an assignment of credit.

| Role | Primary artifacts | Cross-review |
|---|---|---|
| Data and provenance | Rights/access log, manifest, scene audit, EDA | Verify model input/label contracts |
| Detection and compute | Detector adapter, baseline, timing, bounded training pilot | Review nuisance tests and mask failures |
| Captioning and prior art | Baseline captioner, mechanism comparison, claim schema | Review annotation examples and checker scope |
| Verification and evaluation | Checker, blinded audit setup, metrics, integration report | Review split usage and reproducibility |

All members should understand the pipeline, present their own evidence and keep individual worklogs. Use `date, person, task, artifact, experiment ID, finding, blocker, next step`; avoid retrospective invented contributions.

## Information needed when implementation begins

Research can proceed without these answers; the next implementation phase should establish:

1. Which GPU environments and storage are reliably available to the team, and what compute budget is acceptable?
2. Where is the existing shared dataset copy, if it exists, and what exact release/labels does it contain?
3. Who can independently annotate the pilot, and how much review time is available?
4. Does the team prefer factual verification as its main experimental axis after seeing the close-prior-art comparison?

## Handoff checklist

- [ ] Close-method comparison complete enough to freeze a precise hypothesis.
- [ ] Data terms, schema, subset cap and scene grouping recorded.
- [ ] Checkpoint training provenance and official test exposure recorded.
- [ ] One end-to-end development case works before training expansion.
- [ ] Checker evaluated against image-based human references, not its own mask.
- [ ] Useful content, omissions and abstention measured alongside precision.
- [ ] POC findings and failures documented with reproducible artifacts.
- [ ] Main-project promotion reviewed separately; no automatic commit, push or submission.
