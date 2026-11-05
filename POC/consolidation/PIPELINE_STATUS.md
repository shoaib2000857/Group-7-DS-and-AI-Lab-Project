# Pipeline readiness - 5 November 2026

This is the internal implementation audit. It distinguishes executable code, supplied experiment results and architecture decisions.

| Module | Owner / provenance | Implemented and exercised | Evidence boundary |
|---|---|---|---|
| Data preparation | Shoaib; Sujal's 2 Nov fixes | Full MCI/SECOND-CC audit; source adapters, exclusions, task manifests, inference/reference separation and worker-safe loaders. Latest data tests pass. | Historical 14,470-pair cross-source run and 9,544-pair MCI handover predate Sujal's duplicate-policy changes. Do not relabel the old saved outputs as a fresh run of the changed code. |
| A: change detector | Shared architecture / Sarthak work package | Candidate registry and data targets exist. BIT is selected in the M3 architecture on suitability/compute grounds. | No trained detector checkpoint, measured segmentation benchmark or learned A-stage inference artifact was found at the fetched heads. `src/models` and `src/inference` contain scaffolding. |
| B: captioning | Sujal, `8328ba7` / `dc8f3ce` | Released RSCaMa, 20%-budget RSCaMa retrain, Qwen3-VL-4B, SmolVLM2, pixel template and constant control; aggregate LEVIR/SECOND/WHU results; rule claim parser; handoff exporter. | Raw caption/claim caches and trained weights are not included in committed aggregates. Released RSCaMa's training/checkpoint selection saw original train/val; parser revisions used screening examples. These are model-selection screens, not untouched test results. |
| C: crop/evidence witness | Priyanshu, `54b713a` | Padded crops and transforms, visibility checks, LAB mean shift, edge/orientation and spatial anomaly rules, semantic gates, typed evidence records; nine unit tests. | 50-case accuracy is synthetic; three real-image cases use two source pairs and manually authored claims/boxes. The script's B-output descriptions are hard-coded, not a live B-stage invocation. Confidence is heuristic. Quantity is not independently counted. |
| D: decisions | Shoaib | Portable notebook/runtime, frozen visual witness, accept/reject/defer controls, Qwen3 text/VL outputs, fixed operating gate, failures, diagnostics and ZIP exports. | Six ZIPs represent five session IDs and 40 distinct source pairs. Two screen ZIPs are exports of the same session. Proxy/generated-claim labels are not independent human factuality labels. d3 has no completed scored output in these exports; InternVL is partial/failed. |
| Human audit | Team protocol | Four states, two raters, pre-adjudication agreement/kappa and third-rater resolution are defined. | No independent completed rating file was located in the inspected branches. |
| Application / final narrative | Shared design and consolidation | `POC/pipeline/` now provides isolated detector/captioner/referee workers, claim-region routing, evidence inspection, confidence gates, JSON output and a BIT training entrypoint. Four CPU test-double tests exercise routing and failure handling. | `app/app.py` prints an entrypoint message. GPU model workers are compiled and source-checked; no complete learned detector-to-final-report execution was located. |

## Integration points that matter

- B's `rules-v3` emits `added/removed/modified/unchanged`, source spans and coarse location; it deliberately has no pixel coordinates. C needs actual image-space candidate boxes. Region matching and coordinate transforms must be explicit.
- RSCaMa's tested input is the two RGB images, processed by CLIP. It does not accept a mask channel. In the architecture, detector evidence grounds claim verification; mask-guided VLM captioning is a separate configuration.
- C accepts a caller-provided mask run ID but defaults to `detector_open_cd_bit`. This string is not evidence that BIT ran. A real producer ID must replace it in an integrated record.
- C leaves `decision=None`. A `visible_change` boolean is neither a verified exact count nor a complete four-state factual judgement. D's benchmark uses another witness (frozen Qwen3-VL); its measured scores cannot be transferred to the classical C witness without a matched comparison.
- C handles insufficient crop visibility with indeterminate evidence. D should defer when evidence or required claim fields are missing, while preserving the raw output and failure reason.
- B's existing `export_handoff.py` deletes an existing handoff directory before writing. It was **not executed** during consolidation. Use a versioned output destination before running it on valued caches.

## Checks reproduced locally

81 CPU tests pass: 24 root configuration/data checks, nine C checks, 30 caption parser/evaluator checks 14 decision-runtime checks and four orchestration checks. The Torch-dependent RSCaMa shim module is skipped. Collection-path setup errors from the first audit attempt are preserved in `evidence/check_attempts/`; the final runner fixes the module paths and keeps a separate directory for every run.

The suite validates contracts and logic. It does not certify model generalisation, calibrated confidence, end-to-end accuracy, legality inference or a human-validated reliability improvement.
