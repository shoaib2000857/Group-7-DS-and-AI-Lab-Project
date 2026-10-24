# TerraWatch: research and POC decision pack

Prepared 24 October 2026. **Status: research and proposed experiment design; no trained model, reproduced score, or completed experiment is claimed.**

## Recommended direction

Focus the first POC on **auditable descriptions of visible urban change in aligned, high-resolution image pairs**. Use a modest detector and one caption baseline. Investigate whether a checker that can reject uncertain claims improves factual reliability when the detector itself makes mistakes. Keep semantic expansion, long time series, and operational encroachment monitoring as later extensions.

The key distinction is between *agreement with a predicted mask* and *correctness against the images*. A system can satisfy the former while confidently repeating a false detection. The research question should test that failure directly.

**Novelty remains unestablished.** Prior work already includes detection/caption integration, evidence features, counting, localization, and factuality evaluation. A defensible outcome may be a useful adaptation or a careful empirical study. That is preferable to claiming an unverified new research mechanism.

## Read in this order

| File | Purpose |
|---|---|
| [00_context_and_requirements.md](00_context_and_requirements.md) | What the official materials and TA actually require; repository status |
| [01_literature_matrix.md](01_literature_matrix.md) | Comparative research matrix, closest mechanisms, and remaining uncertainties |
| [02_data_and_subset_protocol.md](02_data_and_subset_protocol.md) | Dataset choices, access limitations, leakage, and an explicit 20–30% cap |
| [03_proposed_system.md](03_proposed_system.md) | Concrete proposed pipeline, evidence checks, and implementation choices |
| [04_experiments_and_evaluation.md](04_experiments_and_evaluation.md) | Ablations, factual audit, metrics, and decision criteria |
| [05_execution_roadmap.md](05_execution_roadmap.md) | Sequenced POC tasks and later main-project roadmap |
| [06_m1_revision_notes.md](06_m1_revision_notes.md) | Suggested corrections to the submission, without editing it |
| [sources.md](sources.md) | Primary-source links, access date, reading depth, and unresolved checks |
| [templates/](templates/) | Empty experiment, source-access, subset, and annotation registers |

`source_extracts/` contains searchable extractions from the supplied PDFs/HTML, plus a first-page contact sheet. These are convenience copies, not replacements for the originals. PDF text order may be imperfect. The recording summary is especially noisy; use timestamped transcript context and written TA feedback instead.

## Decisions proposed now

1. Primary data candidate: LEVIR-MCI, with one shared image-pair manifest for masks and captions.
2. Default POC budget: 20% of unique pairs, optionally increased to a hard 30% after checking useful sample coverage. No full training.
3. First implementation: provenance audit → deterministic subset → one detector/caption baseline → mask-only checker → proposed image-backed checker → blinded audit.
4. Primary research contrast: image-backed checking versus mask-only checking, using the same candidate captions. Also measure content retained and changes omitted.
5. Treat exact building counts, added/removed direction, and legal status as unsupported unless the necessary evidence exists.
6. Keep all work local in `POC/`. Promotion into the main project is a later, explicit step.

## What has and has not been verified

The supplied documents were inventoried, PDF text extracted, representative PDF pages inspected, official live course pages retrieved, and primary papers/repositories reviewed at the depths recorded in `sources.md`. The repository currently supplies scaffolding rather than a working detector/caption pipeline.

Dataset archives, full label dictionaries, checkpoint compatibility, local GPU suitability, runtime, and reproducibility of published results **have not been verified**. No dataset or model weights were downloaded. No main-project file was edited, and no commit or push was made.
