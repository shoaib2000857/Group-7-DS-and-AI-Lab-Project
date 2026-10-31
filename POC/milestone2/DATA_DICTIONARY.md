# Prepared dataset dictionary

## Unit and scope

One record is one ordered bi-temporal pair. The strict primary package has 9,544 records: 6,444 train, 1,242 validation and 1,858 test. Five captions are five text targets on a single pair, not five independent images or observations.

The underlying input has two dates and three RGB channels per date: 393,216 uint8 scalar values at native resolution, rather than a fixed table of engineered predictors. Detection has one 256 x 256 class target with three classes. No geographic area, height, acquisition year, ownership or legal-status target is inferred.

## Manifest fields

| Field | Type | Meaning |
|---|---|---|
| pair_id | string | Canonical `LEVIR-MCI/{official_split}/{filename_stem}` key |
| dataset | string | `LEVIR-MCI` |
| source_revision | string | Full immutable Hugging Face revision |
| official_split | enum | `train`, `val`, `test`; retained source membership |
| archive | relative path | Source archive relative to `data/prepared/` |
| t1_member / t2_member | string | Earlier/later image members within the archive |
| target_member | string | Source categorical mask member; absent from inference-only manifests |
| t1_sha256 / t2_sha256 / target_sha256 | hex string | SHA-256 of the acquired PNG file bytes |
| shape | three integers | Native image dimensions `[256,256,3]` |
| mask_encoding | mapping | Source value 0 background; 128 road change; 255 building change |
| mask_class | enum | `unchanged`, `building_only`, `road_only`, `both` based on any nonzero annotated class pixels |
| acquisition_t1 / acquisition_t2 | null | Exact per-tile acquisition dates are absent from the record |
| crs | null | No per-tile coordinate reference system in the acquired JSON/PNG records |
| per_tile_gsd_m | null | No verified per-tile ground sampling distance |
| nominal_source_gsd_m | number | 0.5 m from source description; not a per-tile measurement guarantee |

The grayscale source target is physically stored as RGB with three identical channels. The loader verifies equality and reduces it to a 2-D array. The supplied colour visualisation has black background, yellow road change and red building change; it is not a separate prediction target. During detector loading, values 0/128/255 become indices 0/1/2.

## Inputs and references

- `train.jsonl`, `val.jsonl`, `test.jsonl`: supervised records for the complete strict cohort.
- `*_inference.jsonl`: only pair ID, archive, two image members and their hashes. No masks, class flags or reference captions enter the image-only loader result.
- `references.json`: five source caption strings and source change flag, keyed by pair ID; local and opt-in.
- `poc_train.jsonl`: the 1,363-record selected training-partition pool, before the experiment development split.
- `poc_training.jsonl`: 1,313 actual model-fitting records.
- `poc_development.jsonl`: 50 disjoint training-partition records for development/checkpoint selection.
- `poc_screening.jsonl`: 100 clean validation records.
- `poc_evaluation.jsonl`: 100 clean test records. Their pixels were included in the dataset-quality audit, not model training or selection.

The small cohorts have new frozen IDs after cleaning. They must not be treated as the same population as earlier captioning or decision screens. The manifest/ledger hashes identify which cohort a result uses.

`PreparedPairs(..., include_references=False)` returns uint8 RGB images suitable for a VLM's own image processor. Its `detector_batch()` returns float32 inputs of shape `[B,2,3,256,256]` scaled to [0,1], with int64 class targets `[B,256,256]`. Checkpoint-specific normalisation and tokenisation are applied by the model adapter. The local verification exercises dataset loading, not a trained neural model.

## Quality fields and exclusions

`evidence/full_audit/pair_profile.csv` contains file/pixel hashes; modes/shapes; mask values and pixel counts; colour-map equivalence; brightness/contrast and RGB means; mean absolute temporal pixel difference; 8-connected component diagnostics; image-relative centroid and 3 x 3 changed-pixel grid; reference length and lexical flags.

Image location is in pixel coordinates. The grid has top/centre/bottom rows and left/centre/right columns. It is not a geographic coordinate system. Connected components are not object-instance counts. Lexical numeric flags include phrases such as “two scenes”; they are not verified building-count annotations.

| Exclusion reason | Meaning |
|---|---|
| exact_repeated_pair | Redundant decoded-pixel image pair; unordered temporal signatures detect reversed copies too |
| inconsistent_duplicate_targets | Repeated imagery has differing source masks; entire conflicting group quarantined |
| mask_caption_changeflag_conflict | Nonzero building/road mask disagrees with the source caption's coarse change flag |
| perceptual_cross_split_candidate | Conservative lower-priority removal for a nonexact detected cross-split similarity candidate; none additional occurred in this release at dHash distance <=4 on both dates |

The 533 excluded records are a union of reasons; overlapping counts must not be summed as unique exclusions. Removing inconsistent joint supervision is a cohort policy, not human correction of the source labels. The final strict cohort is distinct from the original published benchmark.

## Real example

`evidence/full_audit/sample_record.json` contains a retained `train_000010` input record and its separately labelled evaluation reference. It demonstrates actual filenames, hashes, null metadata and all five reference strings. The main report quotes one short source caption; this example is not synthetic data.

## SECOND-CC evaluation-only schema

`data/transfer/SECOND-CC/manifest.jsonl` contains 100 original validation keys, local RGB/semantic paths, per-file SHA-256, native dimensions, source change flag and the derived building-change mask path/hash. `references.json` stores five source captions per key. The target is binary building-change supervision only; background is not negative road supervision. The projection uses building-category pixels at either date in the supplied semantic change maps. No source augmented copies are selected.
