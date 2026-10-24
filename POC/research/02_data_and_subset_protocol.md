# Data choice and bounded subset protocol

Recommendations dated 24 October 2026. Dataset pages were inspected; archives and labels have not been downloaded or audited.

## Dataset decisions

| Dataset | Verified source information / limitation | Proposed use |
|---|---|---|
| [LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI) | Extends LEVIR-CC with change masks and captions; author code uses A/B/label folders within train/val/test. HF viewer currently reports an error and missing card metadata; do not infer official splits from its single preview split | Primary candidate, conditional on archive/label/access audit |
| [LEVIR-CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) | 10,077 image pairs and five descriptions per pair; shares images with MCI | Caption references for the same manifest, not an independent transfer dataset |
| [LEVIR-CD](https://github.com/justchenhao/LEVIR) | 637 1024×1024 pairs, nominal 0.5 m resolution, binary building change. Author source also points to supplementary geographic metadata | Detection-only fallback; caption audit would need separately annotated examples |
| [S2Looking / LEVIR-CD+](https://github.com/S2Looking/Dataset) | Author repository supplies download links; payload access and terms untested here | Optional later building-domain transfer test, after overlap/label audit |
| [SYSU-CD](https://github.com/liumency/SYSU-CD) | 20,000 256×256 aerial pairs from Hong Kong at 0.5 m, with several change categories | Later shift test. Its broader change ontology makes raw transfer scores misleading without reconciliation |
| [SECOND](https://captain-whu.github.io/SCD/) | Semantic change labels support land-cover transition questions; public labeled availability differs from total collection in existing materials | Optional semantic extension; verify actual available partitions before planning training |
| [ChangeChat-105k](https://huggingface.co/datasets/hlwu/changechat-105k) | Later release exists; original ChangeChat paper/repository describes 87,195 instructions | Defer fine-tuning; pin release and trace every instruction to its image pair if used |
| [NASA HLS](https://hls.gsfc.nasa.gov/data-products/) | Harmonized observations at 30 m, with analysis-ready products | Separate future large-area monitoring track; unsuitable evidence for validating individual-building detection |
| [RSCC](https://papers.neurips.cc/paper_files/paper/2025/hash/62867024377cac4233195949b9db0ebd-Abstract-Datasets_and_Benchmarks_Track.html) | Paired disaster-caption benchmark | Future disaster application; defer during urban POC |

Choose one aligned primary dataset rather than immediately training four models on four incompatible datasets. A semantic model trained on SECOND does not supply ground truth for LEVIR captions. A binary mask identifies changed pixels within its label definition; it does not establish class, construction direction, or cause.

## Define the 20–30% limit before using data

**Working interpretation of the user's limit:** the POC may use a default 20%, with an absolute ceiling of 30%, of each selected dataset's unique image pairs across all POC activities. A pair seen during EDA, manual annotation, prompt development, training, validation or evaluation counts as used. Multiple captions, crops and augmentations do not create additional independent pairs. Account for shared images across dataset variants in a second global union register.

Do not increase the selected set every run. Freeze a deterministic manifest and reuse it. A small subset must remain small cumulatively.

Let `N_s` be the verified number of pairs in official split `s`, and `f=0.20` initially. Select up to `floor(f*N_s)` pairs from each split; enforce `sum(selected_s) <= floor(f*N_total)`. Scene grouping can reduce counts further. Increase `f` to at most 0.30 only if the initial sample lacks necessary coverage, documenting the reason before examining additional evaluation outputs.

For the reported LEVIR-CC/MCI family total of 10,077 pairs, 20% permits at most 2,015 unique pairs and 30% at most 3,023. These are arithmetic ceilings, not audited manifest counts. Allocate within the actual official partitions after loading their metadata; do not invent split sizes.

Metadata-only inventory of identifiers/hashes can cover the archive to audit leakage without using all images for modeling. Prefer selective downloads; if packaging forces a full archive download, record bytes acquired separately from pairs actually inspected/used. Do not silently treat full-data EDA as a subset experiment.

## Leakage and partition protocol

1. Preserve original official partition membership. Select a bounded subset within each partition, rather than randomly re-splitting all crops.
2. Recover parent-scene/tile identifiers where available. Inspect whether official splits share parent imagery. Check exact image hashes, pair hashes, and near-duplicate/geographic adjacency where possible; exact hashes alone cannot rule out adjacent crop leakage.
3. Keep all descendants, captions and derived instructions associated with the same pair. Never split five captions across partitions.
4. If official partitions have parent overlap, retain an explicitly labeled official-protocol comparison and construct a separate spatially grouped diagnostic if feasible. Do not silently rewrite the benchmark and still claim published comparability.
5. Use the training subset for fitting, the validation subset for hyperparameters/prompts/checker thresholds, and a frozen POC evaluation subset only after decisions are fixed. Repeated inspection turns evaluation data into development data; log it and use a fresh reserved subset within the cap if available.
6. Reserve unused official test examples for the later main project. Publish the POC-used ID list internally so later experiments can exclude contaminated examples or declare reuse.
7. Pretrained checkpoints may already have seen the official training split or a larger LEVIR-derived corpus. Record training-data provenance. A 20% inference pilot with a full-data-trained checkpoint is **not** a 20%-data training result.

Start with uniform seeded sampling within splits/scene groups. Report changed/unchanged balance and class/size coverage. If deliberately stratifying on labels to cover rare failures, record the selection rule and label results as stratified; use a separate representative sample for population-level rates.

## Audit required before the first model

- Record upstream URL, version/commit, retrieval date, archive hash, rights text and redistribution conditions. Distinguish code, imagery, annotations, and checkpoint terms. A repository license does not automatically license all assets; Change-Agent's README itself combines an MIT reference with academic-use wording.
- Verify a complete T1/T2/mask/caption join by ID; reject missing, duplicate, corrupted and mismatched items. Preserve originals.
- Inspect mask values/palette and ignore labels. Determine whether building/road labels represent change class, direction, instances, or semantic state. Never guess a numeric class map from rendered colors.
- Verify image size, channel order, dynamic range and ordering of T1/T2. Check 30–50 selected examples, including unchanged pairs, small changes and touching structures.
- Check whether captions omit visible events or mention changes outside labeled mask classes. Reference captions are not exhaustive truth.
- Audit registration, cloud/shadow/blur and invalid regions. Keep a valid-pixel mask. Do not automatically apply registration that can erase genuine structural changes.
- Preserve metadata uncertainty: dates, coordinates, GSD, CRS and sensor can be unknown. Nominal dataset resolution is not sufficient to invent exact per-image georeferencing.

Proposed manifest columns are in `templates/subset_manifest.csv`. Store relative local paths and asset hashes. Record why each exclusion occurred and how many pairs remain.

## Preprocessing proposal

Use RGB tensors with shape `[B,3,H,W]` for each time point. Apply identical spatial transformations to both images and their labels; nearest-neighbor interpolation for class masks. Keep paired augmentations and deliberately one-sided nuisance perturbations distinct. Fit any normalization statistics on training data only, or document the fixed normalization required by pretrained weights.

Do not use a change-heavy training sampler for validation/evaluation. Preserve the no-change distribution. Do not resize until model input requirements and resulting small-object loss are understood.

## Fallback decisions

- MCI access or label alignment fails: use LEVIR-CD for detection, then a clearly labeled small manual caption/claim audit within the cap. Report loss of direct caption-benchmark comparability.
- Checkpoint unavailable: begin with absolute differencing and templates; this validates plumbing, not learned semantic capability.
- Parent-scene mapping unavailable: retain official split, document residual spatial leakage uncertainty, and avoid claiming strict geographic independence.
- Rights unclear: keep only links/metadata in research and resolve terms before redistributing assets. Do not upload the supplied meeting records or private team copies.
