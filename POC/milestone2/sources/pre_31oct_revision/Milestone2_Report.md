# TerraWatch: Intelligent Satellite Image Change Detection and Analysis

**DS & AI Lab Project (BSDA4001) | Indian Institute of Technology Madras**

**Milestone 2: Dataset Preparation | Group 7 | October 2026**

<p align="center"><img src="assets/iitm_logo.png" alt="IIT Madras emblem" width="130"></p>

| Team member | Student email | GitHub |
|---|---|---|
| Priyanshu Agarwal | 23f2003488@ds.study.iitm.ac.in | 23f2003488 |
| Sarthak Saini | 23f2005347@ds.study.iitm.ac.in | Sarthak-Saini265 |
| Sujal Raj Pradhan | 23f2004759@ds.study.iitm.ac.in | SujalPradhan |
| Shoaib SSM | 24f2000857@ds.study.iitm.ac.in | shoaib2000857 |

## 1. Dataset Selection and Sources

TerraWatch aims to describe visible building and road changes between two satellite images and identify statements that lack supporting evidence. Milestone 2 prepares the images, labels, reference captions and split records needed for that pipeline. The work covers source identification, quality assessment, preprocessing, leakage checks and reproducible model inputs, as required by the course [1].

**Primary dataset.** LEVIR-MCI supplies paired RGB images, building/road change masks and five reference captions for each pair [2,3]. It supports detection and description on the same imagery, avoiding the need to join unrelated image collections. LEVIR-CC provides its caption annotations and is not an independent transfer dataset.

| Dataset and source | Ownership and format | Role and usage constraints |
|---|---|---|
| LEVIR-MCI / LEVIR-CC [2,3] | Liu et al.; Google Earth imagery, paired PNGs, change masks and caption JSON | Primary development and training source. Dataset-specific permission is not explicit on the inspected release pages; the code's academic-use statement is not treated as a data licence. No raw imagery is redistributed. |
| WHU-CDC [4] | Captioned derivative of WHU building-change imagery; RGB pairs, binary masks and caption JSON | Provisional transfer probe. The acquired copy is a third-party upload; caption provenance, release identity and usage terms require confirmation before final reporting. |
| SECOND-CC [5,6] | Karaca et al.; paired RGB images, semantic change labels and captions derived from SECOND | Supplementary transfer probe. The author-linked Zenodo record specifies CC BY 4.0; attribution and underlying source terms are retained. |

### 1.1 Primary dataset characteristics

LEVIR-MCI contains 10,077 image pairs and 50,385 captions. Images are 256 x 256 pixels; the source describes nominal 0.5 m imagery from Texas, USA [2,3]. Exact dates, coordinate reference systems and tile-level spatial metadata are not supplied consistently, so they remain unknown in the manifest.

| Official split | Image pairs | Changed pairs | Reference captions |
|---|---:|---:|---:|
| Training | 6,815 | 3,407 | 34,075 |
| Validation | 1,333 | 667 | 6,665 |
| Test | 1,929 | 964 | 9,645 |
| **Total** | **10,077** | **5,038** | **50,385** |

The primary mask encoding is **0: background; 128: road change; 255: building change**. Background denotes pixels not annotated as building/road change, rather than proof of no other visible change. Captions can describe vegetation and content outside these classes; an absent mask label cannot establish that such a claim is false.

### 1.2 Supplementary label alignment

WHU-CDC is treated as building-only supervision. For SECOND-CC, the prepared building mask uses building-category pixels in its semantic **change** annotations; the underlying SECOND convention marks unchanged areas separately [6]. This mapping must not be applied to arbitrary full-scene land-cover maps, where building presence does not imply change. Road and vegetation claims remain outside this automatic building-only check. Results from the three datasets are reported separately, without pooling their training labels.

<!-- PAGEBREAK -->

## 2. Preparation and Model-Input Alignment

Preparation preserves the original temporal order: T1 is the earlier image and T2 the later image. Archive records are joined by dataset, official split and filename; images and labels are checked before they enter a model. A shared record carries source identifiers, paths, dimensions, label ontology, split membership and SHA-256 hashes.

![Data preparation and evidence separation](assets/data_flow.png)

Figure 1. A shared pair record links model inputs to evaluation references. Reference masks and captions are kept outside prediction requests.

### 2.1 Image and label processing

- **Pair validation:** check that both images and the corresponding label exist, decode correctly and have compatible dimensions. Store hashes of acquired files so later runs can detect substitutions or corruption.
- **Geometry:** preserve release alignment and original coordinates. Any future crop, resize or registration step must record its transform and valid overlap; visual regions use the original T1 frame.
- **Model preprocessing:** use each checkpoint's documented image normalisation and input size. Keep unmodified source images for inspection. Resize categorical masks with nearest-neighbour interpolation.
- **Unknown metadata:** do not infer dates, geographic area or object heights from pixels. Nominal source resolution is distinguished from verified per-image ground sampling distance.

### 2.2 Connections between subtasks

| Subtask | Prepared input | Supervision or evaluation reference |
|---|---|---|
| Change detection | Ordered T1/T2 image tensors | Building/road change mask; explicit valid pixels |
| Change description | Both images; predicted regions when available | Five reference captions for text comparison |
| Visual inspection | Matched image regions and surrounding context | Original images and independently reviewed claims |
| Claim decisions | Parsed claim, predicted evidence and uncertainty; images where supported | Separate accept/reject/defer targets, with annotation provenance |

The captioning package and decision notebook currently use different pilot manifests and pair-ID formats. Before integrated comparisons, both must map to the same canonical dataset/split/filename key. A common source dataset alone does not guarantee aligned experimental samples.

<!-- PAGEBREAK -->

## 3. Dataset Splits and Leakage Controls

The captioning preparation preserves official split membership and uses seeded sampling within each split. The initial smoke set is contained in the development set. Training and checkpoint selection use disjoint subsets of the official training partition; final evaluation IDs are reserved separately.

| Prepared cohort | Official partition | Pairs | Purpose |
|---|---|---:|---|
| Development | Training | 50 | Input checks, prompt/parser development and checkpoint selection; first 10 form the captioning smoke set |
| Subset-training pilot | Training, excluding development | 1,313 | Bounded specialist-captioner training |
| Screening | Validation | 100 | Exploratory system comparison and error inspection |
| Reserved evaluation | Test | 100 | Final assessment after systems and rules are frozen; images remain unopened in the captioning preparation |
| Transfer probes | WHU-CDC and SECOND-CC validation | 100 each | Dataset-specific checks of transfer and annotation compatibility |

Selections use a fixed seed of 42 with split-specific seed strings and sorted source records. Source revision, selection rule and manifest hash are recorded. The primary training and development cohorts together contain 1,363 pairs, or 20% of the official training partition. Separate decision-role pilots must also be included in the team's consolidated usage ledger; they are not assumed to be the same subset.

### 3.1 Duplicate and overlap assessment

The preparation audit compares each image's ZIP-index CRC-32 and uncompressed size, pairing both dates. These are efficient **duplicate-candidate checks**, not cryptographic proof of identical image content. Suspected overlaps require file-hash confirmation before final exclusion lists are frozen.

| Finding | Consequence and handling |
|---|---|
| 16 redundant LEVIR pair records flagged by archive checks; five test/training candidate matches | Keep the candidate list. Confirm content identity, remove redundant training examples and prevent confirmed test counterparts from entering training. The five reported test candidates are outside the reserved 100-pair set. |
| Two LEVIR pairs have matching before/after checksum signatures | Flag for inspection rather than interpreting the signature alone as physical no-change truth. |
| SECOND-CC has reversed pairs and cross-split content matches | Treat temporal reversals and all copies as one pair group. Do not use its current partitions for a new training-versus-test claim without regrouping. |
| Parent scenes cannot be recovered reliably from LEVIR tile names | Spatial independence between partitions remains unproven. Report this limitation and group repeated records by pair at minimum. |

### 3.2 Evaluation exposure

The released RSCaMa checkpoint was trained on LEVIR-CC and selected using its validation partition [7]. A separately subset-trained comparator avoids inheriting that checkpoint's task-specific training, but general pretraining exposure remains possible. Public-data exposure of general vision-language models is not established.

Screening outputs informed parser revisions in the captioning work. Consequently, the current screen is exploratory and must not be described as an untouched validation experiment. Freeze parser rules, prompts, label mappings and decision thresholds before the reserved evaluation is opened. Evaluation-only use of supplementary datasets prevents new project training on them; it does not prove that pretrained models have never seen their imagery.

<!-- PAGEBREAK -->

## 4. Data Quality and Suitability

The preparation audit covers the release index and a 150-pair image/label cohort: 50 development pairs and 100 screening pairs. Archive-index checks assess record completeness; pixel statistics and mask/caption comparisons apply only to that inspected cohort.

![Label composition in the preparation audit](assets/data_profile.png)

Figure 2. Label composition reported by the 150-pair preparation audit. Pixel proportions describe this subset, not the full release.

| Quality dimension | Finding | Implication |
|---|---|---|
| Completeness | No missing A/B/label joins or duplicate caption records reported for LEVIR; every pair has five non-empty captions | Inputs can be joined consistently; decoding and hash checks remain part of ingestion |
| Label balance | 65 masks show no labelled change; 28 building only, 11 road only and 46 both | Report unchanged behaviour and class-specific results; background dominates pixel counts |
| Small changes | 7 of 85 masks with change have under 1% changed pixels | Preserve spatial detail and inspect small-region failures |
| Annotation inconsistency | Mask content and caption change flags disagree in 8 of 150 pairs | Preserve these records with conflict flags; adjudicate rather than silently changing labels |
| Caption repetition | The 5,039 unchanged pairs reuse a fixed five-sentence template | Word-overlap scores can reward generic no-change responses |
| Imaging variation | Registration and illumination differences are visible in reviewed examples | Separate real change from imaging effects; systematic robustness measurement remains pending |

The label imbalance supports class-aware sampling and loss comparisons during detector training; no fixed loss recipe is claimed as validated at this milestone. Masks also have narrower semantic coverage than captions: vegetation, object identity, change direction and exact instance counts require additional evidence.

The captioning screen provides an ingestion check, not an independent factuality benchmark. A constant no-change caption obtains CIDEr-D 0.94 on the 100-pair screen while missing every changed pair in the conflict-free subset. The automatic evaluator uses reference masks and captions, whereas the deployed checker must use predicted evidence and original images. Its 50-pixel class-presence threshold can suppress genuine small changes; its support score does not jointly certify object count, location and direction. Audit-level nonzero-mask conflicts and thresholded evaluation conflicts must be reported separately, with each metric's exact denominator.

**Privacy and bias.** Public availability does not eliminate privacy or usage concerns. No personal identifiers are required as model features; do not add owner identities, precise sensitive-site details or inferred legal status to training records or outputs. Texas-focused imagery, repeated no-change captions and limited mask classes constrain representativeness. Supplementary probes broaden coverage but do not establish generalisation across sensors, seasons or regions.

<!-- PAGEBREAK -->

## 5. Model-Specific Readiness and Reproducibility

### 5.1 Input formats and annotation

| Component | Prepared representation and limits |
|---|---|
| Specialist captioner | RSCaMa uses CLIP image processing at 224 pixels and a 42-token caption limit under its own tokenizer; vocabulary construction is recorded [7] |
| General vision-language captioners | Ordered image pair plus a fixed instruction; generation capped at 96 new tokens. The SmolVLM2 screen uses one 384-pixel view per image, with image splitting disabled and this configuration recorded |
| Claim parser and checker | Structured object, action, location and quantity fields, retaining source text and uncertainty; unavailable fields remain null |
| Human reference audit | Two independent raters assess supported, contradicted, unsupported and unverifiable claims from original images; disagreements and adjudication are retained |

The captioning instruction asks for at most two sentences about visible building or road changes, using image-relative locations. It permits uncertainty and excludes inferred legality, ownership and cause. Exact counts are requested only when individual objects are distinguishable. Future instruction tuning will pair this input with reference responses and check length after the selected tokenizer; word counts are not interchangeable with subword-token limits.

The annotation sheet, viewer and separate answer key are prepared. Human factuality labels remain pending. A model-assisted spot check is a development aid, not a replacement for independent raters. Connected mask components are not accepted as ground-truth building-instance counts.

### 5.2 Synthetic data and augmentation

No synthetic satellite imagery is used in the prepared captioning cohort. The decision notebook includes separately labelled constructed claim/evidence fixtures for software and reasoning checks; these are excluded from satellite-performance claims. Detector augmentation is planned: paired flips and right-angle rotations, with matching label transforms. One-sided brightness, blur and translation perturbations will form a separate robustness set, with transformed labels and valid overlap where required. Reversed time pairs must remain grouped with their originals and have direction-sensitive captions checked.

### 5.3 Reproducibility and handover

The versioned preparation package records source/archive revisions, split-specific seeds, selected identifiers, file hashes, label mappings and processing settings [8]. It separates raw inputs, evaluation references and model outputs. Model, tokenizer and processor revisions, environment versions and configuration-specific image limits are retained with run records; weights and raw imagery are not committed to the repository.

Preparation has supported specialist/general caption generation, a subset-training pilot and decision-model smoke/screen exports. Training readiness is demonstrated for that captioning configuration, not yet for the complete detection/inspection/decision pipeline. A final cleaned shared-dataset size and post-exclusion class distribution cannot be certified until the pending duplicate checks, conflict policy and manifest consolidation are complete.

Before Milestone 3, the team will consolidate the cross-role manifest, confirm duplicate candidates and release permissions, finalise detector/crop adapters, and pilot the two-rater claim audit. Sarthak leads detection ingestion, Sujal caption/reference preparation, Priyanshu visual-region handling and Shoaib decision contracts and integration. The resulting handover must retain the original images, explicit uncertainty and independently stored references throughout the proposed pipeline.

<!-- PAGEBREAK -->

## References

1. IITM DS & AI Lab. *Project Milestones: Milestone 2 - Dataset Preparation*. https://bsc-iitm.github.io/dsai-lab-project/milestones.html#milestone-2
2. Liu, C. et al. (2024). *Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis*. IEEE TGRS. Author code and LEVIR-MCI release: https://github.com/Chen-Yang-Liu/Change-Agent ; https://huggingface.co/datasets/lcybuaa/LEVIR-MCI
3. Liu, C. et al. (2022). *Remote Sensing Image Change Captioning With Dual-Branch Transformers*. LEVIR-CC author dataset: https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset
4. Ji, S., Wei, S. & Lu, M. (2019). *Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set*. IEEE TGRS. WHU source imagery: https://gpcv.whu.edu.cn/data/building_dataset.html ; provisional captioned copy inspected: https://huggingface.co/datasets/hygge10111/RS-CDC
5. Karaca, A. C. et al. (2025). *Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework*. IEEE JSTARS. https://doi.org/10.1109/JSTARS.2025.3600613 ; author-linked release: https://doi.org/10.5281/zenodo.16937571
6. Yang, K. et al. (2022). *Asymmetric Siamese Networks for Semantic Change Detection*. IEEE TGRS. SECOND project and label conventions: https://captain-whu.github.io/SCD/
7. Liu, C. et al. (2024). *RSCaMa: Remote Sensing Image Change Captioning with State Space Model*. IEEE GRSL. https://github.com/Chen-Yang-Liu/RSCaMa
8. TerraWatch, Group 7 (2026). *Versioned dataset preparation, captioning scripts and aggregate screening outputs*. Source snapshot: https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/tree/dc8f3cee62e44ab89dfa0f8f04a6299fc05d258d/POC/b-captioning

Source pages and supplied preparation records reviewed 29 October 2026.

## Appendix A. Prepared Record Schema

Each sample contains two RGB inputs, not a fixed-length table of engineered features. For LEVIR-MCI, the pair has shape 2 x 256 x 256 x 3 before model-specific tensor conversion; the pixel target has shape 256 x 256. The same categorical target encoding is preserved during resizing.

| Field group | Representation and meaning |
|---|---|
| Identity and split | Dataset/revision, official partition, pair ID and experimental role |
| Inputs | T1/T2 file paths, dimensions and SHA-256 hashes |
| Detection target | Mask path/hash and explicit class-value mapping |
| Text references | Five caption strings/IDs stored outside inference requests |
| Provenance and quality | Selection seed/rule, transforms, conflict/exclusion flags; unavailable dates, CRS and per-tile resolution remain null |

Illustrative schema example: a training-partition record links two 256 x 256 RGB PNGs, one categorical mask and five caption references through a single pair key. This is a format example, not an additional dataset record. A training handover must include the frozen manifest, processed files, exclusion ledger and exact loader/processor configuration; scripts or aggregate metrics alone are insufficient.

## Team Review

Previous report version reviewed and verified: **PA · SS · SRP · SSSM**. Confirmation of the technical revision dated 30 October 2026 is pending.
