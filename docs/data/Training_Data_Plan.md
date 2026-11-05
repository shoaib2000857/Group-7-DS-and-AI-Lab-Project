# Training data by pipeline component

LEVIR-MCI is the shared anchor for the primary pipeline. Additional datasets can help individual components, but they do not all provide the same targets and should not be concatenated into one universal label set. These are training-data contracts and experiment choices, not claims that additional data has improved a model.

## Recommended sequence

1. Establish an MCI-only baseline for each applicable component on the fixed clean splits.
2. Pretrain or adapt one component with additional compatible data, then fine-tune it on the MCI task if appropriate.
3. Compare against the same MCI-only baseline without changing prompts, splits or evaluation populations during the comparison.
4. Report each dataset separately. If a dataset is used for training, do not subsequently call it an unseen-domain test.

## Component and source matrix

| Component | Anchor | Additional data and purpose | Constraint |
|---|---|---|---|
| Building change detection | MCI building target | LEVIR-CD, LEVIR-CD+, S2Looking for building-only pretraining; S2Looking adds acquisition/viewpoint variation | Convert MCI building class only. Group overlapping original images/tiles before fitting. Preserve S2Looking's two source masks and use their union for binary building change. |
| Road change detection | MCI road target | Add another road-change dataset only after its labels are verified | Building-only masks are not road-negative annotations. No road labels are invented from WHU, LEVIR-CD or S2Looking. |
| Broad change pretraining | Separate SYSU-CD task | SYSU-CD provides broader visible-change supervision | Its foreground includes more than buildings/roads. Pretraining followed by task-specific adaptation is distinct from pooling it as the MCI target. |
| Semantic change / region interpretation | SECOND semantic change classes | SECOND-CC provides related semantic targets and captions | Preserve two semantic maps, unchanged-region semantics and native ontology. These are not full-scene land-cover labels. SECOND/SECOND-CC are related sources, not independent holdout domains. |
| Change captioning | MCI's five references on ordered images | SECOND-CC provides additional caption/image variation | Separate source metrics; resolve label/caption conflicts and reversed-pair leakage first. Additional caption data is not independently human-rated claim verification. |
| VLM instruction tuning | MCI imagery via ChangeChat | Captioning, localisation, counting and dialogue instructions | Group by source image pair, not instruction ID. Select one training variant; train and train-loc overlap heavily. Keep every test task's parent images out of fitting. |
| Claim verification / final decisions | Predicted evidence and original images | Human-reviewed claim targets and controlled corruption tests | Existing caption references and automatic mask checks are not human factuality labels. Do not train a checker on its own claimed answers and call that independent verification. |
| Large-area environmental context | Separate HLS/Sentinel workflow | Region/date/cloud-quality selected observations | 30 m HLS is not individual-building ground truth. No arbitrary region or date is downloaded as a substitute for a specified use case. |

## Pipeline manifests

`configs/datasets.yaml` defines six task views. `src.data.preprocess` writes a train/val/test directory for each task under `data/processed/tasks/`:

- `building_detection`: binary building targets, including the explicit MCI building projection.
- `road_detection`: MCI road projection only.
- `general_change_pretraining`: SYSU's general binary change target.
- `semantic_change`: two semantic class targets plus a binary change target; the loader returns both heads explicitly.
- `captioning`: ordered images and separately accessed source captions.
- `instruction_tuning`: raw-conversation pointers linked to accepted parent image pairs; one selected training variant.

Task manifest counts reflect available, accepted records, not nominal dataset sizes. An empty training partition is not silently filled from validation/test. Missing sources remain explicit in the notebook coverage table.

`balanced_training_indices` supports reproducible source-balanced fitting batches and rejects validation/test records or an unavailable positively weighted source. It does not prescribe a universal mixture ratio. Begin with an MCI-only comparison, then test one added source/component at a time. Keep MCI fine-tuning and external pretraining phases distinguishable in run records.

## Leakage and evaluation rules

- Keep all five captions and every instruction/variant tied to its original image-pair partition.
- Treat LEVIR-CC and MCI as shared imagery. Related LEVIR building datasets can overlap at the scene/tile level.
- Exact full-frame and aligned 256-pixel tile hashes protect partitions across sources. dHash candidate screening is conservative, not geographic proof.
- Retain source test before validation before training for detected collisions. Quarantine inconsistent duplicate supervision rather than guessing a corrected label.
- Preserve native imagery and resize metadata. Cropping, flips and time reversal can invalidate spatial/directional captions even when masks transform correctly.
- Semantic-model input validity, caption similarity and independently judged factual correctness are different measurements.
- Quality inspection includes test pixels; no model fitting or checkpoint choice is performed in these notebooks. Do not describe every test image as unopened by the team.

## Source references

- [Change-Agent / LEVIR-MCI](https://github.com/Chen-Yang-Liu/Change-Agent)
- [LEVIR-CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)
- [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [S2Looking and LEVIR-CD+](https://github.com/S2Looking/Dataset)
- [SYSU-CD](https://github.com/liumency/SYSU-CD)
- [SECOND](https://captain-whu.github.io/SCD/)
- [SECOND-CC author-linked dataset](https://doi.org/10.5281/zenodo.16937571)
- [ChangeChat annotation release](https://huggingface.co/datasets/hlwu/changechat-105k)
- [HLS products](https://hls.gsfc.nasa.gov/data-products/)
