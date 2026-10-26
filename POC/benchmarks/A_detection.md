# A. Sarthak Saini: change detection

Own paired-image loading, detection adapters, pixel metrics and the primary manifest; Priyanshu co-reviews geometry/quality. Deliver **nine candidate detector rows**, not nine full training projects.

## Candidate list

**P1** = first wave; **P2** = expand after shared inference works; **P3** = optional complexity. Memory tiers are scheduling estimates in the shared protocol, not measured requirements.

| ID | Model | Why it belongs | Priority / starting route |
|---|---|---|---|
| D01 | FC-EF | Simple early-fusion learned control | P1; Open-CD; tier S |
| D02 | FC-Siam-conc | Shared encoders with feature concatenation | P2; same Open-CD environment; S |
| D03 | FC-Siam-diff | Tests explicit feature differencing against concatenation | P2; Open-CD; S |
| D04 | SNUNet-CD C16 | Nested Siamese CNN comparison | P2; Open-CD; S |
| D05 | STANet-PAM | Spatial-temporal attention CNN comparison | P2; Open-CD; S |
| D06 | BIT-R18 | Compact transformer-context comparison | P1; Open-CD; S |
| D07 | ChangeFormer MiT-B0 | Hierarchical transformer detector | P1; Open-CD; S/M |
| D08 | Changer-Ex R18 | Feature-interaction architecture | P2; Open-CD; S/M; native 512 input recipe |
| D09 | ChangeMamba MambaBCD-Tiny | State-space family rather than another CNN variant | P3; author environment and full task checkpoint; M, Linux/CUDA build risk |

The [Open-CD author toolbox](https://github.com/likyoo/open-cd) and its [published weight repository](https://huggingface.co/likyoo/Open-CD_Model_Zoo) expose implementations/checkpoints for D01–D08. Use those matched recipes; an Open-CD ChangeFormer-B0 result is not interchangeable with every author ChangeFormer variant. [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba) provides task checkpoints separately from VMamba encoder pretraining; only the former is a trained change detector.

## Matched Open-CD config and checkpoint paths

Code paths below were found in the upstream tree; checkpoint paths were found in HF metadata on 26 October. Pin revisions from `source_metadata.json`, then verify loading. None of the files has been downloaded or executed here.

| ID | Code config under `likyoo/open-cd` | File under `likyoo/Open-CD_Model_Zoo` |
|---|---|---|
| D01 | `configs/fcsn/fc_ef_256x256_40k_levircd.py` | `LEVIR-CD/fc_ef_256x256_40k_levircd.pth` |
| D02 | `configs/fcsn/fc_siam_conc_256x256_40k_levircd.py` | `LEVIR-CD/fc_siam_conc_256x256_40k_levircd.pth` |
| D03 | `configs/fcsn/fc_siam_diff_256x256_40k_levircd.py` | `LEVIR-CD/fc_siam_diff_256x256_40k_levircd.pth` |
| D04 | `configs/snunet/snunet_c16_256x256_40k_levircd.py` | `LEVIR-CD/snunet_c16_256x256_40k_levircd.pth` |
| D05 | `configs/stanet/stanet_pam_256x256_40k_levircd.py` | `LEVIR-CD/stanet_pam_256x256_40k_levircd.pth` |
| D06 | `configs/bit/bit_r18_256x256_40k_levircd.py` | `LEVIR-CD/bit_r18_256x256_40k_levircd.pth` |
| D07 | `configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py` | `LEVIR-CD/changeformer_mit-b0_256x256_40k_levircd.pth` |
| D08 | `configs/changer/changer_ex_r18_512x512_40k_levircd.py` | `LEVIR-CD/ChangerEx_r18-512x512_40k_levircd.pth` |

D09: use the author's `changedetection` instructions, `MambaBCD_Tiny` config, and the chosen full SYSU/LEVIR-CD+/WHU-CD checkpoint linked in its model zoo. Record which source task trained it. It belongs in a transfer/provenance-aware comparison, not an assumed LEVIR-CD reproduction. Do not infer a full checkpoint from an encoder `.pth` filename.

## Exact tasks

1. Make the pair manifest and verify the mask ontology. Keep all image/caption variants tied to a parent ID; calculate hashes and known overlap. Export small representative thumbnails to the team only where rights allow.
2. Create one CPU diagnostic: absolute RGB difference plus an Otsu threshold, with minimum-region filtering fixed on validation. This is a method, not D10.
3. Prepare an isolated Open-CD environment. Inspect its pinned commit's compatibility constraints before selecting PyTorch/CUDA/MMCV/MMEngine/MMSeg versions. Save the working environment; do not install legacy requirements into the root repo or another member's VLM environment.
4. Adapt D01, D06, D07 first. Preserve upstream RGB/BGR convention, resizing and normalisation. Return maps in original image coordinates and preserve a valid-pixel mask.
5. Native-recipe scores and common-resolution diagnostic scores are separate tables. Resizing every model to the same arbitrary input can break a pretrained recipe; forcing equal shape alone is not fair comparison.
6. Screen D02–D05 and D08 if the environment works. Give D09 a time-boxed install attempt; record `blocked_environment` if compilation/checkpoint access fails.
7. Cache probabilities where available, final masks, class maps, sample IDs, transforms and timings. Return `probabilities_available=false` if a wrapper only exposes labels; do not invent confidence.

## Evaluation and handoff

On Track D (LEVIR-CD), report change-class F1/IoU/precision/recall and no-change false alarms, with image count and sampling uncertainty. On Track M (LEVIR-MCI), evaluate these building-only detectors against an audited **building-change projection**, not a union of roads and buildings. Mark omitted ontology classes explicitly. A mask can be passed to captioning as uncertain evidence even when it covers only buildings.

Report source-trained inference separately from subset-trained runs. LEVIR-family overlap may compromise independence; disclose checkpoint exposure and treat unresolved cases as diagnostics. Inspect tiny objects, touching roofs, false edges and missed changes. Provide `mask.png`, optional `probabilities.npy`, transforms and an events file with region counts, not building counts.

**Done:** at least three first-wave adapters with provenance, one shared evaluator, one successful unchanged/changed case, raw predictions and timings; then a nine-row status table where every model is `run`, `blocked` or `deferred` with a reason. Propose two finalists. A short two-detector learning pilot is optional after inference, not required to start the other work packages.
