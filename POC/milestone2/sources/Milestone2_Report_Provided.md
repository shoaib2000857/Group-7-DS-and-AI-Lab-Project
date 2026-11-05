# TerraWatch: Intelligent Satellite Image Change Detection and Analysis

**DS & AI Lab Project (BSDA4001) | Indian Institute of Technology Madras**

**Milestone 2: Dataset Preparation | Group 7 | October 2026**

<p align="center"><img src="figures/iitm_logo.png" alt="IIT Madras emblem" width="130"></p>

| Team member | Student email | GitHub |
|---|---|---|
| Priyanshu Agarwal | 23f2003488@ds.study.iitm.ac.in | 23f2003488 |
| Sarthak Saini | 23f2005347@ds.study.iitm.ac.in | Sarthak-Saini265 |
| Sujal Raj Pradhan | 23f2004759@ds.study.iitm.ac.in | SujalPradhan |
| Shoaib SSM | 24f2000857@ds.study.iitm.ac.in | shoaib2000857 |

## Summary

TerraWatch converts two images of the same place into a localised description of visible building and road change, and flags claims the images do not support (Milestone 1). This milestone fixes the data that work rests on.

- **Primary dataset:** LEVIR-MCI (10,077 aligned image pairs, each with a building/road change mask and five captions), pinned to an exact revision. The official train/val/test split is preserved.
- **Bounded subset:** a seeded, hash-verified manifest uses at most 20% of the training split and 14.5% of the dataset overall. The 100 sealed test pairs are recorded but have never been downloaded.
- **Quality audit:** the joins are complete (no missing files, records or captions). A checksum audit of every image in the archive finds 16 duplicated pairs, 5 of them shared between test and train. In 8 of the 150 pairs we inspected, the mask and the captions disagree on whether anything changed. Half of all captions are a fixed five-sentence "no change" template.
- **Supplementary datasets:** WHU-CDC and SECOND-CC are used for evaluation only, to test transfer to new places. SECOND-CC's official splits share many images, so it is unsafe for training without regrouping.
- **Reproducibility:** every step is a script with pinned sources, seeds and SHA-256 hashes.

All figures below were computed from the data on 27–29 October 2026. Where a check has not been done, we say so.

## 1. Dataset and Data Sources

### 1.1 Sources, ownership and usage terms

| Dataset | Role in TerraWatch | Source (pinned) | Owner / authors | Format | Usage terms |
|---|---|---|---|---|---|
| **LEVIR-MCI** | Primary: detection masks, captions and claim checking on the same pairs | [huggingface.co/datasets/lcybuaa/LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI), file `LEVIR-MCI-dataset.zip` at revision `dff09c7` (2.77 GB) | Liu et al. [1]; captions from LEVIR-CC [2]; imagery from Google Earth | `images/{train,val,test}/{A,B,label,label_rgb}/*.png` and `LevirCCcaptions.json` | No licence on the Hugging Face card. The authors' Change-Agent repository states the code "can be used for academic purposes only" [1]. We treat the data as academic, non-commercial and not for redistribution. |
| **LEVIR-CC** | Caption source for the same 10,077 pairs; not an independent dataset | [github.com/Chen-Yang-Liu/LEVIR-CC-Dataset](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) | Liu et al. [2] | JSON, five sentences per pair | No licence file; same academic-use treatment |
| **WHU-CDC** | Supplementary: transfer evaluation only | [huggingface.co/datasets/hygge10111/RS-CDC](https://huggingface.co/datasets/hygge10111/RS-CDC), `whu_CDC_dataset.zip` (1.93 GB) | Captions: WHU-CDC authors; imagery and masks: WHU building change dataset [3] | Same layout as LEVIR; binary building-change mask | **Unresolved.** A third-party upload with no licence stated. Imagery is academic-use. We need the authors' official release before citing results as final. |
| **SECOND-CC** | Supplementary: transfer evaluation only | [doi.org/10.5281/zenodo.16937571](https://doi.org/10.5281/zenodo.16937571), `SECOND-CC-AUG.zip` (2.54 GB) | Karaca et al. [4]; imagery and semantic maps from SECOND [5] | `{split}/rgb/{A,B}`, `{split}/sem/{A,B}` and JSON captions | CC-BY-4.0 (Zenodo record) |

Images, masks and model weights stay on members' machines and in the team's private Drive. Git holds only code, manifests (identifiers and hashes) and aggregate results.

### 1.2 Size and characteristics

**LEVIR-MCI**: RGB PNG pairs of 256 × 256 pixels at a nominal 0.5 m, cropped from Google Earth scenes of Texas, USA, 5 to 14 years apart [1,2].

| Split | Pairs | Changed pairs | Captions | Mean words per caption (max) |
|---|---|---|---|---|
| Train | 6,815 | 3,407 (50.0%) | 34,075 | 7.9 (39) |
| Val | 1,333 | 667 (50.0%) | 6,665 | 7.9 (30) |
| Test | 1,929 | 964 (50.0%) | 9,645 | 7.8 (28) |
| **Total** | **10,077** | **5,038** | **50,385** | |

- **Masks:** a single-channel map with values 0 = background, 128 = road change and 255 = building change. A colour version (`label_rgb`) is supplied alongside. Every inspected label used only these values: 65 pairs had no change, 28 building only, 11 road only and 46 both.
- **Class balance:** in the 150 inspected pairs, 90.3% of pixels are background, 5.5% building change and 4.2% road change. In changed pairs a median of 13.5% of pixels changed, and 7 of 85 changed pairs had less than 1% changed. Detection training therefore faces strong background imbalance and many small objects.
- **Captions:** a vocabulary of 983 words, 495 of them appearing at least 5 times. 35.8% of sentences mention buildings or houses, 32.0% roads, and 12.9% trees or vegetation, which the masks do not mark.

**Supplementary datasets** (original pairs only; SECOND-CC's augmented copies are excluded):

| Dataset | Train / val / test pairs | Changed pairs | Mean words per caption | Change label |
|---|---|---|---|---|
| WHU-CDC | 5,947 / 743 / 744 | 33.0% | 6.2 | Binary building change |
| SECOND-CC | 4,219 / 595 / 1,227 | 71.8% | 10.4 | Land-cover maps at both dates (building, low vegetation, ground, tree, water, playground; white = unchanged) |

### 1.3 Quality assessment

| Check | Method | Result |
|---|---|---|
| Missing files | Joined every caption record with its A, B and label files in the archive index | 0 missing files; 0 duplicate records |
| Missing or empty captions | Count per pair | All 10,077 LEVIR pairs have 5 non-empty captions. In SECOND-CC, 22 train and 9 test pairs have 2–4 captions. |
| Duplicate images and pairs | CRC-32 and size of every image in the archive index; metadata only, no images downloaded (§2.3) | LEVIR: 16 pairs duplicated exactly (5 test ↔ train, 11 within train); 2 pairs whose before and after images are identical (`train_007575`, `val_001224`). WHU-CDC: none. SECOND-CC: see §2.3. |
| Mask vs caption consistency | Captions' change flag compared with the mask on the 150 inspected pairs | 8 of 150 (5.3%) disagree on whether anything changed. Example: a road "changed" because the two photos are slightly offset. These pairs are excluded from rate-based metrics and listed in the manifest. |
| Templated captions | Frequency of identical sentences | All 5,039 unchanged LEVIR pairs use the same five sentences ("there is no difference", …). WHU-CDC and SECOND-CC reuse the template. Word-overlap metrics therefore reward a system that always says "no change" (it scores 0.94 CIDEr-D on our screening pairs while missing every change). |
| Captions outside the mask's classes | Keyword counts | 12.9% of LEVIR sentences describe vegetation, which no mask can confirm. Claims about it are marked "unverifiable", not wrong. |
| Image noise | Visual review of selected pairs | Registration offsets and lighting differences between dates occur. They are not yet measured systematically; that is planned under the robustness objective (O2). |

### 1.4 Suitability for the problem

LEVIR-MCI is the only candidate that gives aligned change masks **and** human captions for the same pairs. All four TerraWatch roles (detection, description, visual inspection and claim decisions) can therefore be trained or evaluated on identical inputs, and every claim can be checked against a mask. Its limits are a single region, two change classes and templated no-change captions.

Our first captioning screen measured the cost of the single-region limit. The best captioner on LEVIR saw the share of its building claims that check out fall from 89% to 67% on WHU-CDC and to 50% on SECOND-CC, close to a simple pixel-difference rule. For that reason the two supplementary datasets are kept as independent transfer tests.

### 1.5 Supplementary datasets and augmentation

- **In use:** WHU-CDC (Christchurch, New Zealand; aerial) and SECOND-CC (several Chinese cities; mixed sources). We use 100 seeded validation pairs from each, for evaluation only. Neither dataset marks roads, so only building claims are checked against their masks. SECOND-CC's semantic maps are converted to a building-change mask: a pixel counts as building change if it is labelled building in either date's map.
- **Considered in Milestone 1, not yet acquired:** LEVIR-CD/CD+, S2Looking, SYSU-CD, ChangeChat-105k and OSCD. These remain options for detection transfer once their overlap with LEVIR-MCI has been checked (LEVIR-CD shares LEVIR's source imagery).
- **Augmentation (planned for detector training; none used yet):** identical geometric transforms for both images and the mask (flips, 90° rotations), with nearest-neighbour resampling for masks. One-sided perturbations (brightness, blur, small shifts) are kept separate and used only for robustness testing (§3).

## 2. Dataset Splits

### 2.1 How the splits are built

The official LEVIR-MCI splits are kept. Within each split, pairs are drawn uniformly at random, sorted by filename and seeded with a string per split, so anyone can regenerate the exact same IDs. Each list's SHA-256 hash is checked whenever it is used.

| POC role | Official split | Pairs | Selection rule | Use |
|---|---|---|---|---|
| Development | train | 50 (first 10 = smoke test) | `Random('42-train').sample(sorted train, 50)` | Setting up prompts, parser rules and thresholds; checkpoint selection for training |
| Training (retrain pilot) | train | 1,313 | `Random('42-train-finetune').sample(train − development, 1313)` | One bounded training pilot (§5.2) |
| Screening | val | 100 | `Random('42-val').sample(sorted val, 100)` | Comparing models; never used to tune anything |
| Sealed evaluation | test | 100 | `Random('42-test').sample(sorted test, 100)` | Final evaluation only. IDs recorded; images never downloaded. |
| Transfer (WHU-CDC, SECOND-CC) | val | 100 each | `Random('42-ood-<name>-val').sample(sorted val originals, 100)` | Evaluation only |

**Cap accounting.** The POC limit is 20% of each dataset's unique pairs by default, 30% at most.

| Dataset | Unique pairs used | Share |
|---|---|---|
| LEVIR-MCI train | 1,363 | 20.0% of 6,815 |
| LEVIR-MCI overall, incl. 100 val | 1,463 | 14.5% of 10,077 |
| LEVIR-MCI incl. sealed test IDs | 1,563 | 15.5% (ceiling: 2,015) |
| WHU-CDC | 100 | 1.3% of 7,434 |
| SECOND-CC | 100 | 1.7% of 6,041 |

Only selected pairs are downloaded, using HTTP range requests inside the archives. That came to 47 MB of LEVIR-MCI's 2.77 GB for the 150 inspected pairs (about 410 MB more for the training pilot's pairs), and about 30 MB each for the transfer sets.

### 2.2 Development vs evaluation discipline

Every choice (prompt wording, parser rules, checker thresholds, training checkpoint) is made on the 50 development pairs. The 100 screening pairs are scored once per system with fixed settings. The sealed test pairs will be opened once, after the final systems are frozen.

### 2.3 Leakage checks

**Exact duplicates across splits.** We compared the CRC-32 checksum and size of every image in each archive's index. The index lists all files without downloading them, so this covers the whole dataset without breaking the 20% cap.

- **LEVIR-MCI:** five test pairs are exact copies of training pairs:

  | Test pair | Duplicate of training pair |
  |---|---|
  | `test_000141` | `train_000017` |
  | `test_000142` | `train_000018` |
  | `test_000143` | `train_000019` |
  | `test_000144` | `train_000020` |
  | `test_000292` | `train_001389` |

  None is in our subset. They will be excluded from any final test score, and the exclusion reported. Another 11 duplicate pairs sit within the training split; they need de-duplication before detector training.
- **WHU-CDC:** no duplicate images within or across splits.
- **SECOND-CC:** files marked `_ters_` are the original pair with the before and after images swapped. Counting a pair as the same two images in either order, 277 test-split pairs also appear in another split (246 of them in train). Of 2,423 parent scenes, 753 appear in more than one split. **SECOND-CC's official splits are therefore unsafe for training-then-testing.** We use it for evaluation only. In our 100 validation pairs, 32 share their images with another split and 75 share a parent scene; this does not matter while nothing is trained on SECOND-CC.

**Shared scenes.** LEVIR-MCI's filenames do not identify the larger scene each tile was cut from, so pairs from adjacent tiles may sit in different splits. This leakage cannot be measured from the release and is reported as a limitation.

**What pretrained models have already seen**

| Model | Data seen in training | Effect on our evaluation |
|---|---|---|
| RSCaMa, released checkpoint [6] | LEVIR-CC train, checkpoint selected on val | Our screening pairs (val) are not unseen for it. We therefore retrained it on our training subset only; that copy has seen no val or test images. |
| Qwen3-VL-4B, SmolVLM2-2.2B | Not disclosed by publishers | Possible exposure to public LEVIR imagery; reported as unverified |
| CLIP ViT-B/32 (RSCaMa encoder) | Web image–text pairs | General pretraining; no task labels |

## 3. Synthetic Data

No synthetic data has been generated for this milestone. Two uses are planned and will be reported separately from real-data results:

1. **Robustness sets (O2):** copies of the screening pairs with one image altered (brightness, blur, a small shift). They test whether systems mistake imaging differences for change. Masks and captions are unchanged, because the physical scene is unchanged.
2. **Reversed pairs:** swapping before and after should turn "added" into "removed". SECOND-CC already contains such pairs (§2.3); if we use any, they stay in the same split as their original.

## 4. Multiple Datasets and Subtasks

Every subtask is keyed by the same `pair_id`, so one record carries all the data each role needs:

| Role (Milestone 1 §6) | Inputs from the data | Labels / references |
|---|---|---|
| A: change detection | Before and after RGB images | LEVIR-MCI mask (building, road) |
| B: change description | Before and after images (B1 adds predicted change regions from A) | Five LEVIR-CC captions (reference-based metrics only) |
| C: visual evidence inspection | Image crops at each claim's region | Mask, plus human audit labels |
| D: claim decisions | Claims from B, evidence from A and C | Accept / reject / defer labels derived from the mask and the human audit |

Captions are split into atomic claims (object, action, location, quantity) by a rule-based parser into a shared format, so detection, inspection and decision outputs attach to individual claims. Each transfer dataset is mapped onto LEVIR's label encoding: building change = 255, with no road class, so road claims are marked "unverifiable" on those datasets. The datasets are never pooled for training. Each is reported separately, as Milestone 1 §7 requires.

## 5. Model-Specific Data Requirements

### 5.1 Vision: labels, annotation and quality checks

- **Detection labels:** single-channel masks matching each image's size, read as class values rather than colours. Resizing uses nearest-neighbour interpolation, and the same transform is applied to both images and the mask.
- **Claim annotation:** the reliability objective (O3) needs labels that masks cannot give, such as direction, object identity and counts. Two reviewers label each claim as supported, contradicted, unverifiable or deferred, using the original images, without seeing which system wrote it. Disagreements are resolved and agreement is reported. A pilot annotation sheet, a viewer and a separate answer key are prepared.
- **Quality checks so far:**
  - an automatic reference check combining the mask and the five captions, with its rules unit-tested;
  - the mask/caption conflict list (§1.3);
  - an AI-assisted visual spot check of 36 screening claims. It found the claim parser faithful on 34. On the 29 claims it could judge, it agreed with the automatic verdict on 27.

  The spot check is not a substitute for the human audit, which remains required.

### 5.2 Vision-language models and fine-tuning

- **Input format:** two images (before, after) plus one fixed instruction, prompt `B-prompt-v1`:

  > Image T1 is earlier and T2 is later. Describe visible building or road changes in at most two sentences. Use image-relative locations. If the evidence is insufficient, state that. Do not infer legality, ownership or cause. Do not give an exact count unless individual objects are distinguishable. If no relevant change is visible, say so.

  Output is capped at 96 new tokens. Each model's own preprocessing is applied, and the images are otherwise unaltered.
- **Specialist captioner (RSCaMa):** CLIP preprocessing at 224 px, captions truncated to 42 tokens (the longest LEVIR caption has 39 words), and a vocabulary built from training captions with words seen at least 5 times. That gives 468 tokens on full LEVIR-CC and 234 on our 1,313-pair subset.
- **Training pilot:**
  - **Setup:** RSCaMa was retrained once from public CLIP weights on the 1,313 training pairs (6,565 captions), with the checkpoint selected on development BLEU-4. It ran for about 4 hours on a free Kaggle T4 GPU.
  - **Rationale:** this is a bounded pilot to obtain an uncontaminated baseline, not a full-scale training run.
  - **Result:** it matched the released checkpoint on LEVIR screening (79.2% vs 78.7% of claims checking out).
- **Instruction tuning:** formatting Qwen3-VL fine-tuning data as instruction–response pairs (prompt above → reference caption) is possible from the same manifest but deferred. Our transfer results suggest that tuning on LEVIR alone may narrow the model, so it would be judged on all three datasets.

## 6. Reproducibility

The full pipeline is on branch `feat/poc-b-captioning`, folder `POC/b-captioning/` (README, scripts, tests and the training notebook).

| Step | Command (from `POC/b-captioning/`) | Output |
|---|---|---|
| Freeze LEVIR subset | `python scripts/build_manifest.py` | `data/manifest.csv` with SHA-256 per image (manifest `c5dde811…e1f0`), `references.json`, `mask_stats.json` |
| Build transfer sets | `python scripts/build_ood.py whu_cdc` · `second_cc` | `data_ood/<name>/` in the same layout (manifests `1d7d043a…59a5`, `2c8f4b7c…c7d`) |
| Pinned model weights | `sh scripts/fetch_weights.sh` | Qwen3-VL-4B `ebb281e`, SmolVLM2-2.2B `482adb5`, RSCaMa `3e527e6`; RSCaMa code `aa68e4c` |
| Training pilot | `colab/rscama_finetune.ipynb` (Kaggle or Colab T4) | `best.pth` (SHA-256 `7014a967…9838`), training log |
| Caption, score and summarise | `caption_*.py`, `controls.py`, `evaluate.py`, `summarise.py` (set `B_DATA=data_ood/<name>` for transfer sets) | `outputs/b0_results*.md` |
| Tests | `python -m pytest -q tests` | 32 tests pass |

- **Environment:** Python 3.11.14 and PyTorch 2.8.0. There are two virtual environments: `transformers` 4.57.1 for the VLMs, and 4.39.3 for RSCaMa, matching upstream.
- **Hardware:** the Apple M4 (MPS) runs inference; a Kaggle T4 ran the training pilot.
- **Determinism:** greedy decoding, fixed seeds and string-seeded selection. The RSCaMa CUDA kernels were replaced by a pure-PyTorch equivalent on the Mac. On a CUDA GPU, the Mac version and the original kernels give identical captions on 150 of 150 pairs.

## 7. Open Items Before Milestone 3

| Item | Why it matters | Owner |
|---|---|---|
| Official source and licence for WHU-CDC | Results on the third-party copy stay provisional until confirmed | Sujal |
| Remove the 16 LEVIR duplicate pairs from the training data | Prevents detection training from seeing test copies | Sarthak (detection), Sujal |
| Human claim audit with two reviewers, inter-rater agreement reported | Validates the automatic reference check (O3) | Sujal + one reviewer |
| Detection data pipeline (tiling, class weights, de-duplicated training split) | Needed for O1 and for evidence-conditioned captioning | Sarthak |
| Data format for the visual inspection and decision roles (crops, labels) | Needed for the checker comparison (settings 3–5) | Priyanshu, Shoaib |
| Robustness perturbation sets | O2 | Team |

## References

1. Liu, C. et al. (2024). *Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis*. IEEE TGRS. Code and LEVIR-MCI: https://github.com/Chen-Yang-Liu/Change-Agent ; https://huggingface.co/datasets/lcybuaa/LEVIR-MCI
2. Liu, C. et al. (2022). *Remote Sensing Image Change Captioning With Dual-Branch Transformers* (LEVIR-CC). https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset
3. Ji, S., Wei, S. & Lu, M. (2019). *Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set* (WHU building dataset). IEEE TGRS. WHU-CDC copy used: https://huggingface.co/datasets/hygge10111/RS-CDC
4. Karaca, A. C., Ozelbas, M. E., Berber, S., Karimli, O., Yildirim, T. & Amasyali, M. F. (2025). *Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework*. IEEE JSTARS. https://github.com/ChangeCapsInRS/SecondCC ; https://doi.org/10.5281/zenodo.16937571
5. Yang, K. et al. *Asymmetric Siamese Networks for Semantic Change Detection* (SECOND). https://captain-whu.github.io/SCD/
6. Liu, C. et al. (2024). *RSCaMa: Remote Sensing Image Change Captioning with State Space Model*. IEEE GRSL. https://github.com/Chen-Yang-Liu/RSCaMa ; checkpoint https://huggingface.co/lcybuaa/RSCaMa
7. IITM DS & AI Lab. *Project Milestones: Milestone 2*. https://bsc-iitm.github.io/dsai-lab-project/milestones.html#milestone-2

Online sources accessed 27–29 October 2026.

## Team Section Sign-off

Signed off: PA · SS · SRP · SSSM *(pending: each member confirms before submission)*
