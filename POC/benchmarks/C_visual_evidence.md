# C. Priyanshu Agarwal: visual evidence and preprocessing

Own original-image inspection, visibility/ambiguity judgments, geometric transforms and preprocessing ablations. This role supplies observations to the decision layer; it is not the final ground-truth annotator.

## Six learned visual-inspector candidates

| ID | Exact checkpoint | Why test it | Priority / scheduling tier |
|---|---|---|---|
| W01 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Small established multi-image control | P1; M |
| W02 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Compare a newer Qwen visual backbone | P2; M; shared with C04/J06 |
| W03 | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Alternative visual representation | P2; M; gated terms, shared with C05 |
| W04 | [OpenGVLab/InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B) | Different multi-image/tile pipeline | P1; M; shared with C06/J07 |
| W05 | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | Low-resource visual-observation control | P2; S/M; shared with C07 |
| W06 | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | Larger multi-image challenger | P3; M/L; approximately 8B class; native custom processor |

Author model cards and weight-file metadata were checked; satellite-domain performance and local runtime were not. Qwen-family witnesses share ancestry, and MiniCPM also uses a Qwen language backbone. Different names do not prove independent errors.

## Standard inspection task

Input is the original pair, a **candidate** region and optionally a single claim to inspect. Provide corresponding T1 and T2 crops with the same coordinates and padding, plus downscaled full-image context. Label every image explicitly. Crop order is always T1 then T2.

Ask for constrained observations: visibility at both dates, object type or unknown, evidence for addition/removal or unknown, image-relative location, whether an exact count is distinguishable, and uncertainty reasons. Preserve raw output and parser status. A verbal confidence number is not a calibrated probability.

Compare two conditions on the same claims: full-image direct inspection and crop-plus-context inspection. Keep source pixels, model, token budget and decoding controlled; log each processor's final image-token/tile count. Start with manually selected **development** crops so this task does not wait for Sarthak; keep that oracle-region diagnostic separate from the later predicted-region track.

Test wrong object, wrong direction, wrong location, false change and ambiguous touching roofs. Audit observations against humans viewing originals. Do not present a generated second caption as independent verification, and do not ask the witness to see human labels or reference captions during inference.

## Six preprocessing approaches, not six invented learned models

All conditions still apply the checkpoint's mandatory processor. Compare optional adjustments against P00; preserve raw inputs and reversibly map output coordinates.

| ID | Approach | Hypothesis / restriction |
|---|---|---|
| P00 | No optional adjustment; model-required normalisation and paired tiling/padding | Mandatory control; protect real differences |
| P01 | Shared pair-level percentile clipping/rescaling | Test saturation/contrast sensitivity; use one declared rule for both dates |
| P02 | CLAHE on luminance with fixed settings | Contrast-recovery diagnostic; can amplify noise or alter evidence |
| P03 | T2-to-T1 histogram matching | Deliberately cautious radiometric diagnostic; can erase real change |
| P04 | Bounded ECC translation/euclidean registration | Test small misalignment; confidence/overlap gate, no automatic elastic warp |
| P05 | ORB matching plus robust affine estimation | Alternative registration when stable features exist; reject weak matches and excessive deformation |

Sources: [OpenCV ECC/motion API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html), [OpenCV ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html), [OpenCV CLAHE tutorial](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html), [scikit-image exposure operations](https://scikit-image.org/docs/stable/api/skimage.exposure.html).

Implement these as independent switches, not an automatically stacked enhancement chain. Screen P00/P01/P04 first on one frozen detector, then all six if useful. On aligned benchmarks, registration should often leave inputs unchanged. Test recovery on known synthetic shifts and quantify harm on clean pairs. Use a declared common coordinate frame and exclude non-overlapping pixels; never shift one image while silently leaving labels in a conflicting frame.

## Outputs and acceptance

- `observations.jsonl`: pair/region/claim IDs, raw and parsed observations, processor details, image count, uncertainty and evidence paths.
- `transforms.jsonl`: source dimensions, declared GSD/CRS or null, affine/resize transforms and valid overlap.
- Six-row preprocessing ablation: pixel-F1 delta, witness error delta, clean-input harm and cost. A prettier image is not a success metric.
- Six-row inspector status/score table and a two-model shortlist. Save one confident error and one justified abstention for each model that runs.

No rescaling can recover physical detail absent in the source sensor. Do not infer metres-per-pixel from dimensions, super-resolve objects into the reference truth, or automatically label clouds/seasonal shifts without validated evidence. Suggested vision prompt settings are development choices and must be frozen before the sealed test.
