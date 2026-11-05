# Experiment interpretation and reporting register

Machine-readable sources, hashes, runtime metadata and selected decision metrics are collected in `evidence/experiment_register.json`.

## Captioning - Sujal

Six configurations are compared: released RSCaMa, RSCaMa retrained from public CLIP weights within the 20% budget, Qwen3-VL-4B, SmolVLM2-2.2B, a pixel-difference template and an always-no-change control. LEVIR has 50 development and 100 screening records; SECOND-CC and WHU each have 100 screening records. The supplied retraining description uses 1,313 train pairs plus 50 development pairs, rather than fine-tuning the already exposed released caption checkpoint.

Use the JSON aggregates as the reproducible metric source. `supported_claim_precision_strict` divides supported claims by all factual claims; `supported_claim_precision_in_scope` excludes claims outside the mask-supported evaluation scope. This denominator difference explains the larger percentages in the README. Neither is a human factual-accuracy measurement. Counts are connected-component diagnostics, not independently annotated building instances.

The released specialist's LEVIR screening strict precision is 78.69%, in-scope precision 88.89% and BLEU-4 0.6223. The controlled retrain has 79.2% strict precision and BLEU-4 0.5786. Overlapping intervals do not establish that the retrain is better. Specialist performance drops in transfer screens; restricted targets and reference conflicts affect the denominators. Keep WHU's usage-rights limitations attached to its historical results.

RSCaMa uses CLIP ViT-B/32, three temporal/spatial fusion layers and a one-layer Transformer decoder; the wrapper records greedy decoding and a 42-token maximum. The retraining script specifies Adam at 1e-4, 12 image pairs per step, up to 40 epochs, patience eight, seed zero and resumable best/last checkpoints. A script default is not an observed completed epoch count. The README reports Mac/CUDA caption parity; contradictory older parity notes remain preserved. A fresh GPU kernel parity test was not run in this CPU consolidation.

## Visual evidence - Priyanshu

The 50-case result is produced by `generate_synthetic_cohort`: 20 construction, 12 unchanged-with-false-claim, eight shadow, six jitter and four demolition fixtures. The supplied file records 50/50. These are hand-constructed regression conditions, even though the sample IDs start with `LEVIR-MCI/eval`.

The three real-image records use `train_000010` twice and `val_000005` once. Claims, quantities and regions are written into the evaluation script. The witness emits true/false/true visible-change results and leaves each D decision null. These cases demonstrate code execution on real pixels; they are not an independent real-scene accuracy estimate or evidence of a connected captioner/referee pipeline.

## Decisions - Shoaib

All six original ZIPs pass CRC and export-manifest hash checks. The first three `.1` sessions fail worker setup. The repaired `.2` smoke/screen sessions contain successful J05 (Qwen3-4B text) and J06 (Qwen3-VL-4B) inference. The 16:48:11 screen has two preserved exports; do not add them as independent runs. The union of all session manifests is 40 image pairs.

For the 30-pair screen validation group, raw annotation-proxy decision agreement is 41.33% for J05 common evidence, 45.33% for J06 common evidence and 56.67% for J06 image-backed. All three complete 150 proxy-labelled claims. These are the raw-choice metrics, before the fixed 0.8 probabilistic-accept gate. Report gated coverage/precision separately; model probabilities are not independently calibrated correctness probabilities.

The frozen visual witness and judges share model-family dependencies, so correlated errors are possible. Generated claims, synthetic logic cases and unreviewed natural captions remain separate groups. Missing/failed d3/InternVL outputs are not zero-accuracy benchmarks. No Jev/Clef/Kev run is established by listing those candidates.

## Milestone 3 use

The measured evidence supports choosing a domain captioner, retaining general VLM comparators, a compact visual-evidence module and conservative accept/reject/defer interfaces. Detector selection relies on architecture suitability and primary literature. The M3 report specifies the complete system design and training configuration; it does not relabel these separate component screens as a measured full-pipeline result.
