# Shared experiment protocol

## 1. What is fixed before independent runs

Freeze one manifest version, T1/T2 ordering, label dictionaries, valid-pixel rules, crop coordinate convention and output contracts. Use the same development/screen/evaluation IDs across appropriate tasks. Keep each dataset's official partition membership and parent-scene grouping where recoverable.

**All POC use counts:** training, image inspection, prompts, annotations and evaluation. Default ceiling is `floor(0.20 * unique_pairs)` per dataset, up to an absolute 30% maximum only under the existing POC policy. Apply corresponding within-split ceilings. Shared imagery across LEVIR releases is counted in a global union too; re-running an old sample does not consume a new independent pair.

Do not fill a fresh random subset per member. All four use the same frozen central set. Label-derived hard-case enrichment is a separate diagnostic stratum, not a representative population estimate.

## 2. Two data tracks, not one misleading leaderboard

**Track M, primary pipeline:** LEVIR-MCI with the LEVIR-CC reference-caption mapping. Proposed starting allocation is 50 training/development pairs (including the 10-pair smoke set), 100 validation-screen pairs and 100 sealed test pairs. These **250 proposed IDs are not selected yet**; accept these counts only after auditing official splits and confirming they satisfy the 20%/scene limits. If not, reduce counts. Prefer a representative seeded screen, with extra diagnostic cases chosen from the same budget.

MCI tasks have different label coverage: the Open-CD detector pool is building-change trained, while caption references can describe other changes. Evaluate building-only detectors on an audited building-mask projection. Evaluate broader captions against the relevant human/reference facts. Do not call missed roads a false-negative building prediction, or treat an unlabelled event as proven absent.

**Track D, secondary detection reproduction:** LEVIR-CD, preserving official train/validation/test. Use at most 20% of each split, so evaluation counts can be small; report exact counts and broad uncertainty. Published full-test scores are contextual, not a fair small-subset target. Each pretrained checkpoint's training exposure is recorded. No specialist caption score belongs in this detector-only table.

MCI and CD may share source imagery. Audit parent overlap before claiming transfer or geographical independence. Keep pretrained source-data inference, genuinely unseen transfer and subset-trained results in separate tables. Add at most one compatible held-out domain after the primary pipeline works; nine datasets are a later portfolio, not the first download task.

## 3. Stage selection without a Cartesian explosion

1. First-wave source/load test on the same 10 development pairs.
2. Record install/access/fit failures for every candidate. Limit first install troubleshooting to 90 minutes per candidate before documenting and switching to a feasible next entry. This is a project time budget, not a claim about model performance.
3. Run feasible first-wave models on the validation screen, then expand to the remaining P2 entries. No accuracy claims from smoke fixtures.
4. Shortlist at most two per role. Human-audit a small common sample first (e.g. 10 pairs per candidate) to estimate annotation cost; do not commit to judging every output of every model at full scale.
5. Run a frozen baseline chain. Use cached captions/detections for controlled checker comparisons and one witness/decision choice at a time.
6. At most 2×2 detector/caption combinations initially; use remaining budget for the main checking ablation. Only test more combinations when a specific failure warrants them.
7. Before opening sealed results, freeze the principal contrast, prompts, thresholds, preprocessing and matching rules. Finalists alone get the held-out benchmark. No prompt revisions based on test errors without declaring test reuse.

## 4. Colab Pro+ / WSL run rules

The [Colab FAQ](https://research.google.com/colaboratory/faq.html) states resources are variable and not guaranteed. Record actual device and memory with the probe at the start of **every session**. Colab Pro+ does not imply an A100, identical GPUs for all members, four guaranteed concurrent sessions or permanent storage.

Scheduling estimates, **not measured VRAM minima**:

| Tier | Start here | Escalation |
|---|---|---|
| S | CPU controls; compact detectors; small specialist inference | GPU optional/preferred; measure rather than assume batch 32 fits |
| M | One 2–4B VLM/text judge at a time on a 16GB-class session, limited image tokens and batch 1 | Try native supported precision, then a separately logged supported quantisation/offload route; fit is not guaranteed |
| M/L | 8–9B visual/decision models or legacy multi-image stacks | Prefer 24GB+ or validated lower-bit route; raw BF16 weights alone can approach/exceed a 16GB budget |
| L | 27B models / large visual contexts | Separate 40–80GB-class feasibility queue or supported quantisation; not required first-wave work |

Use FP16 on hardware where BF16 is unsupported; do not force BF16 from a model-card snippet. Treat a quantised model as a separate **configuration**, log the method/digest, and do not mix its scores invisibly with native precision. Input token count, tile count and output limit materially affect fit and latency.

WSL is appropriate for local preparation, repositories and Linux-only dependency troubleshooting; it does not inherit a remote Colab GPU. Use separate environments for Open-CD, specialist captioning, Mamba, Transformers VLMs and decision runtimes. Prefer one known-good model per environment before bulk installation. Keep native Colab notebook execution as fallback if the existing CLI/bridge is unavailable; its access has not been tested by this plan.

Mount authorised storage and copy the selected working subset to session-local disk for timing; save outputs/checkpoints/configs incrementally to durable storage. Do not time Drive file transfer as model inference, and do not put credentials into notebooks or Git. Existing account access is not permission to buy compute units, paid APIs or cloud model calls.

## 5. Standard outputs and fair measurements

Per run save: model ID/role; upstream code commit; checkpoint revision/hash; source training data/exposure status; sample/manifest hash; environment; processor/crop/resolution/normalisation; precision/quantisation; prompts; seed/decoding; start/end/status; raw outputs; parsed outputs; timing/memory; errors and excluded samples.

Coordinates use original T1 pixel frame, bounding boxes `[x0,y0,x1,y1)` and explicit transforms/valid overlap for T2. Unknown dates, CRS/GSD, counts and probabilities are null. `region_count` is not `building_count`. Prediction and reference directories must be separate and references unavailable to inference adapters.

Use batch 1 for comparable latency. Record cold load separately. Warm up on development pairs, synchronise GPU measurements, then measure the screen in a fixed order. Report median/p95, peak allocated memory and device; no cross-device speed leaderboard without an explicit hardware distinction. Cache timings separately from fresh end-to-end latency. Include timeouts/failed parses as system failures, not silent dropped samples; report the common-valid subset and overall completion rate.

### Metric families

- **Detection:** change F1/IoU/precision/recall over valid pixels, with ontology and aggregation stated; unchanged false alarms and small-region failure examples.
- **Captioning:** pinned BLEU/METEOR/ROUGE/CIDEr recipe for comparability; human supported-claim precision, reference-fact recall, false-change/omission rates and mean emitted claims.
- **Witness:** object/action/location correctness, visibility/ambiguity assessment and helpful/incorrect observations, stratified by naturally wrong masks.
- **Decision:** true-claim acceptance, false-claim rejection, false rejection, deferral, precision at matched coverage, factual recall and valid calibration measures where probabilities exist.
- **Robustness:** per-condition clean-to-corrupted drop plus separate real-domain transfer, not one pooled “robustness” number.

Empty output is not 100% factual precision. Compare useful coverage/recall alongside precision. Repeated captions of the same fact cannot inflate recall. On unchanged pairs, evaluate no-change behaviour separately. Ground truth masks may have incomplete object/type annotations; human reference facts require visibility and scope rules.

## 6. Robustness and annotation budget

Start with brightness/contrast, blur/JPEG and one-image small translations, two declared severities each. Select severity on validation; keep synthetic tests separate from real seasonal/cloud/viewpoint evidence. Apply geometric labels/transforms consistently and evaluate only valid overlap. Run the six preprocessing variants on one frozen detector first, then a finalist witness if useful.

Blinded human audit: pilot 10 common pairs per candidate where affordable; then approximately 100 held-out pairs for the final checking comparison if the workload/power pilot supports it. Two raters per output, rotated across members; anonymise system IDs. Establish reference facts from original pairs without predicted masks before judging outputs. Keep original disagreements and adjudication, and report raw agreement plus an appropriate statistic. Bootstrap by scene where available, otherwise by pair with the dependence limitation; claims within a pair are not independent samples.

Separate natural generated errors from injected wrong count/type/action/location/mask examples. Keep wrong-mask and no-change cases because they test the project's actual reliability question. All members help annotate; the integration owner is not the sole authority on correctness.

## 7. Optional short learning pilots

Only after frozen inference baselines and a timed batch estimate: at most two compact detector configurations, seed 42 initially, **500 optimiser steps or two epochs or 45 minutes per run, whichever ends first**. At most one selected caption/VLM adapter experiment, **200 steps or 30 minutes**, after a fit check. These are POC resource caps, not adequate-convergence claims. Lower the budget if required; no automatic extensions, broad tuning or full-model pretraining.

Compare the same training subset and explicit image/step budget. Do not compare a full-source pretrained model and a scratch pilot as if training data/compute were equal. Three seeds are reserved for a later focused confirmation if justified and budgeted.

## 8. Seven-day proposed sequence

| Day | Shared checkpoint |
|---|---|
| 1 | Freeze IDs/ontology/contracts; environment records and first valid output per owner |
| 2 | First-wave 10-pair smoke; resolve joining/shape issues; document blockers |
| 3–4 | First-wave screen; add feasible P2 entries; early common human audit |
| 5 | Freeze at most two finalists per role; preprocessing and wrong-evidence diagnostics |
| 6 | Integrate one chain; controlled checking and matched-budget comparisons |
| 7 | Sealed finalist evaluation if ready; results, uncertainty, failures and next decision |

This is an effort sequence, not a promise of GPU availability or completion. Prioritise the actual October 29 M2 data deliverables and November 5 M3 end-to-end subset over completing every optional model. If shared data/access is blocked, continue environment and development-fixture checks without claiming benchmark results.
