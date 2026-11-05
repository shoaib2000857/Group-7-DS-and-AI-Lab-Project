# D. Shoaib SSM: decisions, calibration and integration

Own the claim/evidence contract, checking baselines, decision-model adapters, calibration, paired evaluation and integration. This is not an instruction to infer image truth from a text-only model.

## Eight learned candidates

| ID | Candidate | Modality and purpose | Priority / qualification |
|---|---|---|---|
| J01 | [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini), 4B-class | Native bounded decisions; text/structured state and image route | P1; M; [Ollama d3](https://ollama.com/library/d3) is an alternative runtime, not another model |
| J02 | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | Native decision model, **text only** | P2; M; base model + adapter + pointer head; not a plain standalone classifier |
| J03 | [Clef-flash](https://ollama.com/library/clef-flash), 9B | Native multimodal decision comparator | P2; M/L; needs a compatible decision endpoint and feasible precision |
| J04 | [Clef](https://ollama.com/library/clef), 27B | Larger native decision challenger | P3; L; local fit test first; cloud use is not automatically authorised |
| J05 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | General **text-only** structured judge | P1; M; checks stated evidence/claim consistency, not unseen pixels |
| J06 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | General multimodal direct judge | P2; M; important control against specialised decision branding |
| J07 | [OpenGVLab/InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B) | Alternative multimodal direct judge | P2; M; reuse checkpoint/adapters from Priyanshu |
| J08 | [Jev 1.14.0 release family](https://typesafe.ai/blog/jev-1-14) | Hosted decision comparator; image support experimental | P3, **deferred_access_and_cost**; pin actual served version, not a guessed model/API identifier |

Author/platform sources support candidate capabilities, not remote-sensing accuracy. Kev's published checkpoint includes an adapter and head; keep the head's probability handling and base revision intact. Do not feed it images or silently use an Apple-only Ollama package on Colab. Ollama pages/SDK surfaces can evolve: probe the installed `/v1/systemone` route with a trivial local example before scheduling inference. Prefer the model publisher's Linux/CUDA route when packaging does not match WSL/Colab.

## Mandatory controls, in addition to the eight learned candidates

- **R00:** no checking; retain baseline claims.
- **R01:** mask-only consistency rules, with unsupported fields marked unknown.
- **R02:** rules using the same image-backed observations as learned judges.
- **R03:** simple direct image-backed VLM judgment without a disagreement-trigger policy.

Rules/templates/threshold variants are systems or configurations, not additional model families.

## Separate two comparisons

**Decision quality on common evidence:** every judge receives exactly the same claim, candidate-region summary, visibility flags and frozen textual observations. J02/J05 compete here fairly with multimodal models in **text-only mode**. No images or human reference facts are leaked into selected conditions. This tests decision logic conditional on perception.

**Image-backed checking:** compatible judges also see original matched crops and full context. Compare this with their own text-only results. A multimodal gain over Kev is partly an information advantage and must not be attributed solely to a better decision architecture.

Writings from a witness are uncertain observations. Explicitly label their source; avoid calling them established facts. An incorrect witness must remain possible so propagation errors can be measured.

## Implementation tasks

1. Freeze `templates/claim_decision.json` with all owners before they build adapters. Use null for unavailable probabilities or metadata. Preserve raw outputs, reason codes and failed parses.
2. Create a small development fixture containing correct claims, naturally wrong claims where available, and separately marked injected object/count/direction/location errors. Group all variants by original pair; never split their claims randomly across validation/test.
3. Run J05 and rules first. Feature-probe J01 and its model-specific runner next; do not assume a decision readout can be replaced with `generate()`.
4. Add models by priority. If a stage fails to load or fit, record that; do not silently switch from local to billable cloud inference.
5. Report acceptance of true claims, rejection of false claims, false rejection, unverifiable/defer rate, coverage, human factual precision and reference-fact recall. All-defer is not perfect performance.
6. Where valid probability vectors exist, compute Brier score and calibration plots on independent pair groups. ECE is supplementary and depends on bins/sample size. Model concentration is not correctness probability; generated confidence is not equivalent to a readout probability.
7. Select thresholds/prompts on validation. Vary option order and paraphrase the decision wording on a fixed diagnostic set; measure decision flips. Never choose the best wording using held-out labels.
8. Compare targeted reinspection with direct full inspection and a random-inspection control at matched inference-call/token budgets. Keep detector, captions and common evidence fixed for the main checker contrast.

## Integration deliverable

Join by pair/claim/region ID, not filenames guessed from presentation order. Build one baseline chain and then a checker ablation table. Include stage-specific latency and fresh end-to-end latency; cached outputs help comparisons but do not establish deployable timing.

Every member should receive an anonymised review batch from another stage. Shoaib coordinates the split and merge but must not be sole annotator and sole final judge. Keep a custom pipeline metric exploratory until it is specified and validated against humans; it cannot replace the standard pixel and factuality metrics.

**Done:** working controls plus first-wave judges; an eight-row run/block/defer register; one calibrated/thresholded comparison when probabilities permit; and a reproducible integrated baseline with both successful and failed examples. Paid/API-only J08 may remain deferred without preventing completion.
