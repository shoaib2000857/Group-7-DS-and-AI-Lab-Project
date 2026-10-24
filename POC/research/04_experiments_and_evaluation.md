# Experiments, factual audit, and decision rules

All experiments below are proposed. Results remain unmeasured. Use the fixed subset protocol in `02_data_and_subset_protocol.md`.

## Primary question

Does image-backed claim checking improve the factual reliability of descriptions compared with mask-only checking when predicted change evidence is imperfect, while retaining useful information?

The POC estimates feasibility and effect direction. It cannot guarantee statistical power for a small effect. Do not reuse the submitted +10 percentage-point, 25% robustness, or sub-5-second targets as evidence-based promises.

## Experiment ladder

| ID | Comparison | Controlled factors | Output / interpretation |
|---|---|---|---|
| P0 | Data and pairing audit | One immutable manifest | Access, rights, labels, scene overlap, subset counts; stop if supervision is incompatible |
| P1 | Absdiff/Otsu and one pretrained detector | Same pairs and valid pixels | F1/IoU and failure examples; pretrained performance is not a subset-trained result |
| P2 | Image-only caption vs evidence-conditioned caption | Same model/version and decoding where interface permits | Isolate conditioning; if different architectures are necessary, label the confound |
| P3 | No checker vs mask-only V0 vs image-backed V1 | Identical cached candidate captions and detections | Primary checker effect, retention, omissions and abstention |
| P4 | Correct vs deliberately corrupted evidence | Same images/captions, controlled mask alterations | Does the checker reject bad evidence or just echo it? Synthetic stress test, not a natural-error estimate |
| P5 | Predicted vs oracle mask | Same caption/checker implementation | Diagnostic sensitivity to detector quality; an oracle condition may itself miss annotation-excluded changes |
| P6 | Clean vs limited nuisance perturbations | Fixed selected pairs, masks transformed consistently | Failure sensitivity, not proof of real-world robustness |
| P7 | Optional bounded training pilot | Same subset and training budget | Check loss/gradient behavior and small-data learning; do not compare with full-data SOTA as a fair contest |

P2 and P3 answer different questions. A template generated directly from events can establish constrained reporting behavior but cannot by itself show improvement over an image-only neural captioner.

## Wrong-evidence and wrong-claim challenges

Create a small, human-verified diagnostic set from selected development/evaluation pairs. Include mask false positives, omitted change, displaced masks, merged/split components, label swaps where the ontology permits, and ambiguous low-quality regions. Mask perturbations must never be mistaken for original annotations.

Separately introduce controlled caption errors: object-type swap, added/removed reversal, count error, location error, invented causal explanation, and false change on an unchanged pair. Keep naturally generated errors in a separate stratum. Synthetic errors are useful for sensitivity/specificity, but their prevalence does not estimate deployment error rates.

Test checks on correct claims too. A checker that rejects everything has high error rejection and no practical value. When feasible, include manual claim extraction as an oracle diagnostic to isolate parsing failures from verification failures.

## Human reference and annotation

Start with a 30–50-pair pilot within the frozen POC budget, covering changed/unchanged, small/large, crowded and ambiguous cases. This is a feasibility sample; a later larger audit needs a power/workload estimate. Label any enriched hard-case selection explicitly.

Two raters independently inspect original T1/T2 plus available reference annotations. First establish reference change facts without seeing the system's predicted mask or treatment name. Then judge anonymized candidate/revised captions in randomized order. Show predicted evidence only in a separate diagnostic round if needed, so the primary truth judgment is not anchored by the detector.

For each atomic claim mark `supported`, `unsupported`, `contradicted`, or `unverifiable`. `Unsupported` means no supporting evidence was found despite adequate visibility; `unverifiable` covers insufficient visibility/metadata or unresolved ambiguity. Mark object, action, location and count correctness separately. Record omissions against a reference inventory that states which changes are observable and within scope.

Raters must allow multiple true phrasings and reference-caption omissions. A third member adjudicates disagreements. Report raw agreement, a suitable agreement statistic such as Cohen's kappa, class distribution and examples; sparse classes can make kappa unstable. Save original rater judgments as well as adjudication.

## Metrics with explicit denominators

| Metric | Definition and handling |
|---|---|
| Change precision/recall/F1/IoU | Aggregate TP/FP/FN over valid pixels for the defined change ontology; state micro versus pair-macro aggregation |
| Empty-mask handling | State convention for pairs with no reference/predicted change. Report no-change false alarm rate separately; avoid undefined divisions |
| Class mIoU | Only for an audited multiclass ontology; state whether unchanged/background is included |
| Supported-claim precision | `supported / all emitted factual claims`; contradicted, unsupported and unverifiable claims remain in denominator for this strict primary measure |
| Evaluability | Share of emitted factual claims that human raters can judge; additionally report unverifiable rate |
| Contradiction / unsupported rate | Each category count divided by all emitted factual claims; retain categories separately |
| Reference-fact recall | Supported reference facts conveyed / adjudicated in-scope observable reference facts; measure omitted information |
| Claim retention | Claims retained after checking / candidate factual claims, plus mean factual claims per pair |
| Coverage / abstention | Share of pairs with an accepted informative report, and share requiring review; show precision versus coverage where scores allow |
| Checker error detection | Precision/recall for rejecting false claims and acceptance rate for true claims, using human labels; stratify natural and injected errors |
| No-change behavior | False change reports per unchanged pair; true negative rate; omitted true change on changed pairs |
| Region-count error | MAE against a fixed connected-component convention, explicitly named as region count |
| Building-count error | Only on manually validated instance counts with adequate visibility; report count coverage and ambiguity exclusions |
| Caption metrics | BLEU-4, METEOR, ROUGE-L and CIDEr only with pinned script/tokenization/reference handling; supplementary, not truth labels |
| Timing / memory | Median and tail warm latency plus cold load; per-stage and end-to-end including checker, parser and output; device/software/batch size stated |

For a system emitting no factual claims, factual precision is **undefined**, not 100%. Report zero coverage and reference-fact recall accordingly. Do not manufacture a new “CGP” metric whose definition duplicates precision without specifying matching and annotation rules.

Report counts and effect sizes, not just percentages. For paired comparisons bootstrap by independent parent scene where known; if unavailable, use pair-level resampling with the spatial dependence limitation stated. Keep all captions from a pair together. A small POC confidence interval is descriptive; it does not establish generalization. Use validation only to select thresholds/prompts and predeclare the main contrast before final POC evaluation.

## Limited robustness pilot

Begin with three nuisance families: brightness/contrast, blur/compression, and small one-image translations. Select a few severities on validation and report each condition. Do not equate synthetic brightness changes with seasonal change or synthetic translation with all viewpoint effects.

For a spatially shifted input, document coordinate conventions and valid overlap. Do not shift the target inconsistently. Real cloud robustness needs cloud/quality annotations or a defensible exclusion policy. Cross-dataset evaluation is a separate test after ontology and overlap audits, not merely another corruption level.

## POC exit decisions

- **Continue with checker research:** an automated candidate is feasible, catches some natural errors beyond V0, and gains do not arise only from near-total abstention. Estimate annotation and runtime cost for the main study.
- **Simplify:** templates are reliable but the visual checker adds no measurable value; position the work as constrained reporting/empirical analysis and consider detector robustness as the research axis.
- **Pivot data:** labels cannot support the desired claims or reliable access fails. Document the mismatch rather than training an unrelated semantic model to patch it.
- **Report negative results:** V1 echoes wrong masks, fails on crowded roofs, or loses too much content. This is useful evidence for main-project planning.

No outcome is prefilled. Empty tracking files in `templates/` are schemas, not experiment records.
