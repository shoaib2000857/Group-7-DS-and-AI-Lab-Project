# Suggested M1 corrections, without changing the submission

These notes respond to written TA feedback and the review discussion. Root reports, slides, README, and problem statement have not been edited. Preserve previous versions if the team later revises them.

## Claim audit

| Existing statement / implication | Problem | Suggested replacement |
|---|---|---|
| “Today's tools stop at a change map” | Change captioning, agents and interactive systems already go beyond masks | “Many detection workflows produce masks; existing captioning and interactive systems also describe changes. We investigate their factual reliability under imperfect evidence.” |
| Modular architecture is a research contribution | A useful design decision is not itself a demonstrated advance | Explain what controlled substitution enables experimentally, and call modularity an engineering choice |
| Captioners ignore evidence / none check captions | Too broad; detection-guided features and fact-based evaluation already exist | Compare exact mechanisms in Change-Agent, ChangeChat, CCExpert, KCFI, ChangeVG and FMScore; state remaining uncertainties |
| Human factual accuracy is novel | Related remote-sensing and general-caption work already measures factuality | Position the specific detector-error experiment and checker mechanism, if supported |
| Mask agreement proves factual correctness | Detector errors can propagate into both caption and checker | Explicitly separate consistency with predicted evidence from independent reference judgment |
| Connected components equal structures | Semantic regions merge/fragment; masks may not encode instances | Name the metric region-count error, or collect validated instance annotations for building-count error |
| Binary change implies new construction | Direction and object class may not be encoded | Use “change region” until visual/semantic evidence establishes type and direction |
| +10-point factual gain, 25% robustness improvement, ≥85% grounding precision | No pilot or comparable baseline establishes these targets | State comparative hypotheses, primary measures, pilot feasibility criteria and uncertainty; set justified numerical goals after measurement |
| Under 5 seconds on T4 | No full-pipeline benchmark or verified hardware | State that per-stage and end-to-end timing will be measured on documented hardware |
| Data access/license is settled | Link presence, payload access, imagery terms and redistribution are different questions | Separate “source identified,” “metadata inspected,” “archive acquired,” “labels audited,” and “rights resolved” |
| A consumer GPU/T4 will train all baselines comfortably | Model and image/sequence requirements differ | Provide measured batch feasibility and a training estimate after a pilot |
| No published work reports relevant factual evaluation | FMScore and adjacent grounded-change evaluation invalidate a broad absence claim | Describe what each existing evaluation measures and what the proposed experiment additionally tests |
| Long-term temporal tracking from two images | Two snapshots do not recover a trajectory or intermediate events | “Two-date change comparison”; time-series tracking remains a separately evaluated extension |
| Illegal encroachment detection | Images alone do not determine authorization, ownership or legal status | “Visible change flags to support officer review”; legality needs external authoritative records |
| HLS / high-resolution building data are interchangeable | Spatial scale and spectral/label requirements differ | Separate building-scale and large-area monitoring tracks; do not claim transfer without evaluation |

## Proposed replacement research paragraph

“TerraWatch investigates the reliability of plain-language descriptions of visible urban change between two aligned high-resolution images. We compare an image-only description baseline, evidence-conditioned descriptions, mask-based consistency checks and a proposed image-backed checker that can abstain when evidence is insufficient. Existing work already combines change detection, captioning and interactive analysis, and includes factuality-oriented evaluation. Our provisional focus is therefore the effect of imperfect predicted evidence on claim correctness and useful information retained. The POC will establish feasibility and whether the checking mechanism adds value; novelty and numerical improvement are not yet demonstrated.”

## Suggested revised M1 structure

1. Concrete task, users, practical value, and limits.
2. Study-comparison matrix with objective, data, design, findings and relevant limitations.
3. Mechanism comparison of close prior art, including what is already solved or partially addressed.
4. One candidate research question, with a defensible contribution category.
5. Data feasibility and the distinction between identified links and audited usable assets.
6. Experimental contrasts, factuality/coverage definitions, risks and negative-result handling.
7. POC boundary and milestone plan, clearly separating proposed from completed work.

Keep evidence traceable and remove repeated promotional claims. A concise coherent revision is more useful than adding pages of model names. Do not submit this research pack verbatim as a completed experiment report.
