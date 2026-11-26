## Appendix J. Human Annotation Protocol and Ethics Checklist

### J.1 Design

The audit measures claim-level groundedness independently of the automated proxy labels used during development (Section 13.1). 120 stratified held-out Test claims are each rated independently by two team members, and a third member adjudicates every disagreement to form the gold consensus set. Each team member rates 60 claims. Raters see only the evidence needed to judge the claim: the earlier image T1, the later image T2, the zoomed contextual crop pair (minimum 32 x 32 pixels, padded by 25%) and the atomic claim text. The system's accept, reject or defer verdict, any referee probability and the other rater's label are hidden.

### J.2 Slice, rater and adjudicator assignment

| Claim slice | Rater A | Rater B | Adjudicator (when A and B disagree) |
|---|---|---|---|
| Claims 001 to 030 | Shoaib (SSSM) | Priyanshu Agarwal (PA) | Sujal Raj Pradhan (SRP) |
| Claims 031 to 060 | Priyanshu Agarwal (PA) | Sarthak Saini (SS) | Shoaib (SSSM) |
| Claims 061 to 090 | Sarthak Saini (SS) | Sujal Raj Pradhan (SRP) | Priyanshu Agarwal (PA) |
| Claims 091 to 120 | Sujal Raj Pradhan (SRP) | Shoaib (SSSM) | Sarthak Saini (SS) |

No claim is adjudicated by one of its own raters, and each member rates two slices (60 claims). The claim set is stratified over the model's accept, reject and defer outcomes and over the scene strata (unchanged, building-only, road-only, mixed, small-change), so that the sheet is not dominated by easy accepts.

### J.3 Rubric

Each claim receives exactly one `human_label`.

| Label | Definition |
|---|---|
| `supported` | The claimed object and change type (appearance, disappearance or no-change) are clearly visible between T1 and T2 |
| `contradicted` | The claim is factually wrong: for example a new building asserted on unchanged ground, the wrong object class, or an evident hallucination |
| `unverifiable` | Resolution, cloud or shadow occlusion, crop ambiguity or an exact numeric count prevents a confident human verdict |

When the label is `contradicted` or `unverifiable`, the rater also assigns one `error_category`; otherwise the category is `none`.

| `error_category` | Definition |
|---|---|
| `none` | Used only with `supported` |
| `hallucinated_object` | The claimed object does not exist in either image |
| `missed_change` | A real change is present but the claim states none, or omits it |
| `wrong_class_road_vs_bldg` | The change is real but is labelled road when it is building, or the reverse |
| `wrong_count` | The stated number of objects is incorrect |
| `wrong_location` | The object exists but the stated location or relation (for example "along the road") is incorrect |
| `shadow_or_seasonal` | The apparent change is shadow, illumination or seasonal appearance |
| `tiny_or_occluded` | The change is too small, partial or occluded to judge |

### J.4 Blinding and submission rules

1. Raters do not view the automated verdict and do not discuss labels with their paired rater until both sheets are submitted.
2. Each rater inspects T1, T2 and the zoomed crop pair for every claim and does not consult reference captions or ground-truth masks.
3. Each rater saves a completed sheet as `POC/milestone5/evidence/human_audit/rater_<initials>.csv` with initials `SSSM`, `PA`, `SS` or `SRP`.
4. The adjudicator sees both rater labels but not the model verdict, assigns the final label and category, and records the adjudication in a separate CSV passed to `m5_compute_human_audit.py --adjudication`.
5. Sheets are not edited after submission; changes are made only through adjudication.

### J.5 Agreement and groundedness metrics

**Cohen's kappa.** For two raters labelling the same claims over the three classes, with observed agreement `p_o` (the fraction of claims given the same label) and chance agreement `p_e = sum over k of p_{A,k} x p_{B,k}` (`p_{A,k}` and `p_{B,k}` are the marginal proportions of class `k` for raters A and B),

`kappa = (p_o - p_e) / (1 - p_e)`.

Kappa is reported pooled across the four slices, per slice and with the 3 x 3 rater confusion matrix, together with percent exact agreement. Values are interpreted on the conventional scale (below 0.40 poor to fair, 0.41 to 0.60 moderate, 0.61 to 0.80 substantial, above 0.80 almost perfect) and the number of claims on which the adjudicator overruled both raters is reported.

**Human-verified groundedness and hallucination rate.** For a verification policy, over the claims it accepts, groundedness is the share whose gold label is `supported` and hallucination rate is the share whose gold label is `contradicted`; `unverifiable` claims are reported separately. Policies compared are the unverified captioner, mask-only, direct referee and the calibrated mask-first hybrid. The agreement between the automated proxy label and the gold consensus is reported as the proxy-versus-human alignment, which determines how far the Milestone 4 proxy precision values can be trusted.

### J.6 Ethics and responsible-deployment checklist

| Item | Requirement | Status in this project |
|---|---|---|
| Data licence | Use only data under research-compatible terms, with attribution retained | `configs/datasets.yaml` records the licence disposition per dataset; SECOND-CC is CC BY 4.0, others are research-only with attribution |
| No raw data or weights in Git | Imagery, checkpoints and archives stay outside version control | Enforced by repository rules; checkpoints are referenced by SHA-256 only |
| Annotator welfare and consent | Raters are project members who are informed of the task and its use; no personal data are shown | Imagery is 0.5 m aerial tiles without identifiable individuals |
| Rater independence | Blind, independent labelling with a third-party adjudicator | Section J.2 and J.4 |
| Label provenance | Every final label traces to two independent labels and one adjudication | Rater and adjudication CSVs retained |
| Model transparency | Checkpoint digests, referee revision and threshold are recorded with every decision | `configs/pipeline.json`; Evidence Review records |
| Human in the loop | No automated accept is the sole basis for a consequential decision | Evidence Review keeps model outputs immutable and records overrides in an append-only store |
| Purpose limitation | Use limited to infrastructure, land-use and disaster change review; no tracking of individuals or protected sites | Section 13.7 |
| Privacy | Resolution below the level at which individuals are identifiable; reassess before using higher-resolution data | Section 13.7 |
| Scope of claims | Metrics are quoted for LEVIR-MCI and SECOND-CC only; local validation before any new region or sensor | Section 13 |
| Bias disclosure | Single region and sensor, class imbalance and 388-word vocabulary are disclosed | Section 13.3 |
| Auditability | Override and export records are reproducible from the audit database | `/api/export` of the Evidence Review application |
