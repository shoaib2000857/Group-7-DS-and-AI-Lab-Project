# Shoaib (SSSM): Colab decision benchmarks

[Notebook](TerraWatch_Shoaib_Decision_Benchmarks.ipynb) · [Readable runtime](benchmark_runtime.py) · [Validation](VALIDATION.md)

[Open the notebook in Google Colab](https://colab.research.google.com/github/shoaib2000857/Group-7-DS-and-AI-Lab-Project/blob/7a375e8a65ddc3e3b208fa55ff9f0c48019f7038/POC/benchmarks/colab/TerraWatch_Shoaib_Decision_Benchmarks.ipynb) · [Source branch](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/tree/feat/shoaib-colab-benchmarks)

This is work package D: claim decisions, evidence checking, calibration diagnostics and initial integration. It does not run all four teammates' model pools. The notebook is self-contained and needs no repository checkout or companion upload.

## Run in Colab

**Runtime 2026-10-27.2:** repairs the environment-creation failure observed in Shoaib's three L4 runs. Open the current notebook, start a fresh smoke run, and confirm the printed runtime ends in `.2`. Private environment creation now uses a pinned virtualenv bootstrap with bundled pip; setup is checked before image loading, and setup stdout/stderr is included in the ZIP. Do not resume the old `.1` archives into the revised runtime. Their checked findings are [recorded here](../results/2026-10-27/shoaib/README.md).

1. Open the notebook in Colab or upload the `.ipynb` using **File > Upload notebook**.
2. Select a GPU runtime. L4/A100-class hardware can attempt d3's BF16 route; a T4 skips the current d3 release and attempts compatible models. No particular device or fit is guaranteed.
3. Keep `PROFILE = "smoke"` and run all cells. This selects four development and six validation pairs. Judge models run sequentially; installation, downloads and GPU speed affect duration. The judge loop is capped at 120 minutes, plus up to 30 minutes for the witness.
4. Download the final `SSSM_..._results.zip` and send the whole ZIP back. Errors and partial results are useful. If needed, use the rescue cell or the Colab Files panel.
5. After reviewing the smoke output, change to `screen` for 10 development and 30 validation pairs. Reuse the exported manifest with the other owners. Do not treat a smoke/screen run as sealed evaluation.

Enable Drive backup if you want checkpoint ZIPs copied to `MyDrive/TerraWatch/SSSM_benchmarks` after each candidate. Working images remain local for timing. Resume with the previous results ZIP and the same configuration; completed cases are keyed by source content, prompt/settings and model revision. Reused measurements are excluded from fresh timing summaries. Runtime environments remain outside the results ZIP.

## Candidate coverage

| ID | Implementation | Default |
|---|---|---|
| J01 d3-mini | Publisher's native System One readout, pinned Hub revision; current checkpoint requires BF16 | Attempt if GPU passes the BF16/memory gate |
| J02 Kev-4B | Author Linux server, base + adapter + pointer head; source commit pinned, head temperature retained | Optional local runtime; conservative 22 GB gate |
| J03 Clef-flash 9B | Local Ollama install/pull, digest recording, System One feature probe | Optional local runtime; conservative 22 GB gate |
| J04 Clef 27B | Same local runtime route, larger memory gate | Optional; conservative 44 GB gate |
| J05 Qwen3-4B text | Three allowed single-token labels, restricted softmax; text only | Attempt |
| J06 Qwen3-VL-4B | Frozen witness generation, then common-text and original-image judging | Attempt |
| J07 InternVL3.5-4B | Separate Transformers environment; original multi-image patch grouping; strict generated label parser | Attempt |
| J08 Jev | Access/cost-deferred register entry; no hosted caller | Deferred |

The gates are scheduling safeguards, not measured VRAM minima. No automatic quantisation, alternate checkpoint or cloud fallback occurs. Optional local installation is explicitly controlled by `ALLOW_OPTIONAL_RUNTIMES`; add J02/J03/J04 to `MODEL_IDS` to schedule them. The current optional server routes need their own Colab fit test. Failed access, installation, native API, parsing, timeout and memory checks remain in the exported register.

The lightweight notebook kernel keeps its installed dependencies. Native HF workers use Transformers 5.17.0; InternVL uses its separate 4.52.1 environment. Colab's CUDA PyTorch/torchvision are reused where compatible. Kev dependencies are isolated and its Linux runtime retains its own load behaviour. No `pip install kev` package-name collision is used: the author repository is installed from a pinned checkout.

Environment bootstrap helpers are installed into the private runtime cache with `--target`, rather than into the notebook kernel. [Virtualenv's seeded environments](https://virtualenv.pypa.io/en/20.35.1/cli_interface.html) supply their own pip; [stdlib venv normally calls ensurepip](https://docs.python.org/3/library/venv.html). The old ZIPs captured the failed command but not its detailed stderr, so an absent/disabled ensurepip remains a hypothesis rather than an established root cause.

## Data and interpretation

- The default source is the author's [LEVIR-MCI archive](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI), which contains the LEVIR-CC annotations. The loader checked actual archive layout and fetches only selected members with bounded HTTP ranges. It refuses a server that ignores ranges; it never silently downloads the full archive. A local copy can be supplied with `LOCAL_DATASET_ZIP`.
- Selection uses IDs and a seed, preserving official train/validation membership and checking within-split POC ceilings. If a central shared manifest exists, supply it with `TEAM_MANIFEST_PATH`. Otherwise the manifest is explicitly provisional and must be shared across the team. Parent-scene overlap and model pretraining exposure remain unresolved; this is not proof of independent geographical transfer.
- The real-image controls include change presence/absence, a reference caption, its explicit logical negation, and an unverifiable legality claim. They are **annotation proxies and controlled stress tests**, not naturally sampled error prevalence. Reference captions can contain several claims and are labelled caption-level. Generated witness captions remain unrated until humans inspect them.
- Frozen witness observations are generated without reference annotations. Missing observations stay unknown. The absolute-difference/Otsu mask is a diagnostic control, not a trained detector. Counts of changed pixels/regions are never treated as building counts.
- Constructed count/location/legality fixtures test decision logic separately from satellite performance. All variants stay grouped by pair. These fixtures must not be pooled into a satellite accuracy headline.
- LEVIR-CD/+, S2Looking, SYSU-CD, SECOND and OSCD require compatible human claim references before decision evaluation. ChangeChat instruction data needs an overlap and annotation audit. They are a portfolio, not nine interchangeable labelled judge datasets. Use the bundle route below when their suitable evidence is ready.

## Team-bundle schema

A ZIP has these files at its root:

```text
manifest.json
cases.jsonl
references.jsonl
images/pair001/T1.png
images/pair001/T2.png
```

`manifest.json` must document the actual audit and include `poc_budget_audited: true` and `split_and_overlap_audited: true`. These flags are an assertion by the bundle preparer, not an automatic proof. Include dataset/release IDs, official split counts, selected IDs, source overlap, ontology, image/transform provenance and independent scene IDs where recoverable. Keep these audit notes in the ZIP.

Each line of `cases.jsonl` is an inference case. The record below illustrates structure, not a measured prediction:

```json
{
  "case_id": "pair001_claim01",
  "pair_id": "pair001",
  "scene_id": "scene001",
  "phase": "val",
  "benchmark_group": "team_natural_claims",
  "claim_text": "Two buildings were added in the upper-right area.",
  "claim": {"object": "building", "action": "added", "quantity": 2, "location": "upper_right"},
  "image_paths": ["images/pair001/T1.png", "images/pair001/T2.png"],
  "mask": {"source": "frozen_detector_run_id", "visible_change": true, "building_count": null},
  "witness": {"source": "frozen_witness_run_id", "summary": "Possible new buildings; one is unclear.", "visible_change": true, "quality_flags": ["ambiguous_count"]}
}
```

Each line of `references.jsonl` contains the separately adjudicated target, joined by case ID:

```json
{"case_id":"pair001_claim01","target":"defer","label_source":"human_adjudicated_two_raters","error_type":"ambiguous_count"}
```

Allowed targets are `accept`, `reject`, `defer`, or null for unreviewed claims. Use `dev`, `val`, `test` phases and keep all claims from the same scene/pair in one phase. Images are ordered T1 earlier, T2 later; unknown dates/GSD stay null. Geometric transforms and crop coordinates must follow the shared original-T1 convention. Native model adapters receive an allowlisted claim/evidence state and images only; references are not in their request.

For held-out use, supply `PROFILE="heldout"`, the team bundle and a previous `FREEZE_PATH`. Test targets require `human_adjudicated_two_raters`, and inference settings must match the freeze. The default loader blocks official test images. Retain real annotations and disagreement logs; a label-source string is not a substitute for the two-rater process. Finalist selection and prompt/threshold tuning belong on development/validation.

HF finalist checkpoint revisions are frozen. Sealed Ollama evaluation remains blocked until a separate immutable runtime/model-digest freeze is implemented; its optional route is for development/validation only.

## Results and limitations

The notebook displays completion, accepted-target precision, true-claim acceptance, false-claim rejection, false rejection, deferral/coverage, valid three-class Brier scores, a reliability diagram, per-call timing/memory and representative image pairs. Outputs remain separated by phase, information track and reference source. Empty acceptance has undefined precision; all-defer has zero useful decision coverage. Missing outputs stay visible. No fictitious probabilities are assigned to deterministic rules or InternVL's generated label.

Qwen's probability is conditional on the three label tokens; native readout probabilities also need task-specific calibration. The reliability chart is an assessment, not proof that a model is calibrated. The 0.80 acceptance threshold is a fixed starting setting. Validation sensitivity and reversed option-order checks do not tune on sealed labels. Pair/scene bootstrapping is exploratory when independent groups are few.

Main metrics apply that fixed gate only to probabilistic accepts; lower-probability accepts become defer. Raw choices remain in each prediction file, `raw_outcome` and `raw_choice_metrics.csv`. The reliability diagram uses the ungated top option. Rules and InternVL's generated code have no fabricated probability and remain ungated. Ollama package tags may already contain publisher-quantised weights; the pulled digest and model details identify that configuration rather than treating it as native BF16.

Targeted/random reinspection replays cached decisions at equal image-call counts. It does not match all token costs, measure fresh end-to-end latency or validate a deployment trigger. J06 being both witness and judge creates correlated errors. Actual trained detector/captioner outputs should replace the diagnostic chain through the bundle route before drawing full-pipeline conclusions.

Human review sheets hide system identity and retain the key separately. Two raters should first establish reference facts from original images. The notebook does not invent human factual precision, reference-fact recall, count-instance truth or a custom pipeline metric. Those require the missing references and later analysis.

## Build and check

```powershell
python POC/benchmarks/colab/build_notebook.py
python -m unittest discover -s POC/benchmarks/colab -p test_benchmark_runtime.py -v
```

The generator embeds the readable runtime and its SHA-256 into the notebook. Regenerate it whenever the runtime changes. Test fixtures are implementation checks and are never published as learned-model results. See [VALIDATION.md](VALIDATION.md) for actual execution evidence and the remaining GPU execution gap.
