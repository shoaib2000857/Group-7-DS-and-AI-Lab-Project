# Main-project data notebooks

Run these in order:

1. `01_EDA.ipynb` - coverage, native dimensions, label/pixel distributions, small changes, missing joins, image variation, spatial density, duplicate/leakage evidence, outliers, correlations, semantic transitions and caption/instruction variants.
2. `02_Preprocessing.ipynb` - deterministic exclusions, target conversion, fixed source partitions, task-specific manifests, complete loader checks and a real input batch.
Exploratory resize/brightness/augmentation checks are kept separately under [`POC/data_quality/`](../POC/data_quality/README.md).

These notebooks call the same modules under `src/data/` as the command-line pipeline. They are not separate copies of processing logic. Their saved tables and plots reflect an executed run; check the coverage table before interpreting a convenience probe as a complete release.

The existing model-experiment and evaluation placeholders are left available for subsequent modelling work; the data-quality notebook does not claim to train a model.

## Google Colab

Open a notebook from the `data-pipeline` branch. The setup cell can clone that branch into `/content/TerraWatch` and install `requirements-data.txt` when the repository is absent. Drive mounting is enabled automatically in Colab and requires your normal Google authorisation.

The default data folder is `/content/drive/MyDrive/TerraWatch_Extracted`, based on the supplied collection. Set `DATA_ROOT` if your shortcut/mount has a different name. Dataset roots can be nested as in the inspected Drive collection; the catalogue lists the supported layouts. `TERRAWATCH_<DATASET_ID>` is an explicit path override, for example `TERRAWATCH_SECOND_CC`.

Run all of `01_EDA` before `02_Preprocessing`. Set `RERUN_AUDIT=True` or `RERUN_PREPARATION=True` after changing sources/configuration; otherwise an existing local audit/preparation is reused transparently. `MAX_PAIRS=None` scans all discovered pairs. An integer requests a bounded and labelled scan.

SECOND-CC's native ZIP exceeds the connector's per-file upload limit. The Drive copy is a checksummed set of parts plus `SECOND-CC-AUG.zip.parts.json`. The storage layer reads it as one seekable original ZIP, verifies its publisher checksum and preserves native member paths. No manual reconstruction or extraction is needed.

## Local execution

Use Python 3.11 or newer.

```powershell
python -m pip install -r requirements-data.txt
python -m src.data.cli audit --output outputs/data
python -m src.data.cli prepare --audit-dir outputs/data --output data/processed
python -m src.data.cli verify --processed data/processed --audit-dir outputs/data
python scripts/run_data_notebooks.py
```

To scan a mounted collection, add `--data-root PATH` to `audit`. To retain native dimensions, pass `--image-size 0` to `prepare`; use task-specific size-compatible batches in that mode. Checkpoint-specific image normalisation and tokenisation remain part of the selected model processor.

For a bounded smoke check, `audit --max-pairs 20` and `verify --limit 20` record their limited scopes. Do not relabel them as full-release execution.

## Data and output locations

Raw archives and images stay in `data/raw/` or the mounted collection. Prepared targets, references and manifests stay under `data/processed/`. Detailed run records, profiles and validation are under `outputs/data/`; HTML inspection copies go to `outputs/notebook_previews/`. Large data/run files are ignored by Git. Executed notebooks retain bounded tables and useful figure outputs, including one illustrative attributed image pair for input inspection.

See [the training-data plan](../docs/data/Training_Data_Plan.md) for component-specific dataset choices and [the pipeline guide](../docs/data/Pipeline.md) for schemas, policies and script entry points.
