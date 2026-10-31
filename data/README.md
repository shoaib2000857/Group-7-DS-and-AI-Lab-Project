# Main-project datasets and preparation

Large source archives, prepared targets and references stay out of Git. The main analysis notebooks are [01_EDA](../notebooks/01_EDA.ipynb), [02_Preprocessing](../notebooks/02_Preprocessing.ipynb) and [data-quality experiments](../notebooks/03_Data_Quality_Experiments.ipynb). Their saved outputs show the execution scope.

## Storage

- `raw/`: immutable acquired sources or a local mirror of mounted Drive data.
- `processed/`: transformed targets, fixed train/val/test records, separate references and component task manifests.
- `sample/`: documentation for temporary synthetic test fixtures; not a source of claimed EDA results.
- `outputs/data/` at repository root: detailed profiles, diagnostics, source coverage, exclusion policy and validation records for a run.

## Configure and run

Install `requirements-data.txt`. Dataset roles, supported folder layouts, supervision coverage, provenance and training eligibility are in `configs/datasets.yaml`. Source roots can be supplied with `--data-root` or `TERRAWATCH_<DATASET_ID>` environment variables. User-provided files remain unchanged.

```powershell
python -m src.data.cli audit --output outputs/data
python -m src.data.cli prepare --audit-dir outputs/data --output data/processed
python -m src.data.cli verify --processed data/processed --audit-dir outputs/data
python scripts/run_data_notebooks.py
```

To run on the shared collection in Colab, mount Google Drive and set `DATA_ROOT=/content/drive/MyDrive/TerraWatch_Extracted`. The notebooks include setup/mount cells; see [notebook instructions](../notebooks/README.md).

## Prepared records

One pair has an ordered T1/T2 image input, source-relative members, a verified local storage root, source hashes, explicit target kind, recorded dimensions/scale and source partition. Target classes are dataset-specific. Inference records exclude source masks, captions and change flags. The reference store is a deliberate, separate training/evaluation input.

Detector inputs are float32 `[B,2,3,H,W]` in [0,1]. Binary/multiclass targets are class indices `[B,H,W]`; semantic tasks return T1/T2 semantic targets separately. A VLM receives uint8 images and its own checkpoint processor. No object counts, dates, geographic positions or legality labels are invented.

## Sources and training roles

LEVIR-MCI supplies the shared building/road/caption foundation. LEVIR-CC shares its imagery. LEVIR-CD/CD+ and S2Looking support building tasks; SYSU supports broader change pretraining; SECOND/SECOND-CC support semantic changes, and SECOND-CC also provides captions. ChangeChat supplies instruction variants linked back to MCI pairs. The WHU caption mirror is excluded from fitting because derivative terms are not established. HLS is contextual imagery requiring an area/date/product choice, not a substitute building-change target.

See the [training-data plan](../docs/data/Training_Data_Plan.md) and [pipeline schema/policies](../docs/data/Pipeline.md). Additional datasets are not blindly pooled: source semantics, related imagery and fixed evaluation populations remain explicit.

## Acquisition and Drive archives

`src.data.acquire.download()` retrieves the pinned LEVIR-MCI or author-linked SECOND-CC archive into a requested local/mounted raw directory, verifies size and publisher checksums, and preserves unsuccessful transfers as `.part`. It does not overwrite an existing source unless requested.

The added SECOND-CC Drive folder uses transfer-sized parts and a `.zip.parts.json` manifest because the connector has per-file transfer limits. `Store` reads these as one original archive, without manual extraction or duplicate reconstructed storage. The virtual archive is checked against the publisher checksum before audit. Native image and annotation member paths are preserved.
