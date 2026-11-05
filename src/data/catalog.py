"""Dataset catalogue, root discovery and explicit source/label contracts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_catalog(path: str | Path | None = None) -> dict:
    with Path(path or ROOT / 'configs/datasets.yaml').open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_source(dataset: dict, data_root: str | Path | None = None) -> Path | None:
    """Prefer an explicit override, then project raw data, then mounted Drive."""
    override = os.getenv('TERRAWATCH_' + dataset['id'].upper())
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f'Explicit source override does not exist: {p}')
        return p
    roots = [Path(data_root)] if data_root else []
    roots += [ROOT / 'data/raw', Path('/content/drive/MyDrive/TerraWatch_Extracted')]
    for root in roots:
        for relative in dataset.get('paths', []):
            p = root / relative
            if p.exists():
                return p.resolve()
    return None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=True) + '\n' for r in rows), encoding='utf-8')
