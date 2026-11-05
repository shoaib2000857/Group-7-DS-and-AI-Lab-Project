"""Publish compact run evidence without copying source imagery or large profiles."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(relative):
        return json.loads((ROOT / relative).read_text(encoding='utf-8'))
    audit = read('outputs/data/audit_summary.json')
    for source in audit.values():
        if isinstance(source, dict) and 'source_path' in source:
            source['source_path'] = '<local source; resolved by catalogue>'
    code = sorted((ROOT/'src/data').glob('*.py')) + sorted((ROOT/'scripts').glob('*.py')) + [ROOT/'configs/datasets.yaml', ROOT/'requirements-data.txt']
    result = {
        'executed_date': datetime.fromtimestamp((ROOT/'outputs/data/audit_summary.json').stat().st_mtime).astimezone().date().isoformat(),
        'scope_note': 'Complete discovered original MCI and SECOND-CC pairs; three convenience image probes; all published ChangeChat JSON variants (overlapping, not additive).',
        'audit': audit,
        'preparation': read('outputs/data/preparation_summary.json'),
        'loader_validation': read('outputs/data/loader_validation.json'),
        'notebooks': read('docs/data/notebook_validation.json'),
        'environment': {'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','Pillow','matplotlib','nbclient','nbformat']}},
        'implementation_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
    }
    (ROOT/'docs/data/run_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
