"""Read-only hardware/package probe; works without installing ML dependencies.

python probe_environment.py --output path/to/environment.json
It does not access tokens, install packages, allocate a model or start GPU work.
"""
import argparse
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {'exit_code': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def probe():
    result = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'nvidia_smi': command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                     '--format=csv,noheader']), 'packages': {}}
    for package in ['torch', 'torchvision', 'transformers', 'accelerate', 'bitsandbytes',
                    'mmcv', 'mmengine', 'mmsegmentation', 'mamba-ssm', 'peft', 'ollama']:
        try:
            result['packages'][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result['packages'][package] = None
    result['ollama_cli'] = command(['ollama', '--version'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(probe(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
