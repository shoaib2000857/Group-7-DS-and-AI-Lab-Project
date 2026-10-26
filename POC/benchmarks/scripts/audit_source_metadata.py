"""Fetch public source metadata only, never model weights or dataset imagery.

python audit_source_metadata.py --output ../source_metadata.json
No credentials are read. Missing/inaccessible sources are recorded as errors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

MODELS = [
    'likyoo/Open-CD_Model_Zoo', 'lcybuaa/RSCaMa', 'lcybuaa/Change-Agent',
    'jirvin16/TEOChat', 'Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct',
    'google/gemma-3-4b-it', 'OpenGVLab/InternVL3_5-4B',
    'HuggingFaceTB/SmolVLM2-2.2B-Instruct', 'openbmb/MiniCPM-V-4_5',
    'jaredpalmer/kev-4b', 'vllm-sr/d3-mini', 'Qwen/Qwen3-4B-Instruct-2507',
]
REPOS = [
    'likyoo/open-cd', 'ChenHongruixuan/ChangeMamba', 'ShizhenChang/Chg2Cap',
    'Chen-Yang-Liu/RSCaMa', 'Chen-Yang-Liu/Change-Agent', 'ermongroup/TEOChat',
]


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'TerraWatch-POC-source-audit'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def model(mid):
    url = 'https://huggingface.co/api/models/' + mid
    try:
        data = fetch(url)
        files = [item['rfilename'] for item in data.get('siblings', [])]
        weights = [p for p in files if p.endswith(('.pth', '.pt', '.bin', '.safetensors'))]
        return {'id': mid, 'metadata_url': url, 'status': 'metadata_retrieved',
                'revision': data.get('sha'), 'gated': data.get('gated'),
                'license_metadata': data.get('cardData', {}).get('license'),
                'weight_files': weights, 'all_file_count': len(files),
                'local_inference_verified': False}
    except Exception as exc:
        return {'id': mid, 'metadata_url': url, 'status': 'metadata_error', 'error': str(exc)}


def repo(name):
    base = 'https://api.github.com/repos/' + name
    try:
        commits = fetch(base + '/commits?per_page=1')
        sha = commits[0]['sha']
        record = {'id': name, 'url': 'https://github.com/' + name,
                  'revision': sha, 'status': 'metadata_retrieved'}
        if name == 'likyoo/open-cd':
            tree = fetch(base + '/git/trees/' + sha + '?recursive=1')
            record['config_files'] = [x['path'] for x in tree.get('tree', [])
                                      if x['path'].startswith('configs/') and x['path'].endswith('.py')]
            record['tree_truncated'] = tree.get('truncated')
        return record
    except Exception as exc:
        return {'id': name, 'status': 'metadata_error', 'error': str(exc)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=6) as executor:
        models = list(executor.map(model, MODELS))
        repos = list(executor.map(repo, REPOS))
    result = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Public metadata only; no model weights, imagery or paid requests.',
              'models': models, 'code_repositories': repos}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in models + repos:
        print(row['id'], row['status'])
