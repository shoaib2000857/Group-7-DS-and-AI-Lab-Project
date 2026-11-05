"""Collect source-backed aggregates without modifying original experiment exports."""
from pathlib import Path
import csv
import hashlib
import io
import json
from datetime import datetime
import re
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'POC/consolidation/evidence'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[];all_pairs=set()
    for path in sorted((ROOT/'POC/benchmarks/results/2026-10-27/shoaib/archives').glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
            mismatches=[]
            for entry in manifest['files']:
                data=z.read(entry['path'])
                if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:mismatches.append(entry['path'])
            assert z.testzip() is None and not mismatches
            config=json.loads(z.read('config.json'))
            health=json.loads(z.read('metadata/run_health.json')) if 'metadata/run_health.json' in z.namelist() else {'status':'legacy_setup_failure'}
            metrics=list(csv.DictReader(io.StringIO(z.read('tables/raw_choice_metrics.csv').decode())))
            shared=json.loads(z.read('shared_manifest.json'))
            pairs=set()
            def visit(obj):
                if isinstance(obj,dict):
                    if obj.get('pair_id'):pairs.add(obj['pair_id'])
                    for value in obj.values():visit(value)
                elif isinstance(obj,list):
                    for value in obj:visit(value)
            visit(shared);all_pairs.update(pairs)
            predictions={n:{'sha256':hashlib.sha256(z.read(n)).hexdigest(),
                            'rows':sum(bool(x.strip()) for x in z.read(n).splitlines())}
                         for n in z.namelist() if n.startswith('predictions/') and n.endswith('.jsonl')}
            runtimes={n:json.loads(z.read(n)) for n in z.namelist() if n.startswith('metadata/') and n.endswith('_runtime.json')}
            runs.append({'archive':path.relative_to(ROOT).as_posix(),'run_id':re.search(r'SSSM_\d{8}_\d{6}',path.name).group(),
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'zip_crc':'passed','export_hashes':'passed',
                         'profile':config['PROFILE'],'runtime':manifest.get('runtime_version'),
                         'pair_ids':sorted(pairs),'health':health,'predictions':predictions,'runtimes':runtimes,
                         'validation_proxy_metrics':[r for r in metrics if r['phase']=='val' and r['benchmark_group']=='satellite_annotation_proxy'],
                         'metric_groups':sorted({r['benchmark_group'] for r in metrics})})
    b={}
    for path in sorted((ROOT/'POC/execution/outputs').glob('*.json')):
        b[path.name]={'source':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':json.loads(path.read_text())}
    c={}
    for path in [ROOT/'POC/c-evidence/results/original/batch_evaluation_summary.json',ROOT/'POC/c-evidence/results/original/real_levir_evaluation_results.json',ROOT/'POC/c-evidence/results/original/stage_c_claim_decision_record.json']:
        data=path.read_bytes();c[path.name]={'source':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'records':json.loads(data)}
        target=ROOT/'POC/c-evidence/results/original'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
    result={'date':datetime.now().astimezone().date().isoformat(),'captioning':b,'visual_evidence':c,'decision_runs':runs,
            'decision_unique_pair_ids':sorted(all_pairs),'decision_archive_count':len(runs),
            'decision_distinct_run_ids':len({r['run_id'] for r in runs}),
            'scope':'Source exports are retained verbatim. Synthetic visual-evidence fixtures, automatic caption-reference checks and decision annotation proxies are separate evidence types.'}
    (OUT/'experiment_register.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Captioning aggregate files:',len(b),'visual evidence files:',len(c),'decision ZIPs:',len(runs),'unique manifest pair IDs:',len(all_pairs))


if __name__=='__main__':main()
