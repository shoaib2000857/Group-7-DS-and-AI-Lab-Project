"""Verify that consolidation retained every inventoried local file unchanged."""
from pathlib import Path
import hashlib
import json
from datetime import datetime

ROOT=Path(__file__).resolve().parents[3]


def main():
    evidence=ROOT/'POC/consolidation/evidence'
    baseline=json.loads((evidence/'local_preservation_inventory.json').read_text())
    problems=[]
    for row in baseline['files']:
        p=ROOT/row['path']
        if not p.is_file():problems.append({'path':row['path'],'problem':'missing'});continue
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=row['sha256']:problems.append({'path':row['path'],'problem':'hash changed'})
    branch=json.loads((evidence/'branch_inventory.json').read_text())
    for row in branch['additional_unique_poc_snapshots']:
        p=ROOT/row['snapshot']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:problems.append({'path':row['snapshot'],'problem':'snapshot missing/changed'})
    conflict_manifest=ROOT/'POC/consolidation/archive/2026-11-05/manifest.json'
    conflicts=json.loads(conflict_manifest.read_text())['files']
    for row in conflicts:
        p=ROOT/row['snapshot']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:problems.append({'path':row['snapshot'],'problem':'conflicting version missing/changed'})
    additional_versions=0
    for manifest in (ROOT/'POC/consolidation/archive/2026-11-06').rglob('manifest.json'):
        entries=json.loads(manifest.read_text())
        entries=entries.get('files',[]) if isinstance(entries,dict) else entries
        for row in entries:
            relative=row.get('snapshot',row.get('path'))
            if relative and row.get('sha256'):
                p=ROOT/relative;additional_versions+=1
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:problems.append({'path':relative,'problem':'cleanup snapshot missing/changed'})
    result={'date':datetime.now().astimezone().date().isoformat(),'inventoried_local_files':len(baseline['files']),
            'inventoried_local_bytes':sum(r['bytes'] for r in baseline['files']),
            'additional_branch_snapshots':len(branch['additional_unique_poc_snapshots']),'conflicting_versions_checked':len(conflicts),'cleanup_versions_checked':additional_versions,
            'problems':problems,'status':'passed' if not problems else 'failed',
            'local_git_bundle':'.git/preservation/2026-11-05/all-branches.bundle'}
    (evidence/'preservation_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return bool(problems)


if __name__=='__main__':raise SystemExit(main())
