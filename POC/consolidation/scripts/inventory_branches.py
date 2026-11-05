"""Preserve unique POC content from every remote head and inventory provenance."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'POC/consolidation/evidence'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def tree(ref, prefix=None):
    args=['ls-tree','-r','-z',ref]
    if prefix:args+=['--',prefix]
    rows=[]
    for line in git(*args).split(b'\0'):
        if not line:continue
        metadata,name=line.split(b'\t',1);mode,kind,oid=metadata.decode().split()
        if kind=='blob':rows.append((name.decode('utf-8'),oid))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    refs=[r for r in git('for-each-ref','--format=%(refname:short)','refs/remotes/origin').decode().splitlines()
          if r not in {'origin','origin/HEAD','origin/poc-consolidation'}]
    known={oid for _,oid in tree('HEAD')};snapshots={};branches=[]
    for ref in refs:
        commit=git('rev-parse',ref).decode().strip();files=tree(ref,'POC')
        row={'ref':ref,'commit':commit,'poc_files':len(files),'new_snapshots':0,
             'commits_outside_main':int(git('rev-list','--count','origin/main..'+ref).decode())}
        for name,oid in files:
            if oid in known:continue
            content=git('cat-file','blob',oid)
            dest=ROOT/'POC/consolidation/archive/branch-heads'/ref.removeprefix('origin/').replace('/','__')/name
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
            snapshots[oid]={'source_ref':ref,'source_commit':commit,'original_path':name,
                            'snapshot':dest.relative_to(ROOT).as_posix(),'git_blob':oid,
                            'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
            known.add(oid);row['new_snapshots']+=1
        branches.append(row)
    result={'date':'2026-11-05','main_commit':git('rev-parse','origin/main').decode().strip(),
            'branches':branches,'additional_unique_poc_snapshots':list(snapshots.values()),
            'policy':'Preserve every unique POC blob at inspected remote heads; current implementations remain canonical. Historical commits remain in Git and the local all-branch bundle.'}
    (OUT/'branch_inventory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    baseline=ROOT/'.git/preservation/2026-11-05/local-files-before.json'
    if baseline.exists():(OUT/'local_preservation_inventory.json').write_bytes(baseline.read_bytes())
    print('Remote heads:',len(refs),'additional unique POC snapshots:',len(snapshots))


if __name__=='__main__':main()
