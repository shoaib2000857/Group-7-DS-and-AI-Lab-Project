"""Acquire pinned public model code without resetting an existing checkout."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SOURCES=[('https://github.com/justchenhao/BIT_CD.git','models/external/BIT_CD','adcd7aea6f234586ffffdd4e9959404f96271711'),
         ('https://github.com/Chen-Yang-Liu/RSCaMa.git','models/external/RSCaMa','aa68e4c6933c7dd0f2e2f9cbd7642d61929c86a5')]


def main():
    records=[]
    for url,relative,revision in SOURCES:
        target=ROOT/relative
        if target.exists():
            actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
            if actual!=revision:raise ValueError(f'Existing checkout preserved at {target}: expected {revision}, found {actual}')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','--filter=blob:none','--no-checkout',url,str(target)],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--detach',revision],check=True)
        records.append({'url':url,'path':relative,'revision':revision})
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
