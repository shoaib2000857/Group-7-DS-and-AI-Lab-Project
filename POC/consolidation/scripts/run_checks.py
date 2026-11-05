"""Run CPU checks for canonical modules without collecting archived test variants."""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
from datetime import datetime,timezone
import sys
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[3]


def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'POC/benchmarks/colab'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/'outputs/data/consolidation_checks'/stamp;out.mkdir(exist_ok=False,parents=True)
    junit=out/'junit.xml'
    code=pytest.main(['tests','POC/execution/tests','POC/benchmarks/colab/test_benchmark_runtime.py',
                     '--import-mode=importlib','-q','-ra','--basetemp='+str(out/'tmp'),
                     '--junitxml='+str(junit)])
    suites=ET.parse(junit).getroot();cases=list(suites.iter('testcase'))
    summary={'date':'2026-11-05','python':platform.python_version(),'pytest_exit_code':int(code),
             'passed':sum(not any(c.tag in {'failure','error','skipped'} for c in case) for case in cases),
             'failures':sum(len(list(c.iter('failure'))) for c in cases),
             'errors':sum(len(list(c.iter('error'))) for c in cases),
             'skipped':[{'test':c.attrib.get('name'),'reason':c.find('skipped').attrib.get('message','')} for c in cases if c.find('skipped') is not None],
             'packages':{p:importlib.metadata.version(p) for p in ['numpy','Pillow','pydantic','opencv-python-headless','pytest']},
             'scope':'CPU logic, data contracts and software tests; no model inference or training.'}
    target=ROOT/'POC/consolidation/evidence/checks.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        attempts=target.parent/'check_attempts';attempts.mkdir(exist_ok=True)
        (attempts/(stamp+'_previous.json')).write_bytes(target.read_bytes())
    summary['junit_path']=junit.relative_to(ROOT).as_posix()
    target.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
    return int(code)


if __name__=='__main__':raise SystemExit(main())
