"""Inspect configured assets before launching independent model environments."""
from pathlib import Path
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def inspect_config(config, root=ROOT, environments=False):
    root = Path(root); checks = []
    for stage in ['detector','captioner','referee']:
        spec = config.get(stage, {})
        python = spec.get('python', '')
        executable = shutil.which(python) or (str(root/python) if python and (root/python).is_file() else None)
        checks.append({'stage':stage,'item':'python','value':python,'ready':bool(executable)})
        for key in (['source_dir','checkpoint'] if stage!='referee' else ['revision']):
            value = spec.get(key)
            ready = bool(value) and (key=='revision' or (root/value).exists())
            if ready and key=='source_dir':
                expected='models/networks.py' if stage=='detector' else 'model/model_encoder_attMamba.py'
                ready=(root/value/expected).is_file()
            checks.append({'stage':stage,'item':key,'value':value,'ready':ready})
        if stage=='captioner' and spec.get('vocab'):
            checks.append({'stage':stage,'item':'vocab','value':spec['vocab'],'ready':(root/spec['vocab']).is_file()})
        if stage=='referee':
            threshold=spec.get('accept_threshold')
            checks.append({'stage':stage,'item':'accept_threshold','value':threshold,
                'ready':isinstance(threshold,(int,float)) and 0<=threshold<=1})
        if environments and executable:
            packages = ['torch','torchvision','einops'] if stage=='detector' else ['torch','clip','transformers'] if stage=='captioner' else ['torch','transformers','accelerate']
            command = 'import importlib.util,json; print(json.dumps({p:importlib.util.find_spec(p) is not None for p in '+repr(packages)+'}))'
            try:
                result=subprocess.run([executable,'-c',command],capture_output=True,text=True,timeout=30,check=True)
                found=json.loads(result.stdout)
                checks.append({'stage':stage,'item':'packages','value':found,'ready':all(found.values())})
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                checks.append({'stage':stage,'item':'packages','value':str(error),'ready':False})
    return {'ready':all(row['ready'] for row in checks),'checks':checks,
            'scope':'asset/environment preflight; use a learned smoke run to validate checkpoint execution'}
