"""Verified downloads into local raw storage or a mounted Drive destination."""
from pathlib import Path
import hashlib
import json
import requests
from .catalog import ROOT

SOURCES={
 'levir_mci':{'url':'https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/resolve/dff09c709999b5b1f1f1806024811c65a6b14dfe/LEVIR-MCI-dataset.zip',
              'name':'levir_mci.zip','sha256':'8f7d52298fa3ca32aec983e84addef80f61ad7d981c0ff97a92c5880c4b83e9d','size':2771943190},
 'second_cc':{'url':'https://zenodo.org/records/16937571/files/SECOND-CC-AUG.zip?download=1','name':'SECOND-CC-AUG.zip','size':2539187782,'md5':'ca930ddb819d68a797938b940d1711f1'},
}


def download(dataset,destination,overwrite=False):
    spec=SOURCES[dataset];destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);target=destination/spec['name']
    if target.exists() and not overwrite:
        if target.stat().st_size!=spec['size']:raise ValueError('Existing source size differs; preserve and inspect it')
        for algorithm in ('sha256','md5'):
            if spec.get(algorithm):
                with target.open('rb') as f:h=hashlib.file_digest(f,algorithm).hexdigest()
                if h!=spec[algorithm]:raise ValueError('Existing source checksum differs')
        return target
    part=target.with_suffix(target.suffix+'.part')
    with requests.get(spec['url'],stream=True,timeout=(30,90)) as response:
        response.raise_for_status()
        with part.open('wb') as f:
            for chunk in response.iter_content(4<<20):f.write(chunk)
    if part.stat().st_size!=spec['size']:raise ValueError('Incomplete source download retained as .part')
    with part.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    if spec.get('sha256') and h!=spec['sha256']:raise ValueError('Publisher SHA-256 mismatch')
    if spec.get('md5'):
        with part.open('rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
        if spec['md5']!=md5:raise ValueError('Publisher MD5 mismatch')
    part.replace(target)
    target.with_suffix(target.suffix+'.provenance.json').write_text(json.dumps({'dataset':dataset,'url':spec['url'],'sha256':h,'bytes':target.stat().st_size},indent=2),encoding='utf-8')
    return target
