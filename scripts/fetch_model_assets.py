"""Download the pinned released RSCaMa checkpoint on explicit request."""
from pathlib import Path
import argparse,hashlib,json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rscama',action='store_true',required=True)
    parser.add_argument('--output',default='models/checkpoints/rscama.pth')
    parser.add_argument('--source-dir',default='models/external/RSCaMa')
    parser.add_argument('--captions',help='Official LevirCCcaptions.json, needed only to reproduce released-checkpoint vocabulary')
    args=parser.parse_args()
    from huggingface_hub import hf_hub_download
    import shutil,sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    path=Path(hf_hub_download('lcybuaa/RSCaMa','RSCaMa.pth',revision='3e527e6656b62cc615eab1d2f3fb73777cbbf62c'))
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    expected='5eeba271fb7c2ff4722b725cbf5ce1a1fdc0a33ddf1b077dcf5381aebd643449'
    if digest!=expected:raise ValueError('Released checkpoint SHA-256 mismatch')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    shutil.copy2(path,out)
    if args.captions:
        from src.training.captioner import upstream_tokenizer
        tokenize,build_vocab,_=upstream_tokenizer(args.source_dir)
        raw=json.loads(Path(args.captions).read_text())['images']
        vocabulary=build_vocab([(r['filename'],[tokenize(s['raw']) for s in r['sentences']]) for r in raw],5)
        vocab=out.with_name('rscama_vocab.json')
        with vocab.open('x',encoding='utf-8') as stream:json.dump(vocabulary,stream,indent=2)
    print(json.dumps({'checkpoint':str(out),'sha256':digest,'vocabulary_scope':'upstream official splits, for released-checkpoint reproduction only'}))

if __name__=='__main__':main()
