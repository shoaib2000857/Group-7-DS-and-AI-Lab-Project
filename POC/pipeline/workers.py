"""Real model workers. Each runs in its model's own Python environment."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def build_detector(spec,device):
    import torch
    source=Path(spec['source_dir']).resolve()
    if not (source/'models/networks.py').is_file():raise FileNotFoundError('BIT source_dir must contain the official models/networks.py')
    sys.path.insert(0,str(source))
    from models.networks import BASE_Transformer
    model=BASE_Transformer(input_nc=3,output_nc=3,with_pos='learned',resnet_stages_num=4,
                           token_len=4,enc_depth=1,dec_depth=8,decoder_dim_head=8,backbone='resnet18')
    return model.to(device)


def detect(req,out):
    import torch
    import numpy as np
    from PIL import Image
    spec=req['spec'];device=req['device'];model=build_detector(spec,device)
    state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(state.get('model',state),strict=True);model.eval()
    tensors=[]
    for path in req['images']:
        image=Image.open(path).convert('RGB').resize((spec['image_size'],spec['image_size']),Image.Resampling.BILINEAR)
        tensors.append(torch.from_numpy(np.array(image).transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/127.5-1)
    with torch.inference_mode():prediction=model(*tensors).argmax(1)[0].cpu().numpy().astype('uint8')
    shape=Image.open(req['images'][0]).size
    target=out.with_suffix('.mask.png');Image.fromarray(prediction).resize(shape,Image.Resampling.NEAREST).save(target)
    return {'mask_path':str(target.resolve()),'model':'BIT-MCI-3class','checkpoint':spec['checkpoint'],'source_kind':'model_prediction'}


def caption(req,out):
    import torch
    from PIL import Image
    spec=req['spec'];device=req['device']
    sys.path.insert(0,str(ROOT/'POC/execution/scripts'))
    import caption_rscama as helper
    helper.install_patches(device)
    checkpoint=Path(spec['checkpoint']).resolve();vocab_path=Path(spec['vocab']).resolve() if spec.get('vocab') else None
    source=Path(spec['source_dir']).resolve();sys.path.insert(0,str(source));os.chdir(source)
    import clip
    from model.model_encoder_attMamba import Encoder,AttentiveEncoder
    from model.model_decoder import DecoderTransformer
    ckpt=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    vocab=ckpt.get('vocab')
    if vocab is None:
        if not spec.get('vocab'):raise ValueError('Released caption checkpoint requires its matching vocabulary JSON')
        vocab=json.loads(vocab_path.read_text())
    _,processor=clip.load('ViT-B/32',device='cpu',jit=False)
    enc=Encoder('CLIP-ViT-B/32');fusion=AttentiveEncoder(n_layers=3,feature_size=[7,7,768],heads=8,dropout=.1)
    dec=DecoderTransformer(decoder_type='transformer_decoder',embed_dim=768,vocab_size=len(vocab),max_lengths=42,
                           word_vocab=vocab,n_head=8,n_layers=1,dropout=.1)
    for model,key in [(enc,'encoder_dict'),(fusion,'encoder_trans_dict'),(dec,'decoder_dict')]:
        model.load_state_dict(ckpt[key],strict=True);model.eval().to(device)
    images=[processor(Image.open(p).convert('RGB')).unsqueeze(0).to(device) for p in req['images']]
    with torch.inference_mode():
        a,b=enc(*images);tokens=dec.sample(fusion(a,b),k=1)
    if hasattr(tokens,'tolist'):tokens=tokens.tolist()
    while tokens and isinstance(tokens[0],list):tokens=tokens[0]
    inv={int(v):k for k,v in vocab.items()};skip={'<START>','<END>','<NULL>'}
    text=' '.join(inv[int(t)] for t in tokens if inv[int(t)] not in skip)
    return {'caption':text,'model':'RSCaMa','checkpoint':spec['checkpoint'],'source_kind':'model_prediction'}


def judge(req,out):
    sys.path.insert(0,str(ROOT/'POC/benchmarks/colab'))
    import benchmark_runtime as rt
    config=rt.default_config();config['SEED']=req['seed'];model=rt.HFJudge(req['spec']['model_id'],req['spec']['revision'],config)
    decisions=[]
    for case in req['cases']:
        paths=req['images'] if req['spec']['model_id']=='J06' else []
        outcome,probabilities,raw,semantics,usage=model.decide(case['state'],paths,False)
        decisions.append({'claim_id':case['claim_id'],'outcome':outcome,'probabilities':probabilities,
                          'probability_semantics':semantics,'raw':raw,'usage':usage})
    return {'decisions':decisions,'model':model.metadata,'source_kind':'model_prediction'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['detector','captioner','referee']);ap.add_argument('request');ap.add_argument('response');a=ap.parse_args()
    request=json.loads(Path(a.request).read_text());out=Path(a.response).resolve()
    torch_seed=request['seed']
    import torch
    torch.manual_seed(torch_seed)
    result={'detector':detect,'captioner':caption,'referee':judge}[a.stage](request,out)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
