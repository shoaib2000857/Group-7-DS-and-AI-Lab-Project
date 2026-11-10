"""Train RSCaMa on prepared MCI pairs with a training-only vocabulary."""
from pathlib import Path
import argparse
import ast
import json
import math
import random
from datetime import datetime, timezone
import numpy as np


def upstream_tokenizer(source_dir):
    source = (Path(source_dir)/'preprocess_data.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = [node for node in tree.body if
        (isinstance(node, ast.FunctionDef) and node.name in {'tokenize','build_vocab','encode'}) or
        (isinstance(node, ast.Assign) and any(getattr(x,'id','') == 'SPECIAL_TOKENS' for x in node.targets))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'upstream_tokenizer', 'exec'), namespace)
    def tokenize(caption):
        return namespace['tokenize'](caption, add_start_token=True, add_end_token=True,
            punct_to_keep=[';',','], punct_to_remove=['?','.'])
    return tokenize, namespace['build_vocab'], namespace['encode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', default='models/external/RSCaMa')
    parser.add_argument('--processed', default='data/processed')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vocab-threshold', type=int, default=5)
    parser.add_argument('--output')
    args = parser.parse_args()
    if min(args.epochs,args.batch,args.accum,args.patience,args.vocab_threshold) <= 0 or args.lr <= 0:
        parser.error('Training counts and learning rate must be positive')
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence
    from nltk.translate.bleu_score import corpus_bleu
    from PIL import Image
    from src.data.dataset import PairDataset
    from src.models.captioner import build_components
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root = Path(args.processed)
    train = PairDataset(root/'train.jsonl', root, include_targets=False, include_references=True)
    val = PairDataset(root/'val.jsonl', root, include_targets=False, include_references=True)
    try:
        for dataset in [train,val]:
            dataset.rows = [row for row in dataset.rows if row['dataset']=='levir_mci']
        if not train.rows or not val.rows: raise ValueError('MCI train and validation pairs are required')
        if {row['pair_id'] for row in train.rows} & {row['pair_id'] for row in val.rows}:
            raise ValueError('Train/validation pair IDs overlap')
        tokenize, build_vocab, encode = upstream_tokenizer(args.source_dir)
        sequences = [(row['pair_id'], [tokenize(caption) for caption in train.refs[row['pair_id']]['captions']]) for row in train.rows]
        if any(not captions for _,captions in sequences): raise ValueError('Every training pair requires reference captions')
        vocabulary = build_vocab(sequences, args.vocab_threshold)
        if set(vocabulary.values()) != set(range(len(vocabulary))): raise ValueError('Vocabulary indices must be contiguous')
        models, processor = build_components(args.source_dir, vocabulary, args.device)
        encoder, fusion, decoder = models; encoder.fine_tune(True)
        for model in models: model.to(args.device)
        optimizer = torch.optim.Adam([p for model in models for p in model.parameters() if p.requires_grad], lr=args.lr)
        inverse = {value:word for word,value in vocabulary.items()}
        skip = {vocabulary[word] for word in ['<START>','<END>','<NULL>']}
        output = Path(args.output or Path('outputs/captioner_training')/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        (output/'config.json').write_text(json.dumps(vars(args)|{'train_pairs':len(train),'val_pairs':len(val),
            'vocabulary_scope':'training pairs only','selection':'whitespace corpus BLEU-4 on fixed validation'}, indent=2))
        (output/'vocab.json').write_text(json.dumps(vocabulary, indent=2))
        def images(dataset, indices):
            samples = [dataset[int(index)] for index in indices]
            first = torch.stack([processor(Image.fromarray(sample['t1'])) for sample in samples]).to(args.device)
            second = torch.stack([processor(Image.fromarray(sample['t2'])) for sample in samples]).to(args.device)
            return samples, first, second
        best, stale = -1., 0
        for epoch in range(args.epochs):
            for model in models: model.train()
            order = np.random.default_rng(args.seed+epoch).permutation(len(train)); steps = math.ceil(len(order)/args.batch)
            optimizer.zero_grad(); loss_sum = 0.
            for step,start in enumerate(range(0,len(order),args.batch)):
                samples,first,second = images(train,order[start:start+args.batch]); encoded=[]; repeats=[]
                for sample in samples:
                    captions = train.refs[sample['pair_id']]['captions']; repeats.append(len(captions))
                    for caption in captions:
                        tokens = encode(tokenize(caption),vocabulary,allow_unk=True)[:42]
                        tokens[-1] = vocabulary['<END>']
                        encoded.append((tokens+[vocabulary['<NULL>']]*(42-len(tokens)),len(tokens)))
                token = torch.tensor([value for value,_ in encoded],device=args.device)
                lengths = torch.tensor([length for _,length in encoded],device=args.device)
                first,second = encoder(first,second)
                features = fusion(first,second).repeat_interleave(torch.tensor(repeats,device=args.device),dim=0)
                scores,sorted_tokens,decode_lengths,_ = decoder(features,token,lengths)
                prediction=pack_padded_sequence(scores,decode_lengths,batch_first=True).data
                target=pack_padded_sequence(sorted_tokens[:,1:],decode_lengths,batch_first=True).data
                loss=torch.nn.functional.cross_entropy(prediction,target.long());loss_sum+=float(loss.detach())
                divisor=min(args.accum,steps-(step//args.accum)*args.accum);(loss/divisor).backward()
                if (step+1)%args.accum==0 or step+1==steps:optimizer.step();optimizer.zero_grad()
            for model in models:model.eval()
            hypotheses,references=[],[]
            with torch.inference_mode():
                for index in range(len(val)):
                    samples,first,second=images(val,[index]);first,second=encoder(first,second)
                    tokens=decoder.sample(fusion(first,second),k=1)
                    if hasattr(tokens,'tolist'):tokens=tokens.tolist()
                    while tokens and isinstance(tokens[0],list):tokens=tokens[0]
                    hypotheses.append([inverse[int(token)] for token in tokens if int(token) not in skip])
                    references.append([tokenize(caption)[1:-1] for caption in val.refs[samples[0]['pair_id']]['captions']])
            score=float(corpus_bleu(references,hypotheses)); row={'epoch':epoch,'train_loss':loss_sum/steps,'val_bleu4':score}
            with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(row,flush=True)
            if score>best:
                best,stale=score,0; checkpoint=output/f'best_epoch_{epoch:03d}.pth'
                torch.save({'encoder_dict':encoder.state_dict(),'encoder_trans_dict':fusion.state_dict(),
                    'decoder_dict':decoder.state_dict(),'vocab':vocabulary,'epoch':epoch,'config':vars(args),
                    'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'val_bleu4':score},checkpoint)
                (output/'best.json').write_text(json.dumps({'checkpoint':str(checkpoint.resolve()),'val_bleu4':score},indent=2))
            else:stale+=1
            if stale>=args.patience:break
    finally:
        train.close();val.close()


if __name__=='__main__': main()
