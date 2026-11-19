"""Score new predictions against prepared targets; references never enter inference."""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage
from src.models.claims import extract_claims,to_contract
from src.evaluation.claim_metrics import check_claim,summarise,CHANGE_ACTIONS

class PreparedTruth:
    def __init__(self,dataset,in_scope=None):
        self.dataset=dataset;self.in_scope=in_scope or {'road':1,'building':2}
        self.index={row['pair_id']:i for i,row in enumerate(dataset.rows)}
        self.refs={pid:dict(dataset.refs[pid],changeflag=dataset.refs[pid].get('source_changeflag',dataset.refs[pid].get('changeflag'))) for pid in self.index}
        if any(value['changeflag'] is None for value in self.refs.values()):raise ValueError('Evaluation requires explicit source change flags')
        self.ref_claims={pid:[claim for caption in self.refs[pid]['captions'] for claim in extract_claims(caption)] for pid in self.index}
        self.cache={}
    def mask(self,pid):
        if pid not in self.cache:self.cache[pid]=self.dataset[self.index[pid]]['targets']['target']
        return self.cache[pid]
    def class_changed(self,pid,cls):return int((self.mask(pid)==self.in_scope[cls]).sum())>=50
    def any_change(self,pid):return any(self.class_changed(pid,cls) for cls in self.in_scope)
    def conflict(self,pid):return self.any_change(pid)!=bool(self.refs[pid]['changeflag'])
    def cells(self,pid,cls):
        mask=self.mask(pid)==self.in_scope[cls];h,w=mask.shape
        return {(r,c) for r in range(3) for c in range(3) if mask[r*h//3:(r+1)*h//3,c*w//3:(c+1)*w//3].mean()>=.02}
    def components(self,pid,cls):
        mask=self.mask(pid)==self.in_scope[cls];labels,n=ndimage.label(mask)
        return int(sum(ndimage.sum(mask,labels,range(1,n+1))>=50)) if n else 0


def text_metrics(rows,references):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tokenizer=PTBTokenizer()
    truth={r['pair_id']:[{'caption':s} for s in references[r['pair_id']]['captions']] for r in rows}
    predicted={r['pair_id']:[{'caption':r['raw_text'] or ''}] for r in rows}
    truth,predicted=tokenizer.tokenize(truth),tokenizer.tokenize(predicted)
    bleu,_=Bleu(4).compute_score(truth,predicted,verbose=0)
    return {**{f'BLEU-{i+1}':float(v) for i,v in enumerate(bleu)},
            'ROUGE-L':float(Rouge().compute_score(truth,predicted)[0]),
            'CIDEr-D':float(Cider().compute_score(truth,predicted)[0])}

def evaluate_predictions(rows, T, output, text_scores=False):
    roles=None;tag=""
    IN_SCOPE=T.in_scope
    if not rows:raise ValueError("Predictions must contain at least one row")
    if len({r["pair_id"] for r in rows})!=len(rows):raise ValueError("Duplicate prediction pair IDs")
    for row in rows:
        if row["pair_id"] not in T.index:raise ValueError("Prediction pair is absent from selected evaluation manifest")
    run_dir=Path(output);run_dir.mkdir(parents=True,exist_ok=False)
    run_dir = Path(run_dir)
    ok = [r for r in rows if r["status"] == "ok"]
    claims_out, per_pair = [], []
    from collections import Counter
    for r in rows:
        pid = r["pair_id"]
        cs = extract_claims(r.get("raw_text") or "", pid, r["run_id"])
        labels = Counter()
        n_inscope = asserts = says_nc = loc_c = loc_ok = q_c = q_ok = 0
        covered = set()
        for c in cs:
            chk = check_claim(c, pid, T)
            labels[chk["label"]] += 1
            obj, act = c["claim"]["object"], c["claim"]["action"]
            if act in CHANGE_ACTIONS and (obj in IN_SCOPE or obj == "region") and not c["parser"]["negated"]:
                asserts = 1
            if act == "no_change" and obj in ("scene", "building", "road"):
                says_nc = 1
            if obj in IN_SCOPE or obj in ("scene", "region"):
                n_inscope += chk["label"] != "abstention"
            if chk["label"] == "supported":
                if act in CHANGE_ACTIONS and obj == "region":
                    covered |= {c2 for c2 in IN_SCOPE if T.class_changed(pid, c2)}
                elif act in CHANGE_ACTIONS:
                    covered.add(obj)
                elif act == "no_change":
                    covered.add("no_change")
            if chk["location_check"] is not None:
                loc_c += 1
                loc_ok += chk["location_check"]
            if chk["quantity_check"]:
                q_c += 1
                q_ok += chk["quantity_check"]["within_tolerance"]
            rec = to_contract(c)
            rec["b_reference_check"] = chk
            claims_out.append(rec)
        truth_change = T.any_change(pid)
        facts = {c for c in IN_SCOPE if T.class_changed(pid, c)} or {"no_change"}
        conflict = T.conflict(pid)
        n_factual = sum(v for k, v in labels.items() if k != "abstention")
        per_pair.append({
            "pair_id": pid, "n_claims": len(cs), "labels": labels,
            "n_supported": labels["supported"], "n_factual": n_factual, "n_inscope": n_inscope,
            "n_facts": 0 if conflict else len(facts),
            "facts_covered": 0 if conflict else len(facts & covered),
            "truth_change": truth_change, "conflict": conflict,
            "asserts_change": asserts, "says_no_change": says_nc,
            "loc_checked": loc_c, "loc_ok": loc_ok, "q_checked": q_c, "q_ok": q_ok,
        })
    with open(run_dir / f"claims{tag}.jsonl", "w") as f:
        for c in claims_out:
            f.write(json.dumps(c) + "\n")
    with open(run_dir / f"per_pair{tag}.jsonl", "w") as f:
        for p in per_pair:
            f.write(json.dumps({**p, "labels": dict(p["labels"])}) + "\n")
    summary = summarise(per_pair)
    summary['text_metrics']=text_metrics(ok,T.refs) if text_scores and ok else None
    summary['completion_rate']=len(ok)/len(rows) if rows else None
    summary['evaluator']={'reference_check':'automatic annotation proxy; oracle targets and reference captions',
        'mask_scope':T.in_scope,'mask_pixel_threshold':50,'target_resolution':'prepared manifest size',
        'bootstrap':'pair-level, 2000 resamples, seed 0','parser':'rules-v3'}
    (run_dir/'eval_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

