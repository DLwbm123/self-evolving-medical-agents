"""Greedy paired action evaluation after all three saved training fits pass 99%."""
import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fit_evidence import RULE, demonstrations
from medical.calculation import Action, FIELDS, State
from medical.evidence import eligibility, evidence_step, observe, synthetic_world


def paired_states():
    known = {prompt for prompt,_ in demonstrations()}
    held = set()
    for equal in (False,True):
        for kind in ('stale','wrong_encounter','ambiguous_unit'):
            for case in range(1000,1020):
                w,_ = synthetic_world(kind,case,equal_value=equal)
                s=State(w.budget)
                for field in FIELDS[w.calculator]:
                    s,_,_=evidence_step(w,s,Action('query',field))
                    good,= [r for r in w.records if r.field==field and all(eligibility(w,r).values())]
                    s,_,_=evidence_step(w,s,Action('bind',field,good.pointer))
                bad,= [r for r in w.records if not all(eligibility(w,r).values())]
                correct=dict(s.bindings)[bad.field]
                for pointer, target in ((correct,'CALCULATE'),(bad.pointer,f'BIND {bad.field} {correct}')):
                    end,_,_=evidence_step(w,s,Action('bind',bad.field,pointer))
                    prompt=observe(w,end)
                    if prompt in known or prompt in held:
                        raise ValueError("overlapping evaluation prompt")
                    held.add(prompt)
                    yield equal,kind,case,pointer==correct,prompt,target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','runs','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    for seed in range(3):
        result=json.loads((args.runs/str(seed)/'fit.json').read_text())
        if result['correct']/result['n'] < .99:
            raise RuntimeError('training accuracy gate failed')
    rows=list(paired_states())
    tok=AutoTokenizer.from_pretrained(args.model,padding_side='left',local_files_only=True)
    with args.output.open('w',newline='') as out:
        writer=csv.writer(out,lineterminator='\n')
        writer.writerow(('seed','equal_value','kind','case','qualified','target','prediction','correct'))
        for seed in range(3):
            base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                            attn_implementation='sdpa',local_files_only=True).to('cuda')
            model=PeftModel.from_pretrained(base,args.runs/str(seed)/'adapter').eval()
            for j in range(0,len(rows),16):
                group=rows[j:j+16]
                prompts=[tok.apply_chat_template([dict(role='system',content=RULE),dict(role='user',content=r[4])],
                          tokenize=False,add_generation_prompt=True) for r in group]
                x=tok(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    ids=model.generate(**x,do_sample=False,max_new_tokens=32,pad_token_id=tok.pad_token_id,
                                       eos_token_id=tok.eos_token_id)[:,x.input_ids.shape[1]:]
                for r, tokens in zip(group,ids.tolist(),strict=True):
                    pred=tok.decode(tokens,skip_special_tokens=True)
                    writer.writerow((seed,*r[:4],r[5],pred,int(pred==r[5] and tok.eos_token_id in tokens)))
            del model,base


if __name__=='__main__':
    main()
