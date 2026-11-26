"""Fit calibration for the synthetic evidence environment; no treatment comparison."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from medical.calculation import Action, FIELDS, State
from medical.evidence import decision_states, eligibility, evidence_step, observe, read_evidence_panel, synthetic_world

RULE = ("Return one tool command. Query and bind each required field in listed order. Use records "
        "from the current encounter, within the inclusive time window, with a documented supported unit. "
        "Correct an ineligible binding before calculating. CALCULATE then SUBMIT the displayed result. "
        "Commands: QUERY <field>, BIND <field> <pointer>, CALCULATE, SUBMIT.")


def demonstrations():
    rows = []
    for kind in ("stale","wrong_encounter","ambiguous_unit"):
        for case in range(8):
            w,_ = synthetic_world(kind, case)
            s = State(w.budget)
            for field in FIELDS[w.calculator]:
                rows.append((observe(w,s), f"QUERY {field}"))
                s,_,_ = evidence_step(w,s,Action("query", field))
                good, = [r for r in w.records if r.field == field and all(eligibility(w,r).values())]
                target = f"BIND {field} {good.pointer}"
                rows.append((observe(w,s),target))
                s,_,_ = evidence_step(w,s,Action("bind",field,good.pointer))
            bad, = [r for r in w.records if not all(eligibility(w,r).values())]
            matched,_,_ = evidence_step(w,s,Action("bind",bad.field,dict(s.bindings)[bad.field]))
            rows.append((observe(w,matched), "CALCULATE"))
            displaced,_,_ = evidence_step(w,s,Action("bind",bad.field,bad.pointer))
            rows.append((observe(w,displaced), f"BIND {bad.field} {dict(s.bindings)[bad.field]}"))
            rows.append((observe(w,s), "CALCULATE"))
            s,_,_ = evidence_step(w,s,Action("calculate"))
            rows.append((observe(w,s), "SUBMIT"))
    return rows


def panel_demonstrations(directory):
    strata = defaultdict(list)
    for world, key in read_evidence_panel(directory / "train.jsonl.gz"):
        group = (world.calculator, key["focus"], tuple(key["faults"]), len(world.records), key["equal"])
        strata[group].append((world, key))
    rng = Random(413)
    selected = [rng.choice(strata[group]) for group in sorted(strata)]
    rows = [(obs, target) for world, key in selected for _, obs, target in decision_states(world, key)]
    return rows, [key["case"] for _, key in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--training-panel", type=Path, help="Use one train world per v2 design stratum for fit calibration")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    examples = []
    if args.training_panel:
        rows, cases = panel_demonstrations(args.training_panel)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"train_cases.json").write_text(json.dumps(cases)+"\n")
        print(json.dumps(dict(training_worlds=len(cases), commands=len(rows))), flush=True)
    else:
        rows = demonstrations()
    for observation, command in rows:
        prefix = tok.apply_chat_template([dict(role="system",content=RULE),
                                          dict(role="user",content=observation)],
                                         tokenize=True,add_generation_prompt=True)
        target = tok.encode(command,add_special_tokens=False) + [tok.eos_token_id]
        examples.append((prefix,target))
    for seed in args.seeds:
        torch.manual_seed(seed)
        base = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",local_files_only=True).to("cuda")
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,
                    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]))
        optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
        def batch_loss(ids):
            seqs = [examples[i] for i in ids]
            length = max(len(p)+len(t) for p,t in seqs)
            width = max(len(t) for _,t in seqs)
            inputs = torch.tensor([[tok.pad_token_id]*(length-len(p)-len(t))+p+t for p,t in seqs],device="cuda")
            mask = torch.tensor([[0]*(length-len(p)-len(t))+[1]*(len(p)+len(t)) for p,t in seqs],device="cuda")
            labels = torch.tensor([[-100]*(width-len(t))+t for _,t in seqs],device="cuda")
            logits = model(input_ids=inputs,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp_min(0),
                           logits_to_keep=width+1,use_cache=False).logits[:,:-1].float()
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)), labels.flatten(),
                                                     ignore_index=-100,reduction="sum")
            exact = ((logits.argmax(-1)==labels)|(labels==-100)).all(-1).sum().item()
            return nll, (labels!=-100).sum().item(), exact
        batches = (len(examples)+7)//8
        for epoch in range(1,25):
            order=list(range(len(examples)))
            Random(10007*seed+epoch).shuffle(order)
            model.train()
            for offset in range(0,len(order),8):
                update=(epoch-1)*batches+offset//8+1
                optim.param_groups[0]["lr"]=3e-4*min(1,update/(2*batches))
                optim.zero_grad(set_to_none=True)
                nll,n,_=batch_loss(order[offset:offset+8])
                (nll/n).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                optim.step()
            model.eval()
            with torch.inference_mode():
                total_loss=tokens=correct=0
                for i in range(0,len(order),8):
                    nll,n,exact=batch_loss(range(i,min(i+8,len(order))))
                    total_loss+=nll.item()
                    tokens+=n
                    correct+=exact
            score=dict(seed=seed,epoch=epoch,correct=correct,n=len(order),token_nll=total_loss/tokens)
            print(json.dumps(score),flush=True)
            if epoch>=10 and correct/len(order)>=.99:
                break
        folder=args.output/str(seed)
        folder.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(folder/"adapter")
        (folder/"fit.json").write_text(json.dumps(score)+"\n")
        del optim,model,base


if __name__ == "__main__":
    main()
