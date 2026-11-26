"""Teacher-forced and closed-loop greedy scoring of saved budget-ablation adapters."""
import argparse
import csv
import json
from pathlib import Path
from random import Random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sample_model_rollouts import (FIELDS, LETTERS, MENU_RULES, Rollout, State, collect,
                                   outcome, parse_action, synthetic_cases, teacher_examples, with_menu)


def split_example(prompt, target):
    state, menu = prompt.split("\nMenu:\n")
    commands = [line.split(". ", 1)[1] for line in menu.splitlines()]
    return state, commands, commands[target]


def productive(prompt, command):
    """Allow either field order when measuring progress toward a grounded answer."""
    s = json.loads(prompt.split("\nState: ")[1])
    fields = FIELDS[s['calculator']]
    seen = {pointer: field for pointer, field, _ in s['seen']}
    bound = dict(s['bindings'])
    act = parse_action(command)
    if act.kind == 'query':
        return act.field in fields and act.field not in seen.values()
    if act.kind == 'bind':
        return act.field not in bound and seen.get(act.value) == act.field
    if act.kind == 'calculate':
        return len(bound) == len(fields) and s['calculated'] == 'None'
    return act.kind == 'finish' and act.value == s['calculated']


def examples_for(mode, recovery):
    rows, cases = [], {}
    settings = dict(hide_budget=mode == 'hidden', independent_menu=mode != 'legacy')
    for panel in (11, 12, 13):
        group, train = synthetic_cases(panel, recovery, budget_mode=mode,
                                       independent_menu=settings['independent_menu'])
        if panel == 11:
            for index, example in enumerate(train):
                rows.append(('train', 0, str(index), 'none', *split_example(*example)))
        for case, (world, lo, hi) in group.items():
            cases[f'{panel}/{case}'] = (world, lo, hi)
            for step, example in enumerate(teacher_examples(world, **settings)):
                rows.append(('normal', panel, case, str(step + 1), *split_example(*example)))
            for idx, name in enumerate(('early_calculate', 'repeat_query', 'wrong_bind')):
                trajectory = list(teacher_examples(world, idx, **settings))
                rows.append(('off_path', panel, case, name, *split_example(*trajectory[idx])))
    assert len(rows) == 192 + 216 + 108
    return rows, cases


class Greedy:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        ids = [tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
        assert all(len(x) == 1 for x in ids)
        self.ids = [x[0] for x in ids]

    @torch.inference_mode()
    def __call__(self, prompts, menus):
        texts = [self.tokenizer.apply_chat_template(
            [dict(role='system', content=MENU_RULES), dict(role='user', content=with_menu(p, m))],
            tokenize=False, add_generation_prompt=True) for p, m in zip(prompts, menus, strict=True)]
        x = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
        pos = (x.attention_mask.cumsum(-1) - 1).clamp_min(0)
        logits = self.model(**x, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, self.ids]
        allowed = torch.arange(len(self.ids), device='cuda')[None, :] < torch.tensor(
            [len(m) for m in menus], device='cuda')[:, None]
        assert torch.isfinite(logits[allowed]).all()
        indices = logits.masked_fill(~allowed, float('-inf')).argmax(-1).tolist()
        return [(menu[i], 1) for menu, i in zip(menus, indices, strict=True)]


def score(model_path, runs, out, side):
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_path, padding_side='left', local_files_only=True)
    for mode in ('legacy', 'fixed', 'random', 'hidden'):
        rows, cases = examples_for(mode, bool(side))
        for seed in range(3):
            base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
            path = runs / str(side) / mode / str(seed) / 'adapter'
            policy = Greedy(PeftModel.from_pretrained(base, path).eval(), tok)
            tag = f'{side}-{mode}-{seed}'
            with (out / f'{tag}-steps.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(('split','panel','case','position','permutation','target','predicted','exact','productive','target_letter','predicted_letter'))
                for order in ('native', '101', '102', '103'):
                    for start in range(0, len(rows), 32):
                        batch = rows[start:start + 32]
                        prompts, menus = [], []
                        for split, panel, case, position, prompt, native, target in batch:
                            commands = list(native)
                            if order != 'native':
                                commands.sort()
                                Random(order + '|' + prompt).shuffle(commands)
                            prompts.append(prompt)
                            menus.append(commands)
                        predictions = policy(prompts, menus)
                        for row, (pred, _), menu in zip(batch, predictions, menus, strict=True):
                            split, panel, case, position, prompt, _, target = row
                            writer.writerow((split,panel,case,position,order,target,pred,
                                             int(pred==target),int(productive(prompt,pred)),
                                             LETTERS[menu.index(target)],LETTERS[menu.index(pred)]))
            trials = [Rollout(case, State(world.budget)) for case,(world,_,_) in cases.items()]
            collect(trials, cases, policy, True, hide_budget=mode=='hidden', independent_menu=mode!='legacy')
            with (out / f'{tag}-greedy.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(('case','success','actions'))
                for row in trials:
                    _, lo, hi = cases[row.instance]
                    writer.writerow((row.instance,int(outcome(row.state,lo,hi)>0),
                                     json.dumps([a.__dict__ for a in row.actions])))
            del policy, base
            print('completed', tag, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','runs','output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--side', type=int, choices=(0,1), required=True)
    args = parser.parse_args()
    score(args.model, args.runs, args.output, args.side)
