"""Measure training signal in sampled simulator trajectories, with immutable forks."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import json
from pathlib import Path
from random import Random
import re
import signal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from medical.calculation import Action, FIELDS, State, World, observation, outcome, read_cases, step
from medical.credit import Step, Trace, assign_credit, choose_anchors


LABELS = {"5": "MAP", "6": "BMI"}
LETTERS = "ABCDEFGH"
SYSTEM = """You control a simulated medical calculator. Reply with exactly one action per turn.
QUERY <field> obtains a value and its evidence id.
BIND <field> <evidence id> binds a previously queried value to the calculator input.
CALCULATE runs the calculator after all inputs are bound.
FINISH <number> returns the result shown by the calculator.
Use the current state to choose one next action. Use the field names listed for this task.
Example action syntax: QUERY weight
Another example: BIND weight e1
Do not include any other text."""
STATE_RULES = """One action per turn in a simulated calculator. Inspect the state before acting.
- If a requested field has no value in seen, issue QUERY followed by that field's exact name.
- When a field has a value in seen but is absent from bindings, issue BIND followed by the field and its observed id. Each seen entry is [id, field name, value]. Use only ids in seen.
- Once all fields are in bindings, issue CALCULATE if calculated is "None".
- Once calculated contains a number, issue FINISH followed by that number.
Action names are QUERY, BIND, CALCULATE, and FINISH. Respond with one action line only.
Do not repeat an action if it does not change the state."""
MENU_RULES = """Choose the next action from the menu. Reply with just its letter.
Query missing fields, bind observed evidence ids to the matching fields, then calculate and finish.
Avoid repeat queries and actions that can not succeed in the current state."""


def parse_action(text):
    value = text.strip()
    if "\n" in value or "\r" in value:
        return Action("invalid")
    if value == "CALCULATE":
        return Action("calculate")
    if match := re.fullmatch(r"QUERY (\S.*)", value):
        return Action("query", match[1])
    if match := re.fullmatch(r"BIND (\S.*) (e[0-9]+)", value):
        return Action("bind", match[1], match[2])
    if match := re.fullmatch(r"FINISH ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", value):
        return Action("finish", value=match[1])
    return Action("invalid")


@dataclass
class Rollout:
    instance: str
    state: State
    fork: tuple[int, int] | None = None
    steps: list[Step] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)


def action_menu(world, state):
    fields = FIELDS[world.calculator]
    commands = [f"QUERY {field}" for field in fields]
    commands.extend(f"BIND {field} {evidence}" for field in fields for evidence in state.seen)
    commands.append("CALCULATE")
    if state.calculated is not None:
        commands.append(f"FINISH {state.calculated}")
    Random("0|" + observation(world, state)).shuffle(commands)
    return commands


def state_prompt(world, state):
    fields = ", ".join(FIELDS[world.calculator])
    return f"Task: calculate {LABELS[world.calculator]}. Fields: {fields}\nState: {observation(world, state)}"


def with_menu(prompt, commands):
    assert len(commands) <= len(LETTERS)
    return prompt + "\nMenu:\n" + "\n".join(
        f"{letter}. {command}" for letter, command in zip(LETTERS, commands)
    )


def collect(rollouts, cases, infer, use_menu=False):
    for _ in range(9):
        waiting = [row for row in rollouts if not row.state.done]
        if not waiting:
            return
        for offset in range(0, len(waiting), 8):
            batch = waiting[offset:offset + 8]
            prompts = []
            menus = []
            for row in batch:
                world = cases[row.instance][0]
                prompts.append(state_prompt(world, row.state))
                if use_menu:
                    menus.append(action_menu(world, row.state))
            for row, (text, tokens) in zip(batch, infer(prompts, menus), strict=True):
                world = cases[row.instance][0]
                key = observation(world, row.state)
                verdicts = {}
                action = parse_action(text)
                row.state, reward, cost = step(world, row.state, action, checks=verdicts)
                row.steps.append(Step(key, reward, cost, tokens, action.kind != "invalid"))
                row.actions.append(action)
                row.verdicts.append(verdicts)
                if action.kind == "invalid":
                    verdicts["action_format"] = -1
    assert all(row.state.done for row in rollouts)


def teacher_examples(world, recovery=None):
    state = State(world.budget)
    commands = [command for pointer, field, _ in world.facts
                for command in (f"QUERY {field}", f"BIND {field} {pointer}")]
    first_pointer, first_field, _ = world.facts[0]
    second_field = world.facts[1][1]
    perturbations = ["CALCULATE", f"QUERY {first_field}", f"BIND {second_field} {first_pointer}"]
    for index, command in enumerate([*commands, "CALCULATE", "FINISH"]):
        if index == recovery:
            mistake = perturbations[index]
            assert mistake in action_menu(world, state)
            state, _, _ = step(world, state, parse_action(mistake))
        if command == "FINISH":
            command = f"FINISH {state.calculated}"
        menu = action_menu(world, state)
        yield with_menu(state_prompt(world, state), menu), menu.index(command)
        state, _, _ = step(world, state, parse_action(command))
    assert state.done and state.submitted is not None


def synthetic_cases(holdout_seed, recovery_warmup):
    """Independent simulator values. No benchmark records are loaded."""
    rng = Random(2)
    cases = {}
    examples = []
    holdout = Random(holdout_seed)
    ranges = {"5": (range(92, 182), range(54, 91)), "6": (range(46, 125), range(144, 200))}
    for key, label in LABELS.items():
        left, right = ranges[key]
        values = [(x, y) for x in left for y in right]
        pilot = rng.sample(values, 20)
        if holdout_seed is None:
            pairs = pilot
        else:
            used = set(pilot)
            pairs = pilot[:16] + holdout.sample([pair for pair in values if pair not in used], 6)
        for index, pair in enumerate(pairs):
            x, y = map(Decimal, pair)
            if key == "6":
                y /= 100
            facts = tuple((f"e{i}", field, value)
                          for i, (field, value) in enumerate(zip(FIELDS[key], (x, y)), start=1))
            world = World(key, facts)
            if index < 16:
                recovery = index // 2 % 3 if recovery_warmup and index % 2 else None
                examples.extend(teacher_examples(world, recovery))
            else:
                answer = (x + 2 * y) / 3 if key == "5" else x / (y * y)
                tolerance = Decimal("0.01")
                cases[f"{label}-{index - 15}"] = world, answer - tolerance, answer + tolerance
    Random(1).shuffle(examples)
    return cases, examples


def warmup_examples(pool, pilot_notes, csv_path, training_ids):
    """Generate one rule trajectory per case using observation-visible tool actions."""
    with csv_path.open(encoding="utf-8", newline="") as stream:
        note_texts = defaultdict(set)
        text_notes = defaultdict(set)
        for row in csv.DictReader(stream):
            if int(row["Row Number"]) in training_ids:
                note = row["Note ID"]
                text = " ".join(row["Patient Note"].split())
                note_texts[note].add(text)
                text_notes[text].add(note)

    notes = set()

    def exclude_related(pending):
        while pending:
            current = pending.pop()
            if current in notes:
                continue
            notes.add(current)
            pending.extend(other for text in note_texts[current] if text for other in text_notes[text]
                           if other not in notes)

    exclude_related(list(pilot_notes))
    rng = Random(1)
    examples = []
    for group in pool.values():
        candidates = sorted(group)
        rng.shuffle(candidates)
        selected = 0
        for note in candidates:
            if note in notes:
                continue
            selected += 1
            exclude_related([note])
            examples.extend(teacher_examples(group[note][0]))
            if selected == 16:
                break
        assert selected == 16
    rng.shuffle(examples)
    return examples


def make_inference(model_path, device, seed, system, examples, output, epochs):
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
        torch.mps.set_per_process_memory_fraction(0.55)
    dtype = {"mps": torch.float16, "cpu": torch.float32, "cuda": torch.bfloat16}[device]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    stops.update(ids[0] for marker in ("\n", "\n\n")
                 if len(ids := tokenizer.encode(marker, add_special_tokens=False)) == 1)

    letter_tokens = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]

    def encode(prompts):
        texts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        ) for prompt in prompts]
        return tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(device)

    if examples:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )).train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=8e-4)
        batches = [examples[batch:batch + 8] for batch in range(0, len(examples), 8)]
        for update, batch in enumerate(batches * epochs, start=1):
            prompts, targets = zip(*batch)
            encoded = encode(prompts)
            position_ids = encoded.attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)
            logits = model(**encoded, position_ids=position_ids, use_cache=False, logits_to_keep=1).logits
            labels = torch.tensor([letter_tokens[target][0] for target in targets], device=device)
            loss = torch.nn.functional.cross_entropy(logits[:, -1].float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite warm-up loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print("warm-up update", update, "loss", round(loss.item(), 5), flush=True)
        model.save_pretrained(output / "adapter")
        model.eval()
        torch.manual_seed(seed)

    def infer(prompts, menus):
        if menus:
            assert all(len(tokens) == 1 for tokens in letter_tokens)
            prompts = [with_menu(prompt, commands) for prompt, commands in zip(prompts, menus, strict=True)]
        inputs = encode(prompts)
        with torch.inference_mode():
            if menus:
                choice_ids = [tokens[0] for tokens in letter_tokens]

                def restrict(_input_ids, scores):
                    filtered = torch.full_like(scores, -float("inf"))
                    for row, commands in enumerate(menus):
                        allowed = choice_ids[:len(commands)]
                        filtered[row, allowed] = scores[row, allowed]
                    return filtered

                tokens = model.generate(
                    **inputs, max_new_tokens=1, temperature=0.8, top_p=0.9, top_k=0,
                    do_sample=True, logits_processor=LogitsProcessorList([restrict]),
                    pad_token_id=tokenizer.pad_token_id,
                )[:, -1].tolist()
                return [(commands[choice_ids.index(token)], 1)
                        for commands, token in zip(menus, tokens, strict=True)]
            sampled = model.generate(
                **inputs, max_new_tokens=20, temperature=0.8, top_p=0.9, top_k=0,
                do_sample=True, eos_token_id=sorted(stops), pad_token_id=tokenizer.pad_token_id,
            )[:, inputs["input_ids"].shape[1]:].tolist()
        outputs = []
        for row in sampled:
            length = next((index + 1 for index, token in enumerate(row) if token in stops), len(row))
            output = tokenizer.decode(row[:length], skip_special_tokens=True)
            outputs.append((output, length))
        return outputs

    return infer


def benchmark_cases(data, rng, warmup):
    manifest = json.loads((data / "split-map-bmi.json").read_text())
    pool = defaultdict(dict)
    csv_path = data / manifest["training_csv"]
    training_ids = set(manifest["splits"]["train"])
    for key, note, case in read_cases(csv_path, training_ids):
        if case is not None:
            pool[key].setdefault(note, case)
    cases = {}
    pilot_notes = set()
    for key, label in LABELS.items():
        for i, note in enumerate(rng.sample(sorted(pool[key]), 4)):
            cases[f"{label}-{i + 1}"] = pool[key][note]
            pilot_notes.add(note)
    return cases, warmup_examples(pool, pilot_notes, csv_path, training_ids) if warmup else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--prompt", choices=("syntax", "state-rules", "menu"), default="syntax")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, choices=(1, 3), default=1)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-holdout-seed", type=int)
    parser.add_argument("--recovery-warmup", action="store_true")
    args = parser.parse_args()
    if args.warmup and (args.prompt != "menu" or args.device == "mps"):
        parser.error("the warm-up diagnostic requires menu and either CPU or CUDA")
    if args.synthetic_holdout_seed is not None and not args.synthetic:
        parser.error("synthetic holdout seed requires --synthetic")
    if args.recovery_warmup and not (args.synthetic and args.warmup):
        parser.error("recovery warm-up requires synthetic warm-up")
    signal.alarm(1800)
    rng = Random(0)
    cases, examples = (synthetic_cases(args.synthetic_holdout_seed, args.recovery_warmup) if args.synthetic
                       else benchmark_cases(args.data, rng, args.warmup))
    system = {"syntax": SYSTEM, "state-rules": STATE_RULES, "menu": MENU_RULES}[args.prompt]
    use_menu = args.prompt == "menu"
    examples = examples if args.warmup else []
    infer = make_inference(args.model, args.device, 0, system, examples, args.output, args.warmup_epochs)
    base = [Rollout(case, State(world.budget))
            for case, (world, _, _) in cases.items() for _ in range(4)]
    collect(base, cases, infer, use_menu)
    print("base episodes collected", len(base), flush=True)

    branches = []
    anchors = []
    for instance, (world, _, _) in cases.items():
        entries = []
        for parent, row in enumerate(base):
            if row.instance != instance:
                continue
            spent = 0
            for t, entry in enumerate(row.steps):
                entries.append(((parent, t), spent))
                spent += entry.cost
        for parent, t in choose_anchors(entries, world.budget, rng):
            replay = State(world.budget)
            for action in base[parent].actions[:t]:
                replay, _, _ = step(world, replay, action)
            assert observation(world, replay) == base[parent].steps[t].observation
            anchors.append((parent, t))
            branches.extend(Rollout(instance, replay, (parent, t)) for _ in range(3))
    collect(branches, cases, infer, use_menu)
    print("branch episodes collected", len(branches), flush=True)

    records = base + branches
    traces = [
        Trace(row.instance, outcome(row.state, *cases[row.instance][1:]), tuple(row.steps), row.fork)
        for row in records
    ]
    credit = assign_credit(traces)
    base_groups = defaultdict(list)
    checks = defaultdict(Counter)
    for row, trace in zip(records, traces, strict=True):
        if row.fork is None:
            base_groups[row.instance].append(trace.terminal)
        for verdicts in row.verdicts:
            for name, value in verdicts.items():
                checks[name]["abstain" if value is None else "pass" if value > 0 else "fail"] += 1
    tier_counts = Counter(credit.tier.values())
    nonzero_by_tier = Counter(
        credit.tier[key] for key, value in credit.local.items() if abs(value) > 1e-6
    )
    affected = sum(
        any(abs(credit.local.get((i, t), 0)) > 1e-6 and s.trainable for t, s in enumerate(trace.steps))
        for i, trace in enumerate(traces[:len(base)])
    )
    results = {
        "seed": 0,
        "device": args.device,
        "prompt": args.prompt,
        "data": "synthetic" if args.synthetic else "medcalc-training-pilot",
        "warmup_examples": len(examples),
        "warmup_epochs": args.warmup_epochs if args.warmup else 0,
        "synthetic_holdout_seed": args.synthetic_holdout_seed,
        "recovery_warmup": args.recovery_warmup,
        "base_groups": dict(base_groups),
        "mixed_groups": sum(len(set(group)) > 1 for group in base_groups.values()),
        "base_successes": sum(t.terminal > 0 for t in traces[:len(base)]),
        "branch_successes": sum(t.terminal > 0 for t in traces[len(base):]),
        "base_episodes": len(base),
        "branch_episodes": len(branches),
        "anchors_replayed": len(anchors),
        "forks_distinct_first_action": sum(
            len({base[parent].actions[t], *(row.actions[0] for row in branches
                                           if row.fork == (parent, t))}) > 1
            for parent, t in anchors
        ),
        "tier_turns": dict(tier_counts),
        "nonzero_local_turns_by_tier": dict(nonzero_by_tier),
        "base_episodes_with_nonzero_trainable_local_credit": affected,
        "predicate_counts": dict(checks),
        "sampled_tokens": credit.sampled_tokens,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "traces.jsonl").open("w") as handle:
        for row, trace in zip(records, traces, strict=True):
            handle.write(json.dumps({
                "instance": row.instance, "fork": row.fork, "success": trace.terminal > 0,
                "turns": [
                    {**asdict(entry), "action": asdict(action), "verdicts": verdicts}
                    for entry, action, verdicts in zip(row.steps, row.actions, row.verdicts, strict=True)
                ],
            }) + "\n")
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
