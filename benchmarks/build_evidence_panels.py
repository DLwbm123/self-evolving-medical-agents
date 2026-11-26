"""Materialize and audit the synthetic evidence panels using the standard library."""
import argparse
import csv
from collections import Counter
from dataclasses import asdict, replace
from decimal import Decimal
import gzip
import json
from pathlib import Path

from medical.calculation import FIELDS, State, calculate
from medical.evidence import (PANELS, UNITS, command_step, decision_states, eligibility,
                              evidence_panel)


def normalized(r):
    return r.value * Decimal(UNITS[r.field].get(r.unit, "1"))


def audit_world(w, key, panel):
    fields = FIELDS[w.calculator]
    good = key["good"]
    by_pointer = {r.pointer: r for r in w.records}
    for r in w.records:
        failures = {k for k, v in eligibility(w, r).items() if not v}
        assert bool(failures) == (r.pointer != good[r.field])
        if r.pointer == key["bad"]:
            assert r.field == key["focus"] and failures == set(key["faults"])
        if panel == "train" and failures:
            assert (int(r.pointer[1:])-1)//2 != (int(good[r.field][1:])-1)//2
    a, b = [normalized(by_pointer[good[f]]) for f in fields]
    assert calculate(w.calculator, a, b) == Decimal(key["answer"])
    bad = by_pointer[key["bad"]]
    assert (normalized(bad) == normalized(by_pointer[good[bad.field]])) == key["equal"]


def audit_pair(reference, variant, panel):
    original, k0 = reference
    w, k = variant
    assert replace(w, records=(), encounter=original.encounter) == replace(original, records=())
    assert k == (dict(k0, good=k["good"], bad=k["bad"]) if panel == "pointer" else k0)
    for a, b in zip(original.records, w.records, strict=True):
        if panel == "age":
            assert replace(b, day=a.day) == a
        elif panel == "encounter":
            assert (b.encounter == w.encounter) == (a.encounter == original.encounter)
            assert replace(b, encounter=a.encounter) == a
        elif panel == "unit":
            assert normalized(a) == normalized(b) and replace(b, value=a.value, unit=a.unit) == a
        elif panel == "pointer":
            assert replace(b, pointer=a.pointer) == a
    if panel != "encounter":
        assert w.encounter == original.encounter


def reward_control(w, key):
    """Same command budget, same final bindings, one accepted bad bind vs none."""
    state = State(w.budget)
    for f in FIELDS[w.calculator]:
        for cmd in (f"QUERY {f}", f"BIND {f} {key['good'][f]}"):
            state, _, _ = command_step(w, state, cmd)
    f = key["focus"]
    clean = f"BIND {f} {key['good'][f]}"
    returns = []
    for first in (clean, f"BIND {f} {key['bad']}"):
        s = state
        potential = cost = bad_events = 0
        for cmd in (first, clean, "CALCULATE", "SUBMIT"):
            checks = {}
            s, reward, charge = command_step(w, s, cmd, checks=checks)
            potential += reward
            cost += charge
            bad_events += int(bool(checks) and not all(checks.values()))
        returns.append((s, potential, cost, bad_events))
    one, two = returns
    assert one[:3] == two[:3] and one[0].submitted == Decimal(key["answer"])
    assert (one[3], two[3]) == (0, 1)


def audit_shortcuts(world, key, scores):
    # Score the actual displayed strings. Ties choose the first displayed
    # record, exactly as a stable min/argmin on the policy input would.
    for field in FIELDS[world.calculator]:
        records = [r for r in world.records if r.field == field]
        decimals = [len(str(r.value).partition(".")[2]) for r in records]
        if set(decimals) != {8}:
            raise ValueError("numeric precision leaks a candidate distinction")
        choices = (records[decimals.index(min(decimals))], max(records, key=lambda r: r.day),
                   min(records, key=lambda r: int(r.pointer[1:])), records[0])
        for scope in ("all_fields", "focus"):
            if scope == "focus" and field != key["focus"]:
                continue
            counts = scores[scope, key["equal"]]
            counts["n"] += 1
            counts["chance"] += 1/len(records)
            for name, record in zip(("fewest_decimals", "latest", "min_pointer", "first"), choices):
                counts[name] += int(record.pointer == key["good"][field])


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    train_prompts = set()
    reference = {}
    layout_support = {}
    summary = {}
    audit_rows = []
    for panel in PANELS:
        worlds = prompts = 0
        layouts = Counter()
        positions = Counter()
        for_worlds = Counter()
        held_prompts = set()
        scores = {}
        for scope in ("all_fields", "focus"):
            for equal in (False, True):
                scores[scope, equal] = Counter()
        with gzip.GzipFile(str(output / f"{panel}.jsonl.gz"), "wb", mtime=0) as stream:
            for w, key in evidence_panel(panel):
                audit_world(w, key, panel)
                audit_shortcuts(w, key, scores)
                if panel == "iid":
                    reference[key["case"]] = (w, key)
                elif panel != "train":
                    audit_pair(reference[key["case"]], (w, key), panel)
                if panel == "train":
                    reward_control(w, key)
                f = key["focus"]
                n = len(w.records)//2
                stratum = (w.calculator, f, tuple(key["faults"]), n)
                positions[stratum, key["good"][f]] += 1
                layouts[n, key["good"][f], key["bad"]] += 1
                for_worlds["+".join(key["faults"])] += 1
                for _, prompt, _ in decision_states(w, key):
                    if panel != "train" and prompt in train_prompts:
                        raise ValueError("training prompt in holdout")
                    if prompt in held_prompts:
                        raise ValueError(f"duplicate prompt within {panel}")
                    held_prompts.add(prompt)
                    prompts += 1
                row = dict(world=asdict(w), key=key)
                stream.write((json.dumps(row, default=str, separators=(",", ":"))+"\n").encode())
                worlds += 1
        if panel == "train":
            train_prompts = held_prompts
        for stratum, _ in positions:
            n = stratum[-1]
            assert len({positions[stratum, f"e{i+1}"] for i in range(2*n)}) == 1
        for n in (2, 3, 4):
            block = [count for (width, _, _), count in layouts.items() if width == n]
            assert len(set(block)) == 1
            assert len(block) == (2*n if panel == "pointer" else 2*n*(2*n-2))
        for (scope, equal), counts in scores.items():
            n = counts["n"]
            for name in ("fewest_decimals", "latest", "min_pointer", "first"):
                gap = (counts[name] - counts["chance"])/n
                # Fixed design tolerance in percentage points; recency is a
                # real verifier feature, so it is reported without this gate.
                if name != "latest" and abs(gap) > 0.05:
                    raise ValueError(f"shortcut exceeds 5pp: {panel}/{scope}/{equal}/{name} {gap:.4f}")
                audit_rows.append((panel, scope, equal, name, n, counts[name],
                                   counts["chance"]/n, counts[name]/n, gap))
        layout_support[panel] = {(g, b) for _, g, b in layouts}
        summary[panel] = dict(worlds=worlds, decisions=prompts, by_designated_defect=for_worlds)
    assert layout_support["train"].isdisjoint(layout_support["pointer"])
    (output/"counts.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    with (output/"shortcuts.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("panel", "scope", "equal", "rule", "n", "correct", "chance", "accuracy", "gap"))
        writer.writerows(audit_rows)
    return {name: data["worlds"] for name, data in summary.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(build(parser.parse_args().output))
