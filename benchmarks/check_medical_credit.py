import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random

from medical.calculation import Action, State, observation, outcome, read_cases, step
from medical.credit import Step, Trace, assign_credit


def trace(world, lower, upper, actions, start=None):
    state = State(world.budget) if start is None else start
    states = [state]
    steps = []
    for action in actions:
        key = observation(world, state)
        state, reward, cost = step(world, state, action)
        states.append(state)
        # No LM planned these diagnostic actions. No model tokens are charged.
        valid_format = action.kind != "finish" or state.submitted is not None
        steps.append(Step(key, reward, cost, 0, trainable=valid_format))
    return steps, states, outcome(state, lower, upper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    assert 2 * (8 + 1) * 0.05 + 8 * 0.02 < 2
    rng = Random(0)
    counts = Counter()
    cases = {}
    manifest = json.loads(args.split_manifest.read_text())
    if manifest["training_csv"] != args.train_csv.name:
        raise ValueError("split manifest refers to a different training file")
    for calculator, note, case in read_cases(args.train_csv, set(manifest["splits"]["train"])):
        counts[calculator, "total"] += 1
        if case is None:
            counts[calculator, "unsupported"] += 1
            continue
        cases.setdefault((calculator, note), case)
    chosen = rng.sample(sorted(cases), 100)
    all_traces = []
    for instance, key in enumerate(chosen):
        world, lower, upper = cases[key]
        work = [Action(kind, field, value)
                for pointer, field, _ in world.facts
                for kind, value in (("query", ""), ("bind", pointer))]
        work.append(Action("calculate"))
        _, states, _ = trace(world, lower, upper, work)
        answer = str(states[-1].calculated)
        plan = [*work, Action("finish", value=answer)]
        turns, states, reward = trace(world, lower, upper, plan)
        counts[key[0], "reference_pass" if reward > 0 else "reference_fail"] += 1
        anchor = rng.randrange(len(plan) - 1)
        prefix = trace(world, lower, upper, plan[:anchor])[1][-1]
        assert observation(world, prefix) == turns[anchor].observation
        counts["replay", "match"] += 1

        parent = len(all_traces)
        all_traces.append(Trace(str(instance), reward, tuple(turns)))
        bad = Action("finish", value="not a number")
        for alternatives in ([*work, bad], [bad], [Action("query", "unavailable"), *plan]):
            turns_alt, _, score = trace(world, lower, upper, alternatives)
            all_traces.append(Trace(str(instance), score, tuple(turns_alt)))
        for alternatives in (plan[anchor:], [bad], [Action("query", "unavailable"), *plan[anchor:]]):
            turns_alt, _, score = trace(world, lower, upper, alternatives, start=prefix)
            all_traces.append(Trace(str(instance), score, tuple(turns_alt), (parent, anchor)))

    configs = {
        "A1": dict(beta=0, process_weight=0, cost_weight=0),
        "A2": dict(beta=1, process_weight=0),
        "A3": dict(beta=0, trajectory_process=True),
        "A4": dict(beta=1),
    }
    arms = {}
    for arm, config in configs.items():
        stage = assign_credit(all_traces, **config)
        arms[arm] = {
            "tiers": dict(Counter(stage.tier.values())),
            "nonzero_steps": sum(value != 0 for value in stage.advantages.values()),
        }
    output = {
        "counts": {g: {k: v for (group, k), v in counts.items() if group == g}
                   for g in ("5", "6", "replay")},
        "traces": len(all_traces),
        "arms": arms,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
