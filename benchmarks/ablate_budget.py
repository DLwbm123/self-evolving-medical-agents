"""Train once per condition and seed, then evaluate paired case-level outcomes on GPU."""
import argparse
import json
from pathlib import Path

import torch

from sample_model_rollouts import (MENU_RULES, Rollout, State, collect, make_inference,
                                   outcome, synthetic_cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--legacy-reference", action="store_true",
                        help="replicate the original fixed-budget, budget-dependent menu ordering")
    args = parser.parse_args()
    modes = ("legacy",) if args.legacy_reference else ("fixed", "random", "hidden")
    for mode in modes:
        independent = mode != "legacy"
        panels = {}
        used = set()
        for panel in (11, 12, 13):
            cases, examples = synthetic_cases(panel, args.recovery, budget_mode=mode, independent_menu=independent)
            worlds = {case[0] for case in cases.values()}
            if worlds & used:
                raise ValueError("holdout panels overlap")
            used.update(worlds)
            panels[panel] = cases
        for seed in (0, 1, 2):
            folder = args.output / mode / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "cases.json"
            if target.is_file():
                continue
            infer = make_inference(args.model, "cuda", seed, MENU_RULES, examples, folder, 3)
            records = []
            for panel, cases in panels.items():
                torch.manual_seed(1000 + panel)
                rows = [Rollout(name, State(world.budget)) for name, (world, _, _) in cases.items()
                        for _ in range(4)]
                collect(rows, cases, infer, True, hide_budget=mode == "hidden", independent_menu=independent)
                for name, (_, lo, hi) in cases.items():
                    successes = sum(outcome(r.state, lo, hi) > 0 for r in rows if r.instance == name)
                    records.append(dict(mode=mode, recovery=args.recovery, train_seed=seed,
                                        panel=panel, case=name, successes=successes, attempts=4))
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2) + "\n")
            temporary.replace(target)
            print("completed", mode, seed, flush=True)
            del infer


if __name__ == "__main__":
    main()
