"""Summarize the paired budget ablation by independent holdout case."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--include-legacy", action="store_true")
    args = parser.parse_args()
    paired = []
    modes = ("fixed", "random", "hidden") + (("legacy",) if args.include_legacy else ())
    for mode in modes:
        for train_seed in range(3):
            arms = []
            for side in (0, 1):
                path = args.root / str(side) / mode / str(train_seed) / "cases.json"
                records = json.loads(path.read_text())
                cases = {(r["panel"], r["case"]): r["successes"] for r in records}
                expected = {(panel, f"{label}-{i}") for panel in (11, 12, 13)
                            for label in ("MAP", "BMI") for i in range(1, 7)}
                assert len(records) == 36 and set(cases) == expected
                assert all(r["mode"] == mode and r["train_seed"] == train_seed
                           and r["attempts"] == 4 and 0 <= r["successes"] <= 4
                           and r["recovery"] == bool(side) for r in records)
                arms.append(cases)
            for panel, case in sorted(arms[0]):
                paired.append(dict(mode=mode, train_seed=train_seed, panel=panel, case=case,
                                   clean=arms[0][panel, case], recovery=arms[1][panel, case], attempts=4))
    with (args.root / "paired_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    for mode in modes:
        rows = [r for r in paired if r["mode"] == mode]
        gaps = [mean((r["recovery"] - r["clean"]) / 4 for r in rows if r["train_seed"] == s)
                for s in range(3)]
        print(mode, "clean", sum(r["clean"] for r in rows), "recovery", sum(r["recovery"] for r in rows),
              "attempts per arm", sum(r["attempts"] for r in rows), "paired gaps by seed", gaps)


if __name__ == "__main__":
    main()
