"""Stratify frozen age-panel focal-field choices by recency, retaining ties."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path

from medical.evidence import read_evidence_panel


def audit(panel, directory):
    directory.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    with (directory / "AGE_RECENCY_CASES.csv").open("w", newline="") as file:
        cases = csv.writer(file, lineterminator="\n")
        cases.writerow(("case", "focus", "faults", "equal", "recency", "n_latest",
                        "candidates", "good_age", "latest_age", "latest_correct"))
        for world, key in read_evidence_panel(panel):
            records = [r for r in world.records if r.field == key["focus"]]
            gold, = [r for r in records if r.pointer == key["good"][key["focus"]]]
            day = max(r.day for r in records)
            latest = [r for r in records if r.day == day]
            rank = "older" if gold.day != day else ("tied_latest" if len(latest)>1 else "unique_latest")
            correct = int(latest[0] == gold)
            temporal = "time_window" in key["faults"]
            groups[temporal, rank].append((correct, 1/len(records)))
            cases.writerow((key["case"], key["focus"], "+".join(key["faults"]), key["equal"], rank,
                            len(latest), len(records), world.day-gold.day, world.day-day, correct))
    with (directory / "AGE_RECENCY.csv").open("w", newline="") as file:
        summary = csv.writer(file, lineterminator="\n")
        summary.writerow(("designated_time_defect", "recency", "worlds", "latest_correct", "chance"))
        for (temporal, rank), values in sorted(groups.items()):
            hits, chance = zip(*values)
            summary.writerow((temporal, rank, len(values), sum(hits), sum(chance)/len(values)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.panel, args.output)
