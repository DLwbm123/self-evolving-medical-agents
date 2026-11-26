import argparse
import csv
import json
from pathlib import Path
from random import Random

from medical.calculation import FIELDS


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def note_keys(row):
    return {
        (kind, value)
        for kind, value in (("id", row["Note ID"].strip()),
                            ("text", " ".join(row["Patient Note"].split())))
        if value
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train, test = read(args.train_csv), read(args.test_csv)
    test_keys = set().union(*(note_keys(row) for row in test))
    candidates = [row for row in train if row["Calculator ID"] in FIELDS]
    included = [row for row in candidates if not note_keys(row) & test_keys]

    parents = list(range(len(included)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(included):
        for key in sorted(note_keys(row)):
            other = seen.setdefault(key, index)
            parents[root(index)] = root(other)
    groups = sorted({root(index) for index in range(len(included))})
    rng = Random(0)
    rng.shuffle(groups)
    validation = set(groups[:max(1, len(groups) // 10)])
    split = {"train": [], "validation": []}
    for index, row in enumerate(included):
        split["validation" if root(index) in validation else "train"].append(row)

    assert split["train"] and split["validation"]
    assert not (set().union(*(note_keys(row) for row in split["train"]))
                & set().union(*(note_keys(row) for row in split["validation"])))
    assert not (set().union(*(note_keys(row) for row in included)) & test_keys)
    held_out = [row for row in test if row["Calculator ID"] in FIELDS]
    manifest = {
        "seed": 0,
        "filtered_training_rows": len(candidates) - len(included),
        "training_csv": args.train_csv.name,
        "held_out_csv": args.test_csv.name,
        "splits": {name: [int(row["Row Number"]) for row in rows]
                   for name, rows in {**split, "held_out": held_out}.items()},
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("removed", manifest["filtered_training_rows"])
    for name, rows in manifest["splits"].items():
        print(name, len(rows))


if __name__ == "__main__":
    main()
