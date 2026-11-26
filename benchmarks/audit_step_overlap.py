"""Re-score saved argmax predictions after excluding all states seen by either arm."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from score_steps import examples_for
from sample_model_rollouts import with_menu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(("mode","recovery","seed","order","split","position","total","own_exact_native",
                         "shared_novel","correct_shared_novel"))
        for mode in ("legacy","fixed","random","hidden"):
            arms = [examples_for(mode, bool(side))[0] for side in (0,1)]
            # Compare the same retained tasks across arms. Permuting letter positions does
            # not turn an identical state into a new task.
            known = {p for arm in arms for split,_,_,_,p,_,_ in arm if split == "train"}
            for side, rows in enumerate(arms):
                train = {with_menu(p,m) for split,_,_,_,p,m,_ in rows if split == "train"}
                held = {(s,str(panel),case,pos): (with_menu(p,m) in train, p not in known)
                        for s,panel,case,pos,p,m,_ in rows if s != "train"}
                for seed in range(3):
                    counts = defaultdict(lambda: [0]*4)
                    with (args.raw / f"{side}-{mode}-{seed}-steps.csv").open() as f:
                        for r in csv.DictReader(f):
                            if r["split"] == "train":
                                continue
                            same, novel = held[tuple(r[k] for k in ("split","panel","case","position"))]
                            v = counts[r["permutation"],r["split"],r["position"]]
                            for i, amount in enumerate((1,same,novel,novel*int(r["exact"]))):
                                v[i] += amount
                    for key, values in sorted(counts.items()):
                        assert values[0] == 36
                        writer.writerow((mode,side,seed,*key,*values))


if __name__ == "__main__":
    main()
