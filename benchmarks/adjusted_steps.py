"""Descriptive model-level adjustment, using common prompt exclusions and equal order weights."""
import csv
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv",type=Path)
    args=parser.parse_args()
    groups=defaultdict(lambda: [0,0,0,0])
    with args.csv.open() as file:
        for r in csv.DictReader(file):
            key = (r["mode"],int(r["seed"]),int(r["recovery"]))
            if r["split"]=="normal" and r["position"]!="1":
                index=0
            elif r["position"]=="wrong_bind":
                index=2
            else:
                continue
            groups[key][index] += int(r["correct_shared_novel"])
            groups[key][index+1] += int(r["shared_novel"])
    x,y=[],[]
    rates={}
    for (mode,seed,side),(a,b,c,d) in sorted(groups.items()):
        rates[mode,seed,side]=(a/b,c/d)
        x.append([1,a/b,side])
        y.append(c/d)
    assert len(rates)==24
    print("OLS [intercept, normal accuracy slope, supervision coefficient]",
          np.linalg.lstsq(x,y,rcond=None)[0])
    differences=np.array([np.subtract(rates[m,s,1],rates[m,s,0])
                          for m,s,side in rates if side==0])
    print("Paired differences [supervision coefficient, slope]",
          np.linalg.lstsq(np.column_stack([np.ones(12),differences[:,0]]),
                          differences[:,1],rcond=None)[0])


if __name__=="__main__":
    main()
