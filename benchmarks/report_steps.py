"""Aggregate saved step evaluations and match each model to its final training loss."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re


def training_losses(path, side):
    losses = {}
    for prefix in ('', 'ref'):
        final_loss = None
        for line in (path / f'{prefix}{side}.log').read_text().splitlines():
            if m := re.fullmatch(r'warm-up update 72 loss (\S+)', line):
                final_loss = float(m[1])
            if m := re.fullmatch(r'completed (\w+) (\d+)', line):
                assert final_loss is not None
                losses[m[1], int(m[2])] = final_loss
                final_loss = None
    return losses


def write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('eval', 'training', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    details, models, trials = [], [], []
    for side in (0, 1):
        losses = training_losses(args.training, side)
        for (mode, seed), loss in sorted(losses.items()):
            tag = f'{side}-{mode}-{seed}'
            model = dict(recovery=side, mode=mode, seed=seed)
            groups = defaultdict(lambda: [0, 0, 0])
            with (args.eval / f'{tag}-steps.csv').open() as handle:
                for r in csv.DictReader(handle):
                    family = r['case'].split('-')[0] if r['panel'] != '0' else 'pooled'
                    action = r['target'].split()[0]
                    for fam, pos, kind in {('pooled','all','all'),('pooled','all',action),
                                           (family,r['position'],action)}:
                        g = groups[r['permutation'], r['split'], fam, pos, kind]
                        g[0] += int(r['exact'])
                        g[1] += int(r['productive'])
                        g[2] += 1
            for (order, split, fam, pos, kind), (correct, progress, n) in sorted(groups.items()):
                details.append(dict(**model, permutation=order, split=split, family=fam,
                                    position=pos, action=kind, correct=correct, productive=progress, n=n))
            with (args.eval / f'{tag}-greedy.csv').open() as handle:
                episodes = list(csv.DictReader(handle))
            assert len(episodes) == 36
            for r in episodes:
                trials.append(dict(**model, case=r['case'], success=r['success'],
                                   action_types=' '.join(a['kind'] for a in json.loads(r['actions']))))
            summary = dict(**model, last_minibatch_loss=loss)
            for split,n in [('train',192), ('normal',216), ('off_path',108)]:
                for order in ('native','101','102','103'):
                    correct, _, size = groups[order, split, 'pooled','all','all']
                    assert size == n
                    summary[f'{split}_{order}'] = correct
                summary[f'{split}_n'] = n
            summary['greedy_completed'] = sum(int(r['success']) for r in episodes)
            models.append(summary)
        assert len(losses) == 12
    write(args.output / 'STEP_MODELS.csv', models)
    write(args.output / 'STEP_ACCURACY.csv', details)
    write(args.output / 'STEP_GREEDY.csv', trials)


if __name__ == '__main__':
    main()
