"""Paired structural controls for permissive binding and evidence verification."""
import json
from collections import Counter
from decimal import Decimal

from medical.calculation import Action, FIELDS, State, outcome
from medical.evidence import command_step, eligibility, evidence_step, synthetic_world


def check_pair(kind, seed, equal_value):
    world, reference = synthetic_world(kind, seed, equal_value=equal_value)
    records = sorted(world.records, key=lambda r: (r.field != FIELDS[world.calculator][0],
                                                  not all(eligibility(world, r).values())))
    good, bad, other = records
    checks = {}
    start = State(world.budget)
    for row in (good, other):
        start, _, _ = evidence_step(world, start, Action("query", row.field))
    start, _, _ = evidence_step(world, start, Action("bind", other.field, other.pointer))
    def bind(s, r):
        return evidence_step(world, s, Action("bind", r.field, r.pointer), checks=checks)
    clean, pos, _ = bind(start, good)
    damaged, neg, _ = bind(start, bad)
    failed = [k for k,v in checks.items() if not v]
    assert failed == [{"stale":"time_window","wrong_encounter":"encounter_match",
                       "ambiguous_unit":"unit_supported"}[kind]]
    assert dict(damaged.bindings)[bad.field] == bad.pointer and neg < 0 < pos
    intermediate, _, _ = evidence_step(world, damaged, Action("calculate"))
    fixed, correction, _ = bind(intermediate, good)
    assert fixed.calculated is None and correction == pos - neg
    repeat, repeated_reward, _ = bind(fixed, good)
    assert repeat.bindings == fixed.bindings and repeated_reward == 0
    def finish(s):
        s, _, _ = evidence_step(world, s, Action("calculate"))
        s, _, _ = command_step(world, s, "SUBMIT")
        return int(outcome(s, reference - Decimal(".01"), reference + Decimal(".01")) > 0)
    return (finish(clean), finish(damaged), finish(fixed))


if __name__ == "__main__":
    for kind in ("stale","wrong_encounter","ambiguous_unit"):
        for equal in (False, True):
            results = Counter(check_pair(kind, seed, equal) for seed in range(100))
            assert results == {(1,int(equal),1):100}
            print(json.dumps(dict(kind=kind, equal_value=equal, pairs=100, clean=100,
                                  unrepaired=100*int(equal), corrected=100)))
