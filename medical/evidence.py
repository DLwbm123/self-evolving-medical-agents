"""Synthetic evidence-selection tasks using the MAP/BMI arithmetic and action budget."""
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import json
from random import Random

from medical.calculation import Action, FIELDS, HEIGHT, WEIGHT, State, World, calculate, step

UNITS = {"weight": WEIGHT, "height": HEIGHT, **{f: {"mmHg": "1"} for f in FIELDS["5"]}}


@dataclass(frozen=True)
class Record:
    pointer: str
    field: str
    value: Decimal
    unit: str | None
    encounter: str
    day: int


@dataclass(frozen=True)
class EvidenceWorld:
    calculator: str
    records: tuple[Record, ...]
    encounter: str
    day: int
    max_age: int
    budget: int = 8

    def __post_init__(self):
        if len({r.pointer for r in self.records}) != len(self.records):
            raise ValueError("record identifiers must be unique")
        if self.max_age < 0 or self.budget < 0:
            raise ValueError("invalid task limits")
        for record in self.records:
            if record.field not in FIELDS[self.calculator] or not record.value.is_finite() or record.value <= 0:
                raise ValueError("invalid record")


def eligibility(world, record):
    return {"time_window": 0 <= world.day - record.day <= world.max_age,
            "encounter_match": record.encounter == world.encounter,
            "unit_supported": record.unit in UNITS[record.field]}


def observe(world, state):
    return json.dumps(dict(calculator=world.calculator, fields=FIELDS[world.calculator],
                           encounter=world.encounter, as_of_day=world.day, max_age=world.max_age,
                           remaining=state.remaining, bindings=state.bindings,
                           records=[asdict(r) for r in world.records if r.pointer in state.seen],
                           calculated=state.calculated, last_result=state.last_result, done=state.done),
                      default=str, sort_keys=True, separators=(",", ":"))


def arithmetic_world(world):
    # The simulated permissive backend assumes canonical units for an unknown unit.
    # Only the verifier checks whether that assumption has evidence.
    facts = tuple((r.pointer, r.field, r.value * Decimal(UNITS[r.field].get(r.unit, "1")))
                  for r in world.records)
    return World(world.calculator, facts, world.budget)


def evidence_step(world, state, action, *, checks=None):
    backend = arithmetic_world(world)
    following, _, cost = step(backend, state, action)
    if action.kind == "query" and cost:
        pointers = tuple(r.pointer for r in world.records if r.field == action.field)
        if pointers:
            following = replace(following, seen=tuple(sorted(set(state.seen).union(pointers))),
                                last_result="records returned")
    reward = 0.0
    if action.kind == "bind" and following.last_result == "bound":
        records = {r.pointer: r for r in world.records}
        verdicts = eligibility(world, records[action.value])
        if checks is not None:
            checks.update(verdicts)
        previous = dict(state.bindings).get(action.field)
        before = 0 if previous is None else (1 if all(eligibility(world, records[previous]).values()) else -1)
        after = 1 if all(verdicts.values()) else -1
        reward = (after - before) / (2 * len(FIELDS[world.calculator]))
    return following, reward, cost


def synthetic_world(kind, seed, *, equal_value=False):
    """Return model-visible world and separate reference computed from source measurements."""
    rng = Random(seed)
    day = rng.randint(100, 999)
    encounter = f"v{rng.randrange(10000)}"
    if kind == "wrong_encounter":
        calc = "5"
        a, b = Decimal(rng.randint(105, 169)), Decimal(rng.randint(60, 89))
        unit_a, unit_b = "mmHg", "mmHg"
    elif kind in ("stale", "ambiguous_unit"):
        calc = "6"
        a, b = Decimal(rng.randint(48, 102)), Decimal(rng.randint(150, 195)) / 100
        unit_a, unit_b = "kg", "m"
    else:
        raise ValueError("unknown evidence scenario")
    first, second = FIELDS[calc]
    bad_value = a if equal_value else (a * 2 if kind == "ambiguous_unit" else a + rng.randint(9, 25))
    records = [Record("", first, a, unit_a, encounter, day),
               Record("", second, b, unit_b, encounter, day),
               Record("", first, bad_value, None if kind == "ambiguous_unit" else unit_a,
                      encounter + "-prior" if kind == "wrong_encounter" else encounter,
                      day - 8 if kind == "stale" else day)]
    rng.shuffle(records)
    records = tuple(replace(r, pointer=f"e{i}") for i,r in enumerate(records, start=1))
    return EvidenceWorld(calc, records, encounter, day, 7), calculate(calc, a, b)


def command_step(world, state, command, *, checks=None):
    """Execute a command in the text interface used by the fitting diagnostic."""
    if command == "SUBMIT":
        action = Action("finish", value=str(state.calculated))
    elif command == "CALCULATE":
        action = Action("calculate")
    elif command.startswith("QUERY "):
        action = Action("query", command[6:])
    elif command.startswith("BIND ") and " " in command[5:]:
        field, pointer = command[5:].rsplit(" ", 1)
        action = Action("bind", field, pointer)
    else:
        action = Action("invalid")
    return evidence_step(world, state, action, checks=checks)
