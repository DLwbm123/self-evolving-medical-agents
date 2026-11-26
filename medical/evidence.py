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
    feed_units: tuple[str, ...] = ()

    def __post_init__(self):
        if len({r.pointer for r in self.records}) != len(self.records):
            raise ValueError("record identifiers must be unique")
        if self.max_age < 0 or self.budget < 0:
            raise ValueError("invalid task limits")
        if self.feed_units and (len(self.feed_units) != len(FIELDS[self.calculator]) or
                                any(u not in UNITS[f] for f, u in zip(FIELDS[self.calculator], self.feed_units))):
            raise ValueError("invalid source feed units")
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


def backend_value(world, record):
    if record.unit in UNITS[record.field]:
        scale = UNITS[record.field][record.unit]
    elif world.feed_units:
        feed_unit = world.feed_units[FIELDS[world.calculator].index(record.field)]
        scale = UNITS[record.field][feed_unit]
    else:
        scale = "1"  # Historical worlds defaulted to canonical units.
    return record.value * Decimal(scale)


def arithmetic_world(world):
    facts = tuple((r.pointer, r.field, backend_value(world, r)) for r in world.records)
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


# A layout is the ordered pair of absolute pointers (qualified, damaged) in
# the focus field. The reserved pairs are e1/e2, e3/e4, ... in either order.
DEFECTS = ("time_window", "encounter_match", "unit_supported")
PANELS = ("train", "iid", "age", "encounter", "unit", "pointer")


def evidence_panel(panel):
    """Yield matched synthetic worlds and private scoring metadata in stable order."""
    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}")
    case = 0
    for calc in FIELDS:
        for focus in range(2):
            for mask in range(1, 8):
                for n in (2, 3, 4):
                    for good in range(2*n):
                        for bad in range(2*n):
                            if good // 2 == bad // 2:
                                continue
                            case += 1
                            if panel == "unit" and calc != "6":
                                continue
                            rng = Random(case + (100000 if panel == "train" else 200000))
                            yield _panel_world(panel, case, calc, focus, mask, n, good, bad, rng)


def _panel_world(panel, case, calc, focus, mask, n, good, bad, rng):
    # One member per reserved pair per field makes all training candidates
    # structurally interchangeable. Hold out complete pairs, in either order.
    blocks = [b for b in range(n) if b not in (good//2, bad//2)]
    rng.shuffle(blocks)
    focus_slots = [good, bad, *(2*b + rng.randrange(2) for b in blocks)]
    other_slots = [p ^ 1 for p in focus_slots]
    rng.shuffle(other_slots)
    if panel == "pointer":
        for i in range(0, n-1, 2):
            mate = focus_slots[i] ^ 1
            j = other_slots.index(mate)
            other_slots[j], focus_slots[i+1] = focus_slots[i+1], mate

    ids = rng.sample(range(2**42), 2)
    current, different = [f"{v:013d}" if panel != "encounter" else f"{v:011x}" for v in ids]
    day = rng.randint(100, 9999)
    equal = bool(rng.getrandbits(1))
    weight_unit = rng.choice(("kg", "lb"))
    height_unit = rng.choice(("m", "ft"))
    alternate_height = rng.choice(("cm", "in"))
    if calc == "6":
        units = ("g", alternate_height) if panel == "unit" else (weight_unit, height_unit)
        weight_grid = range(110, 211) if weight_unit == "lb" else range(50, 96)
        # 3/8 inch increments convert exactly to m, ft, cm and in.
        grids = ((weight_grid, Decimal(WEIGHT[weight_unit])),
                 (range(160, 201), Decimal("0.009525")))
    else:
        units = ("mmHg", "mmHg")
        grids = ((range(105, 170), Decimal(1)), (range(60, 90), Decimal(1)))
    samples = [[x*scale for x in rng.sample(grid, n)] for grid, scale in grids]
    physical = [sample[0] for sample in samples]

    rows = []
    good_pointers = {}
    for i, field in enumerate(FIELDS[calc]):
        positions = focus_slots if i == focus else other_slots
        good_pointers[field] = f"e{positions[0]+1}"
        for rank, position in enumerate(positions):
            random_mask = rng.randint(1, 7)
            defects = 0 if rank == 0 else (mask if i == focus and rank == 1 else random_mask)
            # Draw the same random variates in every matched rendering.
            age_draw = rng.randrange(60)
            if defects & 1:
                age = ((8, 30) if panel == "age" else (9, 14))[age_draw % 2]
            else:
                ages = (1, 2, 4, 6, 7) if panel == "age" else (0, 3, 5)
                age = ages[age_draw % len(ages)]
            unit = None if defects & 4 else units[i]
            factor = Decimal(UNITS[field][units[i]])
            value = samples[i][0 if equal else rank] / factor
            displayed = value.quantize(Decimal("0.00000001"))
            if displayed != value:
                raise ValueError("display precision must preserve the measurement exactly")
            rows.append(Record(f"e{position+1}", field, displayed, unit,
                               different if defects & 2 else current, day-age))
    rng.shuffle(rows)
    w = EvidenceWorld(calc, tuple(rows), current, day, 7, feed_units=units)
    key = dict(case=case, focus=FIELDS[calc][focus], faults=[flag for k, flag in enumerate(DEFECTS)
               if mask & (1 << k)], good=good_pointers, bad=f"e{focus_slots[1]+1}",
               equal=equal, answer=str(calculate(calc, *physical)))
    return w, key


def decision_states(world, key):
    """Eight command targets per world. Only the observation is model input."""
    state = State(world.budget)
    for field in FIELDS[world.calculator]:
        for role, command in (("query", f"QUERY {field}"), ("bind", f"BIND {field} {key['good'][field]}")):
            yield role, observe(world, state), command
            state, _, _ = command_step(world, state, command)
    yield "natural", observe(world, state), "CALCULATE"
    field = key["focus"]
    correct = f"BIND {field} {key['good'][field]}"
    for role, bind, target in (("paired_clean", correct, "CALCULATE"),
                               ("paired_bad", f"BIND {field} {key['bad']}", correct)):
        side, _, _ = command_step(world, state, bind)
        yield role, observe(world, side), target
    end, _, _ = command_step(world, state, "CALCULATE")
    yield "submit", observe(world, end), "SUBMIT"


def read_evidence_panel(path):
    """Read a published JSONL shard into executable worlds and separate keys."""
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            spec = row["world"]
            records = tuple(Record(**dict(r, value=Decimal(r["value"]))) for r in spec["records"])
            yield EvidenceWorld(**dict(spec, records=records, feed_units=tuple(spec.get("feed_units", ())))), row["key"]
