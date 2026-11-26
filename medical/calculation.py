"""Deterministic tool environment for initial MAP and BMI diagnostics."""

import ast
import csv
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, localcontext
import json

from medical.credit import verifier_reward


FIELDS = {
    "5": ("Systolic Blood Pressure", "Diastolic Blood Pressure"),
    "6": ("weight", "height"),
}
WEIGHT = {"kg": "1", "g": "0.001", "lbs": "0.45359237", "lb": "0.45359237"}
HEIGHT = {"m": "1", "cm": "0.01", "in": "0.0254", "ft": "0.3048"}


@dataclass(frozen=True)
class World:
    calculator: str
    facts: tuple[tuple[str, str, Decimal], ...]
    budget: int = 8


@dataclass(frozen=True)
class State:
    remaining: int
    seen: tuple[str, ...] = ()
    bindings: tuple[tuple[str, str], ...] = ()
    calculated: Decimal | None = None
    submitted: Decimal | None = None
    done: bool = False
    last_result: str = ""


@dataclass(frozen=True)
class Action:
    kind: str
    field: str = ""
    value: str = ""


def observation(world, state):
    return json.dumps({
        "calculator": world.calculator,
        "remaining": state.remaining,
        "seen": [(pointer, field, str(value)) for pointer, field, value in world.facts
                 if pointer in state.seen],
        "bindings": state.bindings,
        "calculated": str(state.calculated),
        "last_result": state.last_result,
        "done": state.done,
    }, sort_keys=True, separators=(",", ":"))


def calculate(calculator, first, second):
    with localcontext() as ctx:
        ctx.prec = 28
        if calculator == "5":
            return (first + 2 * second) / Decimal(3)
        if calculator == "6":
            return first / (second * second)
    raise ValueError("unknown calculator")


def step(world, state, action, *, checks=None):
    """Return (next state, process reward, charged cost). World excludes gold."""
    def scored(next_state, cost, *verdicts):
        if checks is not None:
            checks.update((name, score) for name, _, score in verdicts)
        return next_state, verifier_reward((weight, score) for _, weight, score in verdicts), cost

    if state.done:
        raise ValueError("episode is complete")
    facts = {ptr: (field, value) for ptr, field, value in world.facts}
    if action.kind == "finish":
        try:
            answer = Decimal(action.value)
            if not answer.is_finite():
                answer = None
        except InvalidOperation:
            answer = None
        grounded = None
        if answer is not None:
            grounded = 1 if (state.calculated is not None
                             and state.calculated - Decimal("0.01") <= answer
                             <= state.calculated + Decimal("0.01")) else -1
        return scored(replace(state, submitted=answer, done=True, last_result="finished"), 0,
                      ("answer_format", 2, 1 if answer is not None else -1),
                      ("answer_grounding", 2, grounded))
    if state.remaining == 0:
        return scored(replace(state, done=True, last_result="budget exhausted"), 0,
                      ("action_budget", 1, -1))
    next_state = replace(state, remaining=state.remaining - 1)
    if action.kind == "query":
        match = next((ptr for ptr, field, _ in world.facts if field == action.field), None)
        if match is None:
            return scored(replace(next_state, last_result="not found"), 1,
                          ("query_field", 2, -1), ("query_novelty", 1, None))
        repeated = match in state.seen
        seen = state.seen if repeated else tuple(sorted((*state.seen, match)))
        return scored(replace(next_state, seen=seen, last_result=match), 1,
                      ("query_field", 2, 1), ("query_novelty", 1, -1 if repeated else 1))
    if action.kind == "bind":
        valid = (action.value in state.seen
                 and facts[action.value][0] == action.field)
        bindings = dict(state.bindings)
        if valid:
            bindings[action.field] = action.value
        return scored(replace(next_state, bindings=tuple(sorted(bindings.items())),
                              calculated=None if valid else state.calculated,
                              last_result="bound" if valid else "invalid evidence"), 1,
                      ("binding", 2, 1 if valid else -1))
    if action.kind == "calculate":
        bindings = dict(state.bindings)
        fields = FIELDS[world.calculator]
        value = (calculate(world.calculator, *(facts[bindings[field]][1] for field in fields))
                 if all(field in bindings for field in fields) else None)
        return scored(replace(next_state, calculated=value,
                              last_result="computed" if value is not None else "missing input"), 1,
                      ("calculator_inputs", 2, 1 if value is not None else -1))
    return scored(replace(next_state, last_result="invalid action"), 1, ("action_format", 1, -1))


def outcome(state, lower, upper):
    return 1.0 if state.submitted is not None and lower <= state.submitted <= upper else -1.0


def read_cases(csv_path, row_numbers=None):
    """Use the released annotations to build the typed world for a diagnostic."""
    with csv_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            calculator = row["Calculator ID"]
            if calculator not in FIELDS:
                continue
            if row_numbers is not None and int(row["Row Number"]) not in row_numbers:
                continue
            try:
                annotated = ast.literal_eval(row["Relevant Entities"])
                values = []
                for field in FIELDS[calculator]:
                    raw_value, raw_unit = annotated[field]
                    value, unit = Decimal(str(raw_value)), raw_unit.strip().lower()
                    if calculator == "5":
                        if unit not in {"mmhg", "mm hg"}:
                            raise ValueError("unsupported pressure unit")
                    else:
                        value *= Decimal((WEIGHT if field == "weight" else HEIGHT)[unit])
                    if not value.is_finite() or value <= 0:
                        raise ValueError("invalid input")
                    values.append((f"e{len(values) + 1}", field, value))
                lower, upper = Decimal(row["Lower Limit"]), Decimal(row["Upper Limit"])
                if not lower.is_finite() or not upper.is_finite() or lower > upper:
                    raise ValueError("invalid ground-truth bounds")
            except (ValueError, SyntaxError, KeyError, TypeError, InvalidOperation):
                yield calculator, row["Note ID"], None
                continue
            yield calculator, row["Note ID"], (World(calculator, tuple(values)), lower, upper)
