from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, pstdev


@dataclass(frozen=True)
class Step:
    observation: str
    process: float
    cost: int
    generated_tokens: int
    trainable: bool = True


@dataclass(frozen=True)
class Trace:
    instance: str
    terminal: float
    steps: tuple[Step, ...]
    fork: tuple[int, int] | None = None


@dataclass(frozen=True)
class Credit:
    suffix: dict
    local: dict
    episode: dict
    tier: dict
    advantages: dict
    token_counts: dict
    sampled_tokens: int


def verifier_reward(checks):
    """A check is (weight, score). None denotes a counted abstention."""
    pairs = tuple(checks)
    if any(weight <= 0 or (score is not None and not -1 <= score <= 1)
           for weight, score in pairs):
        raise ValueError("invalid verifier weight or score")
    numerator = sum(weight * score for weight, score in pairs if score is not None)
    return numerator / sum(weight for weight, _ in pairs) if pairs else 0.0


def choose_anchors(turns, action_budget, rng):
    """Candidates are (turn index, pre-action cost already spent)."""
    half = action_budget // 2
    early = [turn for turn, spent in turns if spent < half]
    late = [turn for turn, spent in turns if half <= spent < action_budget]
    return [rng.choice(candidates) for candidates in (early, late) if candidates]


def assign_credit(traces, *, process_weight=0.05, cost_weight=0.02, gamma=1.0,
                  beta=1.0, epsilon=1e-8, trajectory_process=False):
    """Assign Eq. 4–7 credit. Branch traces contain only newly sampled turns."""
    if not 0 < gamma <= 1 or min(process_weight, cost_weight, beta) < 0 or epsilon <= 0:
        raise ValueError("invalid credit scale")
    if not traces:
        raise ValueError("empty batch")
    base = defaultdict(list)
    exact = defaultdict(list)
    suffix = {}
    all_steps = {}
    for i, trace in enumerate(traces):
        if not trace.steps or not isfinite(trace.terminal):
            raise ValueError("invalid trace")
        if trace.fork is None:
            base[trace.instance].append(i)
        else:
            parent, t = trace.fork
            if not (0 <= parent < len(traces) and 0 <= t < len(traces[parent].steps)):
                raise ValueError("invalid fork parent or anchor")
            stem = traces[parent]
            if stem.fork is not None or stem.instance != trace.instance:
                raise ValueError("branch stem is outside the base instance")
            if trace.steps[0].observation != stem.steps[t].observation:
                raise ValueError("fork did not restore the pre-action observation")
            exact[trace.fork].append((i, 0))
        accumulated_process = 0.0
        accumulated_cost = 0
        for t in reversed(range(len(trace.steps))):
            step = trace.steps[t]
            if not -1 <= step.process <= 1 or step.cost < 0 or step.generated_tokens < 0:
                raise ValueError("invalid step accounting")
            accumulated_process = step.process + gamma * accumulated_process
            accumulated_cost += step.cost
            suffix[i, t] = (trace.terminal + process_weight * accumulated_process
                            - cost_weight * accumulated_cost)
            all_steps[i, t] = step

    local = {}
    tiers = {}

    def assign_group(group, tier):
        if len(group) < 2:
            return
        total = sum(suffix[key] for key in group)
        for key in group:
            local[key] = suffix[key] - (total - suffix[key]) / (len(group) - 1)
            tiers[key] = tier

    for anchor, siblings in exact.items():
        assign_group([anchor, *siblings], "fork")

    collisions = defaultdict(list)
    for key, step in all_steps.items():
        if key not in local:
            collisions[traces[key[0]].instance, step.observation].append(key)
    for group in collisions.values():
        assign_group(group, "collision")

    pools = defaultdict(list)
    for i, trace in enumerate(traces):
        if trace.fork is None:
            for t in range(len(trace.steps)):
                if (i, t) not in local:
                    pools[trace.instance, t].append((i, t))
    for group in pools.values():
        assign_group(group, "turn")

    episode = {}
    for ids in base.values():
        scores = [suffix[i, 0] if trajectory_process else traces[i].terminal for i in ids]
        mean = fmean(scores)
        scale = pstdev(scores) + epsilon
        episode.update({i: (score - mean) / scale for i, score in zip(ids, scores)})
    local_scale = (pstdev(local.values()) if local else 0.0) + epsilon
    advantages = {}
    token_counts = {}
    for key, step in all_steps.items():
        i, _ = key
        stem = traces[i].fork is None
        valid = step.trainable and (stem or key in local)
        token_counts[key] = step.generated_tokens if valid else 0
        advantages[key] = ((episode[i] if stem else 0.0)
                           + beta * local.get(key, 0.0) / local_scale) if valid else 0.0
    return Credit(
        suffix, local, episode, tiers, advantages, token_counts,
        sum(step.generated_tokens for step in all_steps.values()),
    )
