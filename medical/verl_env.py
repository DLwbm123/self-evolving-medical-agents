"""Text-only environments implementing verl-agent's batched rollout contract."""
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from medical.calculation import State, outcome
from medical.evidence import command_step, observe, read_evidence_panel


class EvidenceEpisode:
    def __init__(self, case):
        self.world, self.key = case
        self.state = State(self.world.budget)
        self.text = self.prompt()

    def prompt(self):
        return ("Return one command. QUERY each required field, BIND <field> <pointer> from "
                "this encounter within the inclusive age limit with a supported unit, then "
                "CALCULATE and SUBMIT.\n" + observe(self.world, self.state))

    def step(self, text):
        self.state, potential, _ = command_step(self.world, self.state, text.strip())
        self.text = self.prompt()
        ref = Decimal(self.key["answer"])
        won = self.state.done and outcome(self.state, ref-Decimal(".01"), ref+Decimal(".01")) == 1
        valid = self.state.last_result not in ("invalid action", "invalid evidence", "not found",
                                              "missing input", "budget exhausted")
        return float(won), self.state.done, dict(won=bool(won), is_action_valid=valid,
                                                evidence_potential_delta=potential)


class FhirEpisode:
    """Official v2 tasks, GET against the configured benchmark service and simulated POST."""
    def __init__(self, case, base, functions, get, scorer, gated):
        self.case, self.base, self.get, self.scorer, self.gated = case, base, get, scorer, gated
        self.history, self.agent_reads = [], []
        self.turns = 0
        self.text = (f"Use one command per turn: GET <url>, POST <url> followed by JSON on a new "
                     f"line, or FINISH(<JSON list>). FHIR base: {base}\n{functions}\n"
                     f"Context: {case['context']}\nQuestion: {case['instruction']}")

    def step(self, text):
        from benchmarks.audit_medagentbench import CHECK_CODES, has_required_read

        self.turns += 1
        cmd = text.strip()
        self.history.append(SimpleNamespace(role="agent", content=cmd))
        done = valid = official = gated_win = False
        message = "Invalid command."
        if cmd.startswith("GET "):
            url = urlsplit(cmd[4:].strip())
            base = urlsplit(self.base)
            # Expose only direct resource searches on the configured FHIR service.
            resource = url.path.removeprefix(base.path)
            if ((url.scheme, url.netloc) == (base.scheme, base.netloc) and
                    url.path.startswith(base.path) and resource.isascii() and resource.isalpha()):
                params = [(k,v) for k,v in parse_qsl(url.query) if k != "_format"]
                target = urlunsplit((base.scheme, base.netloc, url.path,
                                     urlencode(params+[("_format", "json")]), ""))
                status, body = self.get(target)
                self.agent_reads.append((target, status))
                valid = status == 200
                message = body if valid else f"FHIR returned HTTP {status}."
        elif cmd.startswith("POST "):
            try:
                json.loads(cmd.partition("\n")[2])
            except ValueError:
                message = "Invalid POST request"
            else:
                valid = True
                message = "POST request accepted and executed successfully."
        elif cmd.startswith("FINISH(") and cmd.endswith(")"):
            done = True
            result = cmd[7:-1]
            try:
                valid = isinstance(json.loads(result), list)
            except ValueError:
                pass
            if valid:
                family = self.case["id"].split("_")[0]
                official = getattr(self.scorer, family)(
                    self.case, SimpleNamespace(result=result, history=self.history), self.base) is True
                gated_win = official and (family not in CHECK_CODES or has_required_read(
                    family, str(self.case["eval_MRN"]), self.agent_reads))
            message = "Finished."
        else:
            done = True
        self.history.append(SimpleNamespace(role="user", content=message))
        self.text += f"\n{cmd}\n{message}"
        won = gated_win if self.gated else official
        return float(won), done or self.turns >= 8, dict(won=won, official=official, gated=gated_win, is_action_valid=valid)


class MedicalManager:
    def __init__(self, cases, episode, batch, group=1, *, fixed=False):
        if batch < 1 or group < 1 or not cases:
            raise ValueError("a nonempty task pool and positive batch/group size are required")
        self.cases, self.episode = cases, episode
        self.cursor, self.fixed = 0, fixed
        self.batch, self.group = batch, group

    def observations(self):
        texts = [s.text for s in self.slots]
        return dict(text=texts, image=None, anchor=texts)

    def reset(self, kwargs=None):
        if kwargs is None:
            indices = [(self.cursor+i) % len(self.cases) for i in range(self.batch)
                       for _ in range(self.group)]
            if not self.fixed:
                self.cursor = (self.cursor+self.batch) % len(self.cases)
        else:
            indices = [int(k["case_index"]) for k in kwargs]
            if len(indices) != self.batch*self.group or any(not 0 <= i < len(self.cases) for i in indices):
                raise ValueError("invalid rollout case indices")
            if any(len(set(indices[i:i+self.group])) != 1 for i in range(0, len(indices), self.group)):
                raise ValueError("a rollout group must share one initial case")
        self.slots = [self.episode(self.cases[i]) for i in indices]
        self.ended = [False]*len(self.slots)
        return self.observations(), [{} for _ in self.slots]

    def step(self, text_actions):
        rewards, infos = [], []
        for i, (s, action) in enumerate(zip(self.slots, text_actions, strict=True)):
            if self.ended[i]:
                reward, info = 0., dict(won=False, is_action_valid=False)
            else:
                reward, self.ended[i], info = s.step(action)
            rewards.append(reward)
            infos.append(info)
        return self.observations(), np.asarray(rewards), np.asarray(self.ended), infos

    def success_evaluator(self, total_infos, total_batch_list, **kwargs):
        wins = [any(step["won"] for step, item in zip(infos, items, strict=True) if item["active_masks"])
                for infos, items in zip(total_infos, total_batch_list, strict=True)]
        return {"success_rate": np.asarray(wins, dtype=float)}

    def close(self):
        pass


def make_medical_envs(config):
    """Use in agent_system.environments.env_manager.make_envs for medical/evidence and medical/fhir."""
    cfg = config.env.medical
    train = list(read_evidence_panel(Path(cfg.fixtures) / "train.jsonl.gz"))
    train_manager = MedicalManager(train, EvidenceEpisode, config.data.train_batch_size, config.env.rollout.n)
    if config.env.env_name == "medical/evidence":
        panel = cfg.get("panel", "iid")
        if panel not in ("iid", "age", "encounter", "unit", "pointer"):
            raise ValueError("expected a held-out evidence panel")
        val = list(read_evidence_panel(Path(cfg.fixtures) / f"{panel}.jsonl.gz"))
        return train_manager, MedicalManager(val, EvidenceEpisode, config.data.val_batch_size, fixed=True)
    if config.env.env_name != "medical/fhir":
        raise ValueError("unknown medical environment")
    import requests
    from benchmarks.audit_medagentbench import load_scorer

    root, base = Path(cfg.upstream), cfg.fhir_base.rstrip("/") + "/"
    scorer, grader_reads = load_scorer(root, base)
    data = root/"data/medagentbench"
    val = json.loads((data/"test_data_v2.json").read_text())
    functions = (data/"funcs_v1.json").read_text()
    http = requests.Session()
    http.trust_env = False

    def get(url):
        response = http.get(url, timeout=15, allow_redirects=False)
        return response.status_code, response.text

    def episode(case):
        grader_reads.clear()
        return FhirEpisode(case, base, functions, get, scorer, cfg.get("gated", False))

    # All official tasks go exclusively to validation.
    return train_manager, MedicalManager(val, episode, config.data.val_batch_size, fixed=True)
