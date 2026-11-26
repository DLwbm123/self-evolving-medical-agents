import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, build_opener


def load_scorer(root, base):
    scorer_dir = root / "src/server/tasks/medagentbench"
    package = ModuleType("_benchmark_reference")
    package.__path__ = [str(scorer_dir)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".refsol", scorer_dir / "refsol.py"
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    endpoint = urlsplit(base)
    opener = build_opener(ProxyHandler({}))
    reads = []

    def read_fhir(url):
        parsed = urlsplit(url)
        if (parsed.scheme, parsed.netloc) != (endpoint.scheme, endpoint.netloc):
            raise ValueError("grader requested a different endpoint")
        if not parsed.path.startswith(endpoint.path):
            raise ValueError("grader requested a path outside the FHIR base")
        with opener.open(url, timeout=15) as response:
            data = response.read().decode("utf-8")
            if response.headers.get("Content-Type") == "application/json":
                data = json.loads(data)
            reads.append((url, response.status))
            return {"status_code": response.status, "data": data}

    scorer.send_get_request = read_fhir
    scorer.print = lambda *args, **kwargs: None
    return scorer, reads


CHECK_CODES = {"task5": "MG", "task9": "K", "task10": "A1C"}


def has_required_read(family, patient, reads):
    for url, status in reads:
        parsed = urlsplit(url)
        if status != 200 or not parsed.path.endswith("/Observation"):
            continue
        query = parse_qs(parsed.query)
        if (patient in query.get("patient", [])
                and CHECK_CODES[family] in query.get("code", [])):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fhir", required=True)
    args = parser.parse_args()
    data = args.root / "data/medagentbench"
    cases = json.loads((data / "test_data_v2.json").read_text())
    query = set(json.loads((data / "query_task_ids.json").read_text()))
    assert len(cases) == 300 and len(query) == 150
    assert query <= {case["id"] for case in cases}
    scorer, verifier_reads = load_scorer(args.root, args.fhir)
    reply = "FINISH([])"
    result = SimpleNamespace(
        result=reply[len("FINISH("):-1],
        history=[SimpleNamespace(role="agent", content=reply)],
    )
    counts = defaultdict(Counter)
    controls = Counter()
    for case in cases:
        family = case["id"].split("_")[0]
        verifier_reads.clear()
        try:
            outcome = "pass" if getattr(scorer, family)(case, result, args.fhir) is True else "fail"
        except Exception as exc:
            outcome = "error"
            counts["error_types"][type(exc).__name__] += 1
        counts[family][outcome] += 1
        counts["all"][outcome] += 1
        if case["id"] in query:
            counts["query"][outcome] += 1
        if outcome == "pass" and family in CHECK_CODES:
            patient = str(case["eval_MRN"])
            # Empty agent histories contain no agent-issued reads. The verifier's
            # own successful GETs are used exclusively as oracle positive controls.
            controls["gated_null_pass"] += has_required_read(family, patient, [])
            controls["oracle_relevant_read_pass"] += has_required_read(
                family, patient, verifier_reads
            )
            controls["wrong_patient_read_pass"] += has_required_read(
                family, patient + "-control", verifier_reads
            )
            controls["checked_official_passes"] += 1
    counts["evidence_controls"] = controls
    args.output.write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps({key: counts[key] for key in ("all", "query")}))
    if counts["all"]["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
