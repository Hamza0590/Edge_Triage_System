"""Seal complete Module11 evidence after saved-output verification and visual QA."""

import argparse
import json
from pathlib import Path

from preflight_module11 import verify

from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.harness import source_inventory, verify_hashes
from edge_triage.health import hash_file

ROOT = Path(__file__).resolve().parents[1]


def seal(out, checks_path):
    record = json.loads(checks_path.read_bytes())
    checks = record["checks"]
    if not checks or any(c["exit_code"] != 0 for c in checks):
        raise ValueError("final checks did not pass")
    if not any(c["name"] == "pytest" and "passed" in c["output"] for c in checks):
        raise ValueError("missing full regression evidence")
    evidence = json.loads((out / "analysis/verification.json").read_bytes())
    if evidence["counts"] != dict(complete=120, failed=0, skipped=0, pending=0):
        raise ValueError("study incomplete")
    if (
        len(evidence["component_equivalence"]) != 42
        or evidence["validation_candidates"] != 39 * 643
    ):
        raise ValueError("incomplete validation/component verification")
    summary_path, plots_path = Path(evidence["summary"]), Path(evidence["plots"])
    if (
        hash_file(summary_path) != evidence["summary_sha256"]
        or hash_file(plots_path) != evidence["plots_sha256"]
    ):
        raise ValueError("verified summaries changed")
    plots = json.loads(plots_path.read_bytes())
    qa = json.loads((out / "analysis/visual_qa.json").read_bytes())
    if set(qa["inspected"]) != set(plots["plots"]) or not qa["passed"]:
        raise ValueError("all scientific plots require visual inspection")
    verify_hashes(plots_path.parent, plots["plots"])
    resolved = json.loads((out / "study/matrix.resolved.json").read_bytes())
    if source_inventory(ROOT) != resolved["source_hashes"]:
        raise ValueError("runtime source changed since freeze")
    if record["runtime_source"] != resolved["source_hashes"]:
        raise ValueError("final checks did not run on the frozen runtime source")
    verify_hashes(ROOT, resolved["inputs"])
    sources = {
        str(p.relative_to(ROOT)): hash_file(p)
        for directory in ("src", "tests", "scripts", "configs", "docs", "references")
        for p in sorted((ROOT / directory).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".md", ".yaml", ".json")
    }
    sources["README.md"] = hash_file(ROOT / "README.md")
    write_immutable(out / "source_hashes.json", canonical(sources))
    pipeline = [r for r in evidence["subruns"] if r["kind"] == "pipeline"]
    component = [r for r in evidence["subruns"] if r["kind"] == "component"]
    result = {
        "module": 11,
        "complete": True,
        "checks": checks,
        "inherited_integrity": verify(),
        "pipeline_subruns": len(pipeline),
        "component_subruns": len(component),
        "validation_candidates": evidence["validation_candidates"],
        "pipeline_candidate_occurrences": sum(r["occurrences"] for r in pipeline),
        "component_candidate_occurrences": sum(r["occurrences"] for r in component),
        "candidate_failures": sum(r["failures"] for r in pipeline),
        "below_target_subruns": [r["id"] for r in pipeline if not r["equal_recall_eligible"]],
        "scientific_test_evaluated": True,
        "test_tuning": False,
        "test_authorization_point": hash_file(out / "study/matrix.resolved.json"),
        "proposed_uq": "none",
        "proposed_escalation": "none",
        "dynamic_cap": 1,
        "source_files": len(sources),
        "next_module": 12,
        "limitations": [
            "Unknown repeated-object independence; reused validation; no recall guarantee",
            "Three model seeds on one split; two timing repetitions; descriptive comparisons",
            "Logical compression only; no fidelity, codec, network or energy measurement",
            "Desktop CPU serving and inherited dynamic slowdown; no Jetson claim",
            "Component timings exclude pipeline services; no equal-recall component cost claim",
        ],
        "artifacts": {
            str(p.relative_to(out)): hash_file(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "final_validation.json"
        },
    }
    write_immutable(out / "final_validation.json", canonical(result))
    print(json.dumps({"complete": True, "sha256": hash_file(out / "final_validation.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/module11_v3")
    parser.add_argument(
        "--checks", type=Path, required=True, help="record_checks_module11.py output"
    )
    args = parser.parse_args()
    seal(args.output.resolve(), args.checks.resolve())
