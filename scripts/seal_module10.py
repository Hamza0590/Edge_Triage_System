"""Seal completed Module10 evidence after analysis and successful final checks."""

import json
from pathlib import Path

from preflight_module10 import verify

from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/triage/module10_v2"


def main():
    checks = json.loads((OUT / "checks.json").read_bytes())
    assert checks and all(c["exit_code"] == 0 for c in checks)
    assert any("passed" in c["output"] and "pytest" in c["command"] for c in checks)
    inherited = verify()
    analysis = OUT / "analysis"
    summary = json.loads((analysis / "summary.json").read_bytes())
    assert summary["upstream_configurations"] == 36 and summary["fate_variants"] == 144
    assert summary["collection_candidates"] == summary["frozen_runtime_replay_candidates"] == 23148
    assert summary["fit_feasible"] == 144 and not summary["test_evaluated"]
    assert summary["primary_fit"]["metrics"]["survival_recall"] >= 0.95
    assert summary["primary_replay"]["status"] == "verified_feasible"
    assert len(summary["seed_diagnostics"]) == 9
    for name, digest in json.loads((analysis / "artifact_hashes.json").read_bytes()).items():
        assert hash_file(analysis / name) == digest
    for name, digest in json.loads((analysis / "verified_source_hashes.json").read_bytes()).items():
        assert hash_file(ROOT / name) == digest
    failures = 0
    for setting in json.loads((OUT / "matrix.json").read_bytes())["settings"]:
        directory = OUT / "settings" / setting["name"]
        for filename in ("collection.json", "runtime_replay.json"):
            evidence = json.loads((directory / filename).read_bytes())
            failures += evidence["failed"]
            for name, digest in evidence["hashes"].items():
                assert hash_file(Path(evidence["run"]) / name) == digest
    source = {
        str(p.relative_to(ROOT)): hash_file(p)
        for directory in ("src", "tests", "scripts", "configs", "docs", "references")
        for p in sorted((ROOT / directory).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".md", ".yaml", ".json")
    }
    source["README.md"] = hash_file(ROOT / "README.md")
    write_immutable(OUT / "source_hashes.json", canonical(source))
    final = {
        "module": 10,
        "complete": True,
        "checks": checks,
        "inherited_integrity": inherited,
        "candidate_failures": failures,
        "configurations": 36,
        "frozen_fate_artifacts": 144,
        "collection_candidates": 23148,
        "replay_candidates": 23148,
        "seed_diagnostic_candidates": 9 * 643,
        "scientific_test_evaluated": False,
        "proposed_uq": "none",
        "proposed_escalation": "none",
        "dynamic_cap": 1,
        "primary_fit": summary["primary_fit"]["metrics"],
        "primary_replay": summary["primary_replay"],
        "replay_statuses": summary["replay_statuses"],
        "source_files": len(source),
        "raw_md5": {
            name: hash_file(ROOT / name, "md5")
            for name in ("MeerLICHT_images.npy", "MeerLICHT_labels.csv")
        },
        "limitations": [
            "Conditional validation fit/bootstrap; no held-out recall guarantee",
            "Three training seeds, fixed split, unknown repeated-object dependence",
            "Quarter-size logical payload; no codec, fidelity or network measurements",
            "Other fate ablations reuse fixed upstream evidence; measured per-tier full replays",
            "Desktop CPU serving, inherited dynamic admission slowdown; no Jetson claim",
            "Thresholds bind exact validated system and trace; changed behavior needs validation",
        ],
        "artifacts": {
            str(p.relative_to(OUT)): hash_file(p)
            for p in sorted(OUT.rglob("*"))
            if p.is_file() and p.name != "final_validation.json"
        },
    }
    write_immutable(OUT / "final_validation.json", canonical(final))
    print(
        json.dumps(
            {
                "complete": True,
                "source_files": len(source),
                "sha256": hash_file(OUT / "final_validation.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
