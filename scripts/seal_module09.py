"""Seal already verified Module09 results and actual tool check outputs."""

import argparse
import json
from pathlib import Path

from edge_triage.calibration.profiling import hardware_identity
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    checks = json.loads((out / "checks.json").read_bytes())
    assert all(c["exit_code"] == 0 for c in checks)
    assert "317 passed" in checks[0]["output"]
    fixed = json.loads((out / "fixed_verification.json").read_bytes())
    components = json.loads((out / "components_verification.json").read_bytes())
    replay = json.loads((out / "verification.json").read_bytes())
    assert fixed["completed"] == 10080 and fixed["failed"] == 0
    assert components["verified_candidates"] == 17280
    assert replay["completed"] == 1200 and replay["failed"] == 0
    assert replay["gate_d_pass"] is False and set(replay["caps"].values()) == {1}
    for name, digest in fixed["settings"].items():
        assert hash_file(out / name) == digest
    for result in components["settings"]:
        assert not result["skipped"] and hash_file(out / result["file"]) == result["sha256"]
    for run in replay["runs"]:
        for name, digest in run["hashes"].items():
            assert hash_file(Path(run["run"]) / name) == digest
    hardware_id, hardware = hardware_identity()
    assert hardware_id == "63998d2133ba053a", hardware_id
    source = {
        str(path.relative_to(ROOT)): hash_file(path)
        for directory in ("src", "tests", "scripts", "configs", "docs", "references")
        for path in sorted((ROOT / directory).rglob("*"))
        if path.is_file() and path.suffix in (".py", ".md", ".yaml", ".json")
    }
    source["README.md"] = hash_file(ROOT / "README.md")
    write_immutable(out / "source_hashes.json", canonical(source))
    raw_hashes = {
        "MeerLICHT_images.npy": hash_file(ROOT / "MeerLICHT_images.npy", "md5"),
        "MeerLICHT_labels.csv": hash_file(ROOT / "MeerLICHT_labels.csv", "md5"),
    }
    assert list(raw_hashes.values()) == [
        "bbf55c1cecbdea127497d6cd346e3455",
        "cacf76a6aadd97ed18ab24b58e121bc2",
    ]
    final = {
        "module": 9,
        "complete": True,
        "checks": checks,
        "fixed_completed": fixed["completed"],
        "component_outputs": components["verified_candidates"],
        "dynamic_completed": replay["completed"],
        "candidate_failures": 0,
        "gate_d_pass": False,
        "caps": replay["caps"],
        "proposed_uq": "none",
        "proposed_escalation": "none",
        "scientific_test_evaluated": False,
        "hardware_id": hardware_id,
        "hardware": hardware,
        "raw_hashes": raw_hashes,
        "source_files": len(source),
        "limitations": [
            "CPU desktop only; full pipeline reached at most two active workers at caps2/4/8",
            "Dynamic cap1 delays under inherited latency-ratio pressure; no performance benefit",
            "Component timing excludes durable pipeline instrumentation; process memory replicated",
            "Manifest startup and process-pipe failures preserved; approved retries completed",
        ],
        "artifacts": {
            str(p.relative_to(out)): hash_file(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "final_validation.json"
        },
    }
    write_immutable(out / "final_validation.json", canonical(final))
    print(
        json.dumps(
            {
                "complete": True,
                "sha256": hash_file(out / "final_validation.json"),
                "source_files": len(source),
                "hardware_id": hardware_id,
            }
        )
    )


if __name__ == "__main__":
    main()
