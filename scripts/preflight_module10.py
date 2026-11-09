"""Verify frozen prerequisites without rerunning historical measurements."""

import json
from pathlib import Path

from preflight_module09 import verify as verify_inherited

from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.uncertainty.artifacts import load_uncertainty

ROOT = Path(__file__).resolve().parents[1]


def verify():
    inherited = verify_inherited()
    directory = ROOT / "artifacts/runtime/module09_v2"
    final_path = directory / "final_validation.json"
    assert (
        hash_file(final_path) == "15292d816c08d6ff2ed9cbd843dd24b6306613bab5acc26175b940eaf0d951b4"
    )
    final = json.loads(final_path.read_bytes())
    assert final["complete"] and not final["scientific_test_evaluated"]
    assert final["candidate_failures"] == 0 and set(final["caps"].values()) == {1}
    for name, digest in final["artifacts"].items():
        assert hash_file(directory / name) == digest, name
    uq = ROOT / "artifacts/uncertainty/20260925T205209_b5ff8ad7f32c/routing_5.json"
    load_uncertainty(uq, "19f7d021ede74c676e30282b39e6682b3005899722f54ab918cdd2b2e5c54604")
    return {
        "inherited": inherited,
        "module09_final_sha256": hash_file(final_path),
        "module09_artifacts_verified": len(final["artifacts"]),
        "gate_d_pass": False,
        "new_inference": False,
        "test_evaluated": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        write_immutable(args.output, canonical(result))
    print(json.dumps(result, indent=2))
