"""Read-only Module09 prerequisites; validates evidence without rerunning old experiments."""

import json
from pathlib import Path

from preflight_module07 import verify as verify_inherited

from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest, read_index

ROOT = Path(__file__).resolve().parents[1]
UQ = ROOT / "artifacts/uncertainty/20260925T205209_b5ff8ad7f32c"
PINS = {
    "final_validation.json": "931ade434cd19aab28e9fe65789ef3c518efb754ba9a1d220945c3d5ab30e634",
    "verification.json": "a59ae8f5650bd579e5607ec38232f3daa459da1d57ff5e6d2dedad03b18e2640",
    "gate_c.json": "f12e15ce861546911ddc05b9fce3ac0587fce61b6aaaaa9d71284b263031436f",
}


def verify():
    inherited = verify_inherited()
    for name, digest in PINS.items():
        assert hash_file(UQ / name) == digest, name
    final = json.loads((UQ / "final_validation.json").read_bytes())
    replay = json.loads((UQ / "verification.json").read_bytes())
    assert final["complete"] and final["checks"][0]["passed"] == 282
    assert final["gate_c_pass"] is False
    assert final["proposed_uq"] == final["proposed_escalation"] == "none"
    assert replay["completed"] == final["replay_completed"] == 1080
    assert final["replay_failed"] == 0 and not final["scientific_test_evaluated"]
    for name, digest in final["raw_hashes"].items():
        assert hash_file(ROOT / name, "md5") == digest, name
    family = read_index(ROOT / "artifacts/models/family_20260924T204507_5a8771c620a0.json")
    for entry in family["checkpoints"].values():
        directory = Path(entry["directory"])
        manifest = CheckpointManifest.model_validate_json(
            (directory / "checkpoint_manifest.json").read_bytes()
        )
        for name, digest in manifest.file_hashes.items():
            assert hash_file(directory / name) == digest, name
    for run in replay["runs"]:
        directory = Path(run["run"])
        assert hash_file(directory / "manifest.json") == run["manifest_sha256"]
        for name, digest in run["summary"]["artifacts"].items():
            assert hash_file(directory / name) == digest, name
    return {
        "inherited": inherited,
        "module08_pins": PINS,
        "verified_prior_replay_candidates": 1080,
        "new_inference": False,
        "test_evaluated": False,
        "proposed_uq": "none",
        "proposed_escalation": "none",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
