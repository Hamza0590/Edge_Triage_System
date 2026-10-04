"""Read-only integrity preflight; never invokes prior module source-tree verifiers."""

import json
from pathlib import Path

from edge_triage.calibration.profiling import load_profiles
from edge_triage.contracts import ResourceSnapshot
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "artifacts/models/family_20260924T204507_5a8771c620a0.json": (
        "884719bf93667c5b953c229eb38689c130fa5d5fd40a979a54dc4bdda60386fa"
    ),
    "artifacts/calibration/20260925T073859_3c77b7896ab5/index.json": (
        "b5c2aed682bf10c5916688c3ba34d84c6b612dcb5f2cedcfd502a505c1f13779"
    ),
    "artifacts/profiles/20260925T073859_3c77b7896ab5/model_profiles.json": (
        "1f4ec864fe80da168ec9c400b3cc146eb15e46decc3cf9635c3a0ca4b21b16b8"
    ),
    "artifacts/profiles/20260925T073859_3c77b7896ab5/gate_b.json": (
        "38ad1cadf37a37da2f3f5b95b947454d967e688c1ae5f3a2b19d16aa6c6656a6"
    ),
    "artifacts/traces/suite_a515d472c9d5ef05.json": (
        "a515d472c9d5ef05d718e1ae3dc8d34c54fdd5cbd86672125aff71a9021fe088"
    ),
    "artifacts/monitoring/module06_v1/verification_v2.json": (
        "3efda443d260f35dc8e99e4ddbabbd2016d8c42e63125b6b9fe8a3da910966c9"
    ),
}


def verify():
    for path, digest in PINS.items():
        assert hash_file(ROOT / path) == digest, path
    family, calibration, profiles, gate, suite, verification = [
        json.loads((ROOT / path).read_text()) for path in PINS
    ]
    manifests = {}
    for entry in family["checkpoints"].values():
        path = Path(entry["directory"]) / "checkpoint_manifest.json"
        assert hash_file(path) == entry["manifest_sha256"]
        manifest = CheckpointManifest.model_validate_json(path.read_bytes())
        manifests[entry["checkpoint_id"]] = (manifest, entry["manifest_sha256"])
    profile_path = ROOT / list(PINS)[2]
    bundle = load_profiles(profile_path, PINS[list(PINS)[2]], manifests)
    for entry in calibration["calibrations"].values():
        assert hash_file(Path(entry["path"])) == entry["sha256"]
    assert gate["pass"] is True
    split_hash = hash_file(ROOT / "artifacts/data/split_seed20260924_v1.csv")
    for entry in suite["traces"]:
        assert hash_file(Path(entry["path"])) == entry["sha256"]
        assert len(load_trace(Path(entry["path"]), split_hash)) == 120
    samples = 0
    for run in verification["runs"]:
        path = Path(run["run"]) / "resources.jsonl"
        assert hash_file(path) == run["summary"]["artifacts"]["resources.jsonl"]
        for line in path.read_text().splitlines():
            ResourceSnapshot.model_validate_json(line)
            samples += 1
    return {
        "pins": PINS,
        "profiles": len(bundle.profiles),
        "monitor_samples": samples,
        "traces": len(suite["traces"]),
        "gate_b": True,
        "test_evaluated": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
