"""Read-only inherited evidence verification; never rerun historical measurements."""

import argparse
import json
from pathlib import Path

from preflight_module10 import verify as inherited_verify

from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file

ROOT = Path(__file__).resolve().parents[1]


def verify(*, check_historical_source=False):
    inherited = inherited_verify()
    directory = ROOT / "artifacts/triage/module10_v2"
    seal = directory / "final_validation.json"
    if hash_file(seal) != "f085c6df2f29186e7b7c51a1769b18997ac06e3262098e301fe297381b8e331b":
        raise ValueError("Module10 final seal mismatch")
    final = json.loads(seal.read_bytes())
    if not final["complete"] or final["scientific_test_evaluated"]:
        raise ValueError("Module10 prerequisite state mismatch")
    for name, digest in final["artifacts"].items():
        if hash_file(directory / name) != digest:
            raise ValueError(f"Module10 artifact mismatch: {name}")
    sources = json.loads((directory / "source_hashes.json").read_bytes())
    if check_historical_source:
        for name, digest in sources.items():
            if hash_file(ROOT / name) != digest:
                raise ValueError(f"Module10 starting source mismatch: {name}")
    return {
        "inherited": inherited,
        "module10_seal_sha256": hash_file(seal),
        "module10_artifacts_verified": len(final["artifacts"]),
        "module10_source_files_verified": len(sources) if check_historical_source else None,
        "new_inference": False,
        "test_evaluated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-historical-source", action="store_true")
    args = parser.parse_args()
    result = verify(check_historical_source=args.check_historical_source)
    write_immutable(args.output, canonical(result))
    print(json.dumps(result, indent=2))
