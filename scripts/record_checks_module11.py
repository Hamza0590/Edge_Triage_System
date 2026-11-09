"""Run the repository checks and record their actual outcomes immutably for the Module11 seal."""

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.harness import source_inventory

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {
    "pytest": [
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-W",
        "error",
        "-o",
        "faulthandler_timeout=60",
        "-p",
        "no:cacheprovider",
    ],
    "ruff": ["-m", "ruff", "check", "."],
    "mypy": ["-m", "mypy", "src"],
    "dependencies": ["-m", "pip", "check"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for name, arguments in CHECKS.items():
        command = [sys.executable, *arguments]
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-5:])
        results.append(
            {
                "name": name,
                "command": " ".join(arguments),
                "exit_code": done.returncode,
                "output": tail,
            }
        )
        print(
            f"{name}: exit {done.returncode}: {tail.splitlines()[-1] if tail else ''}", flush=True
        )
    record = {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "runtime_source": source_inventory(ROOT),
        "checks": results,
    }
    write_immutable(args.output, canonical(record))
    if any(r["exit_code"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
