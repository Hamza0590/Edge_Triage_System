"""Persist monitor wall-cost fractions at several sampling intervals."""

import argparse
import time
from pathlib import Path

from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import environment_info
from edge_triage.monitoring.collectors import HostCollector, NvidiaCollector
from edge_triage.monitoring.estimators import RuntimeCounters
from edge_triage.monitoring.monitor import ResourceMonitor


def benchmark(output: Path, duration: float = 2) -> None:
    results = []
    for interval in (0.05, 0.1, 0.5):
        monitor = ResourceMonitor(
            [HostCollector(), NvidiaCollector(2)],
            RuntimeCounters(),
            lambda: {"length": 0, "oldest_wait_ms": 0},
            interval_s=interval,
            path=output.parent / f"monitor_interval_{interval}.jsonl",
        )
        monitor.start()
        try:
            time.sleep(duration)
        finally:
            monitor.stop()
        results.append(
            {**monitor.overhead(), "availability": monitor.current().model_dump(mode="json")}
        )
        if monitor.failures or monitor.sample_count < 2:
            raise RuntimeError("monitor overhead benchmark failed")
    write_immutable(
        output,
        canonical(
            {
                "schema_version": 1,
                "results": results,
                "environment": environment_info(),
                "collector_gpu_refresh_s": 2,
                "warning_rule": "sampling wall fraction >10%; retain warning, never claim low cost",
                "scope": "includes startup and JSONL flush; wall time is not CPU utilization",
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.output)
