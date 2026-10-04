"""Decision-only and durable-event costs, using frozen profiles and synthetic snapshots."""

import argparse
import io
import tempfile
import time
from pathlib import Path

from edge_triage.calibration.profiling import summarize
from edge_triage.config import load_config
from edge_triage.contracts import Candidate, ResourceSnapshot, TierLatency, utc_now
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.event_logging import EventLogger
from edge_triage.health import hash_file
from edge_triage.scheduling.policies import (
    MemoryContentionAwareAdmissionPolicy,
    PolicyContext,
    ResourceQueueAwareTierPolicy,
)
from edge_triage.scheduling.profiles import scheduler_profiles


def benchmark(output: Path, repeats: int = 1000) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/module07_v1.yaml")
    profiles = scheduler_profiles(config)
    now = time.monotonic_ns()
    snapshot = ResourceSnapshot(
        timestamp_utc=utc_now(),
        monotonic_ns=now,
        snapshot_id="benchmark-snapshot",
        cpu_percent=10,
        available_ram_bytes=16_000_000_000,
        queue_length=0,
        arrival_rate_per_s=None,
        recent_latency_by_tier=tuple(
            TierLatency(tier=t, mean_ms=p.latency_ms, samples=10) for t, p in profiles.items()
        ),
    )
    candidate = Candidate(
        run_id="benchmark",
        trace_id="benchmark",
        candidate_id="benchmark",
        dataset_index=0,
        arrival_monotonic_ns=now,
    )
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        for logged in (False, True):
            logger = (
                EventLogger(
                    Path(temporary) / "events.jsonl",
                    "benchmark",
                    config.config_hash,
                    console=io.StringIO(),
                )
                if logged
                else None
            )
            context = PolicyContext(
                config.models.tier_portfolio,
                profiles,
                config.scheduler.constraints,
                clock=lambda: now,
                logger=logger,
            )
            selector = ResourceQueueAwareTierPolicy(context)
            admission = MemoryContentionAwareAdmissionPolicy(context)
            samples = {"selection": [], "admission_and_release": []}
            for i in range(repeats + 50):
                start = time.perf_counter_ns()
                selected = selector.select(candidate, snapshot)
                mid = time.perf_counter_ns()
                assert selected.selected_tier is not None
                admitted = admission.admit(candidate, selected.selected_tier, snapshot)
                assert admitted.reservation_id is not None
                admission.release(admitted.reservation_id)
                end = time.perf_counter_ns()
                if i >= 50:
                    samples["selection"].append((mid - start) / 1e6)
                    samples["admission_and_release"].append((end - mid) / 1e6)
            if logger:
                logger.close()
            results.append(
                {
                    "durable_logging": logged,
                    "warmup": 50,
                    "repeats": repeats,
                    "latency_ms": {k: summarize(v).model_dump() for k, v in samples.items()},
                    "raw_ms": samples,
                }
            )
    write_immutable(
        output,
        canonical(
            {
                "scope": "synthetic snapshots; frozen CPU profiles; no inference or labels; "
                "validation, "
                "metadata construction and lease cleanup included; console redirected to StringIO",
                "config_sha256": config.config_hash,
                "profiles_sha256": hash_file(config.scheduler.profiles),
                "results": results,
                "test_evaluated": False,
            }
        ),
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    benchmark(args.output, args.repeats)
