# Module 06 — Resource Monitor, Candidate Queue and Streaming Workload

## Mission

Implement low-overhead resource observation and reproducible candidate-arrival workloads. Connect them to the walking skeleton without yet enabling the final adaptive scheduler.

## Mandatory preflight

1. Read the master, context and prerequisite modules.
2. Verify model profile artifacts exist.
3. Run the sequential real-model smoke test.
4. Mark Module 06 `IN_PROGRESS`.

## Research basis and adoption

| Source | Mechanism | Adoption |
|---|---|---|
| PRAS, DOI `10.1016/j.sysarc.2025.103597` | React after request arrival using current queued work rather than perfect workload prediction | Inspired/adapted |
| OCTOPINF, arXiv `2502.01277` | Workload-aware serving, resource allocation and batching under dynamic edge conditions | Inspired |
| Quilt, DOI `10.1016/j.sysarc.2026.103696` | GPU-memory awareness and contention safety | Inspired |

Do not implement PRAS's augmented graph/MAB or OCTOPINF's full video-serving system.

## Resource monitor

Implement a platform-aware monitor producing the shared `ResourceSnapshot` at configurable intervals.

Required signals when available:

```text
CPU utilization
available/used RAM
process CPU and RSS
GPU utilization
GPU allocated/reserved/free memory
queue length and oldest wait
active jobs total and by tier
recent inference latency by tier
candidate arrival rate
network bandwidth/latency state supplied by the environment
optional temperature/power
```

Requirements:

- timestamps use UTC and monotonic clocks;
- signal availability is explicit; missing is `null/unavailable`, never zero;
- sampling runs without blocking the candidate pipeline;
- access to mutable counters is thread-safe;
- monitor start/stop is idempotent;
- snapshots are bounded in memory;
- sampling overhead is measured at several intervals;
- platform-specific collectors are isolated behind interfaces;
- NVIDIA collection may use NVML or a documented fallback, not fragile parsing spread throughout the code.

## Recent-latency and arrival estimators

Implement transparent rolling-window estimators with configurable windows. They must report sample count alongside estimates. Do not treat an empty history as good performance.

## Candidate queue

Implement an instrumented bounded queue supporting:

- FIFO as the default;
- enqueue/dequeue/complete/fail lifecycle;
- current and peak length;
- oldest wait time;
- backpressure instead of unbounded growth;
- explicit overflow policy configured as block/reject/drop only for controlled experiments;
- cancellation and clean shutdown;
- per-event structured logging.

For scientific safety, default overflow policy must not silently discard candidates.

## Reproducible workload generator

Generate and persist arrival traces separately from execution. Each trace record must contain:

```text
trace_position, candidate_id, relative_arrival_seconds, workload_phase, seed
```

Required profiles:

- low steady;
- moderate steady;
- high steady;
- burst;
- alternating low/high;
- immediate backlog/offline replay.

Support deterministic periodic and seeded Poisson arrivals. Reusing dataset rows for systems load is allowed, but mark repetitions and never count them as independent classification observations.

Every ablation comparing schedulers must replay the exact same persisted arrival trace.

## Integration mode

Add a `monitor_only` pipeline configuration. The scheduler remains static/sequential, but it receives real snapshots and the workload generator drives the queue. This isolates monitor and queue correctness before adaptive decisions.

## Required logs and artifacts

- periodic resource samples, stored efficiently;
- queue events and peak statistics;
- workload trace and checksum;
- monitor overhead benchmark;
- workload summary including realized arrival rate;
- environment and collector availability report.

Do not duplicate full resource snapshots into every event if a timestamped snapshot ID/reference suffices.

## Required tests

- Monitor start/stop, missing-signal and sampling tests.
- Thread-safe active-job counter tests.
- Rolling-window estimator tests with known timestamps.
- Queue ordering, backpressure, overflow and shutdown tests.
- Deterministic trace regeneration and checksum test.
- Repeated-row annotation test.
- Monitor-only end-to-end workload smoke test.
- Overhead measurement that fails or warns when the monitor consumes an unreasonable fraction of runtime.

## Acceptance criteria

- Resource snapshots are valid and availability-aware.
- Queue metrics and arrival estimators are correct.
- All required arrival traces are reproducible and hashed.
- The sequential pipeline processes a streamed trace with monitor enabled.
- Monitor and scheduler overhead can be accounted for separately.
- No candidate is silently lost.

## Context update requirements

Register collectors, signals, monitor interval/overhead, queue policy, trace paths/hashes and unavailable metrics. Set next action to Module 07.

