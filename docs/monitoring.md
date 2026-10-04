# Module 06: monitor-only streaming

Use the Edge_Project interpreter. Generate the trace suite first, then select one
persisted path from the printed index:

```text
python -m edge_triage workloads --config configs/experiments/module06_v1.yaml
python -m edge_triage pipeline monitor-only --config configs/experiments/module06_v1.yaml --trace artifacts/traces/<trace>.json
python scripts/benchmark_monitor.py --output artifacts/monitoring/<new-version>/overhead.json
```

The monitor-only configuration retains the static Tiny selector, sequential
admission/execution, no uncertainty or escalation, and transmit-all reference
classification fate. It does not implement the Module 07 adaptive scheduler.

## Signals and sampling

ResourceSnapshot now allows missing CPU/RAM/arrival estimates and adds process CPU
and RSS, used RAM, GPU free/allocated/reserved memory, optional temperature/power,
oldest queue wait, active counts, latency by tier, estimator sample counts and
availability notes. Older constructors remain valid. Process CPU follows psutil's
multi-core convention and may exceed 100%; system CPU is bounded to 100%.

Host collection uses psutil. First CPU readings are null while priming. One isolated
NVIDIA collector runs nvidia-smi for device index0, with a one-second timeout and
two-second cached refresh. Optional N/A fields are null; collector errors are
recorded as unavailable. GPU observation timestamps expose cache age. PyTorch
allocator readings are process-local and remain unavailable before CUDA is
initialized; system GPU free memory is a distinct driver signal. Network state
comes from workload/environment configuration; unknown values are null.

A background thread collects and appends compact JSONL samples, flushes each
sample, and holds at most256 recent snapshots in memory. The candidate pipeline
reads the latest immutable snapshot without running collectors. Startup uses an
explicitly unavailable snapshot until the first sample arrives. Candidate events
and analytical records reference persisted snapshot IDs; they avoid duplicating
full resource samples. Snapshot timestamps must be checked by future adaptive
consumers. Starting/stopping is idempotent and stop joins the sampler.

Sequential admission uses live thread-safe active-job counters, not the sampler's
potentially stale active counts. Admission reason codes explicitly end in
`LIVE_COUNTERS`; its timing includes the counter read. Hardware observation remains
asynchronous and its original snapshot timestamps are preserved.

Rolling windows default to five seconds (two seconds in the example config).
Arrival rate is (n-1)/(last-first observed arrival), requiring at least two distinct
timestamps. Mean inference latency uses observed execution calls, including failed
attempts. Empty histories are null with sample count0. Bounded estimator overflow
makes the estimate unavailable until the truncated part ages out. RuntimeCounters
serializes arrivals, active-tier counts and latency updates under one lock.

## Queue and replay semantics

BoundedCandidateQueue defaults to blocking FIFO. Producers wait on a condition,
not a polling loop. Reject/drop require `controlled_loss_experiment: true`; both
emit explicit events and counters. Enqueue/dequeue/complete/fail/cancel lifecycle,
current/peak length and backpressure counts are recorded. Shutdown wakes blocked
producers/consumers; normal closure drains the queue. Oldest wait measures residence
after successful enqueue; candidate queue latency also includes producer blocking.

The streaming producer loads only validation rows and drives the queue while the
consumer reuses the existing Pipeline per-candidate execution path. Queue entries
and pending image tensors are bounded by capacity plus active/blocked work. Arrival
metadata and labels remain separate. A failed inference is counted and does not
stop later candidates. Fatal producer/monitor/storage errors fail the run rather
than claiming success. Summary accounting must reconcile every offered occurrence
with completed, failed, rejected or dropped outcomes.

All six profiles (low/moderate/high steady, burst, alternating, backlog) are saved
for periodic and seeded Poisson arrivals. Burst/alternating phases are defined by
trace-position ranges. Each trace pins the split SHA-256, settings and validation
source identity; checksum sidecars are mandatory on replay. Times are relative to
the execution origin. A delayed or blocked producer preserves the intended trace
and records realized rate and maximum lateness; it does not pretend arrivals stayed
on schedule. Backlog requests have nominal arrival time0.

Each occurrence gets a unique candidate ID plus the original source ID, dataset
index, repetition count and `independent_classification_sample: false`. Repeated
rows are system load, never extra statistical observations. A trace must be reused
byte-for-byte for later scheduler comparisons. Generation refuses test rows and
execution verifies source indices/IDs against validation before replay.

## Overhead and artifacts

Each run persists `resources.jsonl`, queue/candidate events, candidate Parquet and
`metrics/monitor_only_summary.json` with monitor wall-cost fraction, separate static
selection/admission timings, collector availability, realized workload and queue
accounting. The trace hash is in the run manifest. Monitor overhead includes
collector calls, JSON serialization and flush, including startup; it is a wall-time
fraction, not process CPU attribution. The benchmark measures intervals .05/.1/.5s
and warns above10% instead of silently asserting low overhead. A collector failure
or fewer than two samples makes the benchmark fail.

## Research scope

[PRAS](https://www.sciencedirect.com/science/article/abs/pii/S1383762125002693)
inspires observing requests after arrival; no augmented-graph/MAB implementation.
[OCTOPINF](https://arxiv.org/abs/2502.01277) motivates variable workloads and resource
observation; no video-serving, co-location or adaptive batching implementation.
[Quilt](https://www.sciencedirect.com/science/article/abs/pii/S1383762126000147)
motivates GPU-memory awareness; no compiler or partitioning reproduction.
Implementation is independent; upstream code was not copied and licenses are not
assumed from paper access.
