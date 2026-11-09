# Module09 concurrent inference runtime

The common streaming pipeline supports sequential, fixed and dynamic concurrency
through configuration. One event loop owns candidate states, escalation, final
events and Parquet writes; bounded thread workers execute one tier prediction and
its optional UQ. No UQ and no escalation remain the proposed defaults from Gate C.
Performance evidence and the Gate D decision are in [Module09 results](module09_results.md).

## Run and configure

```powershell
python -m edge_triage pipeline concurrent --config configs/experiments/module09_proposed_v2.yaml
```

Use `pipeline.mode: concurrent` and change `admission.execution_mode` to
`sequential`, `fixed_concurrency` or `dynamic_concurrency`. Fixed mode uses
`admission.fixed_concurrency`; dynamic mode uses `max_concurrency` bounded by
the measured `scheduler.contention_profile`. The proposed configuration pins its
SHA-256. Hardware/device, model family, profile bundle, shape and CPU thread count
must match the artifact; unknown portfolios conservatively use one worker.
Resource watermarks can lower dynamic admission further. Legacy smoke and
adaptive-sequential modes remain available to reproduce earlier workflows.

## Job lifecycle

`runtime/executors.py` supplies asynchronous `SequentialExecutor`,
`FixedConcurrencyExecutor` and `DynamicConcurrencyExecutor`. They implement the
`AsyncInferenceExecutor` protocol. `Pipeline.process_async` submits these jobs;
the inherited blocking `InferenceExecutor` protocol supports legacy modes.
`runtime/concurrent.py` constructs the selected backend with shared memory guards.

Each `InferenceJob` contains one candidate/tier/model attempt and optional UQ.
Models must already be loaded, and the caller must not mutate their tensors or
input tensors while jobs are pending/running. A tier escalation is a new job with
a new job ID and the same candidate/trace IDs.

```python
executor = FixedConcurrencyExecutor(4, pending_capacity=64)
try:
    handle = await executor.submit(job)
    result = await handle.result()
    await executor.drain()
finally:
    await executor.shutdown()
```

Submission blocks asynchronously when the pending queue is full. A single event
loop owns dispatch and lifecycle state; a bounded thread pool runs native model
work. No model loading, process spawning, tensor serialization or CUDA streams
occur in the candidate path. The CPU process comparison measures separate warmed
workers receiving integer offsets into read-only input mmap files. Job IDs are retained for the executor's
lifetime to reject duplicate submissions; tensors and completed results are not
retained internally after completion. Create one executor per finite run.

`cancel(job_id)` returns true only for pending jobs. Cancellation yields a typed
terminal result. A running forward continues to completion, including UQ and
lease release. Cancelling a task awaiting `handle.result()` does not cancel the
job. Shutdown rejects submissions, drains by default, or cancels pending jobs
with `cancel_pending=True`. A cancelled shutdown caller can await shutdown again;
the internal cleanup continues. The first shutdown call selects drain/cancel
behavior. Owners must await shutdown before closing the event loop.

Results preserve run/trace/candidate/job IDs and tier. They include status,
deterministic prediction, optional UQ, admission evidence, queue/start/end times,
queue wait, inference/UQ/admission duration, failed MC attempts/completions and
error type. Join results by IDs, never completion position. Worker exceptions
become per-job failures. Reservation-release failure stops further launches,
fails queued jobs and raises from shutdown because ledger cleanup is uncertain.

## Admission and resource updates

Dynamic execution calls the supplied Module07 tier-aware policy before each
launch. Fixed execution optionally accepts the same lease policy, independently
of its hard thread limit. Leases survive through UQ and are released once on the
terminal path. No lease is acquired for a pending cancellation or timeout.

Dispatch visits pending jobs in arrival order, allowing an eligible tier past a
temporarily blocked tier. Each wake makes one bounded pass. There is no periodic
admission polling: submissions, completions, cancellations and
`notify_resources()` wake the dispatcher; an admission deadline supplies the
timeout. A monitor thread may call `notify_resources()` safely. A resource update
must be published before notification. Module07's existing memory, per-tier,
hysteresis and pressure rules remain responsible for admission decisions.

The snapshot supplied to a lease policy counts only jobs external to its ledger:
Module07 adds its own reservations when enforcing caps. Concurrent streaming
supplies zero external jobs because its executor owns all runtime leases. Tier
selection also leaves busy-tier waiting to admission. Live counters still record
launched work through UQ; the final summary reads those counters after draining.
Physical memory observations count all allocations; the inherited whole-process RSS profile reservations remain
conservative estimates, not a physical no-OOM guarantee.

## Shared-model and RNG coordination

`models/inference_state.py` uses a writer-preferring process-wide guard.
Deterministic `RealModelAdapter.predict()` calls acquire shared read access and
can overlap. Adapter construction initializes evaluation mode once; serving
rejects external training-state mutation instead of changing module flags on
every forward. MC sampling takes exclusive access through dropout activation,
sampling, diagnostic RNG restoration and restoration of all module flags. MC
jobs serialize globally, including different models, because diagnostic sampling
uses process-global RNG state. Waiting time is retained in inference/UQ duration.

This coordinates the package adapters and sampler. External direct forwards,
training, model construction and arbitrary RNG consumers must not overlap a
serving session. It is not a general lock around every Torch operation. NoUQ
remains the default; MC serialization is an explicit ablation cost.

PyTorch inference mode is thread-local, so adapters enter it inside each worker.
Python's cancellation documentation motivates shielding lifecycle-owned futures
from waiter cancellation. These are engineering choices, not research algorithms.
Sources: [Python 3.13 asyncio](https://docs.python.org/3.13/library/asyncio-task.html),
[PyTorch inference mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html).
The installed project version remains Torch 2.10.0+cu128; runtime behavior is
covered by tests in Edge_Project, and no dependency migration was performed.

## Provenance and research limits

| Source | Current use and adoption | Deviations / excluded features | Evidence and code status |
|---|---|---|---|
| Quilt, DOI 10.1016/j.sysarc.2026.103696 | Inspired: inherited measured-memory admission before whole-model jobs | No compiler, partitioning, shared tensor pool or source scheduler | Module05/07 verified abstract/institution record; September Module09 publisher reopen returned 403. Prior evidence retained. No copied code; upstream license unverified |
| PRAS, DOI 10.1016/j.sysarc.2025.103597 | Inspired: decisions on pending post-arrival requests | No augmented graph or MAB optimizer | Module06/07 verified publisher abstract; September Module09 publisher reopen returned 403. Prior evidence retained. No copied code; upstream license unverified |
| [OCTOPINF](https://arxiv.org/abs/2502.01277) | Inspired: resource-change notification and workload-aware serving boundary | No video pipeline, edge/server placement or source adaptive batching; our baseline uses fixed batches and a fixed waiting deadline | ArXiv abstract evidence from 2026-09-26 retained; no copied code; upstream code/license unverified |
| Project integration | Async lifecycle, failure isolation, lease ownership, state/RNG guard and measured contention caps | Whole-tier thread jobs; measured scope in the results report | Independently implemented and tested here |

Ordinary Python threads do not reproduce these systems. No new adoption level
above `inspired` is claimed for these scheduling sources.

## Cost accounting and validation

Candidate records retain complete prediction, UQ, escalation and execution chains.
`admission_ms` includes snapshot acquisition, all admission attempts and their
durable decision logging. `executor_ms` measures worker completion to result
dispatch/release; it is not total orchestration CPU time. `queue_ms` runs from
queue entry to first worker start and overlaps selection/admission. Inference and
UQ include lock waiting. E2E covers realized arrival to candidate finalization,
including all preceding instrumentation; final record/event persistence follows
that timestamp. These overlapping costs must not be added or subtracted as a
disjoint decomposition. Source backpressure and intended-arrival lateness are
reported separately. Console rendering is excluded during benchmarks; durable
logging remains enabled.

Microbatch replay accepts preprocessed validation arrays and ordered relative
arrivals, with sizes 1/2/4/8 and a 2 ms oldest-arrival formation deadline. It records
position IDs, batch sizes, forward time, wait and E2E latency. The deadline bounds
formation waiting, not queue delay under backlog or OS scheduling delay. Component
timing excludes preprocessing and pipeline logging and cannot establish superiority
over the fully instrumented pipeline by direct throughput comparison.

`scripts/preflight_module09.py` checks frozen family/checkpoint files, calibration,
24 model profiles, 12 traces, prior monitor evidence, Module08 final-validation
hashes and all nine replay manifests/resource/Parquet hashes. It verifies the
saved 1080-candidate result without replaying it or rerunning historical source
inventory checks. It also checks both raw MD5 values and the noUQ recommendation.

New tests cover all three executor contracts; 300 synthetic jobs per backend;
1/2/4/8 fixed levels; out-of-order completion; bounded pending work; queued
cancellation; waiter/shutdown cancellation; memory/per-tier caps; notification
wakeup without polling; pressure reduction/recovery; exact lease release;
inference/UQ/cleanup failures; and dropout-state/RNG restoration while a competing
deterministic prediction waits. Frozen-model integration compares 18
candidate/tier predictions per backend using six validation rows, joining by
candidate/trace/tier and checking logits/probabilities with explicit tolerances.
These are correctness tests, not performance measurements or scientific metrics.

Streaming integration tests exercise all three modes and forced real MC cascades
for all seven portfolios with isolated routing fixtures. Microbatch tests cover
identity, partial batches, formation deadlines and invalid arrivals. Contention
tests reject incompatible or modified artifacts and verify conservative fallback.
`scripts/benchmark_module09.py` produces immutable fixed and component measurements;
`scripts/verify_module09.py` checks component predictions, runs measured-cap dynamic
replays and plots saved curves. Source hashes prevent mixing runtime versions on
resume. The results report records the completed checks and platform limits.
