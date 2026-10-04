# Module09 implementation checkpoint: executor foundation

Module09 is **IN_PROGRESS**. The asynchronous job backends and shared-model
coordination exist and have targeted tests. The streaming pipeline and CLI still
use the inherited sequential executor. No concurrency performance, safe hardware
cap, contention knee, micro-batching comparison or Gate D result is claimed yet.
The proposed default remains the frozen `module08_proposed_v1.yaml` configuration:
no UQ and no escalation. No frozen configuration or artifact has been overwritten.

## Job lifecycle

`runtime/executors.py` supplies asynchronous `SequentialExecutor`,
`FixedConcurrencyExecutor` and `DynamicConcurrencyExecutor`. They implement the
new `AsyncInferenceExecutor` protocol; the existing blocking `InferenceExecutor`
protocol remains unchanged until pipeline integration. The legacy
`runtime.components.SequentialExecutor` is still the pipeline backend.

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
occur in the candidate path. Threads are the initial correctness implementation,
not a measured choice over CPU processes. Job IDs are retained for the executor's
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

The snapshot supplied to a lease policy must count only jobs external to its
ledger: Module07 adds its own reservations when enforcing caps. The existing
`ObservedAdmission` wrapper currently supplies all live counters, so it must be
adapted explicitly during streaming integration. Passing it unchanged would
conservatively double-count owned jobs. Physical memory observations still count
all allocations; the inherited whole-process RSS profile reservations remain
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
| Quilt, DOI 10.1016/j.sysarc.2026.103696 | Inspired: inherited measured-memory admission before whole-model jobs | No compiler, partitioning, shared tensor pool or source scheduler | Module05/07 verified abstract/institution record; publisher reopen returned 403 in this session. No copied code; upstream license unverified |
| PRAS, DOI 10.1016/j.sysarc.2025.103597 | Inspired: decisions on pending post-arrival requests | No augmented graph or MAB optimizer | Module06/07 verified publisher abstract; publisher reopen returned 403. No copied code; upstream license unverified |
| [OCTOPINF](https://arxiv.org/abs/2502.01277) | Inspired: resource-change notification and workload-aware serving boundary | No video pipeline, edge/server placement or adaptive batching implemented in this checkpoint | ArXiv abstract rechecked 2026-09-26; no copied code; upstream code/license unverified |
| Project integration | Async lifecycle, failure isolation, lease ownership and state/RNG guard | Whole-tier thread jobs; no speedup claim | Independently implemented and tested here |

Ordinary Python threads do not reproduce these systems. No new adoption level
above `inspired` is justified by this implementation checkpoint.

## Validation and remaining acceptance work

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

Next, integrate async tier attempts into the common pipeline, preserving its
state-machine/events and a single Parquet owner. Add configuration-only switching,
resource notifications, live counters without duplicate lease accounting,
cascade resubmission and complete executor overhead accounting. Verify all seven
portfolios with isolated UQ ablation fixtures and frozen trace replays. Then
predeclare performance/Gate D criteria and measure CPU threads versus processes,
fixed concurrency curves, micro-batch formation delay and throughput, resource
peaks and failures. Produce a versioned contention profile, consume it in the
dynamic policy, and report Gate D honestly before marking Module09 complete.
