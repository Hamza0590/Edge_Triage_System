# Module 09 — Sequential, Fixed and Dynamic Concurrent Inference Runtime

## Mission

Implement interchangeable execution backends so the exact same pipeline and policies can run sequentially, with fixed concurrency, or with scheduler-controlled dynamic concurrency. Characterize useful parallelism and the contention-dominated region.

## Mandatory preflight

1. Read the master, context and Modules 03–08.
2. Verify sequential adaptive cascades and resource monitoring.
3. Confirm model profile artifacts and memory reservations are valid.
4. Mark Module 09 `IN_PROGRESS`.

## Research basis and adoption

| Source | Mechanism | Adoption | Excluded scope |
|---|---|---|---|
| Quilt, DOI `10.1016/j.sysarc.2026.103696` | Profile multi-model contention and account for peak memory before concurrent execution | Adapted at whole-model job level | Model compiler/partitioning and shared tensor pool |
| OCTOPINF, arXiv `2502.01277` | Workload-aware serving, adaptive batching and GPU co-location awareness | Inspired/adapted | Full video pipeline and edge-server placement |
| PRAS, DOI `10.1016/j.sysarc.2025.103597` | Model/batch decisions under actual request backlog | Inspired | MAB/graph optimizer |

Do not claim that ordinary Python threads reproduce any of these systems.

## Common executor interface

Implement:

```text
InferenceExecutor.submit(job)
InferenceExecutor.cancel(job_id)
InferenceExecutor.drain()
InferenceExecutor.shutdown()
```

Required backends:

- `SequentialExecutor`
- `FixedConcurrencyExecutor`
- `DynamicConcurrencyExecutor`

All backends return the same result contract and preserve trace/candidate IDs. Switching executor must be configuration-only.

## Concurrency architecture

Use asynchronous orchestration for queue waiting and lifecycle management. Choose thread/process/GPU execution deliberately:

- CPU: benchmark threads and processes; avoid unnecessary tensor serialization.
- CUDA: do not spawn multiple processes each loading all models unless memory cost is measured and justified.
- CUDA launches are asynchronous; synchronize only where correctness/timing requires it.
- CUDA streams are optional and may be added only after the fixed-concurrency baseline proves a need.
- Keep model loading outside candidate critical paths.
- Enforce scheduler memory reservations and per-tier caps atomically.

The default implementation should prioritize correctness and measurable behavior over sophisticated CUDA code.

## Fixed-concurrency experiments

Support configured levels at least `1, 2, 4, 8`, subject to safety. For every tier and relevant portfolio, profile:

- throughput;
- queue wait;
- per-candidate inference and end-to-end latency;
- P50/P95/P99 latency;
- GPU/CPU/RAM use;
- GPU memory allocated/reserved/peak;
- failures/OOMs;
- ordering/completion behavior;
- scheduler/executor overhead.

Stop increasing concurrency when safety constraints fail. Record skipped unsafe settings rather than crashing intentionally.

## Dynamic concurrency behavior

The dynamic executor asks the Module 07 admission policy before every launch. It must:

- avoid busy polling;
- wake on resource/job/queue changes;
- obey current total and per-tier limits;
- handle delayed jobs fairly;
- promptly reduce new admissions after pressure;
- never cancel already running jobs merely to satisfy a new soft threshold;
- release reservations exactly once;
- recover after candidate-level failure;
- drain or cancel cleanly according to run configuration.

## Micro-batching comparison

Implement or integrate a small fixed micro-batching baseline using the Module 05 profiles. Report both waiting delay and forward-pass gain. Dynamic concurrency must not be declared superior without comparison to sensible batching for these small 30×30 inputs.

## Contention characterization

For each tested condition compute:

```text
latency_degradation = concurrent_latency / isolated_latency
throughput_gain = concurrent_throughput / sequential_throughput
```

Identify the concurrency knee based on a configured criterion, for example the highest concurrency before throughput saturates/declines or latency degradation breaches the SLO. Feed the resulting profile back to the scheduler as a versioned artifact.

## Correctness under out-of-order completion

Completion order may differ from arrival order. All outputs must join by `candidate_id/trace_id`, never array position. Classification metrics must remain identical across execution modes within numerical tolerance when policies make the same model decisions.

## Gate D — concurrency usefulness

Pass if at least one realistic workload gains sustainable throughput or queue stability without unacceptable safety/latency degradation. A negative result is acceptable and should lead the dynamic policy to concurrency one under those conditions.

## Required tests

- Executor contract tests shared by all backends.
- Max-active-jobs assertions for fixed/dynamic modes.
- Atomic memory reservation and exact-release tests under failure/cancellation.
- Out-of-order completion/join correctness test.
- No candidate loss/duplication test under a long synthetic trace.
- Graceful shutdown/drain/cancel tests.
- Same-policy prediction-equivalence across modes.
- Fixed-level performance smoke matrix.
- Dynamic-admission response to injected pressure.
- Micro-batch comparison smoke test.

## Acceptance criteria

- Three execution modes are configuration-interchangeable.
- Concurrency caps and reservations are never violated in tests.
- Candidate results are complete despite out-of-order completion.
- Fixed-concurrency curves and a contention-knee artifact exist.
- Dynamic concurrency reacts to pressure and uses measured profiles.
- Gate D is documented as pass or negative result.
- Runtime cost and failures are logged completely.

## Context update requirements

Register executors, supported backends, safe caps, contention profile, micro-batch result, Gate D status and unresolved platform limits. Set next action to Module 10.

