# Module 07 — Adaptive Tier Selection and Contention-Aware Admission Scheduler

## Mission

Implement an explainable, deterministic scheduler that selects a permitted model tier and decides whether new inference may safely start. Keep tier selection and concurrency admission independent so ablations can disable either mechanism.

## Mandatory preflight

1. Read the master, context and Modules 03, 05 and 06.
2. Verify model profiles, monitor snapshots and trace artifacts.
3. Confirm Gate B passed or record why scheduler development is still justified.
4. Mark Module 07 `IN_PROGRESS`.

## Research basis and exact scope

| Source | Mechanism used | Adoption | Explicitly excluded |
|---|---|---|---|
| Taylor et al., DOI `10.1145/3211332.3211336` | Runtime selection among pretrained DNNs under quality/time constraints | Adapted to profile-driven tier choice | Their exact learned selector unless separately reproduced |
| Quilt, DOI `10.1016/j.sysarc.2026.103696` | Admit work with memory awareness; profile contention | Inspired/adapted | MLIR compiler, model partitioning, tensor pool |
| Convergo, DOI `10.1109/EDGE67623.2025.00022` | Treat accuracy/recall, throughput and latency as simultaneous constraints | Adapted | Heterogeneous accelerator stack |
| PRAS, DOI `10.1016/j.sysarc.2025.103597` | Make decisions from current queued requests and load | Inspired/adapted | Augmented graph, MAB and optimality claim |

The first production policy is rule/profile-based and explainable. Do not call it the Taylor, Quilt, Convergo or PRAS scheduler.

## Policy separation

Implement:

```text
TierSelectionPolicy.select(candidate, snapshot, portfolio, model_profiles, constraints)
AdmissionPolicy.admit(candidate, selected_tier, snapshot, model_profiles, constraints)
```

Tier selection chooses only from `tier_portfolio`. Admission may admit or delay and may lower the global/per-tier concurrency cap. It must not secretly switch tiers.

## Scheduler constraints

Typed constraints must include:

- target survival recall reference;
- optional latency SLO;
- optional queue-wait SLO;
- minimum memory safety margin;
- CPU/GPU pressure high/low watermarks;
- queue high/low watermarks;
- per-tier concurrency caps;
- maximum total active jobs;
- contention-degradation threshold;
- hysteresis and cooldown periods.

Thresholds must come from configuration and profile evidence. Unknown measurements trigger conservative behavior.

## Required policies

### Baselines

- `StaticTierPolicy`
- `RoundRobinTierPolicy` only for diagnostics, not a scientific baseline unless justified
- `SequentialAdmissionPolicy`
- `FixedConcurrencyAdmissionPolicy(C)`

### Proposed tier policy

Implement a deterministic `ResourceQueueAwareTierPolicy` with documented priority order:

1. filter to enabled portfolio and available compatible checkpoints;
2. reject tiers whose profiled memory plus safety margin cannot fit;
3. check hard resource and latency constraints;
4. under high queue pressure, prefer the cheapest feasible tier;
5. under low pressure and sufficient headroom, permit a stronger tier;
6. preserve a valid escalation target when possible;
7. return the tier and a structured explanation containing evaluated constraints.

### Proposed admission policy

Implement `MemoryContentionAwareAdmissionPolicy`:

1. enforce total and per-tier caps;
2. reserve profiled memory before launch to prevent races;
3. deny/delay if available memory minus reserved memory violates margin;
4. deny/delay when recent latency degradation indicates the contention region;
5. increase allowed concurrency only after low-pressure hysteresis conditions persist;
6. decrease promptly on hard pressure or failure;
7. release reservations on completion/failure/cancellation.

Never depend on a single instantaneous GPU-utilization sample.

## Scheduler decision record

Every call must log:

```text
policy/version
input snapshot ID
enabled portfolio
candidate ID
selected tier or admission result
constraints evaluated
profile values used
reason_code
decision duration
state before/after
```

Required reason codes include at least:

```text
STATIC_TIER
QUEUE_PRESSURE_HIGH
RESOURCES_AVAILABLE
MEMORY_HEADROOM_LOW
GPU_PRESSURE_HIGH
CPU_PRESSURE_HIGH
PER_TIER_CAP_REACHED
TOTAL_CAP_REACHED
CONTENTION_DETECTED
INSUFFICIENT_HISTORY
NO_FEASIBLE_TIER
HYSTERESIS_HOLD
```

## Portfolio behavior

Test all seven portfolios. Pairwise portfolios must never route through a disabled tier. Single-tier portfolios still use admission policies, allowing static-model concurrency baselines.

## Simulation tests before live execution

Build table-driven snapshot scenarios covering:

- empty queue and idle device;
- high queue with headroom;
- memory pressure;
- high GPU utilization with stale/insufficient latency history;
- one tier disabled;
- only Large enabled;
- active Large cap reached while Tiny remains feasible;
- recent latency degradation;
- missing GPU metrics/CPU-only execution;
- completion and reservation release.

For each, assert exact tier/admission/reason code.

## Scheduler overhead

Benchmark decision latency independently. The scheduler must not perform model inference, scan unbounded histories, access the test labels, or execute slow external commands in its critical path.

## Required tests

- All scenario-table tests.
- Portfolio legality property test.
- Memory reservation race/conservation test.
- Hysteresis and cooldown tests.
- Stable results for identical state/config.
- Missing-signal conservative behavior test.
- Decision-log completeness test.
- Scheduler overhead benchmark.
- Streamed sequential integration using adaptive tier choice but sequential admission.

## Acceptance criteria

- Tier selection and admission are independently replaceable.
- All seven portfolios work.
- Every decision is reproducible and explainable.
- No launch violates configured memory/cap constraints in tests.
- Scheduler overhead is measured and included in events.
- Adaptive-tier sequential pipeline completes before concurrency is introduced.
- Literature claims and deviations are accurately documented.

## Context update requirements

Register policy versions, thresholds and evidence, reason codes, overhead, simulation results and limitations. Set next action to Module 08.

