# Module07 scheduler v1

The scheduler has independent tier selection and admission policies. Live Module07
execution uses the existing **sequential executor**. Fixed and adaptive concurrency
admission are implemented and simulated; concurrent inference remains Module09.
No held-out test predictions, tuning, retraining, new splits or calibration fitting
are performed. All system replays use frozen validation-row traces, and repeated
rows are not independent classification observations.

## Configuration and compatible evidence

`configs/experiments/module07_v1.yaml` pins the Module05 profile hash, family,
calibration index, CPU device and hardware ID `63998d2133ba053a`. The loader checks
the profile hash, family/manifest/raw-timing hashes, checkpoint identity, input
shape, stable status, batch1, concurrency1 and FP32/TF32-off precision. Profile
selection requires one CPU thread. Loading/checks occur outside decisions; live
verification separately checks the actual machine hardware ID. CPU qualifies via
the frozen Gate B cost result (Large/Tiny median ratio2.1556); CUDA did not pass
cost separation. No GPU scheduling benefit is asserted.

Each policy instance captures an immutable portfolio, prepared profiles and typed
`SchedulerConstraints`. Its runtime input is Candidate metadata and a resource
snapshot. No candidate image or offline label is given to a scheduler.

Thresholds are predeclared engineering settings, **not learned optimal values**:

| Constraint | Value / evidence |
|---|---|
| Recall reference | .95; not enforced or guaranteed by scheduling |
| Latency SLO | null by default; optional profiled P95 plus oldest queue wait filter |
| Queue-wait pressure | 20ms engineering budget; does not discard late candidates |
| RAM/GPU memory margin | 256MiB engineering reserve |
| CPU/GPU low/high | 50/90 percent engineering watermarks |
| Queue low/high | 1/6; capacity8 from Module06 |
| Tier caps Tiny/Medium/Large | 4/2/1; simulation limits, not measured optimal concurrency |
| Total maximum | 4; adaptive admission begins at1 |
| Degradation | recent mean / frozen batch1 median >=2; minimum5 observations |
| Low-pressure hysteresis / cooldown | .5s / 1s |
| Snapshot / GPU observation age | .5s / 2.5s; GPU collector refresh is2s |
| Preserve escalation target | enabled; reserves a stronger eligible tier as a possible future target |

CPU memory cost is **full profiled process RSS**, rather than an invented
activation-only allocation. CUDA cost is profiled peak reserved allocator memory.
Both are conservative per-job estimates; they can double-count resident weights
and allocator/process state. Profile peaks are observations, not hard upper bounds
on future allocations. External allocations can change between monitor samples.
The tests prove accounting against configured observations/estimates, not a
physical no-OOM guarantee.

## Selection priority

`ResourceQueueAwareTierPolicy` (`resource_queue_aware`, version1):

1. Keep enabled tiers with prepared compatible profiles, enough observed free
   memory for cost plus margin, latency-SLO feasibility and per-tier capacity.
2. If none fit, return null tier and `NO_FEASIBLE_TIER`. The pipeline records a
   failed candidate without inference or silent routing to a disabled tier.
3. High queue count or oldest wait chooses minimum profiled median cost, with
   canonical Tiny/Medium/Large order breaking ties. CPU/GPU pressure or observed
   latency degradation also favors that cheapest feasible tier.
4. Missing/stale evidence chooses conservatively (no memory observation means
   no feasible tier). A single high GPU reading can prevent a tier upgrade;
   it does not establish an admission concurrency increase.
5. Healthy observed service history plus low CPU/GPU/queue pressure sustained
   through hysteresis/cooldown permits the strongest eligible tier. Frozen
   profiles provide costs for tiers not yet observed live; requiring live history
   for every unused tier would lock routing at the cheapest tier forever.
6. If configured, remove the strongest feasible tier from the initial-choice
   pool when another choice exists, retaining it as an escalation target in the
   explanation. No escalation is executed in Module07. Width order is a project
   preference supported only by preliminary validation quality, not a guarantee.

`StaticTierPolicy` remains the static baseline. `RoundRobinTierPolicy` is explicitly
diagnostic. Audited adapters provide the same evidence fields for the unchanged
static and sequential baselines. Disabling adaptive selection does not disable
admission, and admission never changes the chosen tier.

## Admission and lifecycle

`FixedConcurrencyAdmissionPolicy(C)` enforces atomic total and per-tier capacity
leases and intentionally omits memory/contention checks as an ablation.
`MemoryContentionAwareAdmissionPolicy` starts at1. Under its single lock, it:

- rejects unavailable/incompatible tier evidence, stale/unknown free memory,
  insufficient margin, adequately sampled latency degradation, high CPU, or
  sustained high GPU observations;
- counts all outstanding leases and conservatively adds observed external jobs;
- subtracts all outstanding reserved bytes from observed free bytes before a
  new lease; decreases its cap immediately on hard pressure or inference failure;
- increases one slot at a time only after low-pressure history and cooldown;
- requires tier-specific latency history for increases, keeping a single-job
  bootstrap path when CPU utilization/history is initially unavailable;
- requires two distinct timestamped GPU observations and hysteresis before
  treating high GPU utilization as a hard admission condition.

Observed running jobs and reserved bytes may already overlap; adding/subtracting
both is deliberately conservative until Module09 owns the entire executor
lifecycle. Decreases do not cancel existing jobs, and new work waits until counts
fall below the lowered cap. Delays use the inherited bounded non-busy retry path.
The pipeline releases the exact reservation on completion, inference failure,
or cancellation/error before launch. Duplicate release is harmless. If durable
decision logging fails during reservation creation, the lease is rolled back.

The optional `TierAdmissionPolicy.admit(candidate, selected_tier, snapshot)` and
`release(reservation_id, outcome=...)` extension preserves old `decide(candidate,
snapshot)` policies. Profile/portfolio/constraint arguments are bound at assembly
rather than repeatedly passed into the decision loop. Live admission overlays
locked active-job counters on the referenced sampled hardware snapshot; selection
also receives live queue/rolling-latency signals. The actual values are recorded.

## Evidence and cost accounting

Every Module07 policy call emits `scheduler.decision` with policy/version,
snapshot reference, portfolio, candidate, result, reason, constraints, profile
values, exact inputs, duration and state before/after. The bounded payload is
validated and frozen. Identical state/config/snapshot/clock inputs produce the
same decision and state; measured wall duration is naturally variable. Resource
age and GPU observation evidence are required for replay.

`scripts/benchmark_scheduler.py` measures full decisions (including metadata
construction/validation), admission release, and a separate mode including
durable logging. No inference, test labels, external commands or unbounded
history scans occur in decisions. Event `duration_ms` excludes the sink write;
the streaming wrapper measures the full policy call including that write.
Pipeline's separate routing/admission events and monitoring remain additional
costs. Scheduler overhead is reported directly, not subtracted from model time.

`scripts/preflight_module07.py` verifies inherited immutable artifacts without
using earlier modules' obsolete exact-source-tree verifiers.
`scripts/verify_module07.py` replays the frozen120-row backlog through all seven
portfolios and the frozen120-row burst through the full portfolio. It validates
candidate accounting, portfolio legality, sequential admission, complete decision
events, snapshot links and hashes. Tests additionally exercise reservation races,
memory conservation, failures, missing/stale data, exact scenario outcomes and
deterministic hysteresis/cooldown state.

Run with the Edge_Project interpreter:

```text
python -m edge_triage pipeline adaptive-sequential --config configs/experiments/module07_v1.yaml --trace artifacts/traces/burst_periodic_3a5af13ae9c87b09.json
python scripts/benchmark_scheduler.py --output artifacts/scheduling/module07_v1/overhead.json
python scripts/verify_module07.py --output artifacts/scheduling/module07_v1/verification.json
```

Outputs are immutable; use a new output directory for another benchmark/verification.

## Saved Module07 measurements

Final Edge_Project regression: **237 passed in78.81s**, warnings treated as
errors. Ruff, mypy (46 source files) and dependency checks passed. An initial
regression stalled because the new immutable cap mapping could not unpickle in
Windows workers; constructor-based FrozenDict/FrozenList reconstruction fixed it.
The original worker test and a new immutability round-trip test both pass.

`artifacts/scheduling/module07_v1/verification.json` records **960/960 completed**,
zero failed/rejected/dropped/cancelled, across seven120-row backlog runs and one
120-row burst. Every run used sequential admission and the original frozen trace.
The backlogs selected the cheapest enabled tier. The full-portfolio burst selected
Tiny throughout:46 high-queue decisions,69 degradation signals and5 insufficient-
history decisions. Consequently these replays establish integration/accounting,
not adaptive quality or speed improvements. Upgrades and recovery are verified
in controlled snapshot tests.

`CONTENTION_DETECTED` is an observed degradation signal, **not causal evidence of
concurrent interference**. The burst flags occurred even with sequential execution;
desktop activity, model overhead and frozen-profile mismatch to current conditions
can also slow inference. Module09 must examine this before interpreting gains.

The separate1000-decision benchmark (50 warmups) recorded:

| Full call | Median, no sink | P95, no sink | Median, durable event sink | P95, durable event sink |
|---|---:|---:|---:|---:|
| Selection | .0879ms | .0955ms | .8778ms | .9110ms |
| Admission plus release | .0965ms | .1042ms | .9215ms | .9572ms |

Raw timings and exact configuration/profile hashes are retained in `overhead.json`.
Console output was redirected to StringIO. Persisting complete evidence costs
more than these very small models' frozen batch1 inference times; no net runtime
benefit is claimed. This overhead includes substantially larger decision payloads
than the Module05 logger-only microbenchmark.

## Research relationships

All relationships are **inspired principles**, not paper-algorithm reproductions.
These independently written policies are named for their project behavior.

| Source | Borrowed principle | Excluded / deviation |
|---|---|---|
| [Taylor et al. (author institution abstract)](https://research.lancaster-university.uk/en/publications/adaptive-deep-learning-model-selection-on-embedded-systems/) | Choose among pretrained DNNs under accuracy/time constraints | No learned input-specific predictor; rule/profile-based astronomical CPU selection |
| [Quilt (publisher)](https://www.sciencedirect.com/science/article/abs/pii/S1383762126000147) | Memory/time awareness before admitting work | No MLIR, partitions, tensor pool or Quilt scheduling algorithm |
| [Convergo (IEEE abstract)](https://ieeexplore.ieee.org/document/11120486/) | Consider quality, throughput and deadline objectives together | Recall is a reference; no heterogeneous accelerator stack or Convergo optimizer |
| [PRAS (publisher)](https://www.sciencedirect.com/science/article/abs/pii/S1383762125002693) | React to post-arrival queued requests | No augmented graph, MAB or optimality claim |

Taylor and Convergo abstracts were verified for Module07. Quilt/PRAS publisher
evidence is inherited from Modules05-06; direct publisher reopens returned errors
in this session. No upstream code was copied, and upstream code licenses remain
unverified. Paper access terms are not treated as a code license.
