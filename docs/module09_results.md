# Module09 measurements and Gate D

Gate D is **negative** for the measured CPU pipeline: none of the seven portfolios
achieved at least 1.10x throughput in every repeat while satisfying the declared
latency and failure limits. The scheduler-consumed contention artifact therefore
sets concurrency to **one for all portfolios**. This is an accepted module outcome.
No UQ and no escalation remain the default. Module10 fate logic is not implemented.

## Evidence and methodology

Evidence is under `artifacts/runtime/module09_v2/`. The objective is the unchanged
`configs/experiments/module09_objective_v1.json` (SHA-256
`7c8a85b094c0ec18fc06915f2b8d588af4b0e7c2bf17bd17229b5347f1c21c30`).
The incomplete September v1 measurements remain unchanged. Before v2 measurement,
`session_protocol.json` declared fresh comparisons because the interruption left
an eleven-day gap between old baselines and the resumed session.

The experiment uses the frozen validation-only 120-occurrence backlog, the frozen
CNN family and calibration artifacts, CPU FP32, TF32 disabled, and one intra-op
and inter-op thread. Each of seven portfolios runs configured caps 1/2/4/8 three
times: **84 rounds, 10,080 candidates, zero candidate failures or losses**. Round
robin is a diagnostic selector that preserves the same tier assignment across
settings. All prediction comparisons join by identity; logit absolute tolerance is
1e-5 and probability absolute tolerance is 1e-6, with identical 0.5 classifications.
These repeated system-load occurrences are not independent scientific observations.

Each pipeline run retains JSONL events, Parquet records, resource samples, resolved
configuration and a terminal manifest. `fixed_verification.json` rechecks every run,
identity set, final event, resource reference, execution cost and artifact hash.
Model loading is outside the reported candidate interval; the first forward is
included. Console rendering is suppressed, while durable logging is included.
See [cost definitions](concurrency.md#cost-accounting-and-validation) for overlapping
timings, realized-arrival E2E and backpressure limitations.

## Fixed concurrency

Median candidates/s across three rounds:

| Portfolio | C1 | C2 | C4 | C8 | Selected cap |
|---|---:|---:|---:|---:|---:|
| Tiny | 87.48 | 89.21 | 89.59 | 88.99 | 1 |
| Medium | 88.26 | 89.23 | 89.47 | 89.23 | 1 |
| Large | 92.54 | 90.01 | 91.89 | 91.88 | 1 |
| Tiny + Medium | 89.07 | 90.27 | 90.53 | 90.36 | 1 |
| Tiny + Large | 87.60 | 89.50 | 86.61 | 77.78 | 1 |
| Medium + Large | 87.85 | 85.06 | 89.80 | 92.71 | 1 |
| Tiny + Medium + Large | 89.79 | 89.14 | 87.85 | 95.21 | 1 |

The measured peak active jobs was one at C1 and **two at every larger configured
cap**. Thus these curves characterize the instrumented serving path, not a saturated
eight-job hardware contention experiment. The large gap from component throughput
below is consistent with the previously measured orchestration and logging cost;
this experiment does not isolate a causal cost decomposition. The selected cap is
the conservative result of the objective, not a claim of a physical hardware knee.

The objective also requires E2E P95 ratio <=1.25, mean inference degradation <=4,
and zero failures/losses in every repeat. `benchmark.json` retains every ratio and
decision. P50/P95/P99 for queue, inference and E2E, measured admission and completion
dispatch costs, CPU/RAM/GPU observations and ordering are in each setting's JSON.

![Fixed concurrency curves](../artifacts/runtime/module09_v2/plots/fixed_curves.png)

## CPU processes, threads and microbatching

All **48 component settings / 144 rounds / 17,280 outputs** passed identity,
finite-value and prediction checks. The verifier also checks that the mmap arrays
equal the frozen preprocessing of the validation rows and that the backlog and
moderate traces use the same identity order. Cross-backend/batch logit tolerance is
`atol=1e-5, rtol=1e-5`; zero-logit classifications agree exactly.

Each worker warms five forwards before timing. Processes receive integer offsets
into a read-only mmap and load only the selected CPU model. Process startup is
reported separately (about 2.0-2.7 s), as is thread setup (about 0.47-0.51 s).
Task scheduling, IPC and per-job RSS sampling remain in the thread/process component
elapsed interval. Microbatch elapsed time includes formation waiting and tensor
copy/forward/output collection. Neither scope includes durable pipeline logging.

Best measured backlog throughput, selected descriptively by median (candidates/s):

| Tier | Threads | Processes | Batch1 replay | Batch8 replay |
|---|---:|---:|---:|---:|
| Tiny | 3,842 (2 workers) | 8,828 (4 workers) | 3,980 | 8,767 |
| Medium | 3,808 (4 workers) | 8,373 (8 workers) | 2,797 | 5,234 |
| Large | 3,546 (4 workers) | 6,196 (8 workers) | 1,656 | 2,623 |

Eight process workers consume about 4.26-4.29 GB aggregate sampled RSS, versus
0.55 GB for eight threads. These sums of per-process sampled maxima are conservative
accounting observations, not synchronized physical peaks. Processes improve the
isolated component at the cost of replicated memory. The production runtime retains
threads and the measured cap of one; an end-to-end process backend would require a
separate integration and experiment before claiming a pipeline benefit.

Batch8 reduces mean forward time per candidate from 0.2484/0.3545/0.6009 ms to
0.1130/0.1898/0.3799 ms for Tiny/Medium/Large. At the moderate trace's 10 ms spacing,
the 2 ms deadline produces singleton batches even when the maximum is eight.
Throughput stays near the offered 100/s, while P95 wait rises from roughly 0.5 ms
to 2.5 ms. Backlog waiting includes service queue delay and can exceed the formation
deadline. Batching gains depend on available backlog; no direct superiority over
the fully instrumented dynamic pipeline is asserted.

![Component throughput](../artifacts/runtime/module09_v2/plots/components.png)

## Dynamic integration and correctness

`contention.json` is hash-bound to the benchmark, family, hardware/device, input
shape, profile bundle and thread count. `module09_proposed_v2.yaml` pins its SHA-256:
`dfa48161c0f618b477d30ba8648392b17d3bef051871709059425c4b8c30612d`.
The dynamic factory consumes its cap before constructing the executor and admission
policy. Unknown portfolios default to one; incompatible artifacts are rejected.

`verification.json` records measured-cap replays for all seven portfolios on backlog
with matched tier assignments, plus the full portfolio with resource-aware selection
on low, burst and alternating traces. It checks actual capacity, drained counters,
no extra MC forwards, event/Parquet/resource consistency and unchanged frozen hashes.
Injected-pressure tests separately verify admission reduction, recovery, fair
eligible-tier dispatch and no cancellation of already running jobs.

All **1,200 candidates completed with zero failures**. Dynamic backlog throughput
was 3.02/3.06/3.02 candidates/s for Tiny/Medium/Large, 4.85-5.23 for pairs and 7.21
for the full portfolio. Full-portfolio low/burst/alternating replays achieved
2.88/2.89/3.01 candidates/s, with E2E P95 7.23/8.18/6.33 seconds.
The inherited admission rule delays even at capacity one when rolling runtime
latency exceeds twice the older isolated Module05 profile, then resumes when the
history expires. This conservative rule materially reduces throughput. Cap one
is verified; dynamic execution is not performance-equivalent to the fixed C1
baseline. Retaining the measured negative result avoids hiding this known
limitation or changing the declared thresholds after measurement.

The inherited regression covers model-state/RNG isolation, all executor contracts,
900 synthetic jobs, caps, exact lease release, cancellation/drain, out-of-order
joins, all three real streaming modes and seven forced real MC cascades. Added
checks verify measured-cap construction and partial UQ-failure prediction/cost/counter
retention. The final check outputs and source hashes are in `final_validation.json`.
Final regression: **317 passed in 163.63 s**, warnings treated as errors. Ruff,
mypy (55 source files), dependency checks and Git whitespace checks passed. Both
plots were visually inspected. No tests were skipped or weakened.

## Operational failures and limits

Windows denied one manifest replacement before the final C8 setting began
(`runs/20261007T154508_acbfaabeabea`, terminal status failed, no candidate inference).
An approved retry completed that setting in new runs. Windows also blocked the
first multiprocessing pipe; the saved `component_tiny_processes_1_backlog_failure.json`
records the error, and the approved retry completed the comparison. These startup
failures remain in the evidence. The first four thread component settings and most
pipeline rounds ran under the sandbox; subsequent component settings and the final
pipeline setting used approved execution. No inference safety failure was observed.

Results are short desktop software-in-the-loop observations with uncontrolled
external activity and fixed setting order. They do not establish statistical
significance, edge-device timing, CUDA concurrency performance, a no-OOM guarantee,
or recall safety. GPU utilization/free memory are observed; CPU serving has no
meaningful CUDA allocator peak, and unavailable fields stay null. Whole-process
RSS reservations are conservative estimates. No checkpoint retraining, threshold
refitting, held-out test evaluation or Module10 work occurred.

## Reproduce

Use `C:/Users/hamza/anaconda3/envs/Edge_Project/python.exe` and a new output directory
for a changed experiment. Completed files are immutable; partial runs can resume
only with matching runtime source hashes. The scripts use the frozen v1 objective.

```powershell
python scripts/benchmark_module09.py --stage pipeline --output artifacts/runtime/module09_v2
python scripts/benchmark_module09.py --stage components --output artifacts/runtime/module09_v2
python scripts/verify_module09.py --stage fixed --output artifacts/runtime/module09_v2
python scripts/verify_module09.py --stage components --output artifacts/runtime/module09_v2
python scripts/benchmark_module09.py --stage finish --output artifacts/runtime/module09_v2
python scripts/verify_module09.py --stage replay --output artifacts/runtime/module09_v2
python scripts/verify_module09.py --stage plots --output artifacts/runtime/module09_v2
```

The accepted artifacts are already complete; use their hashes for inspection rather
than overwriting them. Later modules legitimately change source inventories.
