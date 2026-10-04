# Module 11 — Reproducible Experiment and Ablation Harness

## Mission

Build one configuration-driven harness that runs every baseline, model portfolio, scheduler mode and ablation through the same pipeline. Perform held-out test evaluation only after all configurations and threshold artifacts are frozen.

## Mandatory preflight

1. Read every preceding brief and `PROJECT_CONTEXT.md`.
2. Verify frozen split, preprocessing, checkpoint, calibration, profile, trace and threshold hashes.
3. Confirm no test result has been used to tune a component.
4. Mark Module 11 `IN_PROGRESS`.

## Research basis and adoption

This module evaluates the integrated system using experimental dimensions motivated by MeerCRAB, Taylor et al., Quilt, Convergo, PRAS, OCTOPINF, calibration and selective-classification literature. It does not claim a new component algorithm. Its responsibility is fair isolation and reproducibility.

## Experiment specification

Represent every run as immutable YAML plus a resolved manifest. Separate these axes:

```text
portfolio
tier_selection_policy
execution_mode
admission_policy
uncertainty_policy
escalation_policy
threshold_policy
queue_awareness
resource_awareness
fate_policy
arrival_trace
environment_profile
model_seed
run_repetition
```

Do not create a different Python script for each baseline.

## Required portfolio matrix

All portfolios must execute:

```text
Tiny
Medium
Large
Tiny + Medium
Tiny + Large
Medium + Large
Tiny + Medium + Large
```

Required controlled cascade orders:

```text
Tiny→Medium
Tiny→Large
Medium→Large
Tiny→Medium→Large
```

## Minimum primary systems

Implement named configurations equivalent to:

| ID | Description |
|---|---|
| B1 | Static Large, sequential |
| B2 | Static Tiny, sequential |
| B3 | Static Medium, sequential |
| B4a–d | Every two-/three-tier cascade, sequential |
| B5 | Adaptive tier selection, sequential |
| B6 | Adaptive tier selection, fixed concurrency |
| B7 | Adaptive tier selection, dynamic concurrency |
| FULL | Dynamic tier + dynamic concurrency + selective UQ + escalation + per-tier thresholds + recall-constrained fate |

Add fixed-concurrency single-tier runs when needed to isolate executor effects.

## Required ablations

- Tier selection: static vs adaptive.
- Portfolio: each single, pairwise and all-three combination.
- Execution: sequential vs fixed vs dynamic concurrency.
- Thresholds: common vs per-tier.
- Uncertainty: off vs selective MC dropout.
- Queue awareness: resource-only vs resource+queue.
- Admission: fixed cap vs memory/contention-aware.
- Fate: classification-only/transmit-all vs three-way policy.
- Micro-batching vs concurrent batch-1.

Use a curated matrix to control runtime; do not silently present a partial matrix as a full factorial experiment.

## Fair-comparison protocol

For each comparable group, enforce:

- same raw dataset and immutable split;
- same candidate test rows and arrival trace order;
- same environment constraint profile;
- same checkpoints when the model component is not the changed variable;
- same warm-up method;
- thresholds calibrated separately on validation to the same survival-recall target;
- threshold/configuration freeze before test;
- multiple timing repetitions after warm-up;
- classification metrics computed on unique test candidates, not replay duplicates;
- systems metrics may use repeated streams but must identify repetitions.

If a baseline cannot reach target recall, report its best achievable recall and do not compare its cost as though recall were equal.

## Full cost accounting

Report separately and jointly:

```text
data preprocessing
monitoring
scheduler
queue wait
initial inference
UQ passes
escalation inference
fate/compression accounting
transmission model
total wall-clock
```

Also record total forward passes per candidate and by tier.

## Metrics

Classification:

- precision, recall, F1, ROC-AUC, PR-AUC, MCC, FNR and FPR.

Calibration/UQ:

- Brier, NLL, ECE, uncertainty-error relationship, escalation/correction/regression rates.

Scientific safety:

- survival recall, real false-discard count, real transmit/compress rates.

Systems:

- throughput and mean/P50/P95/P99 queue, inference and end-to-end latency;
- peak/mean queue length and queue growth;
- tier usage, active concurrency and contention degradation;
- CPU/GPU/RAM/GPU-memory statistics;
- scheduler/monitor overhead and failure/OOM count.

Efficiency:

- total/average forward passes;
- compute-time proxy;
- bytes transmitted and reduction;
- optional energy only if actually measured.

## Statistical treatment

- Use bootstrap confidence intervals for unique-sample classification/fate metrics.
- Use repeated runs and report distributions for timing/system metrics.
- Do not bootstrap replayed duplicates as independent astronomical samples.
- Preserve per-seed results; do not report only the best seed.
- Record whether differences are descriptive or statistically tested.

## Required plots

1. Survival recall vs compute cost.
2. Throughput vs concurrency.
3. P95 latency vs arrival rate.
4. Queue length over time.
5. Tier usage distribution.
6. Survival recall vs transmitted bytes.
7. Uncertainty vs error.
8. Common vs per-tier threshold comparison.
9. Portfolio comparison across all seven combinations.

Plots must include units, sample/repetition counts and confidence intervals where meaningful.

## Run safety and resumability

- Refuse accidental reuse of a completed run ID.
- Write completion markers only after outputs validate.
- Resume only independent completed subruns; never append incompatible configurations.
- Capture exceptions per subrun and continue when safe.
- Produce a final matrix showing complete/failed/skipped with reasons.

## Required CLI

Commands equivalent to:

```text
edge-triage experiment validate-matrix --config <matrix.yaml>
edge-triage experiment run --config <matrix.yaml>
edge-triage experiment summarize --runs <path>
edge-triage experiment plot --summary <path>
```

## Required tests

- Matrix expansion with known expected run count.
- Configuration isolation: changing one ablation axis changes only that axis.
- Same-trace and same-split enforcement.
- Equal-recall and failure-to-reach-target handling.
- Unique-candidate vs replayed-systems metric separation.
- Cost-component sum test.
- Resume/idempotency and completed-run protection.
- Summary/plot schema tests.
- One reduced end-to-end matrix smoke run.

## Acceptance criteria

- All seven portfolios and three execution modes are supported by one harness.
- Primary baselines and ablations are named and reproducible.
- Validation fitting and test evaluation are operationally separated.
- Full cost, safety and systems metrics are produced.
- Results can be regenerated from manifests.
- Failures/skips are visible, not silently omitted.
- Required plots are generated from saved records.

## Context update requirements

Register the frozen experiment matrix, run counts, test-evaluation authorization point, summaries, plots, failures and exact final deployment action. Set next action to Module 12.

