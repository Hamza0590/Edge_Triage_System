# Module 05 — Evaluation, Probability Calibration and Model Profiling

## Mission

Evaluate each trained tier on validation data, calibrate probabilities without test leakage, and profile latency/memory/throughput on the actual machine. Produce frozen artifacts consumed by the scheduler, uncertainty, fate and experiment modules.

Do not perform the final held-out test comparison in this module.

## Mandatory preflight

1. Read the master, context and Modules 01–04.
2. Verify all checkpoint, split and preprocessing hashes.
3. Confirm at least one trained checkpoint exists per tier.
4. Mark Module 05 `IN_PROGRESS`.

## Research basis and adoption

| Source | Mechanism | Adoption |
|---|---|---|
| Guo et al., arXiv `1706.04599` | Post-hoc temperature scaling and calibration evaluation | Adapted for binary logits |
| MeerCRAB | Precision, recall, false classifications and MCC for Real/Bogus models | Reused evaluation framing |
| Quilt | Measure memory and execution-time behavior before scheduling | Inspired; no compiler reproduction |

## Classification evaluation

Create one evaluator that consumes saved predictions rather than re-running models for every metric. Save per-candidate validation predictions with:

```text
candidate_id, label, tier, checkpoint_id, seed, logit,
p_real_uncalibrated, p_real_calibrated
```

Report per tier and seed:

- precision, recall, F1;
- ROC-AUC and PR-AUC;
- MCC;
- FNR and FPR;
- confusion matrix;
- Brier score;
- negative log likelihood;
- ECE with documented binning;
- bootstrap confidence intervals where practical.

Handle undefined metrics explicitly; never replace them silently with zero.

## Temperature scaling

Fit one positive temperature parameter per checkpoint using validation logits only. For a binary model:

```text
calibrated_p_real = sigmoid(logit / T)
```

Persist:

- temperature;
- fitting objective and optimizer;
- pre/post calibration metrics;
- checkpoint and validation split hashes;
- code/config version;
- candidate prediction file hash.

If temperature scaling worsens the declared calibration criteria, preserve both results and allow the downstream configuration to select uncalibrated probabilities. Do not fit on test data.

## Hardware profiling methodology

Profile Tiny, Medium and Large independently on CPU and CUDA when supported.

For each tier/device/input size/batch size:

1. load once outside the timed region;
2. perform documented warm-up iterations;
3. synchronize CUDA before and after each timed measurement;
4. use monotonic high-resolution timing;
5. run enough iterations for stable quantiles;
6. report median, P90, P95, P99 and mean, not only minimum;
7. record model load time separately;
8. record allocated/reserved GPU memory and host memory where measurable;
9. record software/hardware state and input-generation method.

Required batch sizes: `1, 2, 4, 8`; add larger sizes only if useful. Batch-1 is the primary online inference result. Measure preprocessing, host-to-device transfer, model forward and total adapter time separately where reliable.

Profile the pipeline/logger overhead with a no-op/dummy model so later cost accounting can subtract or report it honestly.

## Micro-batching baseline

Because 30×30 CNNs may be extremely fast, produce an explicit micro-batching profile. Do not assume concurrent single-candidate inference beats batching. Save throughput and per-candidate latency for every batch size.

## Model profile artifact

Create a versioned machine-readable profile keyed by:

```text
hardware_id
device
tier
checkpoint_id
input_shape
batch_size
precision_mode
```

Include:

- parameter and checkpoint size;
- warm latency quantiles;
- throughput;
- peak memory;
- preprocessing/transfer/forward/end-to-end components;
- measurement count and warm-up count;
- observed failures;
- artifact hashes.

The scheduler must later consume this artifact rather than hard-coded assumed costs.

## Gate B — tier usefulness

Evaluate two separate requirements:

- **cost separation:** Tiny is materially cheaper than Large under at least one relevant execution condition;
- **quality separation:** Large improves at least one relevant validation quality/calibration metric without violating the target-recall framing.

Define “materially” before looking at results, for example a latency or memory ratio threshold in configuration. If separation is inadequate:

1. do not touch the test set;
2. widen `F` values or dense widths while preserving same-family structure;
3. retrain as a new model-family version;
4. repeat profiling;
5. document the failed version rather than deleting it.

## Required tests

- Metric calculations against small known examples.
- Temperature parameter positivity and calibration round-trip tests.
- No test-split access during fitting.
- CUDA synchronization timing wrapper test where CUDA exists.
- Profile schema and key uniqueness tests.
- Checkpoint/profile compatibility validation.
- Repeated profile smoke test with reasonable stability bounds.
- End-to-end calibrated prediction adapter test.

## Acceptance criteria

- Validation predictions and full per-tier metrics exist.
- Calibration artifacts exist and are immutable.
- CPU/CUDA batch-1 and micro-batch profiles exist.
- Pipeline overhead is measured.
- Gate B is explicitly pass/fail with evidence.
- Scheduler-readable model profiles are registered in context.
- Test data remain untouched for tuning.

## Context update requirements

Record calibration/profile paths and hashes, hardware ID, tier quality/cost summary, Gate B result, any architecture revision and next action Module 06.

