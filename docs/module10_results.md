# Module 10 validation results

Module10 is complete: 36 collections, 144 frozen fate artifacts, 36 full frozen
replays and nine model-seed diagnostics are verified. Final regression: **378 passed
in212.24s**, with warnings treated as errors. Ruff, mypy59 source files and dependency
checks pass. The [completion seal](../artifacts/triage/module10_v2/final_validation.json)
registers actual checks, source hashes, evidence hashes and raw-data integrity.

## Study design

The predeclared [protocol](../configs/experiments/module10_protocol_v1.json) covers
36 upstream configurations and four fate policies per configuration. All 36 actual
runtime collections completed their 643 unique validation candidates (329 Real,
314 Bogus), for 23,148 candidate records with zero failures. Each configuration has
four immutable fitted artifacts. All 36 per-tier runtime replays also meet the target,
covering another 23,148 candidate records with zero failures. No held-out scientific
test evaluation was run.
These are processing totals over the same 643 rows; each setting's bootstrap uses
one occurrence per validation row.

The matrix includes all seven portfolios in sequential, fixed-concurrency and
dynamic-concurrency modes, the proposed adaptive configuration, and seven paired
calibrated noUQ/MC5 comparisons. NoUQ/no escalation remains the proposed default;
the negative Module08/09 gates and measured dynamic cap1 remain unchanged.

The baseline fates reuse the exact saved final routing evidence. Each per-tier
configuration additionally receives a full frozen runtime replay. This separates a
comparison of fate decisions from changes in routing or runtime conditions.

## Proposed fitted operating point

The proposed collection routed all 643 candidates to Tiny. Its calibrated discard
cutoff is `0.47003519710633723`, transmit cutoff is `.9`, and uncertainty cutoff is
zero for the explicitly disabled-UQ condition. Medium/Large lack resolved validation
support in this configuration and preserve the complete payload if reached later.

| Metric | Validation fit |
|---|---:|
| Survival recall | 313/329 = 95.1368% |
| Classifier recall at .5 | 313/329 = 95.1368% |
| Real transmit recall | 288/329 = 87.5380% |
| Real compress recall | 25/329 = 7.5988% |
| False Real discards | 16 |
| Bogus discard rate | 311/314 = 99.0446% |
| Original logical bytes | 77,160,000 |
| Transmitted logical bytes | 35,400,000 |
| Logical byte reduction | 54.1213% |

The survival constraint is an empirical fit-set target. It is not a 95% statistical
lower bound or a held-out guarantee. Classifier and survival recall happen to be
equal for this operating point; they remain separately defined and reported.
The full proposed replay reproduces every metric in the table. The 1,000-resample
conditional 95% bootstrap interval is **92.7670–97.3064%** for survival recall and
**50.4646–58.0113%** for logical byte reduction. Thresholds remain fixed in this
bootstrap; the interval's lower endpoint does not meet .95.

| True label | Discard | Compress | Transmit |
|---|---:|---:|---:|
| Real | 16 | 25 | 288 |
| Bogus | 311 | 3 | 0 |

## Fate and uncertainty comparisons

| Proposed configuration fate | Survival recall | Logical bytes | Byte reduction |
|---|---:|---:|---:|
| Transmit all | 100% | 77,160,000 | 0% |
| Binary discard/transmit | 95.1368% | 37,920,000 | 50.8554% |
| Common three-way | 95.1368% | 35,400,000 | 54.1213% |
| Per-tier three-way | 95.1368% | 35,400,000 | 54.1213% |

The per-tier fit improves logical bytes over common thresholds in six of the 36
settings: 60,000 bytes for Medium+Large and 30,000 bytes for all-three, repeated
across the three core execution modes. The other settings tie. The proposed
configuration therefore shows no additional per-tier saving over the common policy;
its validation routing reaches only Tiny. These small conditional gains do not
establish a general advantage for tier-specific thresholds.

| Calibrated portfolio | NoUQ false Real discards | MC5 false Real discards | NoUQ bytes | MC5 bytes |
|---|---:|---:|---:|---:|
| Tiny | 16 | 1 | 35,400,000 | 46,920,000 |
| Medium | 16 | 16 | 21,450,000 | 33,060,000 |
| Large | 16 | 5 | 27,930,000 | 39,840,000 |
| Tiny+Medium | 16 | 15 | 35,400,000 | 44,880,000 |
| Tiny+Large | 16 | 1 | 35,400,000 | 46,920,000 |
| Medium+Large | 16 | 16 | 21,450,000 | 33,060,000 |
| Tiny+Medium+Large | 16 | 15 | 35,400,000 | 44,880,000 |

MC5 preserves more Real candidates in five pairs, while all seven pairs transmit
more logical bytes. These are separately fitted policies under the same minimum
recall target, not exactly equal achieved recall. Full UQ/escalation costs are
reported separately; this diagnostic does not overturn Module08's negative Gate C.

## Training-seed sensitivity

Each seed uses the same frozen split, architecture/training settings, calibrated
singleton inference, noUQ and transmit cutoff .5. Original checkpoints remain intact.

| Tier | Seed 20260924 cutoff | Seed 20260925 cutoff | Seed 20260926 cutoff |
|---|---:|---:|---:|
| Tiny | .470035 | .482582 | .268038 |
| Medium | .183141 | .274238 | .265429 |
| Large | .205212 | .157046 | .204564 |

Eight fits retain 313/329 Real candidates; Tiny seed20260925 retains 318/329
(96.6565%). Cutoffs are checkpoint-specific and materially sensitive to training
seed. Three seeds provide descriptive evidence, not a stable population estimate.

![Checkpoint-specific discard cutoffs across training seeds](../artifacts/triage/module10_v2/analysis/seed_stability.png)

## Recall and measured cost

| Requested recall target | Achieved survival recall | Logical bytes | False Real discards |
|---|---:|---:|---:|
| .90 | 95.1368% | 35,400,000 | 16 |
| .95 | 95.1368% | 35,400,000 | 16 |
| .98 | 98.1763% | 35,790,000 | 6 |
| 1.00 | 100% | 44,880,000 | 0 |

These are diagnostic threshold sweeps on fixed proposed collection evidence, not
new deployment selections. The below-.5 discard restriction makes the .90 and .95
solutions coincide. The predeclared .95 artifact remains frozen.

![Validation recall versus logical bytes and full runtime E2E](../artifacts/triage/module10_v2/analysis/recall_bytes_cost.png)

The proposed frozen replay took 221.212 seconds including startup, processing
2.907 candidates/second with peak active count1. Sum of candidate E2E was
3,928,694.169 ms (mean 6,109.944 ms, including waits). Named totals were inference
1,137.874 ms, executor1,498.305 ms, admission22,645.292 ms, UQ15.967 ms,
escalation decision16.926 ms and fate2.310 ms. Scheduler selection totaled949.458 ms;
monitor collection/serialization/flush took5.737 s (2.609% of its measured interval).
The noUQ method still incurs wrapper bookkeeping. These overlapping measurements
are not additive CPU compute costs. Seed training costs are recorded separately.

## Accounting and interpretation

The logical raw payload is 120,000 bytes per 100x100x3 float32 candidate. Discard sends
zero bytes, compression sends 30,000 bytes, and transmission sends 120,000 bytes.
Metadata/protocol overhead is declared zero. No codec or network runs; compressed
scientific fidelity is unmeasured and codec CPU time is zero only by this accounting
convention. The measured fate decision cost remains part of runtime timing.
Survival includes compressed candidates; it does not establish that a future codec
preserves all scientific information.

End-to-end time includes queue/admission waits and instrumentation. Named inference,
UQ, executor, fate, scheduler and monitor costs overlap and must not be added to E2E.
For offline fate comparisons, upstream compute is fixed by construction. The dynamic
admission slowdown remains visible; these measurements make no throughput benefit
or Jetson deployment claim. Collections were resumed across sessions, and timing is
descriptive rather than a repeated-trial performance comparison.

The fixed split was reused for model selection, calibration and threshold fitting.
Row bootstrap intervals are conditional on the fitted thresholds; unknown repeated-
object dependence and model-selection uncertainty are not resolved. The separate
three-training-seed diagnostic describes sensitivity on this same split and does
not establish population-level stability.

See [implementation and reproduction commands](triage.md) and the ready-to-replay
[proposed configuration](../configs/experiments/module10_proposed_v2.yaml).

## Verified evidence

All files below are under `artifacts/triage/module10_v2/analysis/`:

- [Summary and bootstrap](../artifacts/triage/module10_v2/analysis/summary.json),
  [compatible artifact registry](../artifacts/triage/module10_v2/analysis/artifact_registry.json).
- [144 fate metrics](../artifacts/triage/module10_v2/analysis/fit_metrics.csv),
  [36 frozen replays](../artifacts/triage/module10_v2/analysis/runtime_replays.csv),
  [thresholds](../artifacts/triage/module10_v2/analysis/thresholds.csv).
- [False-discard review with IDs](../artifacts/triage/module10_v2/analysis/false_discard_review.csv),
  [fates by label/tier](../artifacts/triage/module10_v2/analysis/label_tier_fates.csv).
- [Common/per-tier comparison](../artifacts/triage/module10_v2/analysis/common_vs_per_tier.csv),
  [UQ comparison](../artifacts/triage/module10_v2/analysis/uq_effect.csv).
- [Recall sweep](../artifacts/triage/module10_v2/analysis/recall_curve.csv),
  [full named runtime costs](../artifacts/triage/module10_v2/analysis/costs.csv).
- [Seed diagnostics](../artifacts/triage/module10_v2/analysis/seed_stability.csv),
  [separate training costs](../artifacts/triage/module10_v2/analysis/seed_training_costs.csv).
- [Analysis file hashes](../artifacts/triage/module10_v2/analysis/artifact_hashes.json),
  [verified input artifact hashes](../artifacts/triage/module10_v2/analysis/verified_source_hashes.json).
