# Module08: selective uncertainty and sequential escalation

Gate C **fails** for all 5/10/20-pass budgets. The proposed configuration is
`configs/experiments/module08_proposed_v1.yaml`: no UQ, no escalation, and the
inherited recommended probability modes. `module08_selective_v1.yaml` retains
five-pass calibrated selective cascades as a **negative ablation**. Always-MC
is available only as an explicit ablation. No Module09 concurrent executor or
Module10 recall-constrained fate is implemented here.

## Runtime behavior

`uncertainty/sampling.py` activates dropout modules only, freezes BatchNorm,
uses inference mode, and restores every original module training flag even on
failure. Diagnostic sampling uses a seed derived from the configured seed,
candidate ID and tier, forks/restores the CPU and relevant device RNG, and
retains the actual seed in each UQ result. `seed: null` means production
stochastic sampling; the resolved run config and nullable manifest seed map
both record null. Module09 now coordinates package deterministic forwards and MC
state/RNG changes with a shared/exclusive guard; MC holds exclusive access through
restoration. The production streaming pipeline remains sequential while async
integration is pending. See [Module09 checkpoint](concurrency.md) for scope and limits.

Outputs are probability mean, population variance/std, entropy of the mean in
nats, binary vote variation ratio, pass count, duration and routing reason.
Dispersion is an approximate MC-dropout uncertainty proxy, not an exact measure
of epistemic uncertainty. Mutual information is not implemented.

`AmbiguousOnlyMCDropoutPolicy` sees only a prediction and image, never a label
or `CandidateInput`. It validates checkpoint, calibration and probability mode.
The factory verifies family/split/preprocessing/calibration-index hashes,
individual calibration hashes and configured pass count. Sampling occurs at
inclusive fitted interval boundaries. Std at or above the fitted threshold,
or disagreement between MC-mean and deterministic classes at0.5, requests
escalation. An outside-interval candidate incurs zero additional forwards.
NoUQ's `passes=1` is a deterministic-summary convention; its additional count
is zero.

`CascadeEscalationPolicy` chooses the next enabled stronger tier, records the
visited path, enforces a maximum of two escalations by default and never wraps
back. Singleton/final-tier unresolved cases receive a conservative flag.
The pipeline also guards invalid substitute-policy paths independently.
Admission is repeated for each tier, and leases remain owned through that
tier's UQ before release. Failures and cancellation release leases. Streaming
active counters cover UQ without adding MC cost to deterministic latency history.

Each tier's prediction/UQ/escalation has a correlated event. Parquet retains
complete JSON chains, initial raw scores and final trusted deterministic
probability. Final-event chains are structured arrays to avoid the logger's
string limit. Scores from different tiers are never averaged. MC means route
only. `uq_ms` and `escalation_decision_ms` time complete policy calls separately
from deterministic inference. Additional forwards count attempts, including
failed attempts; failed sampling also records completed/attempted counts and
elapsed cost. A cancellation aborts the stream with cleanup rather than
inventing a completed candidate record.

Final events carry compact resource snapshot references (including monotonic
time when no sampler ID exists); full snapshots remain in their separate
events and Parquet. This avoids silently truncating fallback snapshot strings
in reference-mode three-tier runs.

Classification-only fate still uses the inherited raw classification score
and transmits all bytes. Its decision is distinct from the final trusted
calibrated score. The conservative flag is a future fate input, not a present
discard policy or survival-recall guarantee.

## Fitting, measurements and Gate C

Frozen evidence is in
`artifacts/uncertainty/20260925T205209_b5ff8ad7f32c/`.
The index hashes three routing artifacts, all stochastic probability samples,
5/10/20 analysis reports, calibration baselines, objective/protocol, environment,
resolved config, source inventory and plots. `verification.json` separately
hashes the benchmark index and nine runtime runs. Use `plots_v2/` for the
visually verified plots; its index pins the benchmark and PNG hashes.

The objective was saved before MC measurements. The subsequent analysis protocol
made aggregation, error definition and measurement order explicit before the
first benchmark. Each tier's symmetric interval includes the closest20% of its
643 frozen calibrated validation probabilities to0.5, with inclusive ties:
129/643=20.0622%. Std cutoff is the linear80th percentile within that interval.
Training, splits, checkpoints, temperatures and Module05 thresholds are unchanged.

| Tier | Lower | Upper | Std cutoff,5 passes | Selective-std error AUROC | Confidence AUROC |
|---|---:|---:|---:|---:|---:|
| Tiny | .0198243 | .9801757 | .2911168 | .9062500 | .9352227 |
| Medium | .2236812 | .7763188 | .1900609 | .7932297 | .9189772 |
| Large | .0907861 | .9092139 | .2155804 | .8743459 | .9338038 |

Selective-std score is zero outside the interval. Error means deterministic
calibrated classification at0.5 disagrees with the validation label; the cheaper
score is negative distance from0.5. Reports also include full-MC separation,
error quantiles with ties kept together, raw/calibrated deterministic and MC-mean
NLL/Brier/classwise15-bin ECE, and none/confidence/selective/always cascades for
all seven portfolios. Confidence-only routing sends ambiguous rows directly
to the next tier with no MC forwards; it is a diagnostic baseline.

Gate C requires every tier's selective AUROC>=.55 and confidence-AUROC gain>=.02;
every portfolio must have escalation<=.20, extra passes/candidate<=4,
UQ/initial-deterministic cost<=2 and recall>=.95 at frozen final-tier thresholds.
Undefined error AUROC fails. Among passing budgets the declared selector would
minimize equal-tier mean selective UQ cost, then pass count. All budgets fail
the confidence-gain checks; some cascades additionally fail recall and larger
budgets fail costs/pass limits. No threshold was relaxed after seeing results.

Three-tier, Tiny-start saved-sample diagnostics:

| Mode | Pass budget | Escalation | Extra passes/candidate | Mean UQ ms | Total component ms | Frozen-threshold recall |
|---|---:|---:|---:|---:|---:|---:|
| None | 0 | 0 | 0 | 0 | .417550 | .951368 |
| Confidence | 0 | .200622 | 0 | 0 | .570526 | .951368 |
| Selective | 5 | .063764 | 1.205288 | .460167 | .912604 | .948328 |
| Selective | 10 | .057543 | 2.348367 | .820699 | 1.267746 | .945289 |
| Selective | 20 | .055988 | 4.696734 | 1.616297 | 2.062214 | .945289 |
| Always | 5 | .091757 | 5.497667 | 2.010974 | 2.481141 | .948328 |

The five-pass selective cascade corrects3 initial errors and introduces34
regressions at the0.5 diagnostic threshold. Those counts are not errors at
the recall thresholds. Reports contain correction/initial-error and
regression/initial-correct rates, class/tier escalation counts and rates,
escalated/non-escalated error rates, UQ cost per candidate/per escalation,
conservative counts, paths, latency quantiles and component throughput.

Measurements use CPU FP32 batch1, one intra-op/inter-op thread,20 deterministic
warmups and one warmup call per MC budget. Each row rotates budget execution
order. Diagnostic seeds provide nested masks across budgets, not independent
sampling replicates. Timing includes sampler state/RNG restoration, transfers
and synchronization. Offline cascade costs sum only visited deterministic/MC
components; they exclude runtime queueing, scheduling, logs and monitoring,
which are measured separately in replay summaries and candidate records.
They are not end-to-end throughput predictions.

Explicit calibrated mode is an ablation choice: Module05 recommends raw scores
for Medium/Large because calibration worsened ECE. Frozen deterministic
temperatures are applied to individual MC logits; that does not recalibrate
the MC mean. The five-pass MC-mean NLL/Brier/ECE are worse than the deterministic
calibrated baseline for all tiers. No MC-specific temperature was fitted.

The accepted report consistently uses frozen logits/probabilities for
classification and threshold comparisons. Fresh batch1 logits are retained
as separate observations. Maximum absolute drift vs Module05 batched logits
is Tiny3.81e-6, Medium7.63e-6, Large5.72e-6; one Large candidate crosses the
inclusive frozen threshold. The earlier benchmark directory
`20260925T204827_f00013e2c19d` mixed these score paths in recall reporting and
is superseded, preserved unchanged. No threshold was moved to hide the drift.

Validation was reused for training selection, calibration and routing fitting.
These are fit-set diagnostics from one training seed, not independent evidence
of generalization or a safety guarantee. No test predictions, test metrics or
test-label tuning occurred. Frozen system traces are repeated load, not
independent classification observations. Actual system UTC run IDs fall on
2026-09-25; local session date is2026-09-26.

## Verification and commands

The verifier completed1,080/1,080 candidates: seven120-row backlog portfolios,
one all-tier selective burst and one proposed-default burst. Zero failures,
rejections, drops or cancellations. It checked event/Parquet chain equality,
IDs, resource references, increasing enabled paths, final trusted scores,
conservative flags, timing/pass accounting, artifact hashes and raw-data hashes.
Those frozen traces exercise only a few escalations; separate temporary
real-model fixtures force every pair/triple/singleton path, including full
Tiny→Medium→Large with actual MC sampling. Synthetic fixture thresholds never
enter the research artifact directory.

Use `C:/Users/hamza/anaconda3/envs/Edge_Project/python.exe` throughout:

```text
python scripts/benchmark_uncertainty.py
python scripts/verify_module08.py --index artifacts/uncertainty/20260925T205209_b5ff8ad7f32c/index.json --output <new-verification-path>
python -m pytest -q --tb=short -W error -o faulthandler_timeout=60
python -m ruff check src tests scripts
python -m mypy src
python -m pip check
```

Benchmark runs create new immutable directories. The verifier checks existing
fitted reports before creating new runtime runs; its config outputs refuse
changed content under the same version. Tests use temporary runs/artifacts;
Windows pytest needs the approved temp/cache execution permissions here.

Final validation: **282 passed in101.41s**, warnings treated as errors; Ruff
clean, mypy clean on50 source files, and pip check clean. The inherited tests
remain enabled. `final_validation.json` and `source_hashes.json` record the
final checks and exact source inventory after the nullable-seed and compact
resource-reference fixes. Replay artifacts predate only those serialization
refinements, both exercised by the final regression.

## Research relationships

| Source | Adoption | Implementation and exclusions |
|---|---|---|
| [Gal and Ghahramani, ICML2016](https://proceedings.mlr.press/v48/gal16.html) | Adapted | Repeated dropout inference as approximate Bayesian sampling; no exact posterior claim |
| [Bonnet-Guerrini et al., sections5.1/5.4 and AppendixF.3](https://arxiv.org/html/2607.05393v1) | Adapted moments/evaluation | Probability mean/population variance and calibration evaluation; no injection/co-teaching, ensembles, dual-network hybrid or interpretability reproduction |
| [Guo et al., ICML2017](https://proceedings.mlr.press/v70/guo17a.html) | Inherited adapted calibration | Frozen validation-fitted binary temperature scaling; MC predictive means are not independently recalibrated |

Selective invocation, validation interval/cutoff fitting, visited-tier cascade,
conservative flag and Gate C are project-specific integration. All code is
independently written; no upstream code copied. Upstream code licenses remain
unverified. Evidence and deviations are registered in `references/provenance.yaml`.
