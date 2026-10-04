# Module 05: validation calibration and hardware profiling

Run with the Edge_Project interpreter:

```powershell
& 'C:/Users/hamza/anaconda3/envs/Edge_Project/python.exe' -m edge_triage evaluate --config configs/experiments/module05_v1.yaml
```

The command creates a unique run, calibration directory and profile directory. It
starts from the frozen three-checkpoint family, verifies all checkpoint files and
split/preprocessing/raw hashes, and never regenerates data splits or trains weights.
Only the 643 validation candidates are inferred. Full-file hashing and split/label
metadata verification are integrity operations, not held-out evaluation.

## Evaluation and calibration

Each saved prediction records candidate ID, offline label, tier, checkpoint ID,
training seed, raw logit, and both raw and calibrated probabilities. One evaluator
reads this file to calculate all metrics. ROC-AUC groups tied logits; AP is
non-interpolated average precision. Trapezoidal PR-AUC is also reported separately.
Confusion matrices use rows true Bogus/Real and columns predicted Bogus/Real.
Undefined metrics are JSON null with an explicit reason. Extreme-logit NLL uses
logaddexp; classification and ranking use logits to avoid sigmoid rounding ties.

Temperature fitting minimizes binary NLL on validation only. Derivative bisection
in inverse temperature uses float64, 100 iterations and T bounds [0.01,100]. Boundary
solutions are recorded. Scaling cannot change the sign of logits or recall at 0.5.
The recommendation accepts calibrated probabilities only when NLL, Brier and ECE
are all non-worsening to 1e-12. Both results are retained regardless of that decision.

ECE is **binary classwise** ECE: 15 equal-width bins of p_real, comparing mean p_real
with Real frequency. Bins are left-closed/right-open, except the final bin includes
one. Empty bins contribute zero weight and have null means. This differs from
top-label confidence ECE used in some multiclass literature.

The equal-recall diagnostic chooses the highest inclusive raw-logit threshold that
retains at least ceil(0.95*N_real) positives; ties are all retained. Calibrated and
raw thresholds represent the same ordering. This diagnostic is not the Module 10
discard/compress policy. Percentile 95% confidence intervals use 1,000 seeded IID
candidate bootstrap replicates with fitted T and threshold fixed. They exclude
model-selection, calibration-fitting, grouped-object and training-seed uncertainty.
All calibration metrics are fit-set diagnostics because validation was reused for
checkpoint selection and calibration. They do not establish held-out performance.

For inference, set `calibration.artifact_index` to the generated index. The factory
verifies its per-tier artifact hashes and checkpoint/split/preprocessing identity.
`probability_mode` is `recommended`, `uncalibrated` or `calibrated`. Raw `p_real`
always remains sigmoid(raw logit); `calibrated_p_real` and calibration ID are
separate. The reference smoke fate policy still uses raw p_real and transmits all
bytes; the calibrated probability is available to later policies.

## Profiling

Each tier/device pair runs in a fresh subprocess, sequentially, on CPU and CUDA
when available. FP32, TF32 disabled, deterministic kernels, cuDNN benchmarking off,
one CPU intra-op thread and one inter-op thread are recorded. Batch sizes 1,2,4,8
use seeded synthetic raw NHWC inputs with the frozen crop and standardization.
No test images are used for profiling. Job, batch and component order are shuffled
with the experiment seed. Each component gets 50 warm-ups and two rounds of 300
measurements. Every CUDA sample synchronizes before and after perf_counter_ns.

Components: preprocessing; pageable blocking H2D (CPU .to is a no-op); preloaded
model forward; combined preprocessing/transfer/forward/sigmoid/host output; and the
actual RealModelAdapter on preprocessed batch-1 input. Components are measured
independently, include synchronization/allocator overhead, and are not additive.
Verified load time is separate and includes checkpoint validation/placement.

Profiles report mean/median/P90/P95/P99, total throughput, batch completion latency,
and amortized service time per candidate. Amortized service time is not the latency
experienced by a request waiting to form a batch. Raw timing samples are retained
and hashed. Two-round median ratios must be <=2 for the smoke stability flag.
These checks are not a guarantee of stable tails or a controlled edge device.
OS/GPU activity, clocks, temperature and utilization snapshots disclose desktop
conditions; other desktop processes are not stopped.

CUDA memory is absolute PyTorch peak allocated/reserved memory including resident
weights, input and measured operations. CPU memory is sampled process RSS, not an
isolated model allocation estimate or true temporal peak. Neither is device-wide
free memory. Model load and driver context costs are not inferred from allocator
peaks. No Jetson performance claims are supported.

The overhead script imports its dummy exclusively from tests/fixtures. It measures
synthetic source, FIFO, policies, minimal monitor, dummy adapter, JSONL and Parquet
including final flush, with one warm-up round and two measured 100-candidate rounds.
Logger-only timings are separate. Human-readable console output goes to StringIO,
so terminal rendering is excluded. Queue-inclusive candidate latency and wall time
per completed candidate are distinct. Report these costs separately; subtracting
them from model timings does not establish a causal concurrent-runtime estimate.

`MeasuredProfile` extends the foundation ModelProfile. `ProfileBundle` rejects
duplicate hardware/device/tier/checkpoint/shape/batch/precision keys. `load_profiles`
verifies bundle/raw timing hashes and checkpoint compatibility. A future scheduler
must choose matching hardware/precision/batch profiles and inspect stability.

## Predeclared Gate B

The persisted configuration precedes measurements. Cost passes when a stable
batch-1 pair has Large/Tiny E2E median latency or CUDA peak allocation >=1.25.
Quality passes when Large AP exceeds Tiny by >=0.0005 and its precision at target
recall is no worse; both must attain validation recall >=0.95. These are engineering
point-estimate criteria, not significance tests. A failure requires a documented
new family version under the module brief; never tune on test data or overwrite v1.

## Research adoption

| Source | Relationship | Implemented scope and exclusions |
|---|---|---|
| [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html) | Adapted | Positive scalar logit rescaling with validation NLL; binary fit and fallback are project choices; no original experimental reproduction |
| [MeerCRAB](https://arxiv.org/abs/2104.13950) | Adapted evaluation framing | Real/Bogus metrics including MCC; no claim of matching their dataset or results |
| [Quilt](https://www.sciencedirect.com/science/article/abs/pii/S1383762126000147) | Inspired | Memory/time measurement before scheduling; no compiler, partitioning, shared pooling or scheduler reproduction |

All implementation is independent; upstream code was not copied. Upstream code
licenses remain unknown rather than inferred from paper availability.
