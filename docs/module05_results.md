# Module 05 results

Run `20260925T073859_3c77b7896ab5` uses the original frozen family, seed 20260924,
643 validation rows, and hardware ID `63998d2133ba053a`. No retraining or test
evaluation occurred. Methodology and limitations are in [evaluation.md](evaluation.md).

## Quality at target validation recall

| Tier | AP | Recall | Precision | FPR | Raw probability threshold |
|---|---:|---:|---:|---:|---:|
| Tiny | 0.997140 | 0.951368 | 0.990506 | 0.009554 | 0.502946 |
| Medium | 0.997733 | 0.951368 | 1.000000 | 0.000000 | 0.028547 |
| Large | 0.997846 | 0.951368 | 1.000000 | 0.000000 | 0.063563 |

Each tier retains 313/329 Real candidates. These are fitted validation thresholds,
not a held-out recall guarantee or the final data-fate policy. Full metrics,
reliability bins and conditional bootstrap intervals are retained in the artifacts.

| Tier | Temperature | NLL before → after | Brier before → after | ECE before → after | Recommended |
|---|---:|---|---|---|---|
| Tiny | 0.722039 | .086152 → .078974 | .021743 → .020981 | .035471 → .021947 | Calibrated |
| Medium | 2.366591 | .407040 → .285196 | .121125 → .094854 | .151954 → .161417 | Uncalibrated |
| Large | 2.009669 | .229150 → .176189 | .066400 → .056507 | .087952 → .089340 | Uncalibrated |

Medium/Large temperatures remain available for explicit downstream ablations.
The recommendation follows the predeclared non-worsening NLL/Brier/ECE rule.

## Profile summary

Median batch-1 time includes synthetic raw preprocessing, transfer, forward,
sigmoid and host output. It excludes the orchestration/logger overhead below.

| Device | Tier | Batch-1 median ms | Batch-1 P95 ms | Batch-1 candidates/s | Batch-8 candidates/s |
|---|---|---:|---:|---:|---:|
| CPU, one thread | Tiny | .2442 | .2686 | 4,023 | 8,865 |
| CPU, one thread | Medium | .3151 | .3405 | 3,129 | 5,289 |
| CPU, one thread | Large | .5265 | .5582 | 1,886 | 2,839 |
| RTX 5060 Ti | Tiny | .5952 | .8217 | 1,587 | 10,929 |
| RTX 5060 Ti | Medium | .5767 | .6258 | 1,720 | 11,190 |
| RTX 5060 Ti | Large | .6082 | .6847 | 1,624 | 11,185 |

All 24 tier/device/batch profiles passed the two-round median stability criterion.
Batch-1 CUDA allocated peaks were 34,790,912 / 35,139,584 / 39,174,656 bytes for
Tiny/Medium/Large. These are allocator peaks with runtime costs, not weight sizes.
CUDA timing tails were variable, especially for Tiny; desktop activity remains
uncontrolled. Raw timing distributions and before/after system snapshots are saved.

Reference pipeline overhead was 2.712 and 2.641 ms per completed candidate in the
two measured 100-candidate synthetic rounds, including final flush. Logger-only
median was .0783 ms/event. Terminal rendering is excluded. This overhead exceeds
the model path's batch-1 time, so future work must account for instrumentation.

**Gate B passes** under the declared rule: Large/Tiny CPU batch-1 latency ratio
2.156 exceeds 1.25, and Large gains .000707 AP while improving precision at equal
recall. CUDA batch-1 latency ratio 1.022 and allocated-memory ratio 1.126 do not
pass the cost threshold. No GPU scheduling or concurrency benefit is established.
The existing architecture is retained.

## Verification and artifacts

170 tests passed in 55.10 s; Ruff, mypy (37 source files) and pip check passed.
All saved prediction IDs exactly match validation; no held-out predictions exist.
Three forced-calibrated singleton pipeline smoke runs completed 15/15 candidates.
Frozen raw files, split, preprocessing and model files still match their hashes.

- [Calibration index](../artifacts/calibration/20260925T073859_3c77b7896ab5/index.json)
- [Full metrics and intervals](../artifacts/calibration/20260925T073859_3c77b7896ab5/metrics.json)
- [Scheduler-readable profiles](../artifacts/profiles/20260925T073859_3c77b7896ab5/model_profiles.json)
- [Gate B](../artifacts/profiles/20260925T073859_3c77b7896ab5/gate_b.json)
- [Verification](../artifacts/profiles/20260925T073859_3c77b7896ab5/verification.json)
- [Run summary and hash inventory](../runs/20260925T073859_3c77b7896ab5/metrics/module05_summary.json)

Module 06 is authorized and its preflight is complete; implementation is pending
at the saved context checkpoint.
