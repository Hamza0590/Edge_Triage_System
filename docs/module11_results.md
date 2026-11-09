# Module 11 results — frozen held-out study

Study `artifacts/experiments/module11_v3` (protocol `module11_study_v4`), sealed 2026-11-09:
`final_validation.json` SHA-256 `f67ba4229c342d76749bc02bd91a4ecb695616f818becedf2c42fc5594abeab5`.
Test-authorisation point: `study/matrix.resolved.json` SHA-256 `363d9974…471a7` — written before any
held-out inference; no threshold, configuration or seed was changed afterwards.

## What ran

- 39 pipeline configurations (decision 0069), each fitted on the 643 unique validation rows
  (25 077 validation candidates, all 39 feasible at survival ≥ 0.95) and promoted to the 966 unique test rows.
- 21 component configurations (microbatch 1/2/4/8, threaded batch-1 1/2/4 workers × Tiny/Medium/Large).
- 2 timing repetitions each → **120/120 subruns complete, 0 failed, 0 skipped, 0 candidate failures, 0 OOM.**
- `verify_module11.py` regenerated every metric from saved records without inference; 42 component outputs
  match their pipeline baselines (max |Δlogit| 9.5e-6, identical predicted classes).
- Final checks on the frozen source: 409 tests passed, ruff/mypy (65 files)/pip clean.
- Environment: desktop CPU serving, Windows build 26300, RTX 5060 Ti; software-in-the-loop, **not Jetson
  emulation** (decision 0071, KI-003, KI-008).

## Per-configuration results (first repetition for science; classification metrics are identical across repetitions)

Survival = Real-candidate survival recall (transmit or compress). CI = IID percentile bootstrap over unique
test rows, 1000 repeats, fixed fitted thresholds. Inference/selection ms are per candidate (fine costs
overlap; never sum them into wall time). Throughput and P95 include all instrumented services.

| Case | Survival [95% CI] | False Real discards | Byte reduction [95% CI] | ROC-AUC | MCC | Throughput /s (r0/r1) | P95 E2E ms (r0/r1) | Inference ms/cand | Selection ms/cand | Fwd passes |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 0.9595 [0.942, 0.975] | 20 | 0.5828 [0.553, 0.612] | 0.9955 | 0.865 | 105.3/103.9 | 173/177 | 2.431 | 1.173 | 966 |
| B2 | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 105.0/104.9 | 171/171 | 1.728 | 1.022 | 966 |
| B3 | 0.9595 [0.942, 0.976] | 20 | 0.6617 [0.637, 0.689] | 0.9988 | 0.740 | 103.8/105.3 | 176/172 | 1.968 | 1.127 | 966 |
| B4a | 0.9798 [0.968, 0.990] | 10 | 0.4193 [0.389, 0.450] | 0.9986 | 0.897 | 96.5/94.1 | 198/262 | 1.545 | 1.190 | 1860 |
| B4b | 1.0000 [1.000, 1.000] | 0 | 0.4058 [0.377, 0.438] | 0.9987 | 0.903 | 92.4/94.7 | 232/205 | 1.713 | 1.231 | 1925 |
| B4c | 0.9595 [0.941, 0.976] | 20 | 0.5701 [0.542, 0.599] | 0.9988 | 0.758 | 90.6/91.1 | 280/280 | 2.056 | 1.271 | 2237 |
| B4d | 0.9798 [0.968, 0.990] | 10 | 0.4193 [0.389, 0.450] | 0.9986 | 0.897 | 94.3/94.1 | 203/203 | 1.552 | 1.246 | 1860 |
| B5 | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 96.0/96.0 | 189/188 | 1.699 | 1.367 | 966 |
| B6 | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 93.9/96.0 | 141/134 | 2.155 | 1.426 | 966 |
| B7 | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.493, 0.555] | 0.9988 | 0.949 | 2.9/2.9 | 8194/8196 | 1.572 | 1.397 | 966 |
| FULL | 0.9798 [0.967, 0.990] | 10 | 0.4193 [0.387, 0.451] | 0.9986 | 0.897 | 2.9/2.9 | 8202/8207 | 1.431 | 1.395 | 1860 |
| static_all | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 99.4/99.9 | 183/181 | 1.735 | 1.200 | 966 |
| B5_resource_only | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 96.4/94.2 | 187/195 | 1.929 | 1.321 | 966 |
| B6_memory_admission | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 2.9/2.9 | 8312/8314 | 3.598 | 1.171 | 966 |
| B7_fixed_admission | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 92.2/92.4 | 197/197 | 2.260 | 1.393 | 966 |
| B7_calibrated_noUQ | 0.9595 [0.942, 0.977] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 2.9/2.9 | 8195/8193 | 1.912 | 1.464 | 966 |
| B5_common | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 91.4/92.1 | 210/198 | 2.198 | 1.425 | 966 |
| B5_transmit_all | 1.0000 [1.000, 1.000] | 0 | 0.0000 [0.000, 0.000] | 0.9988 | 0.949 | 78.3/77.2 | 239/254 | 2.515 | 1.646 | 966 |
| B5_binary | 0.9595 [0.942, 0.976] | 20 | 0.5052 [0.473, 0.534] | 0.9988 | 0.949 | 75.4/78.2 | 251/242 | 2.818 | 1.748 | 966 |
| portfolio_tiny | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 83.3/82.4 | 229/238 | 2.318 | 1.441 | 966 |
| portfolio_medium | 0.9595 [0.942, 0.976] | 20 | 0.6617 [0.637, 0.689] | 0.9988 | 0.740 | 95.7/96.8 | 194/190 | 2.381 | 1.319 | 966 |
| portfolio_large | 0.9595 [0.942, 0.975] | 20 | 0.5828 [0.553, 0.612] | 0.9955 | 0.865 | 96.0/96.2 | 194/191 | 3.074 | 1.399 | 966 |
| portfolio_tiny+medium | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 93.2/92.9 | 199/195 | 2.221 | 1.316 | 966 |
| portfolio_tiny+large | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 93.9/94.9 | 195/196 | 2.105 | 1.347 | 966 |
| portfolio_medium+large | 0.9595 [0.942, 0.976] | 20 | 0.6617 [0.637, 0.689] | 0.9988 | 0.740 | 93.4/94.0 | 198/196 | 2.454 | 1.408 | 966 |
| fixed_tiny | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 98.0/98.0 | 132/135 | 2.214 | 1.265 | 966 |
| fixed_medium | 0.9595 [0.942, 0.976] | 20 | 0.6617 [0.637, 0.689] | 0.9988 | 0.740 | 97.8/97.7 | 135/136 | 2.748 | 1.277 | 966 |
| fixed_large | 0.9595 [0.942, 0.975] | 20 | 0.5828 [0.553, 0.612] | 0.9955 | 0.865 | 97.2/95.8 | 134/137 | 3.371 | 1.255 | 966 |
| B5_arrival_100 | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 90.5/90.8 | 201/201 | 2.371 | 1.402 | 966 |
| B5_arrival_20 | 0.9879 [0.978, 0.996] | 6 | 0.3680 [0.337, 0.396] | 0.9894 | 0.783 | 20.0/20.0 | 11/11 | 0.535 | 1.154 | 966 |
| seed_20260924_tiny | 0.9595 [0.942, 0.976] | 20 | 0.5238 [0.491, 0.554] | 0.9988 | 0.949 | 98.0/99.5 | 188/182 | 2.034 | 1.121 | 966 |
| seed_20260924_medium | 0.9595 [0.942, 0.976] | 20 | 0.7301 [0.709, 0.753] | 0.9988 | 0.740 | 98.7/96.6 | 185/195 | 2.210 | 1.208 | 966 |
| seed_20260924_large | 0.9595 [0.942, 0.975] | 20 | 0.6271 [0.599, 0.655] | 0.9955 | 0.865 | 97.8/98.9 | 189/184 | 2.888 | 1.302 | 966 |
| seed_20260925_tiny | 0.9676 [0.952, 0.982] | 16 | 0.5204 [0.488, 0.550] | 0.9976 | 0.955 | 99.4/99.6 | 184/184 | 2.107 | 1.043 | 966 |
| seed_20260925_medium | 0.9636 [0.946, 0.978] | 18 | 0.5587 [0.530, 0.588] | 0.9970 | 0.940 | 96.8/99.7 | 210/184 | 2.321 | 1.249 | 966 |
| seed_20260925_large | 0.9656 [0.949, 0.980] | 17 | 0.7226 [0.701, 0.746] | 0.9961 | 0.745 | 97.3/97.1 | 207/213 | 2.879 | 1.285 | 966 |
| seed_20260926_tiny | 0.9575 [0.940, 0.974] | 21 | 0.5427 [0.513, 0.573] | 0.9976 | 0.941 | 98.8/99.1 | 195/186 | 2.061 | 1.130 | 966 |
| seed_20260926_medium | 0.9676 [0.952, 0.982] | 16 | 0.5512 [0.521, 0.581] | 0.9977 | 0.936 | 99.6/98.9 | 184/184 | 2.233 | 1.196 | 966 |
| seed_20260926_large | 0.9595 [0.942, 0.976] | 20 | 0.6004 [0.571, 0.629] | 0.9980 | 0.896 | 98.3/98.4 | 188/185 | 2.905 | 1.295 | 966 |

Component-only throughput (candidates/s, preprocessed tensors, excludes scheduler/monitor/fate/logging;
no equal-recall claim): Tiny microbatch 1→8: 3.8k→8.5k; Large microbatch 1→8: 1.6k→2.6k; Large threads
1→4: 1.5k→2.9k; Tiny threads 1→4: 3.5k→3.1k.

## Findings (measured)

1. **Safety target held everywhere.** All 78 pipeline subruns reach survival ≥ 0.95 on held-out rows
   (min 0.9575, `seed_20260926_tiny`); none is ineligible for equal-recall comparison. Lower CI bounds
   (~0.94) fall below 0.95: no statistical guarantee is claimed.
2. **Adaptive selection reduced to static Tiny under backlog.** B5/B6/B7, `static_all`, every
   Tiny-containing portfolio, and the queue-aware/resource-only controls trust Tiny for all 966 rows and give
   science identical to B2. Versus always-Large (B1) at equal survival 0.9595: inference 1.70 vs
   2.43 ms/candidate (−30 %), but end-to-end throughput 96 vs 105/s and byte reduction 0.524 vs 0.583. With
   instrumentation (~10 ms/candidate wall) dominating, the cheaper model does not reduce full system cost.
3. **Portfolio choice decides bytes, not survival.** Medium-only and Medium+Large portfolios give the largest
   byte reduction (0.662) at the same survival; adaptive selection picks the cheapest tier, which is not the
   most selective one for fate.
4. **Dynamic concurrency and memory/contention-aware admission are much slower.** B7, FULL,
   B6_memory_admission, B7_calibrated_noUQ: 2.9/s with P95 ≈ 8.2 s, vs 92–105/s for sequential/fixed
   execution with fixed admission (B7_fixed_admission 92/s). Same inherited cause as Module 09 (R-028): runtime
   latency exceeds isolated profiles, so admission pauses at cap 1. Fixed concurrency 2 lowers P95 (≈135 vs
   ≈175 ms) without raising throughput.
5. **UQ/escalation trades bytes for survival.** FULL vs B7_calibrated_noUQ: survival 0.9798 vs 0.9595
   (10 vs 20 false Real discards), but byte reduction 0.419 vs 0.524 and 1860 vs 966 forward passes. The
   Tiny→Large cascade B4b reaches survival 1.0 (0 false discards) at byte reduction 0.406. MC-dropout standard
   deviation does not separate errors from correct visits (uncertainty plot), consistent with negative Gate C.
6. **Per-tier vs common thresholds: no difference observable** (B5 vs B5_common identical) because only one
   tier is trusted under backlog.
7. **Light load changes tier choice (held-out finding).** At 20 candidates/s the selector picked Medium for 51 %
   / 82 % of rows in the two repetitions, although validation at the same rate chose Tiny for all 643 rows.
   Medium has no validated threshold in that fate artifact, so Module 10's conservative rule transmitted those
   rows: survival 0.988/0.996, byte reduction 0.368/0.094. Safety held; savings depend on the live machine
   state the selector reads. Not corrected after test (KI-009).
8. **Seeds.** Nine frozen singletons over three training seeds: survival 0.9575–0.9676, byte reduction
   0.520–0.730. Fate cutoffs and savings are checkpoint-specific; descriptive n = 3, no best-seed selection.
9. **Arrival rate.** P95 end-to-end ≈ 11 ms at 20/s versus ≈ 190–200 ms in backlog and at 100/s (queue at its
   cap of 8): latency here is queueing, not inference.

## Plots

`study/summaries/plots_a00c2393c46389be/` — 9 required plots + component and seed comparisons, all visually
inspected (`analysis/visual_qa.json`). Known cosmetic issues (not corrected; `figures.py` is pinned frozen
source): 39-case legends reuse 10 colours; second-repetition colours missing from the arrival legend; the
tier-usage legend overlaps the last bars. Use the table above for case-level reading.

## Limitations

Unknown repeated-object independence (KI-001); validation reused for selection/calibration/fitting (KI-002);
two timing repetitions and three seeds, descriptive comparisons only; logical compression accounting, no
codec, fidelity, network or energy measurement (KI-005); desktop background activity and OS/driver drift
(KI-003, KI-008); component timings exclude pipeline services.
