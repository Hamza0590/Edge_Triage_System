# Hypotheses

Source: `extras/Edge_Astronomical_Triage_Final_Project_Spec.md` §8. Status changes only
because of logged experiment results (`experiments/LOG.md`). Negative results are valid
outcomes. Baseline/ablation IDs are those of `specs/11_EXPERIMENT_ABLATION_HARNESS.md`.

Final evidence: Module 11 frozen held-out study (`docs/module11_results.md`, EXP-M11-008;
966 unique test rows, 2 timing repetitions, desktop CPU serving, descriptive comparisons, no
significance tests). Earlier validation-stage evidence (Modules 05–10) is kept for context.

## H1 — Adaptive tiering
- Claim: adaptive Tiny/Medium/Large reduces average inference cost vs always-Large at ≈ the same target recall.
- Tested by: B1 vs B5; static_all vs B5; portfolio ablation (all 7).
- Status: **rejected under full-cost accounting** (inference-only proxy improves)
- Evidence: at equal survival 0.9595, B5 used 1.70 vs B1 2.43 inference ms/candidate (−30 %), but
  end-to-end throughput was 96 vs 105/s, P95 189 vs 173 ms, and byte reduction 0.524 vs 0.583. B5 trusted Tiny
  for all 966 rows under backlog, so it is identical in science to static Tiny (B2); adaptivity added
  selection overhead (~1.4 ms/candidate) but no benefit. Validation context: Gate B passed on CPU only (D-027).

## H2 — Dynamic concurrency
- Claim: resource-aware dynamic concurrency improves throughput and queue growth vs sequential, especially in bursts.
- Tested by: B5 vs B6 vs B7; B5/B6/B7_fixed_admission; component microbatch/threads.
- Status: **rejected**
- Evidence: B7 2.9/s (P95 ≈ 8.2 s) vs B5 96/s and B6 94–96/s; with fixed admission, the dynamic executor
  (B7_fixed_admission) ran 92/s, still not above sequential. Fixed concurrency 2 lowered P95 (≈135 vs ≈175 ms)
  without raising throughput. Component-only threading raised Large throughput 1.5k→2.9k/s, but pipeline
  services (~10 ms/candidate) dominate. Bursty traces were not part of this study (periodic only).
  Consistent with Gate D (D-050).

## H3 — Contention awareness
- Claim: unrestricted parallelism degrades latency/stability; CPU/GPU/RAM/queue-aware admission keeps a better operating region.
- Tested by: B6 vs B6_memory_admission; B7 vs B7_fixed_admission; B5 vs B5_resource_only.
- Status: **rejected** (no degradation to protect against at the tested caps; aware admission was harmful)
- Evidence: memory/contention-aware admission cut throughput from 92–96/s to 2.9/s with no stability gain;
  fixed caps showed no degradation (0 failures, 0 OOM). Removing queue awareness changed nothing (both choose Tiny).

## H4 — Tier-aware trust
- Claim: per-tier thresholds preserve more Real candidates at equal resource cost than one common threshold.
- Tested by: B5 vs B5_common.
- Status: **inconclusive**
- Evidence: identical outcomes (survival 0.9595, byte reduction 0.524), because only one tier is trusted under
  backlog, so per-tier and common thresholds coincide. Validation context: per-tier saved 30–60 KB in 6/36
  settings (D-060).

## H5 — Uncertainty-guided escalation
- Claim: high-uncertainty samples hold a disproportionate share of errors, so selective escalation beats always-Large.
- Tested by: FULL vs B7_calibrated_noUQ; B4a–d cascades; uncertainty-vs-error plot.
- Status: **rejected** (as an efficiency claim)
- Evidence: FULL raised survival 0.9595 → 0.9798 (10 fewer false Real discards), but byte reduction fell
  0.524 → 0.419 and forward passes rose 966 → 1860. Against always-Large B1, FULL transmits more (0.419 vs
  0.583 reduction) with ~2× passes. MC standard deviation does not separate errors from correct visits.
  Consistent with Gate C (D-040). Note: the Tiny→Large cascade B4b reached survival 1.0 — a safety/bytes
  trade-off, not an efficiency gain.
