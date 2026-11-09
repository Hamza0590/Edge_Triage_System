# 0071: Honest hardware labels after an OS/driver update; restart the study as module11_v3

Date: 2026-11-09
Source: Claude Code — validation fit failure on `seed_20260926_tiny`
Status: accepted (delegated)

## Problem
`hardware_identity()` hashes CPU, core count, RAM, GPU name/memory and OS build. Between the
2026-10-10 profiling and today, Windows moved 10.0.26200 → 10.0.26300 and the GPU's reported
memory changed 17 102 864 384 → 17 074 749 440 bytes (driver update). Same physical machine.
Inherited profiles (Modules 05/09) and the seed-20260925 bundle carry label `63998d2133ba053a`;
the seed-20260926 bundle, measured today, honestly carries `4aa40cb2ec4ec9cc`. The declared
seed-20260926 configs pinned the old label, so the scheduler (correctly) found no compatible
profile. The runtime never checks the live machine against the label; each run manifest
records its actual OS/GPU, so all 36 completed v2 collections ran on build 26300 with
old-label profiles. That is how the frozen-profile design works, not a defect.

## Options considered
- Re-label or re-measure seed 20260926 under the old ID: rejected — falsifies provenance.
- Re-profile every inherited family on the new OS: rejected — it would replace frozen Module 05/09
  evidence (contention cap, Gate D) that later modules were validated against; out of scope.
- Skip the seed-20260926 cases: rejected — it silently weakens the declared three-seed stability check.
- Give each config the label its own profile bundle was measured under, read from the bundle,
  and record host and label drift in the protocol: chosen.

## Decision
- Declarations are immutable and the driver pins its own hash, so the study restarts in
  `artifacts/experiments/module11_v3` (protocol `module11_study_v4`). The v2 seed profiles are reused
  unchanged (measurements don't depend on case definitions).
- Seed cases 20260925/20260926 take `scheduler.hardware_id` from their bundle (one label per bundle, enforced).
- `protocol.json` records the declaration host identity, the inherited label and the per-seed labels.
- `module11_v2` (36 frozen validation fits, orphaned B7 run, failed seed_20260926_tiny attempt) is
  preserved as superseded evidence, never mixed with v3.
- Before declaring v3, the freeze/promotion, run, summarize and plot stages were QA'd on v2 fits with
  no test inference (`artifacts/experiments/module11_qa_20261109/`), so further driver changes don't force a v4.

## Consequences
The study runs on Windows build 26300 with cost profiles measured on build 26200 (inherited and seed
20260925) or 26300 (seed 20260926). Profiles are scheduler cost estimates; reported latencies are
measured at runtime. This is a disclosed limitation (KI-008), and the seed comparison stays descriptive.

## Revisit if
Module 12 re-profiles under controlled environment profiles: those get new hardware IDs by design.
