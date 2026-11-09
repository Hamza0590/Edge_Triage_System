# Experiment log

Append-only. Never edit or delete a past entry. Newest last.
Measured outputs live in `artifacts/` and `runs/` (immutable). Experiments before
2026-11-09 (Modules 04–10) are documented in `docs/models.md`, `docs/module05_results.md`,
`docs/monitoring.md`, `docs/scheduling.md`, `docs/uncertainty.md`, `docs/module09_results.md`,
`docs/module10_results.md`, and the archived context `specs/history/PROJECT_CONTEXT_codex.md`.

Entry format:

```
## EXP-NNN - title (date)
- Hypothesis:
- System or baseline:
- Change from previous:
- Config:  Seed:  Code version:  Workload:
- Operating point:
- Result:
- Conclusion:
- Notes:
```

## EXP-M11-000 - Module 11 seed profiling, partial (2026-10-10, Codex)
- Hypothesis: none (prerequisite: measured CPU cost profiles for diagnostic seed families).
- System or baseline: CPU profiling of existing seed-20260925/20260926 Tiny/Medium/Large checkpoints; no retraining.
- Config: `artifacts/experiments/module11_v2/seed_profile_protocol_v2.json` (synthetic inputs, 50 warm-ups, 2×300 samples, batches 1/2/4/8). Protocol v1 superseded before measuring (schema requires all 4 batch sizes).
- Result: seed 20260925 complete (`seed_profiles/20260925/`); seed 20260926 not started — run interrupted by user (Ctrl-C, exit 1).
- Conclusion: resume for 20260926; verify 20260925 before use.

## EXP-M11-001 - Module 11 seed profiling completed (2026-11-09, Claude Code)
- Hypothesis: none (prerequisite for seed-stability cases).
- System or baseline: CPU profile worker, protocol `seed_profile_protocol_v2.json`, unchanged from EXP-M11-000.
- Change from previous: resumed; seed 20260925 outputs reused after verification, seed 20260926 measured.
- Config: `artifacts/experiments/module11_v2/seed_profiles/<seed>/config.yaml`  Seeds: 20260925, 20260926  Code: commit 2ba2d95 + uncommitted Module 11 tree.
- Result: 12/12 profiles per seed, all `stable`, no failures, worker and raw-sample hashes verified.
  Batch-1 CPU p50 (ms) Tiny/Medium/Large: 20260925 0.262/0.348/0.557; 20260926 0.242/0.315/0.527.
  Stability ratios 1.00–1.25 (20260925), 1.00–1.04 (20260926).
- Conclusion: prerequisites for declaration satisfied. Tier cost order Tiny < Medium < Large holds on CPU for both seeds.
- Notes: synthetic inputs; desktop background activity uncontrolled (KI-003).

## EXP-M11-002 - Module 11 study declaration (2026-11-09, Claude Code)
- Hypothesis: H1–H5 (declares the cases that test them; see `docs/hypotheses.md`).
- System or baseline: 39 pipeline cases (decision 0069), protocol `module11_study_v3`, 2 repetitions.
- Config: `artifacts/experiments/module11_v2/{declaration.json,protocol.json,inputs/}`; runtime source pinned in `runtime_source.json`.
- Result: all 39 configs validated; all 14 comparison groups pass single-axis `enforce_comparison`.
  Pre-declaration checks: 409 passed (267 s), ruff/mypy (65 files)/pip clean.
- Conclusion: proceed to validation fit. No `src/` edits allowed from here.

## EXP-M11-003 - Validation fit, first attempt interrupted (2026-11-09, Claude Code)
- Hypothesis: H1–H5 (validation threshold fitting; no hypothesis tested yet).
- System or baseline: `study_module11.py --stage fit`, protocol `module11_study_v3`.
- Result: 9/39 cases collected and frozen (B1, B2, B3, B4a–d, B5, B6). B7 collection interrupted when the
  Claude Code session closed (background process killed, no traceback). Orphaned run preserved:
  `module11_v2/runtime_runs/20261109T104100_43e9081a4155` (manifest `running`, 21 388 events, never attested).
  `validation/B7/` holds only `collection_config.yaml`.
- Conclusion: resumed as a detached process (PID 24408); completed cases are reused after
  hash/attestation checks, and B7 gets a fresh immutable run. The orphaned run is never used as evidence.

## EXP-M11-004 - module11_v2 validation fit stopped by hardware-label mismatch (2026-11-09)
- Hypothesis: none (validation threshold fitting).
- Result: 36/39 cases frozen (all except the three seed-20260926 cases). `seed_20260926_tiny` raised
  `ValueError: no unique stable compatible profile for tiny`. Cause: an OS update (Windows 10.0.26200 → 26300)
  and a GPU driver change moved `hardware_identity()` from `63998d2133ba053a` to `4aa40cb2ec4ec9cc`; the
  seed-20260926 profiles (measured today) carry the new label, while the declaration pinned the old one.
- Conclusion: decision 0071 — restart as `module11_v3` with each seed config taking its bundle's own
  label. The v2 fits are superseded, preserved and never mixed with v3. Total v2 fit time ≈ 25 min.

## EXP-M11-005 - Downstream stage QA on superseded v2 fits (2026-11-09)
- Purpose: exercise freeze/run/summarize/plot before declaring v3, without test inference.
- Config: `scratchpad/qa_downstream.py` → `artifacts/experiments/module11_qa_20261109/`.
- Result: 36 promotions + a 57-case / 114-subrun / 14-comparison test matrix validated (**not run**);
  reduced validation matrix (B1, B7, component_large_microbatch_2 × 2 reps) 6/6 complete, 0 failed;
  summary and 10 plots generated.
- Conclusion: stages work; v3 declared. QA artifacts are not study evidence.

## EXP-M11-006 - module11_v3 declaration (2026-11-09)
- Config: `artifacts/experiments/module11_v3/{protocol.json,declaration.json}`, protocol `module11_study_v4`, 39 cases.
- Hardware recorded: host `4aa40cb2ec4ec9cc` (Windows 10.0.26300); inherited profiles `63998d2133ba053a`;
  seed 20260925 `63998d…`, seed 20260926 `4aa40c…`. Runtime source unchanged since the 409-pass checks.
- Status: validation fit running (detached PID 13092).

## EXP-M11-007 - module11_v3 validation fit and freeze (2026-11-09)
- Result: 39/39 validation collections × 643 unique rows = 25 077 candidates; 39 frozen fate fits, all
  feasible (validation survival ≥ 0.9514; transmit-all 1.0). Validation byte reduction: static Tiny/Medium/Large
  0.5412/0.6614/0.5984; adaptive B5/B6/B7 0.5412 (equal to Tiny — the selector appears to choose Tiny);
  cascades B4a/b/c/d 0.4184/0.3919/0.5715/0.4184; FULL 0.4184; binary 0.5086. Validation-conditional only.
- Freeze: 39 promotions, 39 test configs; `study/matrix.resolved.json` 120 subruns (78 pipeline + 42 component),
  all `ready`. **Test-authorisation point: matrix.resolved.json SHA-256
  363d99744e3e2c1e4e55716ccc894e5dbd6034a239ef56f2fd13003a39f471a7.** No refit after this.

## EXP-M11-008 - Module 11 frozen held-out study (2026-11-09)
- Hypothesis: H1–H5 (see `docs/hypotheses.md`).
- System or baseline: 39 pipeline + 21 component cases × 2 repetitions, `module11_v3`, matrix frozen at 363d9974…471a7.
- Config: `artifacts/experiments/module11_v3/{matrix.yaml,test_configs/}`  Seeds: 20260924 (+20260925/26 singletons)  Code: commit 2ba2d95 + frozen uncommitted tree (runtime_source.json)  Workload: canonical 966-row test partition, periodic 0 / 0.01 / 0.05 s.
- Operating point: survival ≥ 0.95, thresholds fitted on validation only.
- Result: 120/120 complete, 0 failed, 0 candidate failures, 0 OOM; all 78 pipeline subruns ≥ 0.95 (min 0.9575).
  B1 0.9595/0.583 bytes/105 per s; B5 (adaptive = Tiny) 0.9595/0.524/96 per s; B7 and memory-aware admission 2.9/s;
  FULL 0.9798/0.419, 1860 passes; B4b 1.000/0.406; arrival 20/s picked Medium for 51 %/82 % of rows → conservative
  transmit (0.988/0.996 survival, 0.368/0.094 bytes). Full table: `docs/module11_results.md`.
- Verification: `verify_module11.py` regenerated all metrics without inference; 42 component equivalences
  (max |Δlogit| 9.5e-6); visual QA of 11 plots; final checks 409 passed; sealed `final_validation.json` f67ba422…abeab5.
- Conclusion: H1 rejected under full-cost accounting, H2 rejected, H3 rejected, H4 inconclusive, H5 rejected.
  The safety target holds for every configuration on held-out data.
- Notes: run took ≈ 75 min, dominated by 8 memory-aware admission subruns (~5.5 min each).
