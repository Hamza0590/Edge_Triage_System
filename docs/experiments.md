# Module 11 experiment and ablation harness

Module 11 is COMPLETE (2026-11-09). The frozen held-out study `artifacts/experiments/module11_v3`
ran 120/120 subruns with 0 failures, was independently verified and sealed. Results, hypotheses
and limitations: [module11_results.md](module11_results.md) and [hypotheses.md](hypotheses.md).
Decisions: [0069](decisions/0069-portfolio-ablation-adaptive-selection.md),
[0070](decisions/0070-seal-requires-fresh-checks-on-frozen-source.md),
[0071](decisions/0071-hardware-label-drift-new-study-version.md).
`module11_v2` is superseded evidence (hardware-label mismatch, decision 0071) and is never mixed in.

## Study protocol

The curated study declares 39 pipeline cases and 21 component cases, each with two
timing repetitions: 120 subruns, not a full factorial design. Each pipeline configuration collects643 unique
validation rows, fits its own frozen fate thresholds at target survival recall.95,
and is promoted to966 unique held-out rows before evaluation. No test-dependent
refitting, seed selection or adjustment of the negative Gates C/D is permitted.

B1/B2/B3 are static Large/Tiny/Medium sequential baselines. B4a-d cover all pairwise
and three-tier selective cascades. B5/B6/B7 cover adaptive selection with sequential,
fixed and dynamic execution. FULL adds selective UQ and escalation as a negative
ablation; noUQ/no escalation remains the default. Other cases isolate selection,
queue awareness, admission, thresholds, fate, all seven portfolios (adaptive selection,
decision 0069), arrival rate,
and three frozen training seeds. B5/B6/B7 jointly change executor and effective
admission; a separate B5/B6/B7_fixed_admission group holds admission fixed.

Microbatch1/2/4/8 and threaded batch1 with1/2/4 workers use the same singleton models
and held-out backlog. These component timings exclude scheduler, monitor, fate,
preprocessing and durable logging. They cannot support full-pipeline or equal-recall
cost claims. Saved logits are compared against the matching pipeline baseline.

## Commands

Use `C:/Users/hamza/anaconda3/envs/Edge_Project/python.exe` throughout:

```text
python scripts/study_module11.py --stage profiles   # seed profiles, stored in module11_v2
python scripts/record_checks_module11.py --output <out>/checks_predeclare_<date>.json
python scripts/study_module11.py --stage declare
python scripts/study_module11.py --stage fit
python scripts/study_module11.py --stage freeze     # test-authorisation point
python scripts/study_module11.py --stage run
python scripts/verify_module11.py
# inspect every plot, write <out>/analysis/visual_qa.json
python scripts/record_checks_module11.py --output <out>/checks_final_<date>.json
python scripts/seal_module11.py --checks <out>/checks_final_<date>.json
```

Profiles must precede declaration: the two additional frozen seed families need
measured compatible CPU profiles. Profiling uses synthetic inputs,50 warm-ups and
two rounds of300 samples. No retraining occurs. Study source/protocol/configuration
pins are written before validation collection. A changed measurement source requires
a new version. Validation and test runtime warm-ups use five forwards per enabled
tier, plus an explicit sampler warm-up for UQ cases. Component worker warm-ups are
separately recorded.

Equivalent reusable CLI commands are `experiment validate-matrix`, `freeze`, `run`,
`summarize --runs <study>`, and `plot --summary <saved-summary.json>`. Every case uses
the common pipeline or the explicitly identified component backend.

## Integrity and recovery

Promotion checks every behavioral field and permits only partition, experiment
name/purpose and matched periodic-trace/count changes. Both traces must contain the
complete unique partition in canonical order. Promotion pins source configuration,
fit, collection, runtime evidence and threshold files. Runtime checks promotion
before loading test tensors. Threshold loading still validates the original fitted
configuration's exact identity.

The frozen matrix records inputs, transitive checkpoint/profile pins, source and
software versions. Completed subruns are reused only after every saved hash passes.
Incomplete attempts produce new immutable run IDs; failures and below-target outcomes
remain visible. `execution.lock` prevents concurrent runners and is never stolen.
After a crash, inspect its owner and preserved partial artifacts before removing a
confirmed stale lock. Never infer a live process solely from an unfinished manifest.

## Metrics and interpretation

Per-run metrics include classification/calibration, survival and false discards,
logical bytes, unique-row bootstrap intervals, UQ error/correction/regression,
latency distributions, queue trajectories, sampled resources, failures and forward
counts. Timing repetitions stay separate; repeated rows never become independent
astronomical observations. Below-target outcomes are ineligible for equal-recall
cost comparisons and are never repaired by test-dependent threshold changes.

Preprocessing, monitoring, selection, admission, initial/escalation inference, UQ,
fate, executor dispatch and wall time are reported separately. These fine costs
overlap and must not be summed into wall time. A disjoint four-region candidate
clock partition is checked against each end-to-end latency. Candidate-latency sums
are not concurrent stream wall time. Network time is null; compression is logical
quarter-size accounting with no codec or fidelity measurement.

The nine required scientific plots plus component and seed comparisons are generated
from saved summaries. The verifier regenerates metrics from manifests/records,
checks frozen fates/bytes and compares component logits without new inference.
Validation reuse, unknown object grouping, descriptive three-seed evidence, desktop
activity, and inherited dynamic admission slowdown remain explicit limitations.
