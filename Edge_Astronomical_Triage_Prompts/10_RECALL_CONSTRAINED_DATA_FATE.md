# Module 10 — Recall-Constrained Discard, Compress and Transmit Policy

## Mission

Convert the final calibrated prediction, uncertainty/escalation history and tier trust into a conservative data-fate decision. Fit all thresholds on validation data so the system preserves the target fraction of real candidates while reducing transmitted bytes.

## Mandatory preflight

1. Read the master, context and Modules 05, 08 and 09.
2. Verify calibration, UQ and executor artifacts.
3. Confirm `p_real` semantics with contract tests.
4. Mark Module 10 `IN_PROGRESS`.

## Research basis and adoption

| Source | Mechanism | Adoption |
|---|---|---|
| El-Yaniv and Wiener, JMLR 2010 | Selective classification/reject-option risk–coverage framing | Adapted |
| Guo et al., arXiv `1706.04599` | Calibrated probabilities used for threshold decisions | Adapted |
| Bonnet-Guerrini et al., arXiv `2607.05393` | Conservative treatment of uncertainty in Real/Bogus classification | Inspired/adapted |
| MeerCRAB | Astronomical Real/Bogus safety context | Reused task framing |

The three-way data-fate mapping is project-specific integration. Do not claim it appears unchanged in these papers.

## Non-negotiable probability semantics

All decisions use `p_real`:

```text
low p_real + low uncertainty + sufficient tier trust → eligible for discard
high p_real                                      → transmit
ambiguous/high uncertainty/final-tier unresolved → transmit or compress, never unsafe discard
```

Use explicit threshold names:

```text
max_p_real_for_discard
min_p_real_for_transmit
max_uncertainty_for_discard
```

Never use an unlabeled `threshold` or interpret a high `p_real` as evidence for discard.

## Survival-recall definition

Headline scientific safety metric:

```text
survival_recall = real candidates with fate != discard / all real candidates
```

Also report separately:

- classifier recall;
- real transmit recall;
- real compress recall;
- bogus discard rate;
- false-discard count/rate;
- overall byte reduction.

Do not conflate these metrics.

## Byte accounting

Implement versioned logical accounting first:

- `bytes_original`: exact candidate tensor bytes plus documented metadata overhead or a clearly defined logical raw payload;
- `discard`: zero transmitted payload;
- `compress`: configured fraction or explicit channel/thumbnail logical size;
- `transmit`: full configured payload;
- add fixed protocol overhead only if consistently modeled.

All assumptions belong in the run manifest. Compression CPU time is zero only when explicitly labeled `logical accounting`; do not claim a codec speedup.

## Threshold fitting

Fit separately for every relevant system configuration because available tiers and escalation behavior change the final score distribution.

Procedure:

1. Run the complete configuration on the validation split.
2. Collect final trusted tier, calibrated `p_real`, uncertainty and escalation path.
3. Search candidate thresholds deterministically.
4. Retain configurations meeting target survival recall `>=0.95`.
5. Among feasible settings, minimize the declared cost objective, initially transmitted bytes with compute as a reported secondary objective.
6. Apply conservative tie-breaking: fewer real discards, then lower uncertainty, then lower compute.
7. Persist a threshold artifact with validation metrics, configuration/checkpoint/split hashes and algorithm version.
8. Freeze it before any test run.

If a configuration cannot meet the target, record its best achievable recall; do not relax the target invisibly.

## Tier-aware trust

Support:

- `CommonThresholdPolicy` for ablation;
- `PerTierThresholdPolicy` for the proposed system.

The weaker tier should normally require stronger evidence of bogusness before discard, but enforce this only through a validated monotonic constraint or report if validation selects otherwise. Avoid arbitrary hard-coded example values.

## Required fate policies

- `TransmitAllPolicy` — bandwidth upper baseline.
- `BinaryDiscardTransmitPolicy` — simple threshold baseline.
- `CommonThresholdThreeWayPolicy`.
- `PerTierRecallConstrainedPolicy` — proposed.

Define deterministic behavior for missing calibration, missing uncertainty, failed inference and final-tier ambiguity. Default such cases to the conservative non-discard outcome.

## Required analyses

- survival recall vs bytes transmitted;
- survival recall vs full compute cost;
- fate distribution by true label and tier;
- false-discard review table with candidate IDs;
- common vs per-tier thresholds;
- effect of UQ/escalation on false discard and bytes;
- threshold stability across model seeds;
- validation bootstrap confidence intervals.

## Required tests

- Probability-direction regression test preventing high-`p_real` discard.
- Exact survival-recall and byte-accounting unit tests.
- Validation-only fitter access test.
- Artifact compatibility/hash tests.
- Conservative missing-data/failure behavior.
- Common/per-tier policy interchangeability.
- Monotonic constraint test where enabled.
- Frozen threshold cannot be modified during a test run.
- End-to-end fate output for all seven portfolios and execution modes.

## Acceptance criteria

- Every system configuration has a compatible frozen threshold artifact or an explicit infeasible status.
- The primary policy reaches the validation target before test evaluation.
- Survival recall and classifier recall are separately reported.
- Logical compression assumptions are explicit.
- Tier-aware thresholds are tested against a common threshold.
- No failure/ambiguity path silently discards data.

## Context update requirements

Register fate policy versions, byte assumptions, threshold artifacts/hashes, feasible/infeasible configurations and validation outcomes. Set next action to Module 11.

