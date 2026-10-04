# Module 08 — Selective Uncertainty Quantification and Tier Escalation

## Mission

Implement uncertainty only where it can improve safety or routing. Use validation data to define ambiguous regions, run MC dropout selectively, and escalate through the currently enabled portfolio without changing the common pipeline.

## Mandatory preflight

1. Read the master, context and Modules 04–07.
2. Verify calibrated checkpoints and adaptive sequential execution.
3. Mark Module 08 `IN_PROGRESS`.

## Research basis and adoption

| Source | Mechanism | Adoption |
|---|---|---|
| Gal and Ghahramani, arXiv `1506.02142` | Dropout at inference as approximate Bayesian uncertainty sampling | Adapted |
| Bonnet-Guerrini et al., arXiv `2607.05393` | Real/Bogus UQ evaluation and comparison of MC dropout/ensembles/hybrid strategies | Selective MC-dropout ideas adapted; no dual-network reproduction |
| Guo et al., arXiv `1706.04599` | Calibrated probabilities before interpreting confidence | Adapted |

## Correct MC-dropout behavior

Implement a context manager that:

- puts the model in evaluation mode;
- re-enables dropout layers only;
- keeps BatchNorm in evaluation mode;
- restores the original training/evaluation states afterward;
- performs stochastic forward passes without gradients;
- uses explicit seeds for reproducible diagnostic runs while allowing stochastic production sampling.

Do not call `model.train()` globally during inference.

## Uncertainty outputs

For `N` stochastic probabilities, calculate at least:

- predictive mean `mean_p_real`;
- predictive standard deviation;
- predictive entropy of the mean;
- variation ratio or agreement statistic where meaningful;
- pass count and elapsed cost.

If mutual information is implemented, define and test it. Do not label standard deviation as epistemic uncertainty without noting the approximation.

## Selective invocation

MC dropout must not run for every candidate by default. Implement:

```text
NoUncertaintyPolicy
AmbiguousOnlyMCDropoutPolicy
AlwaysMCDropoutPolicy            # ablation only
```

Fit the ambiguous probability interval per tier using validation predictions. Store explicit lower/upper bounds, calibration reference and objective. Candidate labels must not be available to the runtime policy.

Benchmark pass counts `5`, `10`, and `20`, measuring:

- uncertainty/error separation;
- calibration effect;
- escalation rate;
- additional forward passes;
- latency and throughput cost.

Select the default using a predeclared validation objective, not test performance.

## Escalation policy

Implement portfolio-aware cascade escalation:

```text
[tiny, medium]        tiny → medium
[tiny, large]         tiny → large
[medium, large]       medium → large
[tiny, medium, large] tiny → medium → large
```

Single-tier portfolios cannot escalate locally. At the final tier, an unresolved candidate must be marked for conservative fate handling/flagging, never wrapped back to a weaker tier.

Escalation conditions may use:

- membership in the calibrated ambiguous interval;
- MC-dropout uncertainty above a validation-fitted threshold;
- disagreement between predictive mean and deterministic prediction;
- final-tier conservative fallback.

Prevent cycles with an explicit visited-tier path and maximum escalation count.

## Score semantics after escalation

The final decision record must retain:

- initial tier/logit/probability;
- every UQ result;
- every escalation reason;
- every subsequent tier/logit/probability;
- final trusted tier and probability.

Do not average scores across different tiers unless a separately cited/validated ensemble policy is intentionally implemented.

## Validation analyses

Produce:

- error rate vs uncertainty quantile;
- error rate among escalated vs non-escalated candidates;
- escalation rate by tier and class;
- correction rate: initial errors corrected after escalation;
- regression rate: initially correct predictions made incorrect;
- UQ cost per candidate and per escalated candidate;
- validation recall/cost curves;
- plots with uncertainty separated by correct/incorrect prediction.

## Gate C — useful uncertainty

Pass only if uncertainty identifies errors better than chance or a cheaper confidence baseline and the selected escalation rate/cost remains acceptable. If it fails, retain `NoUncertaintyPolicy` as the proposed-system default and report MC dropout as a negative ablation; do not force it into the pipeline.

## Required tests

- Dropout active/BatchNorm frozen state test.
- Model state restoration after success and exception.
- Reproducible diagnostic sampling test.
- Finite uncertainty metric tests.
- Ambiguous-zone boundary tests.
- Correct pairwise and three-tier escalation paths.
- No cycles and final-tier fallback.
- Complete escalation event chain and cost-accounting tests.
- No validation/test label access inside runtime policies.
- Sequential end-to-end cascade smoke tests for all pairwise and three-tier portfolios.

## Acceptance criteria

- MC dropout is correctly and selectively executed.
- Ambiguous zones and uncertainty thresholds are validation-fitted artifacts.
- Every portfolio escalates only through enabled tiers.
- UQ/escalation cost is fully logged.
- Gate C is explicitly pass/fail.
- Sequential adaptive cascades complete and preserve candidate traces.

## Context update requirements

Register UQ policy/version, pass count, ambiguous bounds, threshold artifact, Gate C result, costs and escalation analyses. Set next action to Module 09.

