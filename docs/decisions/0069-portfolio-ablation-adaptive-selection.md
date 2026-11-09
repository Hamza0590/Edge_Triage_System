# 0069: Portfolio ablation uses adaptive selection over all seven portfolios

Date: 2026-11-09
Source: Claude Code review of `scripts/study_module11.py` before declaration
Status: accepted (delegated)

## Problem
Brief 11 requires a portfolio ablation ("each single, pairwise and all-three combination")
and plot 9 "portfolio comparison across all seven combinations". The Codex driver declared
the pairwise cases with the `fixed` selector and `fixed_tier = portfolio[0]`, UQ and
escalation off. Such a case only ever runs its first tier (`portfolio_tiny+large` ≡ Tiny),
so the portfolio had no behavioural effect and the ablation could not answer its question.

## Options considered
- Keep static-first pairs: rejected — measures model loading, not the portfolio.
- Compare B1–B3 (static) against adaptive pairs: rejected — changes two axes (portfolio and
  tier-selection policy), violating the single-axis rule of brief 11's fair-comparison protocol.
- Adaptive `resource_queue_aware` selection (the Module 07 policy, adapted from Taylor et al.)
  for all seven portfolios, with B5 as the all-three member: chosen. Only `portfolio` changes,
  which `enforce_comparison` verifies at declaration.

## Decision
Add `portfolio_tiny`, `portfolio_medium`, `portfolio_large` (adaptive singletons) and make the
three pair cases adaptive. The `portfolio` comparison is these six plus B5. Pipeline cases
36 → 39; subruns (39 + 21 components) × 2 repetitions = 120; validation rows 39 × 643 = 25 077.
Protocol version string `module11_study_v3`. B1–B3 static singletons, `static_all` (selection
ablation) and all other cases are unchanged. Updated: driver, `figures.py` portfolio plot keys,
`seal_module11.py` counts, figure QA test names.

## Consequences
Done before declaration, so no measurement is affected (only the seed-profile stage had run,
which doesn't depend on case definitions). About 3 extra validation collections and 6 extra test subruns.

## Revisit if
The adaptive selector cannot run a single-tier portfolio (Module 07 ran all seven, so this is not expected).
