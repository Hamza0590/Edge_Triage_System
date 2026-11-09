# 0070: Module 11 seal requires freshly recorded checks bound to the frozen source

Date: 2026-11-09
Source: Claude Code review of `scripts/seal_module11.py`
Status: accepted (delegated)

## Problem
The seal read `module11_v2/checks.json`, a hand-assembled file from 2026-10-10 describing an
older tree. A seal could therefore cite test results that never ran on the frozen code.

## Options considered
- Keep the hand-written file: rejected — not tied to the code that produced the results.
- Script that runs pytest/ruff/mypy/pip check, records real exit codes, output tails and the
  `source_inventory()` it ran on: chosen (`scripts/record_checks_module11.py`).

## Decision
`seal_module11.py --checks <record>` refuses unless every check exited 0, pytest reports
"passed", and the record's runtime source equals the frozen matrix's `source_hashes`.
Pre-declaration record: `module11_v2/checks_predeclare_20261109.json` (409 passed, ruff/mypy/pip clean).
The original `checks.json` stays as historical evidence.

## Revisit if
No known condition.
