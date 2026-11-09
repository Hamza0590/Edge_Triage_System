# 0068: Adopt systems-research state/decision/worklog structure

Date: 2026-11-09
Source: Claude Code session (user request: "adjust accordingly ... decision, session history")
Status: accepted (delegated)

## Problem
The project is both engineering (a system) and research (claims tested against baselines and
ablations). Codex kept everything in one file; there were no decision files, session history,
experiment log or hypothesis register.

## Options considered
- Single `STATE.md`: rejected — 12 modules and >100 subruns need separate next-task and issue files.
- Full re-scaffold (`docs/modules/`, `docs/interfaces.md`, `configs/baselines/` ...): rejected —
  modules 01–10 are complete and documented in `specs/` + `docs/*.md`; duplicating them adds drift risk.
- Retrofit only the missing pieces: chosen.

## Decision
Add: `docs/state/{current-state,next-task,known-issues}.md`, `docs/decisions/`, `docs/worklog/`,
`docs/hypotheses.md` (H1–H5 from the project spec §8), `experiments/LOG.md` (append-only),
`.claude/commands/handoff.md`. Module specs stay in `specs/`; per-module docs stay in `docs/`;
measured artifacts stay in `artifacts/` and `runs/`. Baselines/ablations remain cases of the
one Module 11 matrix, not separate configs.

## Consequences
`CLAUDE.md` names this layout. Each session ends with `/handoff`.

## Revisit if
Module 12 adds enough environment profiles to need `configs/environments/` (brief 12 asks for it).
