# 0067: Reorganise repo for Claude Code; delete Codex-only files

Date: 2026-11-09
Source: Claude Code session (user request)
Status: accepted (delegated)

## Problem
The repo carried Codex-specific scaffolding: a 200 KB `PROJECT_CONTEXT.md` handoff, editor
history snapshots, empty sandbox folders, a one-off AI prompt and an obsolete script.

## Options considered
- Keep everything: rejected — the context file is too large to load per session and duplicates docs.
- Delete PROJECT_CONTEXT.md: rejected — it holds the only decision log D-001–D-066, artifact registry and risk register needed for the final report.
- Archive the context, keep the module briefs as specs, delete only junk: chosen.

## Decision
- `Edge_Astronomical_Triage_Prompts/` → `specs/` (briefs 00–12 are the acceptance criteria).
- `PROJECT_CONTEXT.md` → `specs/history/PROJECT_CONTEXT_codex.md` (read-only archive).
- Deleted: `.history/`, `pytest-cache-files-*/`, `extras/MeerLICHT_Dataset_Inspection_Prompt.md`,
  `scripts/checkpoint_module11.py` (targeted the abandoned 42-run smoke study).
- No `src/`, `tests/`, `configs/`, `artifacts/` or `runs/` file touched.

## Consequences
Verified against `artifacts/experiments/module11_v2/session_stop_20261010.json`: all 24 pinned
artifacts and 168/172 source files unchanged; the 4 differences are the link edits and the
deleted script. Runtime `source_inventory()` (only `src/edge_triage`) is unaffected.

## Revisit if
No known condition.
