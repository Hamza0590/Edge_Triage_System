# Module 01 — Foundation, Contracts, Logging and Research Provenance

## Mission

Create a clean, runnable Python project that all later modules can extend without changing core semantics. Implement shared data contracts, typed configuration, structured cross-module logging, immutable run manifests and research-provenance records.

Do not train a CNN or implement scheduling logic in this module.

## Mandatory preflight

1. Read `00_MASTER_IMPLEMENTATION_INDEX.md` and `PROJECT_CONTEXT.md`.
2. Inspect the project root and preserve valid existing code.
3. Verify both raw dataset files exist, but do not load their full contents.
4. Record the module as `IN_PROGRESS` in `PROJECT_CONTEXT.md`.

## Research and engineering basis

This module is infrastructure, not a claimed paper reproduction. Its provenance system exists to support the research methods listed in the master index. Use official Python/PyTorch package practices and explicitly label this module `engineering infrastructure` in the provenance registry.

## Required implementation

### 1. Package and dependency foundation

Create or extend `pyproject.toml` with:

- a package under `src/edge_triage`;
- supported Python version matching the chosen environment;
- runtime dependencies needed now, not every possible future dependency;
- development dependencies for `pytest`, linting and type checking;
- console entry point `edge-triage` or an equivalent `python -m edge_triage` path;
- deterministic test configuration.

Do not silently depend on the old MeerCRAB Python 3.6 environment. This project is a current PyTorch reimplementation/adaptation.

### 2. Shared contracts

Implement immutable dataclasses or equivalent typed models for:

```text
Candidate
ModelTier
PredictionResult
UncertaintyResult
ResourceSnapshot
SchedulingDecision
AdmissionDecision
EscalationDecision
TriageDecision
CandidateEvent
RunManifest
ModelProfile
```

Minimum semantics:

- `ModelTier`: `tiny`, `medium`, `large`.
- `PredictionResult`: logit, calibrated/un-calibrated `p_real`, tier, checkpoint ID, timings.
- `ResourceSnapshot`: CPU, available RAM, GPU utilization when available, GPU memory, queue length, arrival rate, active jobs by tier, recent latency, network state and sample timestamps.
- `SchedulingDecision`: selected tier and explicit `reason_code`.
- `AdmissionDecision`: admitted/delayed/rejected, permitted concurrency and explicit `reason_code`.
- `TriageDecision`: `discard`, `compress`, or `transmit`, with the exact threshold artifact used.

Validate probability ranges, non-negative durations and legal state transitions. Keep tensor objects outside serializable contracts.

### 3. Probability and label conventions

Centralize these constants and test them:

```text
BOGUS_LABEL = 0
REAL_LABEL = 1
p_real = sigmoid(logit)
```

Do not permit a generic ambiguous field named only `score`. Use `p_real`, `p_bogus`, or `logit` explicitly. If `p_bogus` is exposed, compute it as `1 - p_real` and never store inconsistent independent values.

### 4. Typed configuration

Implement YAML-backed configuration with validation for:

- paths;
- dataset and split parameters;
- preprocessing profile;
- model tiers;
- training;
- calibration;
- monitor;
- tier selection;
- admission/concurrency;
- uncertainty/escalation;
- threshold/fate policy;
- workload trace;
- experiment metadata.

Reject unknown keys by default. Resolve relative paths from the configuration file or documented project root, never from an accidental current directory. Produce a canonical serialized form and SHA-256 `config_hash`.

### 5. Structured logging

Create one logging API used by every module. It must:

- write append-only UTF-8 JSONL;
- echo concise human-readable messages to the console;
- include UTC wall-clock and monotonic timestamps;
- propagate `run_id`, `trace_id`, and `candidate_id`;
- accept module, event type, reason code, duration and bounded payload;
- serialize enums and dataclasses safely;
- redact known secret-like fields;
- never serialize raw tensors/images;
- record exceptions without killing the logger;
- flush safely on shutdown.

Define and document event families:

```text
run.*
data.*
model.*
monitor.*
scheduler.*
inference.*
uncertainty.*
triage.*
experiment.*
error.*
```

Add `schema_version` to every record. Add tests that parse every produced line as JSON.

### 6. Run directory and manifest

Create a run-directory manager that generates:

```text
runs/<run_id>/
├── manifest.json
├── config.resolved.yaml
├── events.jsonl
├── candidates.parquet          # may be created later
├── metrics/
└── artifacts/
```

The manifest must record:

- run ID and purpose;
- UTC start/end times and status;
- configuration hash;
- dataset and label hashes;
- split-manifest hash when available;
- checkpoint and threshold hashes when available;
- git commit and dirty-state flag when git exists;
- Python, PyTorch, CUDA and important dependency versions;
- OS, CPU count, GPU name/memory and RAM when observable;
- all random seeds;
- research provenance IDs.

Runs are immutable after completion except for an explicit appended correction record.

### 7. Research-provenance registry

Implement a machine-readable provenance file, for example `references/provenance.yaml`, with fields:

```yaml
id:
title:
authors:
year:
venue:
doi_or_arxiv:
url:
component:
adoption_level: reproduced|adapted|inspired|infrastructure
method_borrowed:
deviations:
excluded_scope:
code_url:
license_status:
verification_notes:
```

Seed it with the core sources listed in the master index. Do not invent missing licenses or code availability; use `unknown_pending_verification` when required.

### 8. Initial CLI and health check

Implement commands equivalent to:

```text
edge-triage --help
edge-triage doctor --config <path>
edge-triage show-config --config <path>
```

`doctor` should validate configuration, directories, dataset existence/hashes, dependency versions and CUDA availability without training or changing data.

## Required tests

- Contract round-trip serialization tests.
- Invalid probability, duration, tier and decision tests.
- Configuration unknown-key and path-resolution tests.
- Stable configuration hash test.
- JSONL schema and correlation-ID propagation tests.
- Secret-redaction and tensor-rejection tests.
- Run-manifest creation/finalization tests.
- Dataset hash health check using the frozen files.
- CLI smoke tests.

## Acceptance criteria

- A fresh environment can install the package.
- `doctor` verifies the frozen files without modifying them.
- A sample run creates a manifest and valid JSONL events.
- All shared contracts are typed, documented and tested.
- No module uses print statements as its primary logging mechanism.
- Provenance records distinguish adaptation from reproduction.
- All tests pass, and the result is recorded in `PROJECT_CONTEXT.md`.

## Context update requirements

Register the logging schema version, contract module paths, CLI, run-directory format, provenance registry and test results. Add the exact next action: Module 02.

