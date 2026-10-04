# Module 12 — Controlled Edge Environment, Deployment and Final Validation

## Mission

Package the system for reproducible controlled resource experiments, apply CPU/RAM/network/background-load conditions safely, and execute final validation. Preserve a clear boundary between PC-based software-in-the-loop results and later physical edge-hardware results.

## Mandatory preflight

1. Read the entire prompt package and project context.
2. Verify the experiment harness and reduced matrix smoke run.
3. Confirm Docker/WSL/Linux availability; record actual status rather than assuming it.
4. Mark Module 12 `IN_PROGRESS`.

## Research basis and adoption

| Source | Mechanism | Adoption |
|---|---|---|
| Quilt | Embedded-GPU contention and memory-safety motivation, including Jetson Orin Nano evaluation context | Inspired; not reproduced |
| Convergo | Multiple simultaneous SLOs on edge accelerator platforms | Adapted evaluation framing |
| PRAS | Bursty and variable online request workloads | Adapted workload framing |
| OCTOPINF | Workload-aware serving under edge resource/network variation | Inspired |

Container, cgroup and network configuration are engineering infrastructure. Cite official tool documentation where needed; do not misrepresent them as research-paper methods.

## Environment profiles

Create versioned profiles such as:

```text
native_unconstrained_reference
cpu_constrained_low
cpu_constrained_medium
ram_constrained
background_cpu_contention
background_memory_contention
network_low_bandwidth_high_latency
bursty_mixed_contention
```

Each profile records exact limits, workload commands, start/stop procedure, verification measurements and supported platforms.

## Safe controlled environment

Preferred Linux/WSL/Docker tools:

- container CPU quotas/cpuset;
- container memory limit and OOM behavior;
- `stress-ng` for controlled background CPU/memory load;
- `tc/netem` for bandwidth, delay, jitter and loss;
- process/container monitoring;
- NVIDIA device access where available.

Requirements:

- all stressors have explicit duration and cleanup;
- never run destructive disk stress;
- verify actual applied constraints before experiments;
- capture host-level background state;
- stop stressors after failure/interruption;
- isolate run artifacts from container teardown;
- do not claim GPU architecture, shared-memory, thermal or power equivalence to Jetson.

A Windows-native fallback may test CPU/RAM pressure and simulated network accounting, but must document which Linux controls were unavailable.

## Packaging

Provide:

- reproducible environment/dependency lock;
- Dockerfile and optional compose/profile definitions if supported;
- mounted read-only raw dataset;
- writable mounted artifacts/runs;
- non-root container execution where practical;
- health check invoking `edge-triage doctor`;
- documented CUDA/CPU fallbacks;
- configuration-driven paths, never host-specific paths baked into source.

## Experiment execution order

1. Native unconstrained reference.
2. Monitor overhead validation.
3. Isolated tier profiling under each relevant profile.
4. Fixed-concurrency sweep to confirm the contention knee changes under constraints.
5. Reduced primary baseline matrix.
6. Full approved ablation matrix.
7. Test-set evaluation exactly once per frozen configuration/seed unless a failure invalidates the run.
8. Artifact validation and report generation.

Do not launch the largest matrix until a reduced smoke matrix passes under the same environment.

## Optional physical hardware

If a Jetson or Raspberry Pi later becomes available:

- create a new `hardware_id` and environment profiles;
- rerun model profiling and threshold compatibility checks;
- do not reuse PC latency/memory profiles;
- execute a reduced representative matrix first;
- report physical hardware separately from PC simulation.

## Final validation checklist

### Data and models

- Raw hashes unchanged.
- Split/preprocessing/checkpoint/calibration/profile hashes verified.
- Test labels never used in fitting.

### Architecture

- Seven portfolios run.
- Three execution modes run or unsupported modes are explicitly documented.
- Policy swaps require configuration only.
- Candidate traces remain complete.

### Safety

- Survival recall calculation verified.
- False-discard candidates enumerated.
- Failure/ambiguity defaults are conservative.
- No silent queue drops or OOM omissions.

### Systems evidence

- Monitor/scheduler/UQ/escalation costs included.
- Concurrency knee measured.
- Queue and burst results use persisted traces.
- Hardware/environment described accurately.

### Reproducibility

- Run manifests complete.
- Source revision captured.
- Matrix status complete/failed/skipped visible.
- Reports regenerate from saved records without rerunning inference.

### Research claims

- Dataset, architecture and methods cited.
- Reproduced/adapted/inspired labels accurate.
- No unsupported “first” claim.
- PC results not presented as Jetson emulation.

## Final deliverables

Produce:

```text
artifacts/reports/final_results.parquet
artifacts/reports/final_summary.csv
artifacts/reports/final_summary.md
artifacts/reports/figures/
artifacts/reports/environment_report.md
artifacts/reports/reproducibility_manifest.json
artifacts/reports/provenance_matrix.md
```

The final summary must distinguish measured findings, interpretation, limitations, negative results and future hardware validation. Do not fill an expected-claim template with unmeasured values.

## Required tests

- Image build/environment installation test where applicable.
- Read-only dataset mount test.
- Constraint verification and cleanup tests.
- Interrupted-run cleanup test.
- Reduced matrix under one constrained profile.
- Report regeneration from saved records.
- Complete artifact/hash/provenance audit.
- Final full repository test suite.

## Acceptance criteria

- At least one constrained environment profile is applied and verified.
- A reduced matrix completes before full experiments.
- Final outputs are manifest-backed and regenerable.
- Claims stay within measured evidence.
- Every module is `COMPLETE` or has an explicit unresolved status and impact.
- `PROJECT_CONTEXT.md` ends with a self-contained final handoff.

## Context update requirements

Record exact environment profiles, executed matrices, final artifact paths/hashes, failures, limitations and final research-ready status. Do not mark the project complete while required work remains.

