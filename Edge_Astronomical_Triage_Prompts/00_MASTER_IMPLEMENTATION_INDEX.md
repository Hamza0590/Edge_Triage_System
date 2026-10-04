# Edge Astronomical Triage — Master Implementation Index

## Purpose

This folder is the implementation contract for the project **Resource- and Contention-Aware Concurrent Edge Triage for Astronomical Transient Candidates**. Each numbered Markdown file is an execution brief for Codex. The briefs are deliberately ordered, testable, and integration-first.

The project does not claim novelty for its dataset, CNN building blocks, uncertainty method, model-selection concept, or resource-aware scheduling mechanisms. Its research contribution is the documented integration and evaluation of established methods for recall-constrained astronomical edge triage.

## Mandatory operating procedure for every Codex run

1. Read this file completely.
2. Read `PROJECT_CONTEXT.md` completely.
3. Read the requested module brief completely.
4. Inspect the current repository and preserve valid existing work.
5. Change the module status in `PROJECT_CONTEXT.md` to `IN_PROGRESS` before implementation.
6. Implement the module; do not stop at a plan or pseudocode.
7. Run all module tests and the integration tests inherited from earlier modules.
8. Update `PROJECT_CONTEXT.md` with decisions, files, commands, results, artifacts, limitations, and the exact next action.
9. Mark a module `COMPLETE` only when every acceptance criterion in its brief is satisfied.
10. If blocked, leave the repository runnable, record the blocker precisely, and do not fabricate a successful result.

## Immutable project facts

- Project root: `D:\Uni\SEM 7\Edge\project`
- Dataset images: `D:\Uni\SEM 7\Edge\project\MeerLICHT_images.npy`
- Dataset labels: `D:\Uni\SEM 7\Edge\project\MeerLICHT_labels.csv`
- Images MD5: `bbf55c1cecbdea127497d6cd346e3455`
- Labels MD5: `cacf76a6aadd97ed18ab24b58e121bc2`
- Stored tensor shape: `(3219, 100, 100, 3)` in NHWC order
- Channels: `0=New/science`, `1=Reference`, `2=Difference`
- Labels: 1,647 `Real`, 1,572 `Bogus`; encode `Bogus=0`, `Real=1`
- Primary model input: center-cropped NRD triplets, default `3×30×30`, configurable
- Default split/training seed: `20260924`; every run must also record its explicit seed
- Primary safety target: real-candidate survival recall `>= 0.95`
- Development GPU observed: NVIDIA GeForce RTX 5060 Ti, 16 GB
- PC experiments are controlled software-in-the-loop edge experiments, not Jetson emulation.

Do not move, rename, overwrite, normalize in place, or otherwise mutate the two raw dataset files.

## Source-claim vocabulary

Every module must classify each research relationship correctly:

- **Reproduced:** the implementation follows the paper's specified algorithm closely enough to make a reproduction claim, with all deviations disclosed.
- **Adapted:** the central mechanism is retained but modified for this dataset, hardware, objective, or project scope.
- **Inspired:** only a design principle is borrowed; do not imply algorithmic reproduction.
- **Project-specific integration:** the mechanism combines components for this project and is not attributed to a source as though it appeared there.

Every algorithmic module must maintain a provenance table containing source, method borrowed, adoption level, deviations, excluded features, code/license status, and verification evidence.

## Module order and dependencies

| Order | Brief | Main output | Depends on |
|---:|---|---|---|
| 1 | `01_FOUNDATION_CONTRACTS_LOGGING_PROVENANCE.md` | Runnable package skeleton, schemas, structured logging, configuration and provenance | None |
| 2 | `02_DATASET_AUDIT_PREPROCESSING_SPLITS.md` | Validated dataset loader, immutable split manifest and preprocessing artifacts | 01 |
| 3 | `03_WALKING_SKELETON_PIPELINE.md` | End-to-end queue→scheduler→executor→prediction→fate→log path | 01–02 |
| 4 | `04_SCALABLE_CNN_FAMILY_TRAINING.md` | Trained Tiny/Medium/Large checkpoints | 01–03 |
| 5 | `05_EVALUATION_CALIBRATION_PROFILING.md` | Metrics, calibration artifacts and hardware model profiles | 02–04 |
| 6 | `06_RESOURCE_MONITOR_STREAMING_WORKLOAD.md` | Resource snapshots and reproducible arrival traces | 01–05 |
| 7 | `07_ADAPTIVE_SCHEDULER.md` | Pluggable tier-selection and admission policies | 03, 05–06 |
| 8 | `08_UNCERTAINTY_ESCALATION.md` | Selective MC-dropout and tier escalation | 04–07 |
| 9 | `09_CONCURRENT_INFERENCE_RUNTIME.md` | Sequential, fixed and dynamic concurrent executors | 03–08 |
| 10 | `10_RECALL_CONSTRAINED_DATA_FATE.md` | Calibrated discard/compress/transmit policy | 05, 08–09 |
| 11 | `11_EXPERIMENT_ABLATION_HARNESS.md` | Baselines, portfolio matrix, ablations, reports and plots | 01–10 |
| 12 | `12_CONSTRAINED_ENVIRONMENT_DEPLOYMENT.md` | Controlled resource/network experiments and final validation | 01–11 |

The scheduler interface appears in Module 03. Its calibrated resource-aware policy is implemented in Module 07 after model profiles and monitor signals exist.

## Expected repository layout

```text
project/
├── MeerLICHT_images.npy                 # immutable raw data
├── MeerLICHT_labels.csv                 # immutable raw labels
├── Prompts/                             # this implementation package
├── pyproject.toml
├── README.md
├── configs/
│   ├── data/
│   ├── models/
│   ├── policies/
│   ├── experiments/
│   └── environments/
├── src/edge_triage/
│   ├── contracts/
│   ├── data/
│   ├── models/
│   ├── calibration/
│   ├── monitoring/
│   ├── scheduling/
│   ├── uncertainty/
│   ├── runtime/
│   ├── triage/
│   ├── experiments/
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── artifacts/                           # generated, normally ignored by git
│   ├── data/
│   ├── models/
│   ├── calibration/
│   ├── profiles/
│   ├── traces/
│   └── reports/
├── runs/                                # one immutable directory per run
└── logs/                                # local live/debug logs
```

Do not create multiple competing source trees. Adapt this layout only when the existing repository already has a coherent equivalent.

## Ablation-first configuration contract

The pipeline must be assembled from independent policies, not a monolithic conditional block. At minimum, configuration must select:

```yaml
tier_portfolio: [tiny, medium, large]
tier_selection_policy: resource_queue_aware
execution_mode: dynamic_concurrency
admission_policy: memory_contention_aware
uncertainty_policy: selective_mc_dropout
escalation_policy: cascade
threshold_policy: per_tier
fate_policy: recall_constrained
target_survival_recall: 0.95
```

Supported portfolios:

- `[tiny]`, `[medium]`, `[large]`
- `[tiny, medium]`, `[tiny, large]`, `[medium, large]`
- `[tiny, medium, large]`

Supported execution modes:

- `sequential`
- `fixed_concurrency`
- `dynamic_concurrency`

Default fixed cascade orders:

- Tiny→Medium
- Tiny→Large
- Medium→Large
- Tiny→Medium→Large

## Cross-cutting logging contract

Every candidate keeps the same `run_id`, `trace_id`, and `candidate_id` across modules. Every structured event must include:

```text
schema_version, timestamp_utc, monotonic_ns, level,
run_id, trace_id, candidate_id, module, event_type,
config_hash, message, reason_code, duration_ms,
payload, error_type, error_message
```

Required storage:

- JSONL for append-only live/debug events.
- Parquet for candidate-level analytical records.
- CSV or JSON for compact summary metrics.
- A run manifest containing hashes of dataset, split, configuration, checkpoints, thresholds, arrival trace, source commit, environment and seeds.

Never log raw image tensors, secrets, or unbounded stack traces into candidate-event records.

## Core research sources

1. Hosenie et al., **MeerCRAB: MeerLICHT Classification of Real and Bogus Transients using Deep Learning**, Experimental Astronomy 51 (2021), DOI `10.1007/s10686-021-09757-1`, arXiv `2104.13950`.
2. Stoppa et al., MeerLICHT/ATLAS/Pan-STARRS transient image release, Zenodo record `14714279`, DOI `10.5281/zenodo.14714279`.
3. Bonnet-Guerrini et al., **Interpretable Human-Label-Free Deep Learning for Real-Bogus Classification with Uncertainty Quantification**, arXiv `2607.05393` (2026).
4. Taylor et al., **Adaptive Deep Learning Model Selection on Embedded Systems**, LCTES 2018, DOI `10.1145/3211332.3211336`.
5. Lee et al., **Peak-memory-aware partitioning and scheduling for multi-tenant DNN model inference**, Journal of Systems Architecture 173 (2026), DOI `10.1016/j.sysarc.2026.103696`.
6. Jiang et al., **Convergo: Multi-SLO-Aware Scheduling for Heterogeneous AI Accelerators on Edge Devices**, IEEE EDGE 2025, DOI `10.1109/EDGE67623.2025.00022`.
7. Sun et al., **Adaptive scheduling of online inference pipelines at the edge: A post-hoc request-oriented approach**, Journal of Systems Architecture 169 (2025), DOI `10.1016/j.sysarc.2025.103597`.
8. Nguyen et al., **OCTOPINF: Workload-Aware Inference Serving for Edge Video Analytics**, arXiv `2502.01277` (2025).
9. Gal and Ghahramani, **Dropout as a Bayesian Approximation**, ICML 2016, arXiv `1506.02142`.
10. Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017, arXiv `1706.04599`.
11. El-Yaniv and Wiener, **On the Foundations of Noise-free Selective Classification**, JMLR 11 (2010).

## Global definition of done

The project is complete only when:

- every module's acceptance criteria pass;
- all seven tier portfolios execute through one common pipeline;
- sequential, fixed-concurrency and dynamic-concurrency modes execute through interchangeable backends;
- thresholds are fitted only on validation data and frozen before test evaluation;
- every primary comparison is made at the same target survival recall or reports inability to reach it;
- the full cost includes monitoring, scheduling, UQ, escalation, compression accounting and transmission accounting;
- the test set is evaluated without tuning;
- all runs are reproducible from a manifest;
- the final report distinguishes measured results, inference, limitation and future work;
- no PC result is described as accurate Jetson emulation.
