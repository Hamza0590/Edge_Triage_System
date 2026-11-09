# Edge Astronomical Triage

Resource- and contention-aware astronomical Real/Bogus triage. The research
contribution is integration and equal-recall evaluation of established methods.
Modules01-08 supply infrastructure, frozen data/models, calibration/profiling,
monitoring, adaptive sequential scheduling and evaluated uncertainty ablations.
Modules09–10 complete concurrent execution and validated recall-constrained data fate.

## Setup (PowerShell)

Python 3.11+; the project's dedicated Anaconda environment uses Python 3.13:

```powershell
conda env create -f environment.yml
conda activate Edge_Project
edge-triage --help
edge-triage doctor --config configs/default.yaml
edge-triage show-config --config configs/default.yaml
edge-triage sample-run --config configs/default.yaml
edge-triage pipeline smoke --config configs/experiments/smoke.yaml --limit 20
edge-triage train --config configs/training/family_v1.yaml
python -m pytest
python -m ruff check .
python -m mypy src
```

See [data pipeline setup and usage](docs/data.md) for Module 02 audit, preprocessing and loaders.
See [reference pipeline documentation](docs/pipeline.md) for Module 03 protocols,
non-scientific smoke configurations, candidate records and timing definitions.
See [CNN training and results](docs/models.md) for Module 04 checkpoints, validation
metrics, registry checks, reproducibility and unresolved tier-separation limitations.
`python -m edge_triage` provides the same CLI. Foundation dependencies are Pydantic,
PyYAML and psutil. PyTorch is detected when installed but is optional in this
foundation module; install and validate its GPU build during the training module.
The old MeerCRAB Python 3.6 environment is not used.

## Data and configuration

Keep `MeerLICHT_images.npy` and `MeerLICHT_labels.csv` unchanged. `doctor` streams
their MD5 hashes in 1 MiB chunks without loading the array or writing either file.
It reports dependency versions, CUDA availability, hardware and output-directory
readiness. Missing PyTorch is reported and is not a foundation failure; an explicit
`training.device: cuda` request fails when CUDA is unavailable.
GPU name/memory can also be observed through `nvidia-smi`; `cuda_available` means
CUDA usable by PyTorch in the active environment, not merely an installed GPU.

Unknown or duplicate YAML keys are errors. `paths.project_root` resolves from the
configuration file's parent; every other path resolves from that project root.
Canonical sorted JSON of the fully resolved configuration defines `config_hash`.
Thus an identical resolved configuration has an identical hash, while moving
absolute input/output paths deliberately changes it. `show-config` prints every default.

The default 50/20/30 stratified split and 3x30x30 NRD crop are declarations only in
Module 01. `p_real` always denotes probability of Real (label 1); Bogus is label 0.
See [contract documentation](docs/contracts.md) and [logging/run documentation](docs/logging.md).

## Reproducibility and provenance

Each `sample-run` creates a unique `runs/<run_id>` containing a manifest, resolved
configuration, JSONL events, and `metrics/` and `artifacts/` directories. Candidate
Parquet is produced by pipeline runs. Completed/failed runs
reject subsequent manager/logger writes; only `append_correction` may add a separate
audit note. This is an API guarantee, not protection against manual filesystem edits.

The manifest records dataset hashes, available artifact hashes, all configured
seeds, source commit/dirty state when Git exists, dependency versions and observable
hardware. No Git repository is required; unavailable source metadata is explicit null.
Training applies seeds to Python, NumPy, PyTorch, workers, shuffling and augmentation.

[references/provenance.yaml](references/provenance.yaml) seeds all 11 research
sources and labels Module 01 `engineering infrastructure`. Planned adoption levels
are provisional until each algorithmic module verifies its source. Unknown code and
license information stays `unknown_pending_verification`; no reproduction is claimed.
Packaging follows [PyPA guidance](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
Frozen models and validation follow [Pydantic documentation](https://docs.pydantic.dev/latest/concepts/models/).

The persistent implementation status and exact next action live in
[docs/state/](docs/state/current-state.md) (Codex-era record archived in [specs/history](specs/history/PROJECT_CONTEXT_codex.md)).

Module 05 provides validation-only calibration and CPU/CUDA model profiling:
see [methodology and usage](docs/evaluation.md) and [verified results](docs/module05_results.md).

Module08 adds selective MC dropout and bounded sequential cascades. Gate C fails
the declared validation objective, so the proposed default keeps UQ/escalation
disabled. See [implementation, evidence and commands](docs/uncertainty.md).

Module09 provides interchangeable sequential, fixed and dynamic streaming execution,
shared-model state/RNG coordination, measured contention caps and a microbatch baseline.
See [runtime usage](docs/concurrency.md) and [measurements and Gate D](docs/module09_results.md).

Module10 adds conservative fate policies, logical byte accounting and immutable
validation-fitted thresholds. All36 frozen runtime replays meet the target; the
proposed policy retains95.14% of Real candidates with54.12% logical byte reduction.
Final regression:378 tests. These are conditional validation results; no held-out
evaluation or compression codec was run. See [usage](docs/triage.md),
[verified results](docs/module10_results.md), and the
[proposed configuration](configs/experiments/module10_proposed_v2.yaml).

Module 11 is complete: one harness runs all seven portfolios and three execution modes as a
curated 39-case pipeline + 21-case component matrix, fitted on validation only and evaluated once on
the 966 held-out rows (120/120 subruns, independently verified and sealed). Every configuration keeps
survival recall >= 0.95 on held-out data; adaptive selection, dynamic concurrency, contention-aware
admission and UQ escalation give no full-cost benefit on this desktop (negative results, reported as such).
See [results](docs/module11_results.md), [hypotheses](docs/hypotheses.md) and [usage](docs/experiments.md).
