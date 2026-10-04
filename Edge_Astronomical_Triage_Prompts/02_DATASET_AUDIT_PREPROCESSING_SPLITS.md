# Module 02 — Dataset Audit, Preprocessing and Frozen Splits

## Mission

Turn the frozen MeerLICHT files into a validated, reproducible PyTorch data pipeline without modifying the raw files. Produce an immutable split manifest and preprocessing artifact used by every later model and ablation.

## Mandatory preflight

1. Read the master index, context file and Module 01 artifacts.
2. Run the Module 01 health check and test suite.
3. Confirm raw-file hashes exactly match the context file.
4. Mark Module 02 `IN_PROGRESS`.

## Research basis and adoption

| Source | Use | Adoption |
|---|---|---|
| Stoppa et al., Zenodo `14714279` | Exact downloaded triplets and labels; channel order | Reused dataset with citation |
| Hosenie et al., MeerCRAB, DOI `10.1007/s10686-021-09757-1` | NRD task framing, center crop and flip augmentation | Adapted |

Do not state that this exact 3,219-row release is T9 or T10. Record the absence of object metadata and the resulting leakage limitation.

## Required implementation

### 1. Dataset inspector

Implement a read-only audit command:

```text
edge-triage data audit --config configs/data/meerlicht.yaml
```

It must verify and save:

- file paths, sizes and cryptographic hashes;
- NPY header, shape, dtype and memory order;
- CSV columns, types, row count and label values;
- exact image/label alignment through sequential `index_no`;
- class counts and ratios;
- per-channel/global min, max, mean, standard deviation and selected percentiles;
- NaN, Inf, all-zero, constant-channel and exact-duplicate counts;
- explicit metadata inventory;
- a warning that repeated-object leakage cannot be checked.

Write `artifacts/data/dataset_audit.json` and a concise Markdown report. Use memory mapping and chunked statistics; do not load unnecessary copies of the 386 MB array.

Expected values must be asserted, not merely printed:

```text
shape = (3219, 100, 100, 3)
dtype = float32
Real = 1647
Bogus = 1572
```

### 2. Immutable split manifest

Create one deterministic stratified split with proportions:

```text
train = 50%
validation = 20%
test = 30%
```

Use default seed `20260924`, stored explicitly in configuration and the manifest. A different seed creates a new version rather than overwriting the primary split. Generate a manifest containing:

```text
candidate_id,index_no,label,split
```

Write split statistics and SHA-256 hash. Assert:

- every index occurs exactly once;
- no split overlap;
- union equals all 3,219 rows;
- both classes occur in every split;
- ratios remain close to the full dataset;
- subsequent runs with the same inputs reproduce the identical manifest.

Never regenerate this manifest per experiment. Refuse to overwrite a different existing manifest unless the user explicitly authorizes a new split version.

### 3. Dataset loader

Implement a memory-mapped dataset class returning:

```text
Candidate metadata
image tensor in CHW order
integer label
```

Requirements:

- source is NHWC; model tensor is CHW;
- select channels by explicit names, default `new,reference,difference`;
- validate channel order against the dataset configuration;
- configurable center crop, default 30×30;
- no label-dependent preprocessing;
- no augmentation outside the training split;
- deterministic worker seeding;
- optional return of the original byte count for fate accounting.

### 4. Preprocessing profiles

Implement named, versioned preprocessing profiles rather than hidden transformations.

Required baseline profile:

```text
center crop 30×30
float32
training-only random horizontal flip
training-only random vertical flip
train-fitted per-channel standardization
```

Fit normalization statistics on the training split only. Persist them in `artifacts/data/preprocessing_<version>.json` with split hash and dataset hashes. Validation/test transformations must load the frozen artifact.

Because the raw distribution is heavy-tailed, collect percentile diagnostics. Do not add clipping/winsorization silently. If clipping is later tested, it must be a separately named preprocessing ablation fitted on training data only.

### 5. Data loaders and reproducibility

Provide loader factories for train, validation and test. Configuration must control batch size, workers, pinning and prefetching. Tests must show that validation/test batches and labels are identical across repeated runs.

### 6. Visual QA artifact

Generate a deterministic contact sheet with at least three Real and three Bogus candidates, each showing New, Reference and Difference separately in grayscale. The visualization may use a display stretch, but the title must say that display scaling is not model preprocessing.

## Leakage and test-integrity rules

- Never tune preprocessing, architecture, thresholds or ambiguous zones on the test set.
- The lack of object IDs is a documented limitation, not evidence that leakage is absent.
- Keep test labels accessible to offline evaluators but out of runtime scheduling decisions.
- Any synthetic repetition for load experiments may affect system metrics only; repeated rows are not new statistical classification samples.

## Required tests

- Expected hash/shape/count assertions.
- Failure on a wrong hash or wrong channel configuration.
- Split coverage, exclusivity, stratification and deterministic-hash tests.
- Correct NHWC→CHW and center-crop tests using known index patterns.
- No augmentation in validation/test.
- Train-only normalization fit test.
- Dataset and multi-worker loader smoke tests.
- Raw files unchanged after the full test suite.

## Acceptance criteria

- Audit and visualization artifacts exist.
- A single immutable split manifest exists and is hashed.
- Data loaders produce finite `float32` tensors of expected shape and integer labels.
- Preprocessing artifact is fitted only from training rows.
- The original dataset hashes remain unchanged.
- All tests and the audit command pass.
- Context records exact manifest/preprocessing paths and hashes.

## Context update requirements

Register dataset audit, split manifest, preprocessing version, class counts per split, visualization path, tests and limitations. Set the next action to Module 03.
