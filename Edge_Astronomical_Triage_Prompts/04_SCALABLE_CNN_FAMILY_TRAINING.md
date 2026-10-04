# Module 04 — Scalable Tiny/Medium/Large CNN Family and Training

## Mission

Implement and train a single scalable CNN family for Real/Bogus classification. Tiny, Medium and Large must differ primarily in capacity while keeping topology, preprocessing, splits, training protocol and output semantics comparable.

## Mandatory preflight

1. Read the master index, context and Modules 01–03.
2. Verify the raw dataset, split and preprocessing hashes.
3. Run the walking-skeleton smoke test.
4. Mark Module 04 `IN_PROGRESS`.

## Research basis and adoption

| Source | Mechanism | Adoption |
|---|---|---|
| Bonnet-Guerrini et al., arXiv `2607.05393` | Three convolutional blocks with width progression `F→2F→4F`, BatchNorm, ReLU, pooling, spatial dropout and dense classifier | Adapted to 3-channel MeerLICHT inputs and supervised labels |
| Hosenie et al., MeerCRAB, arXiv `2104.13950` | Real/Bogus task, NRD inputs, 30×30 central crops, flip augmentation and shallow/deep CNN comparison | Adapted/reimplemented in PyTorch |

Do not implement the Bonnet-Guerrini injection pipeline, co-teaching, contaminated-class framework or dual-network training. Do not claim exact MeerCRAB reproduction because this dataset release and framework differ.

## Required model architecture

Implement one `ScalableRealBogusCNN` controlled by a typed tier specification.

Reference block:

```text
Conv2d(3×3, padding=1, bias consistent with BatchNorm)
BatchNorm2d
ReLU
MaxPool2d(2×2)
Dropout2d
```

Use three blocks with output channels:

```text
F, 2F, 4F
```

Follow with flattening, configurable dense layer(s), ordinary dropout and one output logit. Do not put a sigmoid inside the training model when using `BCEWithLogitsLoss`. The inference adapter converts the logit to `p_real`.

Initial tier candidates:

| Tier | Initial F | Initial dense width | Purpose |
|---|---:|---:|---|
| Tiny | 8 | 64 | Cheapest initial tier |
| Medium | 16 | 128 | Balanced tier |
| Large | 32 | 256 | Strongest tier |

These are hypotheses, not permanent facts. Store each architecture in YAML and emit parameter count, serialized checkpoint size and theoretical MAC/FLOP estimate where a reliable tool is available. Architecture changes require a versioned decision and must preserve same-family comparability.

Use initial spatial dropout `0.20` and dense dropout `0.50`, both configuration-controlled. These values are starting adaptations within the dropout range studied in the source architectures, not claimed paper-optimal values. Tune them only on training/validation data and version any accepted change.

## Training protocol

Use the frozen split and preprocessing artifact only.

Minimum training configuration:

- binary cross entropy with logits;
- Adam or AdamW, explicitly configured and cited as an implementation choice;
- learning rate, weight decay, batch size and maximum epochs in YAML;
- validation-based early stopping;
- checkpoint chosen by a declared validation metric, default PR-AUC with recall/FNR tie-breaking;
- deterministic seed propagation to Python, NumPy, PyTorch and data-loader workers;
- mixed precision configurable, with an FP32 reference run;
- horizontal/vertical flips only in the paper-aligned baseline;
- no test-set access during training or checkpoint selection.

Because classes are close to balanced, do not add class weighting by default. If weighting is tested, make it an explicit training ablation.

Train at least one complete seed for each tier to unlock integration. The final research experiments should use three seeds per tier if time permits. Never overwrite a checkpoint from another seed or configuration.

## Artifact format

Each checkpoint directory must contain:

```text
model_state.pt
model_config.yaml
training_config.yaml
preprocessing_reference.json
split_reference.json
metrics.json
history.csv
checkpoint_manifest.json
```

The manifest records tier, architecture version, parameter count, dataset/split/preprocessing hashes, seed, code commit, best epoch, selection metric and file hashes.

Avoid serializing arbitrary Python objects when a state dictionary and explicit configuration suffice.

## Training logs

Emit structured events for:

- training start/end;
- epoch start/end;
- train/validation loss and metrics;
- learning rate;
- early-stopping state;
- checkpoint save/load;
- NaN/Inf detection;
- peak host/GPU memory when observable;
- elapsed time.

Training histories are analytical artifacts; do not emit a JSON event per batch unless debug mode is explicitly enabled.

## Model registry integration

Implement a `ModelRegistry` that:

- loads by tier and checkpoint ID;
- verifies manifest and file hashes;
- places a model on the requested device;
- sets inference mode correctly;
- exposes preprocessing and architecture compatibility;
- rejects a checkpoint trained with a different split/preprocessing artifact unless explicitly allowed for diagnostics;
- supports all seven portfolios without changing model code.

Replace the walking skeleton's dummy adapters with real adapters while retaining dummy adapters only under test fixtures.

## Sanity gates

Before declaring success, report:

1. parameter/size differences among tiers;
2. validation metrics for all tiers;
3. a preliminary batch-1 latency check;
4. evidence that logits and gradients remain finite;
5. evidence that models can overfit a very small diagnostic subset when debug mode is used;
6. whether Tiny/Large appear different enough to justify adaptive selection.

Do not tune on test data to force tier separation. If separation is weak, record Gate B as unresolved for Module 05 profiling.

## Required tests

- Output shape and finite-logit tests for all tiers.
- Parameter counts strictly increase Tiny→Medium→Large.
- Forward/backward smoke tests on CPU and CUDA when available.
- Loss decreases on a tiny overfit fixture.
- Deterministic initialization and repeatable validation inference.
- Checkpoint save/load prediction-equivalence test.
- Registry hash and incompatibility rejection tests.
- No sigmoid-double-application regression test.
- End-to-end walking-skeleton run using each real model.

## Acceptance criteria

- Three versioned same-family model configurations exist.
- At least one valid trained checkpoint per tier exists.
- Model manifests link to the frozen data artifacts.
- No test metric influenced training or checkpoint choice.
- Real-model sequential smoke runs complete with structured logs.
- All tests pass and training outcomes are recorded honestly.

## Context update requirements

Register model versions, checkpoint paths/hashes, parameter counts, best validation results, seeds, training duration and unresolved tier-separation concerns. Set next action to Module 05.
