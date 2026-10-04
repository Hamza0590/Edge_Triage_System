# Module 04: trained CNN family

## Usage

Use the existing `Edge_Project` Conda environment. No additional dependencies were
needed beyond the Module 02/03 PyTorch/NumPy/PyArrow stack.

```powershell
edge-triage train --config configs/training/family_v1.yaml
edge-triage pipeline smoke --config configs/experiments/smoke_tiny.yaml --limit 20
python scripts/verify_module04.py --index artifacts/models/family_20260924T204507_5a8771c620a0.json --output artifacts/models/module04_verification_new.json
```

Training creates new unique per-tier directories and prints a family index path;
it never replaces an earlier seed/configuration. Point `models.checkpoint_index` in
the seven smoke configs to a new index when intentionally changing checkpoints.
Current indices contain absolute local directories: relocation requires a new index,
not modifications to the immutable checkpoint files.

## Architecture and provenance

`models/cnn.py` implements a typed `TierSpec` and one `ScalableRealBogusCNN`.
Each block is bias-free 3x3 padded convolution, BatchNorm, ReLU, 2x2 max-pool and
Dropout2d. Three blocks have F/2F/4F channels; flattening feeds a hidden dense/ReLU/
dropout layer and one logit. Spatial/dense dropout are 0.20/0.50. Three YAML files
under `configs/models/` define version `scalable_cnn_v1`.

The block structure is adapted from [Bonnet-Guerrini et al., section 3.2](https://arxiv.org/html/2607.05393v1#S3.SS2).
Their two-channel input becomes three-channel NRD here. Widths, dropout choices,
supervised labels and checkpoint selection are project adaptations; injection,
co-teaching and dual-network training are excluded. [MeerCRAB](https://arxiv.org/pdf/2104.13950)
informs the NRD crop/flips, not an exact architecture/dataset reproduction. No upstream
code was copied. Paper availability does not establish upstream code licensing.

The model contains no sigmoid: [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BCEWithLogitsLoss.html)
handles that during loss calculation; inference applies sigmoid exactly once.
[AdamW](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html) is an explicit
implementation choice, not a claimed paper optimum.

## Training and checkpoints

- Frozen 1610-row training and 643-row validation partitions; fixed Module 02 crop,
  channel order and train-fitted normalization. Test tensors/predictions/metrics are
  never requested by training. Whole-file checksum reads and split/label metadata
  validation include the raw release for integrity only, not model selection.
- Seed 20260924; AdamW, LR 0.001, weight decay 0.0001, batch 64, maximum 30 epochs,
  patience 8. No class weighting, rotations, scheduler or test-dependent adjustment.
- Maximize validation **average precision** (non-interpolated PR area, grouped ties),
  then recall at 0.5 (equivalently lower FNR), then lower BCE. Earliest exact tie wins.
  This is not trapezoidal PR-AUC. Every completed epoch is recorded in history.csv.
- FP32 reference, TF32 off; seeded Python/NumPy/PyTorch/workers and epoch shuffling.
  Deterministic algorithms enabled. Reproducibility is scoped to the software/hardware
  environment, not guaranteed across platforms ([PyTorch guidance](https://docs.pytorch.org/docs/2.10/notes/randomness.html)).
- Optional `precision: amp` uses CUDA autocast/GradScaler; CPU AMP is rejected.
  Nonfinite logits/loss/gradients fail fast and emit an error rather than saving a
  silently corrupted checkpoint. Only FP32 reference checkpoints are reported here.
- Separate debug models fit 8 training examples (4/class), 100 updates, with dropout
  disabled and BatchNorm frozen. All reached 100% accuracy and reduced BCE below
  1e-14. Debug weights are never selected or saved as trained production checkpoints.

Each directory contains state_dict-only `model_state.pt`, model/training YAMLs,
split/preprocessing references, metrics, history and a checkpoint manifest. A family
index pins manifest SHA-256; each manifest pins every required file and the original
dataset MD5s, split/profile SHA-256, seed, architecture, count and best epoch. Git
commit is explicitly null because this directory is not a Git repository.

`ModelRegistry` verifies the manifest, exact file inventory and checksums before
`torch.load(weights_only=True)`, rejects incompatible data/split/preprocessing, checks
architecture/parameter count and finite state, then moves to the selected device and
sets eval mode. A deliberate incompatible-diagnostic opt-in is available at the API,
not enabled in the pipeline. Registry returns the typed architecture specification.
Run manifests link checkpoint manifests; structured events use `model.*`/`error.*`.

## Observed results (one seed, validation only)

| Tier | Parameters | Checkpoint bytes | Conv/Linear MACs | Best epoch | AP | Recall @0.5 | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 24,649 | 105,953 | 697,888 | 26 | 0.997140 | 0.951368 | 0.086152 |
| Medium | 97,681 | 398,369 | 2,402,624 | 27 | 0.997733 | 0.689970 | 0.407040 |
| Large | 388,897 | 1,564,129 | 8,832,640 | 21 | 0.997846 | 0.829787 | 0.229150 |

Tiny/Medium ran 30 epochs; Large stopped at 29 after 8 stale epochs. Measured training
durations (including setup and repeated best validation, excluding later diagnostics/
serialization) were 10.94/10.85/10.03 seconds on this RTX 5060 Ti. These are engineering
observations, not controlled cross-tier performance benchmarks. Host RSS was sampled
per batch; CUDA peak allocated/reserved values include training workspace and are not
inference memory profiles.

MACs are analytic counts for Conv/Linear only; multiply-add = 2 FLOPs. They exclude
BatchNorm, ReLU, pooling, bias and data movement. Exact parameter/MAC values are tested.
No third-party whole-model FLOP estimator was used or implied.

Warm synchronized batch-1 preliminary medians were approximately **0.50-0.53 ms**
at training completion and **0.52-0.56 ms** on verification. These wall times include
Python/kernel-launch/synchronization overhead, omit transfers/preprocessing, and were
not measured under controlled isolation. The historical report field named
`quiet_preliminary_latency` does not establish a quiet or isolated GPU. All values
are preliminary; no scheduling speedup is established.

**Gate B remains unresolved.** Large has about 15.8x Tiny's parameters and 12.7x its
Conv/Linear MACs, but AP differs by only 0.000707 and latency overlaps. Wider tiers are
not consistently better at threshold 0.5; the ranking-based selection metric can pick
poorly calibrated checkpoints. No thresholds were tuned to hide this. Module 05 must
evaluate calibration and equal-recall tradeoffs before justifying adaptive selection.
Only one seed is trained; final experiments should add seeds. Repeated-object leakage
cannot be excluded from the supplied metadata. No test-set performance claim is made.

Verification report: `artifacts/models/module04_verification_v1.json`. All three saved
checkpoints exactly reproduce recorded full validation metrics after loading. Seven
real-model portfolio smoke runs each completed 20 candidates, zero failures, with
verified event/record contracts and immutable data hashes. Dummy inference is now
test-only. Smoke triage still transmits every byte and uses an uncalibrated threshold;
it is not a scientific recall-constrained triage evaluation.
