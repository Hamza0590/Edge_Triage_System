# Frozen MeerLICHT data pipeline (Module 02)

Activate `Edge_Project`. Install `torch==2.10.0` using the official CUDA 12.8 wheel
index, then `python -m pip install -e ".[dev,data]"`. Run:

```powershell
edge-triage data audit --config configs/data/meerlicht.yaml
edge-triage data prepare --config configs/data/meerlicht.yaml
```

Both commands verify the raw MD5s and use read-only memory maps. Audit writes JSON
and Markdown with shape/header/dtype/order, CSV alignment, counts, quality and
metadata inventory. Chunk size is 32 images. Min/max/mean/std cover every pixel;
percentiles are approximate, computed on at most one million evenly spaced pixels
per channel. This is explicitly recorded and never used to tune preprocessing.

Prepare writes one immutable CSV split version per seed. NumPy PCG64 shuffles
indices separately for each class; largest remainders allocate 50/20/30 counts,
with partition order breaking ties. The manifest is sorted by original index.
IDs `meerlicht-000000` through `meerlicht-003218` remain stable across runs.
Same-seed reruns verify byte-identical content. Different content is never overwritten;
a different seed produces a separate version. Loaders require the saved manifest
and verify it; they never write or replace a split during an experiment.

The `nrd_center_crop_train_standardize_v1` artifact records crop size, explicit
channel names, raw hashes, split hash, training-index hash and float64-fitted mean/std.
Only unaugmented training crops fit normalization. Transforms output contiguous
float32 CHW tensors; training alone uses random horizontal/vertical flips. No clipping,
winsorization, rotation or translation is applied. Holdout pixels never fit statistics.
The required Module 02 baseline uses `configs/data/meerlicht.yaml`; Module 01's
`configs/default.yaml` remains an infrastructure demonstration with normalization none.

`make_loader(config, partition, run_id=..., epoch=...)` returns batches containing
Candidate metadata, image tensors, int64 labels and optional original byte counts.
Candidate.label stays null: offline labels are separate from scheduling metadata.
Offline arrival time is zero; the runtime assigns actual arrival timestamps later.
Explicit epoch/sample seeds make training augmentation reproducible across worker
counts. Workers reopen memory maps under Windows spawn; no image array is pickled.
No persistent workers are used; pass each epoch to make_loader. Workers seed NumPy
and Python from PyTorch's generator. Validation/test never shuffle or augment.

The contact sheet shows three Real and three Bogus training examples, all NRD planes,
using only a per-panel display stretch. Its title distinguishes display scaling from
model preprocessing. It is not a test-set selection or scientific performance result.

## Provenance and limits

| Source | Adoption and mechanism | Deviations / excluded scope | Code/license | Evidence |
|---|---|---|---|---|
| [Stoppa release](https://zenodo.org/records/14714279) | Reused frozen dataset | No T9/T10 assignment or object-level leakage claim | unknown_pending_verification | Zenodo indexed record verified NRD triplets and exact file MD5s on 2026-09-24; local shape/count checks |
| [Hosenie et al.](https://arxiv.org/pdf/2104.13950) | Adapted NRD crop and flips | New release, PyTorch and frozen PCG64 splits; train-fitted normalization is project-specific integration | No upstream code copied; upstream license unknown_pending_verification | Paper pp. 7-8: center crop, 50/20/30 division and horizontal/vertical flips verified 2026-09-24 |

Only index and label metadata exist. Repeated-object leakage cannot be checked;
absence of exact duplicate triplets does not establish object independence.
Synthetic workload repetitions are system load only, never additional statistical samples.
