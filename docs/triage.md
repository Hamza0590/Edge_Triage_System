# Module 10: recall-constrained fate

Module 10 is **COMPLETE**. All 144 operating points are fitted and frozen on complete
configuration outputs, all 36 full runtime replays meet the validation target, and
the final regression passes378 tests. Ruff, mypy59 source files and dependency checks
are clean. Smaller integration tests use synthetic, test-owned operating points;
the scientific validation evidence comes from the separately attested full study.

**2026-10-08 resume:** The user's completion request supersedes the session stop.
The validation collection/fitting/replay workflow and CLI have targeted regression
coverage. The predeclared study has 36 upstream configurations and 144 fate variants.
Six saved collections were verified and preserved; all 36 collections are now complete.
The three-seed diagnostic and all frozen runtime replays are complete and verified.
Historical evidence remains in `artifacts/triage/module10_v2/session_stop.json`.

## Runtime behavior

`triage/policies.py` supplies `TransmitAllPolicy`, `BinaryDiscardTransmitPolicy`,
`CommonThresholdThreeWayPolicy` and `PerTierRecallConstrainedPolicy`. Factory names
are `transmit_all`, `binary`, `common_three_way` and `recall_constrained`.
The legacy `classification_only` path remains available for historical smoke runs.

`ContextFatePolicy` receives a label-free `FateContext`: final prediction, final UQ,
terminal conservative flag and failure status. The common pipeline supplies it for
both legacy and asynchronous executors. Failed Module10 candidates remain failed
in execution metrics but explicitly transmit their entire logical payload. A
failure decision has a null score when prediction evidence is unavailable.

All thresholds name their direction. Discard requires an inclusive
`p_real <= max_p_real_for_discard`, complete compatible evidence, resolved UQ,
and `standard_deviation <= max_uncertainty_for_discard`. The discard bound is
restricted below .5; `None` disables discard even at exactly-zero probability.
`p_real >= min_p_real_for_transmit` transmits. The middle region compresses in
three-way policies and transmits in the binary baseline.

Missing prediction, threshold, UQ or required calibration; mismatched checkpoint,
score mode or candidate identity; high dispersion; ambiguous/unresolved UQ; and
terminal conservative fallback all transmit. Explicit `method=none` evidence
permits the noUQ ablation: its zero dispersion denotes disabled sampling and is
not an epistemic certainty claim. The old prediction-only fate call always
preserves the candidate because it lacks UQ evidence.

The final tier must also match the resolved threshold. The fitter disables discard
authority for enabled tiers with no compatible, resolved validation evidence;
these tiers preserve full payloads even if routing reaches them later.

Raw probabilities are allowed only through an explicitly bound uncalibrated mode,
preserving Module05's recommended fallback for Medium/Large. Calibrated mode
requires the exact calibration artifact ID. Module08's negative Gate C remains
unchanged: noUQ/no escalation is still the proposed default.

## Logical payload and costs

`logical_raw_tensor_v1` counts the original 100x100x3 float32 payload: 120,000
bytes per dataset candidate, irrespective of the model's 30x30 crop. Configured
metadata is added to both original and surviving payloads. Defaults are zero
metadata bytes, .25 compression fraction and zero protocol overhead.

| Fate | Logical bytes transmitted |
|---|---|
| Discard | 0 |
| Compress | ceil(raw tensor bytes * compression fraction) + metadata |
| Transmit | raw tensor bytes + metadata |

These defaults are accounting assumptions, not measured compression or scientific
fidelity. A codec and network are not executed. `compression_cpu_ms=0` is explicitly
logical accounting. `fate_ms` records policy evaluation/accounting wall time;
runtime E2E includes that work and existing waits/instrumentation. Do not add E2E
and overlapping component times or label E2E as CPU compute. Resolved run config
records the fraction/metadata and `compression: logical_bytes`; the copied frozen
threshold definition is registered in the run manifest.

## Frozen artifacts and fitting

`triage/artifacts.py` supplies scientific common/per-tier threshold policies,
separate from the legacy smoke classifier's common .5 threshold. Artifacts use
canonical JSON and immutable writes, bind family/split/preprocessing/calibration
and validation-record hashes, and bind system behavior via `system_identity`.
Output directories, experiment name, partition and the artifact's own path/hash
are excluded from system identity; tier/routing/UQ/execution/workload/accounting
choices are included. A new behavior or trace therefore requires new validation.
Trace file contents are hashed as well as its configured path.
Changing validation to test permits loading a frozen artifact, never refitting it.

`fit_thresholds` requires all validation candidate IDs exactly once, correct labels
and the expected frozen split hash. It rejects non-validation requests before
iterating supplied predictions. Inputs must be the final records of the complete
configuration. `triage/workflow.py` verifies completed manifests, full candidate
membership, trace/config identity, final events, resource references and drain/no-loss
accounting. Collection attestations bind all evidence file hashes and recomputed
metrics/costs; missing or altered entries are rejected before fitting.

Search considers observed final scores below .5 plus disabled discard, and an
explicit transmit/uncertainty grid. The API defaults fix transmit at .5 and UQ at
0; these defaults are not a claim of optimal production search ranges. Dynamic
programming minimizes bytes under the global integer false-discard budget. Ties
prefer fewer lost Real candidates, then lower summed UQ cutoffs, then deterministic
lexicographic bounds. Compute is invariant across fate settings on a fixed run.
Optional monotonicity requires weaker-tier discard bounds no greater than stronger
ones. Common thresholds use exactly the same operating point across tiers.

`evaluate_fates` separately reports survival recall, .5 classifier recall, Real
transmit/compress recall, bogus discard rate, false-discard count/rate and review
IDs, byte reduction, label/tier fate counts and observed E2E cost. Missing class
denominators produce null metrics. `bootstrap_fates` supplies seeded IID percentile
intervals with fixed fitted thresholds; these are conditional validation diagnostics,
not a generalization guarantee or model-seed stability estimate.

## Reproducible workflow

Use the `Edge_Project` environment. The declared protocol is
`configs/experiments/module10_protocol_v1.json`; evidence is stored under
`artifacts/triage/module10_v2/`. Run stages in this order, with no simultaneous timed
serving and training/testing jobs:

```powershell
python scripts/benchmark_module10.py --stage init
python scripts/benchmark_module10.py --stage collect
python scripts/benchmark_module10.py --stage fit
python scripts/benchmark_module10.py --stage replay
python scripts/module10_seeds.py
python scripts/analyze_module10.py
```

The benchmark also accepts `--only <setting substring>` for a bounded resume.
Existing complete collections/replays are verified and skipped. The runtime-source
guard rejects implementation changes that would mix timed runtime versions. Artifact
writes are immutable: identical outputs are idempotent; differing content is rejected.
The interrupted seventh startup run and initial v1 console-noisy pilot are preserved
as historical evidence and excluded from the v2 results.

The CLI exposes the same library boundaries:

```text
edge-triage triage collect --config <transmit-all-config> --output <collection-directory>
edge-triage triage fit --config <target-config> --collection <collection.json> --protocol <protocol.json> --output <fit-directory>
edge-triage triage replay --config <frozen-config.yaml> --output <runtime_replay.json>
```

The 36 settings comprise 21 core configurations (seven portfolios x three execution
modes, diagnostic round-robin selection), one proposed adaptive all-tier configuration,
and 14 paired forced-calibrated noUQ/MC5 configurations (fixed-first sequential
selection). Each setting processes all 643 validation candidates once: 329 Real and
314 Bogus. The existing frozen split is not regenerated. Recommended probability
modes are used in the core/proposed studies; forced calibration in the UQ pairs makes
the score mode comparable across those two conditions.

Four fate policies are fitted on each setting's complete final evidence. Full runtime
replays use the frozen per-tier policy for all 36 settings. The other three fate
ablations apply to the same attested routing outputs, so their upstream compute is
held fixed. This is a fate-policy comparison, not a rerouting performance estimate.
Fitting uses inclusive observed discard scores below .5, transmit cutoffs .5/.75/.9,
and standard-deviation cutoffs 0/.1/.25/.5. The global recall target is .95 with
monotonic tier discard cutoffs. No thresholds are adjusted after replay misses.

The seed diagnostic trains two additional families with seeds 20260925 and 20260926
in separate directories and compares them with original seed 20260924. Architectures,
training protocol, split and preprocessing are fixed. Each singleton is calibrated
on validation, then fitted with noUQ and transmit cutoff .5 to isolate seed sensitivity.
These diagnostic families never replace the original family or select a new default.

## Acceptance evidence

| Requirement | Verified evidence |
|---|---|
| Compatible frozen point per declared configuration | 144 artifacts across 36 complete upstream configurations; immutable configs, hashes and fit reports |
| Primary meets target before held-out evaluation | Fit and full frozen replay retain313/329 Real (95.1368%); all36 replays feasible; no held-out evaluation |
| Separate survival/classifier metrics | Fit/replay CSVs and JSON metrics include both, plus Real transmit/compress, bogus discard, false-discard IDs and logical bytes |
| Explicit logical accounting |120000 raw /30000 compressed bytes, metadata/protocol0, no codec/network, logical codec CPU0 |
| Common vs per-tier tested |36 paired fits; gain only30000–60000 bytes in six diagnostic settings; proposed configuration ties |
| Conservative failure and ambiguity handling | Direction, identity/calibration/UQ, missing-data, failure, terminal ambiguity, unsupported-tier and immutable-artifact tests; all measured candidates accounted |
| Required analyses | Recall/bytes/cost curves; label/tier fate table; false-discard IDs; UQ pairs;1000-resample conditional bootstrap;9 singleton diagnostics across3 training seeds |

See [verified results and evidence links](module10_results.md). The ready-to-replay
configuration is `configs/experiments/module10_proposed_v2.yaml`; its threshold
SHA-256 is `a45cbaf0bba9ff8645ea699226a90b6b9132e2cd41b17776c538d78567926321`.
Revalidating the saved proposed replay is idempotent:

```powershell
python -m edge_triage triage replay --config configs/experiments/module10_proposed_v2.yaml --output artifacts/triage/module10_v2/settings/proposed_all_dynamic/runtime_replay.json
```

No held-out scientific evaluation was performed. Module11 remains outside the
current scope. Existing dynamic cap1 admission slowdown is preserved. These artifacts
bind the exact validated configuration/trace; changed routing, workload or models
require compatible validation. Conditional fit/bootstrap results do not guarantee
held-out recall or the scientific fidelity of a future compression codec.

## Provenance

| Source | Relationship | Limits |
|---|---|---|
| [El-Yaniv and Wiener, JMLR 2010](https://www.jmlr.org/papers/v11/el-yaniv10a.html) | Adapted reject-option/risk-coverage framing | This project's noisy finite-data survival/byte objective has no noise-free theoretical guarantee |
| [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | Consumes adapted Module05 temperature calibration | Raw fallback and fate thresholds are project-specific |
| Bonnet-Guerrini et al., arXiv2607.05393 | Inherited Module08 uncertainty evidence inspires preservation of ambiguity | No claim that the paper supplies this three-way rule; negative Gate C retained |
| MeerCRAB | Inherited astronomical Real/Bogus task framing | No reproduction of a three-way fate mechanism |

The first two primary source abstracts were checked on 2026-10-08. Detailed
calibration/UQ/task evidence remains in the corresponding earlier-module docs.
All fate code is independently implemented; upstream code/license remains unverified.
