"""Strict curated matrices with guarded held-out promotion and frozen inventories."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from edge_triage.config import AppConfig, UniqueKeyLoader, load_config
from edge_triage.contracts import SHA256, Contract, ModelTier
from edge_triage.data.artifacts import canonical
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import hash_file
from edge_triage.models.registry import read_index
from edge_triage.triage.artifacts import load_fate_artifact


class FileReference(Contract):
    path: Path
    sha256: SHA256


class ExperimentCase(Contract):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_+.-]{0,99}$")
    description: str = Field(min_length=1)
    config: FileReference | None = None
    status: Literal["ready", "skipped"] = "ready"
    reason: str | None = None
    comparison_group: str | None = None
    component: Literal["microbatch", "batch1_threads"] | None = None
    component_level: Literal[1, 2, 4, 8] = 1

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.status == "ready" and (self.config is None or self.reason is not None):
            raise ValueError("ready case needs a pinned config and no skip reason")
        if self.status == "skipped" and not self.reason:
            raise ValueError("skipped case needs an explicit reason")
        return self


class ComparisonGroup(Contract):
    id: str
    changed_axes: tuple[str, ...]
    case_ids: tuple[str, ...] = ()


class ExperimentMatrix(Contract):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    project_root: Path
    output_dir: Path
    partition: Literal["validation", "test"] = "validation"
    purpose: Literal["systems_smoke", "unique_validation", "unique_test"] = "systems_smoke"
    design: Literal["curated_not_full_factorial"] = "curated_not_full_factorial"
    repetitions: int = Field(default=3, ge=2, le=100)
    warmup_forwards_per_tier: int = Field(default=5, ge=1, le=1000)
    bootstrap_repeats: int = Field(default=1000, ge=10, le=10000)
    bootstrap_seed: int = Field(default=20260924, ge=0)
    target_survival_recall: float = Field(default=0.95, gt=0, le=1)
    cases: tuple[ExperimentCase, ...]
    comparisons: tuple[ComparisonGroup, ...] = ()

    @model_validator(mode="after")
    def identifiers(self) -> Self:
        if (self.partition == "test") != (self.purpose == "unique_test"):
            raise ValueError("test partition requires explicit unique_test purpose")
        if not self.cases or len({c.id.casefold() for c in self.cases}) != len(self.cases):
            raise ValueError("matrix requires unique case IDs (case insensitive)")
        groups = {g.id for g in self.comparisons}
        if len(groups) != len(self.comparisons):
            raise ValueError("duplicate comparison group")
        if any(c.comparison_group and c.comparison_group not in groups for c in self.cases):
            raise ValueError("unknown comparison group")
        if any(set(g.case_ids) - {c.id for c in self.cases} for g in self.comparisons):
            raise ValueError("comparison references unknown cases")
        return self


def load_matrix(path: Path) -> ExperimentMatrix:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    matrix = ExperimentMatrix.model_validate(value)
    raw = matrix.model_dump()
    root = (path.resolve().parent / matrix.project_root).resolve()
    raw.update(project_root=root, output_dir=(root / matrix.output_dir).resolve())
    for case in raw["cases"]:
        if case["config"]:
            case["config"]["path"] = (root / case["config"]["path"]).resolve()
    return ExperimentMatrix.model_validate(raw)


def axes(config: AppConfig) -> dict[str, Any]:
    """Describe effective runtime choices, including inherited coupled admission.

    The concurrent executor currently chooses admission from execution_mode. Do
    not advertise the unused legacy admission.policy field as an independent axis.
    """
    dynamic = config.admission.execution_mode == "dynamic_concurrency"
    return {
        "portfolio": [t.value for t in config.models.tier_portfolio],
        "tier_selection_policy": config.tier_selection.policy,
        "execution_mode": config.admission.execution_mode,
        "admission_policy": config.admission.concurrent_policy
        if config.admission.concurrent_policy != "auto"
        else "memory_contention_aware"
        if dynamic
        else "fixed_memory",
        "uncertainty_policy": config.uncertainty.policy,
        "escalation_policy": config.escalation.policy,
        "threshold_policy": config.threshold.policy,
        "queue_awareness": config.tier_selection.policy == "resource_queue_aware",
        "resource_awareness": True,  # all three executors enforce memory admission
        "fate_policy": config.fate.policy,
        "arrival_trace": hash_file(config.workload.trace) if config.workload.trace else None,
        "environment_profile": {
            "hardware_id": config.scheduler.hardware_id,
            "device": config.training.device,
            "cpu_threads": config.training.cpu_threads,
            "monitor": config.monitor.model_dump(mode="json"),
            "queue": config.queue.model_dump(mode="json"),
            "constraints": config.scheduler.constraints.model_dump(mode="json"),
        },
        "model_seed": config.training.seed,
    }


def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


# Explicit leaf ownership makes a purported single-axis ablation auditable.
AXIS_PREFIXES = {
    "portfolio": ("models.tier_portfolio", "tier_selection.fixed_tier"),
    "tier_selection_policy": ("tier_selection.policy", "tier_selection.fixed_tier"),
    "execution_mode": ("admission.execution_mode",),
    "admission_policy": ("admission.concurrent_policy",),
    "queue_awareness": ("tier_selection.policy",),
    "arrival_trace": (
        "workload.trace",
        "workload.count",
        "workload.arrival_rate_per_s",
        "workload.profile",
    ),
    "uncertainty_policy": ("uncertainty.",),
    "escalation_policy": ("escalation.",),
    "threshold_policy": ("threshold.policy",),
    "fate_policy": ("fate.policy",),
    "concurrency": ("admission.fixed_concurrency", "admission.max_concurrency"),
    "model_seed": ("training.seed", "models.checkpoint_index", "calibration.artifact_index"),
    "probability_mode": ("calibration.probability_mode",),
}


def changed_fields(left: AppConfig, right: AppConfig) -> set[str]:
    ignored = {
        "experiment.name",
        "paths.runs_dir",
        "paths.artifacts_dir",
        "paths.logs_dir",
        "threshold.artifact",
        "threshold.artifact_sha256",
        "pipeline.promotion",
        "pipeline.promotion_sha256",
    }
    a, b = flatten(left.model_dump(mode="json")), flatten(right.model_dump(mode="json"))
    return {key for key in a.keys() | b.keys() if key not in ignored and a.get(key) != b.get(key)}


def enforce_comparison(configs: list[AppConfig], changed_axes: tuple[str, ...]) -> None:
    if not configs:
        raise ValueError("empty comparison group")
    if any(axis not in AXIS_PREFIXES for axis in changed_axes):
        raise ValueError("unsupported independent comparison axis")
    prefixes = tuple(p for axis in changed_axes for p in AXIS_PREFIXES[axis])
    for config in configs[1:]:
        unexpected = {k for k in changed_fields(configs[0], config) if not k.startswith(prefixes)}
        if unexpected:
            raise ValueError(f"comparison changes undeclared fields: {sorted(unexpected)}")
        if (
            axes(configs[0])["admission_policy"] != axes(config)["admission_policy"]
            and "admission_policy" not in changed_axes
        ):
            raise ValueError("comparison changes effective admission policy without declaring it")


def input_inventory(config: AppConfig) -> dict[str, str]:
    """Transitive file pins, checked again before reusing any completed subrun."""
    files: dict[str, str] = {}

    def pin(path: Path | None, expected: str | None = None) -> None:
        if path is None:
            return
        digest = hash_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"input checksum mismatch: {path}")
        files[str(path.resolve())] = digest

    for path in (
        config.dataset.images,
        config.dataset.labels,
        config.split.manifest,
        config.preprocessing.artifact,
        config.models.checkpoint_index,
        config.calibration.artifact_index,
        config.workload.trace,
        config.paths.provenance_file,
    ):
        pin(path)
    pin(config.scheduler.profiles, config.scheduler.profiles_sha256)
    pin(config.scheduler.contention_profile, config.scheduler.contention_sha256)
    pin(config.uncertainty.artifact, config.uncertainty.artifact_sha256)
    pin(config.threshold.artifact, config.threshold.artifact_sha256)
    pin(config.pipeline.promotion, config.pipeline.promotion_sha256)
    if config.pipeline.promotion:
        for name, digest in json.loads(config.pipeline.promotion.read_bytes())["inputs"].items():
            pin(Path(name), digest)
    if config.split.manifest:
        pin(config.split.manifest.with_suffix(".json"))
    if config.preprocessing.artifact:
        pin(config.preprocessing.artifact.with_suffix(".sha256"))
    if config.workload.trace:
        pin(config.workload.trace.with_suffix(".sha256"))
    if config.models.checkpoint_index:
        for tier, entry in read_index(config.models.checkpoint_index)["checkpoints"].items():
            directory = Path(entry["directory"])
            manifest = directory / "checkpoint_manifest.json"
            pin(manifest, entry["manifest_sha256"])
            value = json.loads(manifest.read_bytes())
            if (
                ModelTier(tier) in config.models.tier_portfolio
                and value["seed"] != config.training.seed
            ):
                raise ValueError("declared model seed differs from checkpoint seed")
            for name, digest in value["file_hashes"].items():
                pin(directory / name, digest)
    if config.scheduler.profiles:
        for profile in json.loads(config.scheduler.profiles.read_bytes())["profiles"]:
            pin(Path(profile["raw_samples_path"]), profile["raw_samples_sha256"])
    if config.calibration.artifact_index:
        index = json.loads(config.calibration.artifact_index.read_bytes())
        for entry in index["calibrations"].values():
            pin(Path(entry["path"]), entry["sha256"])
    return files


def validate_matrix(matrix: ExperimentMatrix) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    configs: dict[str, AppConfig] = {}
    inputs: dict[str, str] = {}
    for case in matrix.cases:
        if case.status == "skipped":
            cases.append({"id": case.id, "status": "skipped", "reason": case.reason})
            continue
        assert case.config is not None
        if hash_file(case.config.path) != case.config.sha256:
            raise ValueError(f"config checksum mismatch: {case.id}")
        config = load_config(case.config.path)
        if config.paths.project_root != matrix.project_root:
            raise ValueError("matrix and pipeline project roots differ")
        if config.pipeline.partition != matrix.partition:
            raise ValueError("matrix and pipeline partitions differ")
        if matrix.partition == "test":
            from edge_triage.experiments.promotion import verify_promotion

            verify_promotion(config)
        if config.pipeline.mode != "concurrent" or not config.monitor.enabled:
            raise ValueError("matrix requires common concurrent pipeline with monitoring")
        if config.pipeline.model_adapter != "cnn" or config.queue.overflow != "block":
            raise ValueError("matrix requires real models and lossless blocking queue")
        if config.fate.target_survival_recall != matrix.target_survival_recall:
            raise ValueError("unequal target survival recall")
        if config.tier_selection.fixed_tier not in config.models.tier_portfolio:
            raise ValueError("fixed initial tier outside portfolio")
        if config.workload.trace is None or config.split.manifest is None:
            raise ValueError("matrix requires explicit frozen trace and split")
        trace = load_trace(
            config.workload.trace, hash_file(config.split.manifest), partition=matrix.partition
        )
        with config.split.manifest.open(encoding="utf-8", newline="") as stream:
            expected = {
                r["candidate_id"]: r
                for r in csv.DictReader(stream)
                if r["split"] == matrix.partition
            }
        for row in trace:
            if (
                row.source_candidate_id not in expected
                or int(expected[row.source_candidate_id]["index_no"]) != row.dataset_index
            ):
                raise ValueError("trace source outside frozen validation split")
        if matrix.purpose in {"unique_validation", "unique_test"} and (
            len(trace) != len(expected) or {r.source_candidate_id for r in trace} != set(expected)
        ):
            raise ValueError("unique validation requires every validation source exactly once")
        if config.fate.policy in {"binary", "common_three_way", "recall_constrained"}:
            if config.threshold.artifact is None or config.threshold.artifact_sha256 is None:
                raise ValueError("fate experiment needs a compatible frozen threshold")
            load_fate_artifact(config.threshold.artifact, config.threshold.artifact_sha256, config)
        inventory = input_inventory(config)
        inputs.update(inventory)
        configs[case.id] = config
        cases.append(
            {
                "id": case.id,
                "status": "ready",
                "config_hash": config.config_hash,
                "axes": axes(config),
                "occurrences": len(trace),
                "unique_sources": len({r.source_candidate_id for r in trace}),
                "component": case.component,
                "component_level": case.component_level,
            }
        )
    for group in matrix.comparisons:
        enforce_comparison(
            [
                configs[c.id]
                for c in matrix.cases
                if (c.comparison_group == group.id or c.id in group.case_ids) and c.id in configs
            ],
            group.changed_axes,
        )
    expanded = [
        {
            "id": f"{case.id}__r{repeat:03d}",
            "case": case.id,
            "run_repetition": repeat,
            "status": case.status,
            "reason": case.reason,
            "component": case.component,
            "component_level": case.component_level,
        }
        for case in matrix.cases
        for repeat in range(matrix.repetitions)
    ]
    definition = matrix.model_dump(mode="json")
    return {
        "schema_version": 1,
        "matrix": definition,
        "matrix_sha256": hashlib.sha256(canonical(definition)).hexdigest(),
        "cases": cases,
        "subruns": expanded,
        "inputs": inputs,
        "counts": {
            "cases": len(cases),
            "subruns": len(expanded),
            "ready": sum(r["status"] == "ready" for r in expanded),
            "skipped": sum(r["status"] == "skipped" for r in expanded),
        },
        "test_evaluated": False,
    }
