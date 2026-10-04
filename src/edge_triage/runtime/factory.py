"""Configuration registry and explicitly non-scientific end-to-end smoke entry point."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from edge_triage.calibration.temperature import CalibratedModelAdapter, load_calibration
from edge_triage.config import AppConfig
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import (
    canonical,
    frozen_split,
    load_raw,
    split_path,
    write_immutable,
)
from edge_triage.data.preprocessing import load_profile, profile_identity, profile_path
from edge_triage.event_logging import EventLogger
from edge_triage.health import hash_file
from edge_triage.models.registry import (
    CheckpointManifest,
    ModelRegistry,
    RealModelAdapter,
    read_index,
)
from edge_triage.models.training import resolve_training_data
from edge_triage.runs import RunManager
from edge_triage.runtime.components import DatasetSource, FifoCandidateQueue, SequentialExecutor
from edge_triage.runtime.interfaces import (
    AdmissionPolicy,
    EscalationPolicy,
    FatePolicy,
    InferenceExecutor,
    ModelAdapter,
    ThresholdPolicy,
    TierSelectionPolicy,
    UncertaintyPolicy,
)
from edge_triage.runtime.pipeline import Pipeline, PipelineComponents
from edge_triage.runtime.records import ParquetRecordWriter
from edge_triage.runtime.reference import (
    SMOKE_PURPOSE,
    ClassificationOnlyFatePolicy,
    CommonThresholdPolicy,
    NoEscalationPolicy,
    NoUncertaintyPolicy,
    SequentialAdmissionPolicy,
    StaticTierPolicy,
)
from edge_triage.scheduling.policies import (
    AuditedAdmission,
    AuditedSelection,
    FixedConcurrencyAdmissionPolicy,
    MemoryContentionAwareAdmissionPolicy,
    PolicyContext,
    ResourceQueueAwareTierPolicy,
    RoundRobinTierPolicy,
)
from edge_triage.scheduling.profiles import scheduler_profiles
from edge_triage.uncertainty.artifacts import load_uncertainty
from edge_triage.uncertainty.policies import (
    AlwaysMCDropoutPolicy,
    AmbiguousOnlyMCDropoutPolicy,
    CascadeEscalationPolicy,
)

SELECTION: dict[str, Callable[[AppConfig], TierSelectionPolicy]] = {
    "fixed": lambda config: StaticTierPolicy(config.tier_selection.fixed_tier)
}
ADMISSION: dict[str, Callable[[AppConfig], AdmissionPolicy]] = {
    "sequential": lambda config: SequentialAdmissionPolicy()
}
EXECUTORS: dict[str, Callable[[AppConfig], InferenceExecutor]] = {
    "sequential": lambda config: SequentialExecutor()
}
MODELS: dict[str, Callable[[AppConfig], ModelAdapter]] = {}  # Test-only adapters may be injected.
UNCERTAINTY: dict[str, Callable[[AppConfig], UncertaintyPolicy]] = {
    "none": lambda config: NoUncertaintyPolicy()
}
ESCALATION: dict[str, Callable[[AppConfig], EscalationPolicy]] = {
    "none": lambda config: NoEscalationPolicy(),
    "cascade": lambda config: CascadeEscalationPolicy(
        config.models.tier_portfolio, config.escalation.max_escalations
    ),
}
THRESHOLDS: dict[str, Callable[[AppConfig], ThresholdPolicy]] = {
    "common": lambda config: CommonThresholdPolicy()
}
FATE: dict[str, Callable[[AppConfig], FatePolicy]] = {
    "classification_only": lambda config: ClassificationOnlyFatePolicy()
}
T = TypeVar("T")


def _build(registry: dict[str, Callable[[AppConfig], T]], name: str, config: AppConfig) -> T:
    if name not in registry:
        raise ValueError(f"policy/backend {name!r} not implemented in reference pipeline")
    return registry[name](config)


def assemble(config: AppConfig, logger: EventLogger | None = None) -> PipelineComponents:
    if config.experiment.purpose != SMOKE_PURPOSE:
        raise ValueError("reference pipeline requires purpose NON_SCIENTIFIC_SMOKE_TEST")
    models: dict[ModelTier, ModelAdapter] = {}
    if config.pipeline.model_adapter == "cnn":
        if config.models.checkpoint_index is None:
            raise ValueError("CNN adapter requires checkpoint_index")
        resolved, hashes = resolve_training_data(config)
        assert resolved.split.manifest is not None and resolved.preprocessing.artifact is not None
        registry = ModelRegistry(
            split_sha256=hash_file(resolved.split.manifest),
            preprocessing_sha256=hash_file(resolved.preprocessing.artifact),
            dataset_hashes=hashes,
            device=config.training.device,
            logger=logger,
        )
        index = read_index(config.models.checkpoint_index)
        for tier in config.models.tier_portfolio:
            entry = index["checkpoints"][tier.value]
            registry.register(
                tier, entry["checkpoint_id"], Path(entry["directory"]), entry["manifest_sha256"]
            )
            model, spec = registry.load(tier, entry["checkpoint_id"])
            if spec.crop_size != config.preprocessing.crop_size or spec.input_channels != len(
                config.preprocessing.selected_channels
            ):
                raise ValueError("incompatible inference input shape")
            models[tier] = RealModelAdapter(model, entry["checkpoint_id"])
            if config.calibration.artifact_index is not None:
                import json

                calibration_index = json.loads(config.calibration.artifact_index.read_text("utf-8"))
                if calibration_index.get("schema_version") != 1:
                    raise ValueError("invalid calibration index")
                calibration_entry = calibration_index["calibrations"][tier.value]
                manifest = CheckpointManifest.model_validate_json(
                    (Path(entry["directory"]) / "checkpoint_manifest.json").read_bytes()
                )
                artifact = load_calibration(
                    Path(calibration_entry["path"]),
                    calibration_entry["sha256"],
                    manifest,
                    entry["manifest_sha256"],
                )
                models[tier] = CalibratedModelAdapter(
                    RealModelAdapter(model, entry["checkpoint_id"]),
                    artifact,
                    config.calibration.probability_mode,
                )
    else:
        models = {
            tier: _build(MODELS, config.pipeline.model_adapter, config)
            for tier in config.models.tier_portfolio
        }
    adaptive = config.tier_selection.policy in {"resource_queue_aware", "round_robin"}
    profiled = (
        adaptive
        or config.admission.policy in {"fixed", "memory_contention_aware"}
        or config.pipeline.mode == "adaptive_sequential"
    )
    context = None
    if profiled:
        context = PolicyContext(
            config.models.tier_portfolio,
            scheduler_profiles(config),
            config.scheduler.constraints,
            device=config.scheduler.device,
            logger=logger,
        )
    selector: TierSelectionPolicy
    if adaptive:
        assert context is not None
        selector = (
            ResourceQueueAwareTierPolicy(context)
            if config.tier_selection.policy == "resource_queue_aware"
            else RoundRobinTierPolicy(context)
        )
    else:
        selector = _build(SELECTION, config.tier_selection.policy, config)
        if context is not None:
            selector = AuditedSelection(selector, context)
    admission: AdmissionPolicy
    if config.admission.policy in {"fixed", "memory_contention_aware"}:
        assert context is not None
        admission = (
            MemoryContentionAwareAdmissionPolicy(context)
            if config.admission.policy == "memory_contention_aware"
            else FixedConcurrencyAdmissionPolicy(context, config.admission.fixed_concurrency)
        )
    else:
        admission = _build(ADMISSION, config.admission.policy, config)
        if context is not None:
            admission = AuditedAdmission(admission, context)
    uncertainty: UncertaintyPolicy
    if config.uncertainty.policy != "none":
        if (
            config.uncertainty.artifact is None
            or config.uncertainty.artifact_sha256 is None
            or config.calibration.artifact_index is None
            or config.models.checkpoint_index is None
        ):
            raise ValueError("MC policies require a frozen UQ artifact and calibrated family")
        uq_artifact = load_uncertainty(
            config.uncertainty.artifact, config.uncertainty.artifact_sha256
        )
        resolved, _ = resolve_training_data(config)
        assert resolved.split.manifest and resolved.preprocessing.artifact
        if (
            uq_artifact.passes != config.uncertainty.passes
            or uq_artifact.family_sha256 != hash_file(config.models.checkpoint_index)
            or uq_artifact.split_sha256 != hash_file(resolved.split.manifest)
            or uq_artifact.preprocessing_sha256 != hash_file(resolved.preprocessing.artifact)
            or uq_artifact.calibration_index_sha256 != hash_file(config.calibration.artifact_index)
        ):
            raise ValueError("incompatible frozen UQ configuration")
        for uq_spec in uq_artifact.tiers:
            entry = calibration_index["calibrations"][uq_spec.tier.value]
            if uq_spec.calibration_sha256 != entry["sha256"]:
                raise ValueError("incompatible UQ calibration hash")
        policy = (
            AlwaysMCDropoutPolicy
            if config.uncertainty.policy == "always_mc_dropout"
            else AmbiguousOnlyMCDropoutPolicy
        )
        uncertainty = policy(models, uq_artifact, config.uncertainty.seed)
    else:
        uncertainty = _build(UNCERTAINTY, config.uncertainty.policy, config)
    return PipelineComponents(
        queue=FifoCandidateQueue(),
        selector=selector,
        admission=admission,
        executor=(
            SequentialExecutor()
            if config.pipeline.mode == "concurrent"
            else _build(EXECUTORS, config.admission.execution_mode, config)
        ),
        models=models,
        uncertainty=uncertainty,
        escalation=_build(ESCALATION, config.escalation.policy, config),
        thresholds=_build(THRESHOLDS, config.threshold.policy, config),
        fate=_build(FATE, config.fate.policy, config),
    )


def run_smoke(config: AppConfig, limit: int = 20) -> Path:
    if limit < 1:
        raise ValueError("limit must be positive")
    _, labels, hashes = load_raw(config)
    _, split_hash = frozen_split(config, labels)
    identity = profile_identity(config, split_hash, hashes)
    load_profile(config, identity)
    resolved = config.model_dump()
    resolved["split"]["manifest"] = split_path(config)
    resolved["preprocessing"]["artifact"] = profile_path(config, identity)
    config = AppConfig.model_validate(resolved)
    with RunManager(config) as run:
        components = assemble(config, run.logger)
        if config.uncertainty.artifact is not None:
            run.register_threshold(config.uncertainty.artifact)
        if config.pipeline.model_adapter == "cnn":
            assert config.models.checkpoint_index is not None
            index = read_index(config.models.checkpoint_index)
            for tier in config.models.tier_portfolio:
                entry = index["checkpoints"][tier.value]
                run.register_checkpoint(
                    Path(entry["directory"]) / "checkpoint_manifest.json", entry["checkpoint_id"]
                )
        if not isinstance(components.thresholds, CommonThresholdPolicy):
            raise ValueError("smoke threshold persistence requires CommonThresholdPolicy")
        threshold_path = run.directory / "artifacts" / "common_threshold.txt"
        write_immutable(threshold_path, components.thresholds.definition())
        run.register_threshold(threshold_path)
        source = DatasetSource(config, run.manifest.run_id, limit)
        writer = ParquetRecordWriter(
            run.directory / "candidates.parquet", config.pipeline.record_batch_size
        )
        completed, failed = Pipeline(config, components, run.logger, writer).run(source)
        write_immutable(
            run.directory / "metrics" / "smoke_summary.json",
            canonical(
                {
                    "purpose": SMOKE_PURPOSE,
                    "completed": completed,
                    "failed": failed,
                    "candidates": writer.count,
                    "scientific_metrics": False,
                }
            ),
        )
    return run.directory
