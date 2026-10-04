"""Strict YAML configuration. All paths resolve from paths.project_root.

The project root itself resolves relative to the YAML file's parent directory.
Resolved configuration can be loaded again without changing its hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from edge_triage.contracts import (
    Contract,
    Count,
    Identifier,
    ModelTier,
    NonNegative,
    PositiveCount,
    Probability,
)
from edge_triage.scheduling.settings import SchedulerConfig

IMAGES_MD5 = "bbf55c1cecbdea127497d6cd346e3455"
LABELS_MD5 = "cacf76a6aadd97ed18ab24b58e121bc2"
DEFAULT_SEED = 20260924


class PathsConfig(Contract):
    project_root: Path = Path("..")
    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("artifacts")
    logs_dir: Path = Path("logs")
    provenance_file: Path = Path("references/provenance.yaml")


class DatasetConfig(Contract):
    channel_order: tuple[Literal["new"], Literal["reference"], Literal["difference"]] = (
        "new",
        "reference",
        "difference",
    )
    images: Path = Path("MeerLICHT_images.npy")
    labels: Path = Path("MeerLICHT_labels.csv")
    images_md5: Literal["bbf55c1cecbdea127497d6cd346e3455"] = "bbf55c1cecbdea127497d6cd346e3455"
    labels_md5: Literal["cacf76a6aadd97ed18ab24b58e121bc2"] = "cacf76a6aadd97ed18ab24b58e121bc2"


class SplitConfig(Contract):
    train_fraction: Probability = 0.5
    validation_fraction: Probability = 0.2
    test_fraction: Probability = 0.3
    seed: Count = DEFAULT_SEED
    stratified: Literal[True] = True
    manifest: Path | None = None

    @model_validator(mode="after")
    def check_fractions(self) -> Self:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1):
            raise ValueError("positive split fractions must sum to one")
        return self


class PreprocessingConfig(Contract):
    profile: Literal["nrd_center_crop"] = "nrd_center_crop"
    crop_size: int = Field(default=30, ge=1, le=100, strict=True)
    channels: Literal["NRD"] = "NRD"
    normalization: Literal["none", "train_standardize"] = "none"
    horizontal_flip: bool = True
    vertical_flip: bool = True
    selected_channels: tuple[Literal["new", "reference", "difference"], ...] = (
        "new",
        "reference",
        "difference",
    )
    artifact: Path | None = None

    @model_validator(mode="after")
    def check_channels(self) -> Self:
        if not self.selected_channels or len(set(self.selected_channels)) != len(
            self.selected_channels
        ):
            raise ValueError("selected channels must be nonempty and unique")
        return self


class LoaderConfig(Contract):
    batch_size: PositiveCount = 64
    workers: Count = 0
    pin_memory: bool = False
    prefetch_factor: PositiveCount = 2
    include_original_bytes: bool = True


class ModelsConfig(Contract):
    tier_portfolio: tuple[ModelTier, ...] = tuple(ModelTier)
    checkpoint_index: Path | None = None

    @model_validator(mode="after")
    def check_portfolio(self) -> Self:
        if not self.tier_portfolio or len(set(self.tier_portfolio)) != len(self.tier_portfolio):
            raise ValueError("portfolio must contain distinct supported tiers")
        if tuple(tier for tier in ModelTier if tier in self.tier_portfolio) != self.tier_portfolio:
            raise ValueError("portfolio order must be tiny, medium, large (omissions allowed)")
        return self


class TrainingConfig(Contract):
    seed: Count = DEFAULT_SEED
    epochs: PositiveCount = 30
    batch_size: PositiveCount = 64
    learning_rate: float = Field(default=0.001, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    optimizer: Literal["adam", "adamw"] = "adamw"
    weight_decay: float = Field(default=0.0001, ge=0)
    patience: PositiveCount = 8
    selection_metric: Literal["average_precision"] = "average_precision"
    precision: Literal["fp32", "amp"] = "fp32"
    deterministic: bool = True
    cpu_threads: PositiveCount = 4
    debug_overfit: bool = False


class CalibrationConfig(Contract):
    method: Literal["none", "temperature_scaling"] = "temperature_scaling"
    fit_partition: Literal["validation"] = "validation"
    seed: Count = DEFAULT_SEED
    bins: PositiveCount = 15
    artifact_index: Path | None = None
    probability_mode: Literal["recommended", "uncalibrated", "calibrated"] = "recommended"


class EvaluationConfig(Contract):
    partition: Literal["validation"] = "validation"
    bootstrap_repeats: PositiveCount = 1000
    temperature_min: float = Field(default=0.01, gt=0)
    temperature_max: float = Field(default=100, gt=1)
    gate_cost_ratio: float = Field(default=1.25, gt=1)
    gate_ap_improvement: float = Field(default=0.0005, gt=0)

    @model_validator(mode="after")
    def check_temperature_bounds(self) -> Self:
        if not self.temperature_min < 1 < self.temperature_max:
            raise ValueError("temperature search bounds must contain one")
        return self


class ProfilingConfig(Contract):
    batch_sizes: tuple[PositiveCount, ...] = (1, 2, 4, 8)
    warmup: PositiveCount = 50
    repeats: int = Field(default=300, ge=20)
    rounds: int = Field(default=2, ge=2)
    cpu_threads: PositiveCount = 1
    overhead_candidates: int = Field(default=100, ge=20)
    stability_ratio: float = Field(default=2, gt=1)

    @model_validator(mode="after")
    def check_batches(self) -> Self:
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("duplicate batch size")
        if not {1, 2, 4, 8} <= set(self.batch_sizes):
            raise ValueError("required batches are 1,2,4,8")
        return self


class MonitorConfig(Contract):
    enabled: bool = True
    interval_s: float = Field(default=0.5, gt=0)
    history_limit: PositiveCount = 256
    estimator_window_s: float = Field(default=5, gt=0)
    estimator_max_samples: int = Field(default=10000, ge=2, strict=True)
    gpu_refresh_s: float = Field(default=2, gt=0)
    overhead_warning_fraction: float = Field(default=0.1, gt=0, le=1)


class QueueConfig(Contract):
    capacity: PositiveCount = 64
    overflow: Literal["block", "reject", "drop"] = "block"
    controlled_loss_experiment: bool = False

    @model_validator(mode="after")
    def check_loss(self) -> Self:
        if self.overflow != "block" and not self.controlled_loss_experiment:
            raise ValueError("reject/drop require controlled_loss_experiment=true")
        return self


class TierSelectionConfig(Contract):
    policy: Literal["fixed", "round_robin", "resource_queue_aware"] = "resource_queue_aware"
    fixed_tier: ModelTier = ModelTier.TINY


class AdmissionConfig(Contract):
    policy: Literal["unbounded", "fixed", "sequential", "memory_contention_aware"] = (
        "memory_contention_aware"
    )
    execution_mode: Literal["sequential", "fixed_concurrency", "dynamic_concurrency"] = (
        "dynamic_concurrency"
    )
    fixed_concurrency: PositiveCount = 1
    max_concurrency: PositiveCount = 4

    @model_validator(mode="after")
    def check_concurrency(self) -> Self:
        if self.fixed_concurrency > self.max_concurrency:
            raise ValueError("fixed concurrency exceeds maximum")
        return self


class UncertaintyConfig(Contract):
    policy: Literal["none", "selective_mc_dropout", "always_mc_dropout"] = "selective_mc_dropout"
    passes: PositiveCount = 10
    seed: Count | None = None
    artifact: Path | None = None
    artifact_sha256: str | None = None


class EscalationConfig(Contract):
    policy: Literal["none", "cascade"] = "cascade"
    max_escalations: Count = 2


class ThresholdConfig(Contract):
    policy: Literal["global", "common", "per_tier"] = "per_tier"
    artifact: Path | None = None
    fit_partition: Literal["validation"] = "validation"


class FateConfig(Contract):
    policy: Literal["transmit_all", "classification_only", "recall_constrained"] = (
        "recall_constrained"
    )
    target_survival_recall: Probability = 0.95
    compression: Literal["logical_bytes"] = "logical_bytes"


class WorkloadConfig(Contract):
    trace: Path | None = None
    arrival_rate_per_s: NonNegative = 10
    seed: Count = DEFAULT_SEED
    count: PositiveCount = 100
    profile: Literal["low", "moderate", "high", "burst", "alternating", "backlog"] = "burst"
    arrival_process: Literal["periodic", "poisson"] = "periodic"
    low_rate: float = Field(default=5, gt=0)
    moderate_rate: float = Field(default=50, gt=0)
    high_rate: float = Field(default=500, gt=0)
    network_available: bool | None = None
    network_bandwidth_bytes_per_s: NonNegative | None = None
    network_latency_ms: NonNegative | None = None


class ExperimentConfig(Contract):
    name: Identifier = "foundation"
    purpose: Identifier = "Module 01 infrastructure validation"
    seed: Count = DEFAULT_SEED
    provenance_ids: tuple[Identifier, ...] = ("engineering-foundation",)


class PipelineConfig(Contract):
    mode: Literal["reference", "monitor_only", "adaptive_sequential", "concurrent"] = "reference"
    model_adapter: Literal["dummy", "cnn"] = "cnn"
    partition: Literal["train", "validation", "test"] = "validation"
    admission_retry_s: float = Field(default=0.01, gt=0, le=60)
    admission_timeout_s: float = Field(default=1, gt=0, le=3600)
    record_batch_size: PositiveCount = 64


class AppConfig(Contract):
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    tier_selection: TierSelectionConfig = Field(default_factory=TierSelectionConfig)
    admission: AdmissionConfig = Field(default_factory=AdmissionConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    fate: FateConfig = Field(default_factory=FateConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @model_validator(mode="after")
    def check_fixed_tier(self) -> Self:
        if (
            self.tier_selection.policy == "fixed"
            and self.tier_selection.fixed_tier not in self.models.tier_portfolio
        ):
            raise ValueError("fixed tier must belong to selected portfolio")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def resolved_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True, allow_unicode=True)


class UniqueKeyLoader(yaml.SafeLoader):
    """Prevent silently overwriting repeated YAML keys."""


def _unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict[str, object]:
    result: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("configuration requires unique string mapping keys")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve(strict=True)
    raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    config = AppConfig.model_validate(raw)
    root = (config_path.parent / config.paths.project_root).resolve()
    resolved = config.model_dump(mode="python")
    for section in resolved.values():
        for key, value in section.items():
            if isinstance(value, Path):
                section[key] = root if key == "project_root" else (root / value).resolve()
    config = AppConfig.model_validate(resolved)
    raw_paths = {config.dataset.images, config.dataset.labels}
    if len(raw_paths) != 2:
        raise ValueError("images and labels paths must differ")
    for output in (config.paths.runs_dir, config.paths.artifacts_dir, config.paths.logs_dir):
        if any(output == raw_path or output in raw_path.parents for raw_path in raw_paths):
            raise ValueError("output directory must not contain the raw dataset")
    return config
