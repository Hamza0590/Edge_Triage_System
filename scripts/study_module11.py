"""Declare, fit, freeze and execute one curated Module11 study. Never tune on test."""

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

from edge_triage.calibration.profiling import ProfileBundle, hardware_identity
from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.harness import freeze_matrix, run_matrix, source_inventory
from edge_triage.experiments.matrix import ExperimentMatrix, enforce_comparison
from edge_triage.experiments.promotion import promote
from edge_triage.experiments.workloads import TraceRecord
from edge_triage.health import hash_file
from edge_triage.triage.workflow import collect, fit_collection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/module11_v3"
# Seed CPU profiles were measured once under module11_v2 and are reused unchanged (decision 0071).
SEED_PROFILES = ROOT / "artifacts/experiments/module11_v2"
TIERS = tuple(ModelTier)


def profile_seeds(out):
    """Fresh synthetic-input CPU profiles for existing seed families; no retraining."""
    base = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml")
    write_immutable(
        out / "seed_profile_protocol_v2.json",
        canonical(
            {
                "seeds": [20260925, 20260926],
                "device": "cpu",
                "batches": [1, 2, 4, 8],
                "scheduler_batch": 1,
                "supersedes": "seed_profile_protocol.json (schema rejected before measurements)",
                "warmup": 50,
                "repeats": 300,
                "rounds": 2,
                "cpu_threads": 1,
                "input": "seeded synthetic tensor; no test inputs",
                "selection": "no seed selection",
            }
        ),
    )
    for seed in (20260925, 20260926):
        inputs = json.loads(
            (ROOT / f"artifacts/triage/module10_v2/seeds/{seed}/inputs.json").read_bytes()
        )
        raw = base.model_dump()
        raw["models"]["checkpoint_index"] = Path(inputs["family"])
        raw["training"]["seed"] = seed
        raw["profiling"].update(batch_sizes=(1, 2, 4, 8), warmup=50, repeats=300, rounds=2)
        config = AppConfig.model_validate(raw)
        directory = out / "seed_profiles" / str(seed)
        path = directory / "config.yaml"
        write_immutable(path, config.resolved_yaml().encode())
        profiles, workers = [], []
        for tier in TIERS:
            output = directory / f"{tier.value}.json"
            if not output.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "edge_triage.calibration.profile_worker",
                        "--config",
                        str(path),
                        "--tier",
                        tier.value,
                        "--device",
                        "cpu",
                        "--output",
                        str(output),
                    ],
                    check=True,
                    timeout=600,
                )
            worker = json.loads(output.read_bytes())
            profiles.extend(worker["profiles"])
            workers.append({"path": str(output), "sha256": hash_file(output)})
        bundle = ProfileBundle(
            config_sha256=config.config_hash,
            code_sha256=hash_file(ROOT / "src/edge_triage/calibration/profile_worker.py"),
            family_index_sha256=hash_file(config.models.checkpoint_index),
            profiles=profiles,
            environment={"workers": workers},
            methodology={"protocol": str(out / "seed_profile_protocol_v2.json")},
            overhead={"scope": "component profile; runtime instrumentation measured by harness"},
        )
        write_immutable(directory / "profiles.json", canonical(bundle.model_dump(mode="json")))


def seed_hardware():
    """The single hardware label under which each reused seed bundle was measured."""
    result = {}
    for seed in (20260925, 20260926):
        bundle = json.loads(
            (SEED_PROFILES / "seed_profiles" / str(seed) / "profiles.json").read_bytes()
        )
        labels = {p["hardware_id"] for p in bundle["profiles"]}
        if len(labels) != 1:
            raise ValueError(f"seed {seed} profiles span several hardware labels: {labels}")
        result[str(seed)] = labels.pop()
    return result


def trace(config, out, partition, interval):
    with config.split.manifest.open(encoding="utf-8", newline="") as stream:
        rows = sorted(
            (r for r in csv.DictReader(stream) if r["split"] == partition),
            key=lambda r: r["candidate_id"],
        )
    value = {
        "schema_version": 1,
        "partition": partition,
        "split_sha256": hash_file(config.split.manifest),
        "promotion_workload": {"process": "periodic", "interval_seconds": interval},
        "purpose": "UNIQUE_PARTITION_FROZEN_MODULE11_PROTOCOL",
        "records": [
            TraceRecord(
                trace_position=i,
                candidate_id=r["candidate_id"],
                source_candidate_id=r["candidate_id"],
                dataset_index=int(r["index_no"]),
                relative_arrival_seconds=i * interval,
                workload_phase="backlog" if interval == 0 else "moderate",
                seed=20260924,
                repetition=0,
            ).model_dump(mode="json")
            for i, r in enumerate(rows)
        ],
    }
    path = out / "inputs" / f"{partition}_{interval}.json"
    write_immutable(path, canonical(value))
    write_immutable(path.with_suffix(".sha256"), (hash_file(path) + "\n").encode())
    return path.resolve(), len(rows)


def declare(out):
    base = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml")
    uq_base = load_config(ROOT / "configs/experiments/module08_selective_v1.yaml")
    protocol = json.loads((ROOT / "configs/experiments/module10_protocol_v1.json").read_bytes())
    protocol.update(
        version="module11_study_v4",
        module=11,
        design="curated_not_full_factorial",
        repetitions=2,
        warmup_forwards_per_tier=5,
        test_tuning_forbidden=True,
        seed_scope="three frozen families; all seed results retained; no best-seed selection",
        component_scope="matched held-out component microbatch/concurrent batch1; separate timing",
        matrix={
            "pipeline_cases": 39,
            "component_cases": 21,
            "definition": "declaration.json and matrix.yaml; named B1-B7/FULL and matched controls",
            "primary_systems": "B5/B6/B7 change executor and effective admission together",
            "independent_controls": "admission overrides; singleton executor controls",
        },
        trace="canonical unique partition rows; periodic intervals 0, .01 and .05 seconds",
        hardware={
            "declaration_host": hardware_identity()[1],
            "declaration_host_id": hardware_identity()[0],
            "inherited_profile_hardware_id": base.scheduler.hardware_id,
            "seed_profile_hardware_ids": seed_hardware(),
            "note": "hardware_id labels the machine state a profile bundle was measured on; "
            "run manifests record the actual OS/GPU of every run (decision 0071)",
        },
        seed_stability={
            "training_seeds": [20260924, 20260925, 20260926],
            "scope": "frozen singletons; calibrated noUQ; shared search grid; no seed selection",
        },
    )
    write_immutable(out / "protocol.json", canonical(protocol))
    write_immutable(out / "runtime_source.json", canonical(source_inventory(ROOT)))
    write_immutable(out / "driver_source.json", canonical({"sha256": hash_file(Path(__file__))}))
    specs = []

    def case(
        name,
        portfolio=TIERS,
        selector="fixed",
        mode="sequential",
        uq=False,
        fate="recall_constrained",
        admission="fixed_memory",
        interval=0,
        probability="recommended",
    ):
        specs.append(
            dict(
                id=name,
                portfolio=portfolio,
                selector=selector,
                mode=mode,
                uq=uq,
                fate=fate,
                admission=admission,
                interval=interval,
                probability=probability,
            )
        )

    case("B1", (TIERS[2],))
    case("B2", (TIERS[0],))
    case("B3", (TIERS[1],))
    for letter, portfolio in zip("abcd", [*itertools.combinations(TIERS, 2), TIERS]):
        case(f"B4{letter}", portfolio, uq=True, probability="calibrated")
    case("B5", selector="resource_queue_aware")
    case("B6", selector="resource_queue_aware", mode="fixed_concurrency")
    case(
        "B7",
        selector="resource_queue_aware",
        mode="dynamic_concurrency",
        admission="memory_contention_aware",
    )
    case(
        "FULL",
        selector="resource_queue_aware",
        mode="dynamic_concurrency",
        uq=True,
        probability="calibrated",
        admission="memory_contention_aware",
    )
    # Matched single-axis controls for the primary systems.
    case("static_all")
    case("B5_resource_only", selector="resource_only")
    case(
        "B6_memory_admission",
        selector="resource_queue_aware",
        mode="fixed_concurrency",
        admission="memory_contention_aware",
    )
    case(
        "B7_fixed_admission",
        selector="resource_queue_aware",
        mode="dynamic_concurrency",
        admission="fixed_memory",
    )
    case(
        "B7_calibrated_noUQ",
        selector="resource_queue_aware",
        mode="dynamic_concurrency",
        probability="calibrated",
        admission="memory_contention_aware",
    )
    case("B5_common", selector="resource_queue_aware", fate="common_three_way")
    case("B5_transmit_all", selector="resource_queue_aware", fate="transmit_all")
    case("B5_binary", selector="resource_queue_aware", fate="binary")
    # Portfolio ablation (decision 0069): adaptive selection over each of the seven
    # portfolios; B5 is the all-three member. Only the portfolio differs.
    for size in (1, 2):
        for portfolio in itertools.combinations(TIERS, size):
            case(
                "portfolio_" + "+".join(t.value for t in portfolio),
                portfolio,
                selector="resource_queue_aware",
            )
    for tier in TIERS:
        case(f"fixed_{tier.value}", (tier,), mode="fixed_concurrency")
    for interval in (0.01, 0.05):
        case(f"B5_arrival_{int(1 / interval)}", selector="resource_queue_aware", interval=interval)
    for seed in (20260924, 20260925, 20260926):
        for tier in TIERS:
            case(f"seed_{seed}_{tier.value}", (tier,), probability="calibrated")
            specs[-1]["seed"] = seed
    configs = []
    for spec in specs:
        raw = base.model_dump()
        raw["models"]["tier_portfolio"] = spec["portfolio"]
        raw["tier_selection"].update(policy=spec["selector"], fixed_tier=spec["portfolio"][0])
        raw["admission"].update(
            execution_mode=spec["mode"], fixed_concurrency=2, concurrent_policy=spec["admission"]
        )
        raw["calibration"]["probability_mode"] = spec["probability"]
        if "seed" in spec:
            seed = spec["seed"]
            raw["scheduler"].update(contention_profile=None, contention_sha256=None)
            if seed != 20260924:
                inputs = json.loads(
                    (ROOT / f"artifacts/triage/module10_v2/seeds/{seed}/inputs.json").read_bytes()
                )
                raw["training"]["seed"] = seed
                raw["models"]["checkpoint_index"] = Path(inputs["family"])
                raw["calibration"]["artifact_index"] = Path(inputs["calibration"])
                profile = (SEED_PROFILES / "seed_profiles" / str(seed) / "profiles.json").resolve()
                raw["scheduler"].update(
                    profiles=profile,
                    profiles_sha256=hash_file(profile),
                    hardware_id=seed_hardware()[str(seed)],
                )
        if spec["uq"]:
            raw["uncertainty"] = uq_base.uncertainty.model_dump()
            raw["uncertainty"]["seed"] = 20260924
            raw["escalation"].update(policy="cascade", max_escalations=2)
        raw["experiment"].update(name=spec["id"], purpose="VALIDATION_FATE_STUDY")
        raw["paths"]["runs_dir"] = (out / "runtime_runs").resolve()
        trace_path, count = trace(base, out, "validation", spec["interval"])
        raw["workload"].update(
            trace=trace_path,
            count=count,
            profile="backlog" if spec["interval"] == 0 else "moderate",
            arrival_rate_per_s=0 if spec["interval"] == 0 else 1 / spec["interval"],
        )
        raw["fate"]["policy"] = spec["fate"]
        raw["threshold"].update(
            policy="common"
            if spec["fate"] in {"common_three_way", "transmit_all", "binary"}
            else "per_tier",
            artifact=None,
            artifact_sha256=None,
        )
        config = AppConfig.model_validate(raw)
        path = out / "inputs" / f"{spec['id']}.yaml"
        write_immutable(path, config.resolved_yaml().encode())
        configs.append(
            {
                "id": spec["id"],
                "config": str(path.resolve()),
                "sha256": hash_file(path),
                "interval": spec["interval"],
            }
        )
    write_immutable(out / "declaration.json", canonical(configs))
    by_id = {spec["id"]: load_config(spec["config"]) for spec in configs}
    for comparison in comparisons():
        enforce_comparison(
            [by_id[key] for key in comparison["case_ids"]], tuple(comparison["changed_axes"])
        )
    return configs, protocol


def fit(out):
    from edge_triage.triage.artifacts import load_fate_artifact
    from edge_triage.triage.workflow import read_collection

    specs, protocol = declare(out)
    for spec in specs:
        if json.loads((out / "runtime_source.json").read_bytes()) != source_inventory(ROOT):
            raise ValueError("runtime changed after study declaration")
        target = load_config(spec["config"])
        directory = out / "validation" / spec["id"]
        if (directory / "fit/fit.json").exists():
            fitted = load_config(directory / "fit/config.yaml")
            load_fate_artifact(fitted.threshold.artifact, fitted.threshold.artifact_sha256, fitted)
            read_collection(directory / "collection.json", fitted)
            continue
        print(f"VALIDATION {spec['id']}", flush=True)
        raw = target.model_dump()
        raw["fate"]["policy"] = "transmit_all"
        raw["threshold"].update(policy="common", artifact=None, artifact_sha256=None)
        collection = collect(
            AppConfig.model_validate(raw),
            directory,
            warmup_forwards_per_tier=protocol["warmup_forwards_per_tier"],
        )
        fit_collection(collection, target, directory / "fit", protocol)
        print(f"FROZEN {spec['id']}", flush=True)


def freeze(out):
    specs, protocol = declare(out)
    cases = []
    for spec in specs:
        source_path = out / "validation" / spec["id"] / "fit/config.yaml"
        source = load_config(source_path)
        raw = source.model_dump()
        trace_path, count = trace(source, out, "test", spec["interval"])
        raw["pipeline"]["partition"] = "test"
        raw["experiment"]["purpose"] = "HELD_OUT_EXPERIMENT"
        raw["workload"].update(trace=trace_path, count=count)
        promoted = promote(
            source_path, AppConfig.model_validate(raw), out / "promotions" / f"{spec['id']}.json"
        )
        path = out / "test_configs" / f"{spec['id']}.yaml"
        write_immutable(path, promoted.resolved_yaml().encode())
        cases.append(
            {
                "id": spec["id"],
                "description": f"Frozen held-out {spec['id']}; see protocol and declaration",
                "config": {"path": str(path.resolve()), "sha256": hash_file(path)},
            }
        )
    for baseline, tier in (("B1", "large"), ("B2", "tiny"), ("B3", "medium")):
        ref = next(c["config"] for c in cases if c["id"] == baseline)
        for backend, levels in (("microbatch", (1, 2, 4, 8)), ("batch1_threads", (1, 2, 4))):
            for level in levels:
                cases.append(
                    {
                        "id": f"component_{tier}_{backend}_{level}",
                        "description": "Component timing only; no equal-recall cost claim",
                        "config": ref,
                        "component": backend,
                        "component_level": level,
                    }
                )
    matrix = ExperimentMatrix(
        name=out.name,
        project_root=ROOT,
        output_dir=(out / "study").resolve(),
        partition="test",
        purpose="unique_test",
        repetitions=protocol["repetitions"],
        warmup_forwards_per_tier=5,
        bootstrap_repeats=1000,
        cases=cases,
        comparisons=comparisons(),
    )
    path = out / "matrix.yaml"
    write_immutable(path, yaml.safe_dump(matrix.model_dump(mode="json"), sort_keys=False).encode())
    freeze_matrix(path)
    print(path, flush=True)
    return path


def comparisons():
    result = [
        {
            "id": "execution_fixed_admission",
            "case_ids": ["B5", "B6", "B7_fixed_admission"],
            "changed_axes": ["execution_mode"],
        },
        {
            "id": "primary_systems",
            "case_ids": ["B5", "B6", "B7"],
            "changed_axes": ["execution_mode", "admission_policy"],
        },
        {
            "id": "selection",
            "case_ids": ["static_all", "B5"],
            "changed_axes": ["tier_selection_policy"],
        },
        {
            "id": "queue",
            "case_ids": ["B5", "B5_resource_only"],
            "changed_axes": ["queue_awareness"],
        },
        {
            "id": "fixed_admission",
            "case_ids": ["B6", "B6_memory_admission"],
            "changed_axes": ["admission_policy"],
        },
        {
            "id": "dynamic_admission",
            "case_ids": ["B7", "B7_fixed_admission"],
            "changed_axes": ["admission_policy"],
        },
        {
            "id": "uq",
            "case_ids": ["B7_calibrated_noUQ", "FULL"],
            "changed_axes": ["uncertainty_policy", "escalation_policy"],
        },
        {
            "id": "threshold",
            "case_ids": ["B5", "B5_common"],
            "changed_axes": ["threshold_policy", "fate_policy"],
        },
        {
            "id": "fate",
            "case_ids": ["B5", "B5_transmit_all", "B5_binary"],
            "changed_axes": ["threshold_policy", "fate_policy"],
        },
        {
            "id": "arrival",
            "case_ids": ["B5", "B5_arrival_20", "B5_arrival_100"],
            "changed_axes": ["arrival_trace"],
        },
        {
            "id": "portfolio",
            "case_ids": [
                "portfolio_tiny",
                "portfolio_medium",
                "portfolio_large",
                "portfolio_tiny+medium",
                "portfolio_tiny+large",
                "portfolio_medium+large",
                "B5",
            ],
            "changed_axes": ["portfolio"],
        },
    ]
    for baseline, tier in (("B1", "large"), ("B2", "tiny"), ("B3", "medium")):
        result.append(
            {
                "id": f"executor_{tier}",
                "case_ids": [baseline, f"fixed_{tier}"],
                "changed_axes": ["execution_mode"],
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=["profiles", "declare", "fit", "freeze", "run"]
    )
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage == "profiles":
        profile_seeds(SEED_PROFILES)
    elif args.stage == "declare":
        declare(out)
    elif args.stage == "fit":
        fit(out)
    elif args.stage == "freeze":
        freeze(out)
    else:
        report = run_matrix(out / "matrix.yaml")
        print(report)
        if any(r["status"] == "failed" for r in json.loads(report.read_bytes())["subruns"]):
            raise SystemExit(1)
