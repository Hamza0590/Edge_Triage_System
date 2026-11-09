"""CLI diagnostics and foundation demonstration; no training or scheduling here."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from edge_triage.config import load_config
from edge_triage.contracts import Candidate
from edge_triage.health import doctor
from edge_triage.runs import RunManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edge-triage", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("doctor", "Read-only configuration, dataset and environment checks"),
        ("show-config", "Show resolved YAML and canonical SHA-256 configuration hash"),
        ("sample-run", "Create and finalize an infrastructure-only demonstration run"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
    data = subcommands.add_parser("data", help="Audit and prepare frozen MeerLICHT data")
    actions = data.add_subparsers(dest="data_action", required=True)
    for name in ("audit", "prepare"):
        actions.add_parser(name).add_argument("--config", required=True)
    pipeline = subcommands.add_parser("pipeline", help="Non-scientific reference pipeline")
    modes = pipeline.add_subparsers(dest="pipeline_action", required=True)
    smoke = modes.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--limit", type=int, default=20)
    monitored = modes.add_parser("monitor-only")
    monitored.add_argument("--config", required=True)
    monitored.add_argument("--trace", required=True)
    concurrent = modes.add_parser("concurrent")
    concurrent.add_argument("--config", required=True)
    concurrent.add_argument("--trace", required=True)
    adaptive = modes.add_parser("adaptive-sequential")
    adaptive.add_argument("--config", required=True)
    adaptive.add_argument("--trace", required=True)
    workloads = subcommands.add_parser("workloads", help="Generate immutable system trace suite")
    workloads.add_argument("--config", required=True)
    train = subcommands.add_parser("train", help="Train CNN portfolio using train/validation only")
    train.add_argument("--config", required=True)
    evaluate = subcommands.add_parser(
        "evaluate", help="Module 05 validation calibration and profiling"
    )
    evaluate.add_argument("--config", required=True)
    experiment = subcommands.add_parser("experiment", help="Frozen validation experiment harness")
    experiment_actions = experiment.add_subparsers(dest="experiment_action", required=True)
    for action in ("validate-matrix", "freeze", "run"):
        experiment_actions.add_parser(action).add_argument("--config", required=True, type=Path)
    experiment_actions.add_parser("summarize").add_argument("--runs", required=True, type=Path)
    experiment_actions.add_parser("plot").add_argument("--summary", required=True, type=Path)
    triage = subcommands.add_parser("triage", help="Validation-only fate collection and fitting")
    triage_actions = triage.add_subparsers(dest="triage_action", required=True)
    for action in ("collect", "fit", "replay"):
        command = triage_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--output", required=True, type=Path)
        if action == "fit":
            command.add_argument("--collection", required=True, type=Path)
            command.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "experiment":
            from edge_triage.experiments.harness import (
                freeze_matrix,
                plot_summary,
                run_matrix,
                summarize_study,
            )
            from edge_triage.experiments.matrix import load_matrix, validate_matrix

            if args.experiment_action == "validate-matrix":
                result = validate_matrix(load_matrix(args.config))
                sys.stdout.write(json.dumps(result["counts"], indent=2) + "\n")
            elif args.experiment_action == "freeze":
                matrix, _ = freeze_matrix(args.config)
                sys.stdout.write(str(matrix.output_dir / "matrix.resolved.json") + "\n")
            elif args.experiment_action == "run":
                result_path = run_matrix(args.config)
                sys.stdout.write(str(result_path) + "\n")
                if any(
                    r["status"] == "failed" for r in json.loads(result_path.read_bytes())["subruns"]
                ):
                    return 1
            elif args.experiment_action == "summarize":
                sys.stdout.write(str(summarize_study(args.runs)) + "\n")
            else:
                sys.stdout.write(str(plot_summary(args.summary)) + "\n")
            return 0
        config = load_config(args.config)
        if args.command == "triage":
            from edge_triage.triage.workflow import collect, fit_collection, replay

            if args.triage_action == "collect":
                result_path = collect(config, args.output)
            elif args.triage_action == "fit":
                result_path = fit_collection(
                    args.collection, config, args.output, json.loads(args.protocol.read_bytes())
                )
            else:
                result_path = replay(config, args.output)
            sys.stdout.write(str(result_path) + "\n")
        elif args.command == "evaluate":
            from edge_triage.calibration.workflow import run_module05

            sys.stdout.write(str(run_module05(config)) + "\n")
        elif args.command == "train":
            from edge_triage.models.training import train_family

            sys.stdout.write(str(train_family(config)) + "\n")
        elif args.command == "workloads":
            from edge_triage.experiments.workloads import generate_suite

            sys.stdout.write(str(generate_suite(config)) + "\n")
        elif args.command == "pipeline" and args.pipeline_action in {
            "monitor-only",
            "adaptive-sequential",
            "concurrent",
        }:
            from edge_triage.runtime.streaming import (
                run_adaptive_sequential,
                run_concurrent,
                run_monitor_only,
            )

            config = config.model_copy(
                update={
                    "workload": config.workload.model_copy(
                        update={"trace": (config.paths.project_root / Path(args.trace)).resolve()}
                    )
                }
            )
            execute = (
                run_monitor_only
                if args.pipeline_action == "monitor-only"
                else run_adaptive_sequential
            )
            if args.pipeline_action == "concurrent":
                execute = run_concurrent
            sys.stdout.write(str(execute(config)) + "\n")
        elif args.command == "pipeline":
            from edge_triage.runtime.factory import run_smoke

            sys.stdout.write(str(run_smoke(config, args.limit)) + "\n")
        elif args.command == "data":
            from edge_triage.data.audit import audit
            from edge_triage.data.preprocessing import prepare

            with RunManager(config) as run:
                result = audit(config) if args.data_action == "audit" else prepare(config)
                run.logger.emit(
                    module="data",
                    event_type=f"data.{args.data_action}_completed",
                    reason_code="FROZEN_DATA_VERIFIED",
                    message=f"Data {args.data_action} complete",
                    payload={"artifact_directory": str(config.paths.artifacts_dir / "data")},
                )
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
        elif args.command == "show-config":
            sys.stdout.write(f"# config_hash: {config.config_hash}\n" + config.resolved_yaml())
        elif args.command == "doctor":
            sys.stdout.write(json.dumps(doctor(config), indent=2) + "\n")
        else:
            with RunManager(config) as run:
                candidate = Candidate(
                    run_id=run.manifest.run_id,
                    trace_id="sample-trace",
                    candidate_id="sample-0",
                    dataset_index=0,
                    arrival_monotonic_ns=time.monotonic_ns(),
                )
                run.logger.bind(candidate.trace_id, candidate.candidate_id).emit(
                    module="foundation",
                    event_type="data.reference_registered",
                    reason_code="METADATA_ONLY",
                    message="Sample candidate reference; no inference",
                    payload=candidate,
                )
            sys.stdout.write(str(run.directory) + "\n")
        return 0
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        # Before a run exists, stderr diagnostics use standard logging.
        from edge_triage.event_logging import safe_text

        logging.getLogger("edge_triage.cli").error(
            "%s: %s", type(error).__name__, safe_text(str(error))
        )
        return 1
