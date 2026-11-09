"""Resumable predeclared Module10 validation study; every output is immutable."""

import argparse
import itertools
import json
from pathlib import Path

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import write_immutable
from edge_triage.health import hash_file
from edge_triage.triage.workflow import (
    collect,
    fit_collection,
    replay,
    save_json,
    validation_trace,
)

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIOS = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]
FATES = ("transmit_all", "binary", "common_three_way", "recall_constrained")


def initialize(out):
    # Protect the timed runtime when continuing the interrupted v2 study. Analysis
    # code may evolve without silently changing the model/serving measurements.
    stopped = out / "session_stop.json"

    def relevant(name):
        return name.startswith("src/edge_triage/") and name not in {
            "src/edge_triage/triage/fitting.py",
            "src/edge_triage/triage/workflow.py",
            "src/edge_triage/cli.py",
        }

    inventory = {
        str(p.relative_to(ROOT)).replace("\\", "/"): hash_file(p)
        for p in sorted((ROOT / "src/edge_triage").rglob("*.py"))
        if relevant(str(p.relative_to(ROOT)).replace("\\", "/"))
    }
    if stopped.exists():
        prior = json.loads(stopped.read_bytes())["source_hashes"]
        prior = {
            name.replace("\\", "/"): digest
            for name, digest in prior.items()
            if relevant(name.replace("\\", "/"))
        }
        if inventory != prior:
            raise ValueError("timed runtime changed since stop; declare a new study version")
    save_json(out / "timed_runtime_source.json", inventory)
    protocol_path = ROOT / "configs/experiments/module10_protocol_v1.json"
    write_immutable(out / "protocol.json", protocol_path.read_bytes())
    base = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml")
    trace = validation_trace(base, out)
    settings = []
    for mode in ("sequential", "fixed_concurrency", "dynamic_concurrency"):
        for portfolio in PORTFOLIOS:
            settings.append(
                (
                    f"core_{'+'.join(t.value for t in portfolio)}_{mode}",
                    portfolio,
                    mode,
                    "round_robin",
                    "recommended",
                    False,
                )
            )
    settings.append(
        (
            "proposed_all_dynamic",
            tuple(ModelTier),
            "dynamic_concurrency",
            "resource_queue_aware",
            "recommended",
            False,
        )
    )
    for uq in (False, True):
        for portfolio in PORTFOLIOS:
            settings.append(
                (
                    f"{'mc5' if uq else 'cal_none'}_{'+'.join(t.value for t in portfolio)}",
                    portfolio,
                    "sequential",
                    "fixed",
                    "calibrated",
                    uq,
                )
            )
    entries = []
    for name, portfolio, mode, selector, probability, uq in settings:
        raw = base.model_dump()
        raw["models"]["tier_portfolio"] = portfolio
        raw["tier_selection"].update(policy=selector, fixed_tier=portfolio[0])
        raw["admission"].update(execution_mode=mode, fixed_concurrency=2)
        raw["workload"].update(trace=trace, profile="backlog", count=643)
        raw["calibration"]["probability_mode"] = probability
        raw["fate"].update(policy="transmit_all", compression_fraction=0.25, metadata_bytes=0)
        raw["threshold"].update(policy="common", artifact=None, artifact_sha256=None)
        raw["experiment"].update(name=f"module10_{name}", purpose="VALIDATION_FATE_STUDY")
        raw["experiment"]["provenance_ids"] = (
            *raw["experiment"]["provenance_ids"],
            "selective-classification",
        )
        if uq:
            raw["uncertainty"].update(
                policy="selective_mc_dropout",
                passes=5,
                seed=20260924,
                artifact=ROOT / "artifacts/uncertainty/20260925T205209_b5ff8ad7f32c/routing_5.json",
                artifact_sha256="19f7d021ede74c676e30282b39e6682b3005899722f54ab918cdd2b2e5c54604",
            )
            raw["escalation"]["policy"] = "cascade"
        config = AppConfig.model_validate(raw)
        path = out / "settings" / name / "collection_config.yaml"
        write_immutable(path, config.resolved_yaml().encode())
        entries.append({"name": name, "config": str(path), "sha256": hash_file(path)})
    save_json(
        out / "matrix.json", {"protocol_sha256": hash_file(protocol_path), "settings": entries}
    )
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/triage/module10_v2")
    parser.add_argument("--stage", choices=["init", "collect", "fit", "replay"], required=True)
    parser.add_argument("--only")
    args = parser.parse_args()
    out = args.output.resolve()
    entries = initialize(out)
    protocol = json.loads((out / "protocol.json").read_bytes())
    if args.stage == "init":
        print(
            f"Declared {len(entries)} upstream configurations and {len(entries) * 4} fate variants",
            flush=True,
        )
        return
    for entry in entries:
        name = entry["name"]
        if args.only and args.only not in name:
            continue
        directory = out / "settings" / name
        config = load_config(entry["config"])
        print(f"{args.stage} START {name}", flush=True)
        if args.stage == "collect":
            result = collect(config, directory)
        elif args.stage == "fit":
            for fate in FATES:
                raw = config.model_dump()
                raw["fate"]["policy"] = fate
                raw["threshold"]["policy"] = (
                    "per_tier" if fate == "recall_constrained" else "common"
                )
                result = fit_collection(
                    directory / "collection.json",
                    AppConfig.model_validate(raw),
                    directory / fate,
                    protocol,
                )
                print(f"fit DONE {name} {fate}", flush=True)
        else:
            target = load_config(directory / "recall_constrained/config.yaml")
            result = replay(target, directory / "runtime_replay.json")
        print(f"{args.stage} DONE {name}: {result}", flush=True)


if __name__ == "__main__":
    main()
