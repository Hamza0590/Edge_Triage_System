"""One fresh subprocess per tier/device; no other project inference runs concurrently."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from edge_triage.calibration.profiling import (
    MeasuredProfile,
    hardware_identity,
    machine_state,
    measure,
    summarize,
    synchronize,
)
from edge_triage.config import load_config
from edge_triage.contracts import Candidate, ModelTier, utc_now
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.data.preprocessing import crop_chw
from edge_triage.health import environment_info, hash_file
from edge_triage.models.registry import (
    CheckpointManifest,
    ModelRegistry,
    RealModelAdapter,
    read_index,
)
from edge_triage.models.training import resolve_training_data, seed_everything
from edge_triage.runtime.interfaces import CandidateInput


def profile(config_path: Path, tier: ModelTier, device_name: str, output: Path) -> None:
    config, hashes = resolve_training_data(load_config(config_path))
    assert config.models.checkpoint_index and config.split.manifest
    assert config.preprocessing.artifact
    torch.set_num_threads(config.profiling.cpu_threads)
    torch.set_num_interop_threads(1)
    seed_everything(config.experiment.seed)
    device = torch.device(device_name)
    entry = read_index(config.models.checkpoint_index)["checkpoints"][tier.value]
    directory = Path(entry["directory"])
    manifest = CheckpointManifest.model_validate_json(
        (directory / "checkpoint_manifest.json").read_bytes()
    )
    registry = ModelRegistry(
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        dataset_hashes=hashes,
        device=device_name,
    )
    registry.register(tier, entry["checkpoint_id"], directory, entry["manifest_sha256"])
    before = machine_state()
    synchronize(device)
    start = time.perf_counter_ns()
    model, spec = registry.load(tier, entry["checkpoint_id"])
    synchronize(device)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    prep = json.loads(config.preprocessing.artifact.read_text())
    mean = np.asarray(prep["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(prep["std"], dtype=np.float32)[:, None, None]
    rng = np.random.default_rng(config.experiment.seed)
    hardware_id, hardware = hardware_identity()
    rows = []
    batches = list(config.profiling.batch_sizes)
    rng.shuffle(batches)
    for batch in batches:
        raw = rng.normal(size=(batch, 100, 100, 3)).astype(np.float32)

        def preprocess() -> Any:
            return torch.stack(
                [
                    torch.from_numpy(
                        np.ascontiguousarray(
                            (crop_chw(x, spec.crop_size, tuple(prep["channels"])) - mean) / std
                        )
                    )
                    for x in raw
                ]
            )

        host = preprocess()
        tensor = host.to(device)

        def total() -> Any:
            # Output probabilities copied to host; no candidate/event construction.
            return torch.sigmoid(model(preprocess().to(device))).cpu().numpy()

        operations = {
            "preprocessing": preprocess,
            "transfer": lambda: host.to(device),
            "forward": lambda: model(tensor),
            "end_to_end": total,
        }
        if batch == 1:
            adapter = RealModelAdapter(model, manifest.checkpoint_id)
            item = CandidateInput(
                Candidate(
                    run_id="profile",
                    trace_id="profile",
                    candidate_id="synthetic",
                    dataset_index=0,
                    arrival_monotonic_ns=0,
                ),
                host[0],
                0,
                120000,
            )
            operations["real_adapter"] = lambda: adapter.predict(item, tier)
        values: dict[str, list[float]] = {name: [] for name in operations}
        round_medians = []
        round_order = []
        rss_before = psutil.Process().memory_info().rss
        rss_peak = rss_before
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(config.profiling.rounds):
                names = list(operations)
                rng.shuffle(names)
                round_order.append(names)
                for name in names:
                    samples = measure(
                        operations[name],
                        device,
                        warmup=config.profiling.warmup,
                        repeats=config.profiling.repeats,
                    )
                    values[name].extend(samples)
                    if name == "end_to_end":
                        round_medians.append(float(np.median(samples)))
                    rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
        raw_path = output.with_name(f"{tier.value}_{device_name}_batch{batch}_samples.json")
        write_immutable(
            raw_path,
            canonical(
                {
                    "samples_ms": values,
                    "round_order": round_order,
                    "input_sha256": hashlib.sha256(raw).hexdigest(),
                }
            ),
        )
        components = {name: summarize(samples) for name, samples in values.items()}
        latency = components["end_to_end"]
        allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        stability = max(round_medians) / min(round_medians)
        rows.append(
            MeasuredProfile(
                profile_id=f"{output.parent.name}:{tier.value}:{device_name}:{batch}",
                tier=tier,
                checkpoint_id=manifest.checkpoint_id,
                device=device_name,
                batch_size=batch,
                concurrency=1,
                input_shape=(spec.input_channels, spec.crop_size, spec.crop_size),
                latency_p50_ms=latency.median_ms,
                latency_p95_ms=latency.p95_ms,
                peak_memory_bytes=allocated if allocated is not None else rss_peak,
                samples=config.profiling.repeats * config.profiling.rounds,
                measured_utc=utc_now(),
                hardware_id=hardware_id,
                checkpoint_manifest_sha256=entry["manifest_sha256"],
                split_sha256=manifest.split_sha256,
                preprocessing_sha256=manifest.preprocessing_sha256,
                parameter_count=model.parameter_count,
                checkpoint_bytes=(directory / "model_state.pt").stat().st_size,
                load_ms=load_ms,
                warmup=config.profiling.warmup,
                rounds=config.profiling.rounds,
                components=components,
                throughput_candidates_s=batch * 1000 / latency.mean_ms,
                amortized_ms_per_candidate=latency.mean_ms / batch,
                batch_completion_median_ms=latency.median_ms,
                cuda_allocated_peak_bytes=allocated,
                cuda_reserved_peak_bytes=reserved,
                host_rss_before_bytes=rss_before,
                host_rss_after_bytes=psutil.Process().memory_info().rss,
                host_rss_sampled_peak_bytes=rss_peak,
                round_medians=tuple(round_medians),
                stability_ratio=stability,
                stable=stability <= config.profiling.stability_ratio,
                raw_samples_path=str(raw_path.resolve()),
                raw_samples_sha256=hash_file(raw_path),
            ).model_dump(mode="json")
        )
        tensor, host = None, None
    write_immutable(
        output,
        canonical(
            {
                "profiles": rows,
                "before": before,
                "after": machine_state(),
                "hardware": hardware,
                "environment": environment_info(),
                "batch_order": batches,
                "cpu_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tier", choices=[t.value for t in ModelTier], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile(args.config, ModelTier(args.tier), args.device, args.output)
