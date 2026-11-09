"""Isolated CPU thread/process and micro-batch component benchmark, validation only."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

from edge_triage.config import AppConfig, load_config
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.models.inference_state import INFERENCE_STATE
from edge_triage.runtime.factory import assemble
from edge_triage.runtime.microbatch import replay_batches

MODEL = IMAGES = None


def initialize(config_path, tier, images_path, barrier=None):
    global MODEL, IMAGES
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(config_path)
    raw = config.model_dump()
    raw["models"]["tier_portfolio"] = (tier,)
    raw["tier_selection"].update(policy="fixed", fixed_tier=tier)
    parts = assemble(AppConfig.model_validate(raw))
    adapter = next(iter(parts.models.values()))
    MODEL = getattr(adapter, "base", adapter).model
    IMAGES = np.load(images_path, mmap_mode="r")
    for _ in range(5):
        infer(0)
    if barrier is not None:
        barrier.wait(timeout=90)


def infer(index, queued_ns=None):
    start = time.perf_counter_ns()
    with INFERENCE_STATE.read(), torch.inference_mode():
        value = float(MODEL(torch.from_numpy(IMAGES[index : index + 1].copy())).item())
    ended = time.perf_counter_ns()
    return {
        "position": index,
        "logit": value,
        "inference_ms": (ended - start) / 1e6,
        "wait_ms": (start - queued_ns) / 1e6 if queued_ns is not None else 0,
        "end_to_end_ms": (ended - queued_ns) / 1e6 if queued_ns is not None else 0,
        "pid": os.getpid(),
        "rss": psutil.Process().memory_info().rss,
    }


def warm_thread(barrier):
    for _ in range(5):
        infer(0)
    barrier.wait(timeout=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--backend", choices=["threads", "processes", "microbatch"], required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--arrivals")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = len(np.load(args.images, mmap_mode="r"))
    start = time.perf_counter()
    result = {"backend": args.backend, "tier": args.tier, "level": args.level, "rounds": []}
    result["inputs"] = {
        "config_sha256": hash_file(Path(args.config)),
        "images_sha256": hash_file(Path(args.images)),
        "arrivals_sha256": hash_file(Path(args.arrivals)),
        "script_sha256": hash_file(Path(__file__)),
    }
    result["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_bytes": psutil.virtual_memory().total,
        "cpu_threads": 1,
        "interop_threads": 1,
        "precision": "fp32_tf32_off",
        "device": "cpu",
    }
    if args.backend == "processes":
        context = mp.get_context("spawn")
        barrier = context.Barrier(args.level)
        pool = ProcessPoolExecutor(
            args.level,
            mp_context=context,
            initializer=initialize,
            initargs=(args.config, args.tier, args.images, barrier),
        )
        # Force every process to load/warm before timed rounds; jobs carry integers only.
        warm_rows = list(pool.map(infer, range(args.level)))
    else:
        initialize(args.config, args.tier, args.images)
        pool = ThreadPoolExecutor(
            args.level, initializer=warm_thread, initargs=(threading.Barrier(args.level),)
        )
        warm_rows = list(pool.map(infer, range(args.level)))
        if args.backend == "microbatch":
            with INFERENCE_STATE.read(), torch.inference_mode():
                for _ in range(5):
                    MODEL(torch.from_numpy(IMAGES[: args.level].copy()))
    result["startup_s"] = time.perf_counter() - start
    result["worker_pids"] = sorted({r["pid"] for r in warm_rows})
    if args.backend == "processes":
        result["worker_pids"] = sorted(pool._processes)
    reference = None
    try:
        for _ in range(3):
            if args.backend == "microbatch":
                arrivals = json.loads(Path(args.arrivals).read_text())
                row = replay_batches(MODEL, IMAGES, arrivals, args.level, 0.002)
            else:
                start = time.perf_counter()
                queued_ns = time.perf_counter_ns()
                futures = [pool.submit(infer, i, queued_ns) for i in range(count)]
                rows = [f.result() for f in futures]
                elapsed = time.perf_counter() - start
                row = {
                    "rows": rows,
                    "elapsed_s": elapsed,
                    "throughput_per_s": count / elapsed,
                    "aggregate_worker_rss": sum(
                        max(
                            [psutil.Process(pid).memory_info().rss]
                            + [r["rss"] for r in rows if r["pid"] == pid]
                        )
                        for pid in result["worker_pids"]
                    ),
                }
            assert len(row["rows"]) == count
            assert {r["position"] for r in row["rows"]} == set(range(count))
            logits = [r["logit"] for r in sorted(row["rows"], key=lambda r: r["position"])]
            assert np.isfinite(logits).all()
            if reference is not None:
                np.testing.assert_allclose(logits, reference, rtol=1e-5, atol=1e-5)
            reference = logits
            result["rounds"].append(row)
    finally:
        pool.shutdown()
    write_immutable(Path(args.output), canonical(result))
    print(json.dumps({"output": args.output, "startup_s": result["startup_s"]}))


if __name__ == "__main__":
    main()
