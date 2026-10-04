"""Centralized optional platform collectors. Failures remain explicit and nullable."""

from __future__ import annotations

import csv
import subprocess
import time
from typing import Any, Protocol

import psutil
import torch


class Collector(Protocol):
    def collect(self) -> dict[str, Any]: ...


class HostCollector:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.primed = False

    def collect(self) -> dict[str, Any]:
        try:
            cpu = psutil.cpu_percent(interval=None)
            process_cpu = self.process.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            result = {
                "cpu_percent": cpu if self.primed else None,
                "process_cpu_percent": process_cpu if self.primed else None,
                "available_ram_bytes": ram.available,
                "used_ram_bytes": ram.used,
                "process_rss_bytes": self.process.memory_info().rss,
                "availability": {
                    "host": "psutil",
                    "cpu": "available" if self.primed else "priming",
                },
            }
            self.primed = True
            return result
        except (OSError, psutil.Error) as error:
            return {
                "cpu_percent": None,
                "available_ram_bytes": None,
                "availability": {"host": f"unavailable:{type(error).__name__}"},
            }


class NvidiaCollector:
    def __init__(self, refresh_s: float = 2) -> None:
        self.refresh_s = refresh_s
        self.previous = -float("inf")
        self.cache: dict[str, Any] = {}

    def collect(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self.previous >= self.refresh_s:
            self.previous = now
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id=0",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=1,
                )
                row = next(csv.reader(result.stdout.splitlines()))
                keys = (
                    "gpu_utilization_percent",
                    "gpu_memory_used_bytes",
                    "gpu_memory_total_bytes",
                    "gpu_memory_free_bytes",
                    "gpu_temperature_c",
                    "gpu_power_watts",
                )
                self.cache = {}
                availability = {"gpu": "nvidia-smi:index0"}
                for key, raw in zip(keys, row, strict=True):
                    try:
                        value = float(raw.strip())
                        self.cache[key] = int(value * 1024 * 1024) if "bytes" in key else value
                    except ValueError:
                        self.cache[key] = None
                        availability[key] = "unavailable"
                self.cache.update(
                    gpu_observed_monotonic_ns=time.monotonic_ns(), availability=availability
                )
            except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as error:
                self.cache = {"availability": {"gpu": f"unavailable:{type(error).__name__}"}}
        values = {**self.cache, "availability": dict(self.cache.get("availability", {}))}
        if torch.cuda.is_initialized():
            try:
                values.update(
                    gpu_allocated_bytes=int(torch.cuda.memory_allocated(0)),
                    gpu_reserved_bytes=int(torch.cuda.memory_reserved(0)),
                )
                values["availability"]["allocator"] = "torch:process-local:index0"
            except RuntimeError:
                values["availability"]["allocator"] = "unavailable"
        else:
            values["availability"]["allocator"] = "unavailable:CUDA_not_initialized"
        return values
