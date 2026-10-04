from subprocess import CompletedProcess

from edge_triage.health import nvidia_devices


def test_gpu_probe_converts_mib_to_bytes(monkeypatch):
    monkeypatch.setattr(
        "edge_triage.health.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, '"Test GPU, Rev 1", 16384\n'),
    )
    assert nvidia_devices() == [
        {"name": "Test GPU, Rev 1", "memory_bytes": 17179869184, "source": "nvidia-smi"}
    ]


def test_gpu_probe_tolerates_unavailable_hardware(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("edge_triage.health.subprocess.run", missing)
    assert nvidia_devices() == []
