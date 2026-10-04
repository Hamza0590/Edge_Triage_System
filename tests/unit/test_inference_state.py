import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from edge_triage.contracts import Candidate, ModelTier
from edge_triage.models.cnn import ScalableRealBogusCNN, load_spec
from edge_triage.models.inference_state import InferenceStateGuard
from edge_triage.models.registry import RealModelAdapter
from edge_triage.runtime.interfaces import CandidateInput
from edge_triage.uncertainty.sampling import SamplingError, sample_probabilities

ROOT = Path(__file__).resolve().parents[2]


def test_readers_overlap_writer_waits_and_nested_cleanup():
    guard = InferenceStateGuard()
    barrier = threading.Barrier(3)
    release = threading.Event()
    written = threading.Event()

    def reader():
        with guard.read(), guard.read():
            barrier.wait(timeout=5)
            assert release.wait(5)

    def writer():
        with guard.write(), guard.write(), guard.read():
            written.set()

    with ThreadPoolExecutor(3) as pool:
        readers = [pool.submit(reader) for _ in range(2)]
        barrier.wait(timeout=5)
        future = pool.submit(writer)
        try:
            assert not written.wait(0.03)
        finally:
            release.set()
        for result in readers:
            result.result(timeout=5)
        future.result(timeout=5)
        assert written.is_set()
    with guard.read(), pytest.raises(RuntimeError, match="read-to-write"):
        with guard.write():
            pass
    with pytest.raises(ValueError), guard.write():
        raise ValueError("cleanup")
    with guard.read():
        pass


@pytest.mark.parametrize("fail", [False, True])
def test_deterministic_forward_waits_for_mc_state_and_rng_restore(fail):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng_before_test = torch.get_rng_state()
    try:
        model = ScalableRealBogusCNN(load_spec(ROOT / "configs/models/tiny_v1.yaml"))
        adapter = RealModelAdapter(model, "synthetic")
        item = CandidateInput(
            Candidate(
                run_id="r", trace_id="t", candidate_id="c", dataset_index=0, arrival_monotonic_ns=0
            ),
            torch.ones(3, 30, 30),
            0,
            120000,
        )
        baseline = adapter.predict(item, ModelTier.TINY)
        entered, release, prediction_started = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        states = [m.training for m in model.modules()]
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        rng = torch.get_rng_state()

        def hook(module, inputs):
            if any(isinstance(m, torch.nn.Dropout) and m.training for m in module.modules()):
                entered.set()
                assert release.wait(5)
                if fail:
                    raise RuntimeError("injected sampling failure")

        def predict():
            prediction_started.set()
            return adapter.predict(item, ModelTier.TINY)

        hook_handle = model.register_forward_pre_hook(hook)
        try:
            with ThreadPoolExecutor(2) as pool:
                sampling = pool.submit(sample_probabilities, model, item.image, 5, seed=27)
                assert entered.wait(5)
                deterministic = pool.submit(predict)
                try:
                    assert prediction_started.wait(5)
                    assert not deterministic.done()
                finally:
                    release.set()
                if fail:
                    with pytest.raises(SamplingError):
                        sampling.result(timeout=5)
                else:
                    assert len(sampling.result(timeout=5)[0]) == 5
                result = deterministic.result(timeout=5)
                assert result.logit == baseline.logit and result.p_real == baseline.p_real
        finally:
            release.set()
            hook_handle.remove()
        assert [m.training for m in model.modules()] == states
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())
    finally:
        torch.set_rng_state(rng_before_test)
        torch.set_num_threads(threads)
