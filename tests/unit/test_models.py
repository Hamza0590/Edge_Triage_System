from pathlib import Path

import numpy as np
import pytest
import torch

from edge_triage.contracts import Candidate, ModelTier, p_real_from_logit
from edge_triage.models.cnn import ScalableRealBogusCNN, load_spec
from edge_triage.models.registry import RealModelAdapter
from edge_triage.models.training import binary_metrics, overfit_diagnostic, seed_everything
from edge_triage.runtime.interfaces import CandidateInput

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("tier", list(ModelTier))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_backward_finite_deterministic_and_repeatable(tier, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    spec = load_spec(ROOT / f"configs/models/{tier.value}_v1.yaml")
    seed_everything(123)
    model = ScalableRealBogusCNN(spec)
    seed_everything(123)
    duplicate = ScalableRealBogusCNN(spec)
    assert all(torch.equal(v, duplicate.state_dict()[k]) for k, v in model.state_dict().items())
    model.to(device).train()
    x = torch.randn(4, 3, 30, 30, device=device)
    logits = model(x)
    assert logits.shape == (4,) and torch.isfinite(logits).all()
    torch.nn.BCEWithLogitsLoss()(
        logits, torch.tensor([0.0, 1.0, 1.0, 0.0], device=device)
    ).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(x), model(x))
    assert not any(isinstance(layer, torch.nn.Sigmoid) for layer in model.modules())
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.zeros(1, 3, 20, 20, device=device))


def test_parameter_and_exact_mac_counts_increase():
    models = [
        ScalableRealBogusCNN(load_spec(ROOT / f"configs/models/{t.value}_v1.yaml"))
        for t in ModelTier
    ]
    assert [m.parameter_count for m in models] == [24649, 97681, 388897]
    assert [m.operation_counts()["conv_linear_macs"] for m in models] == [697888, 2402624, 8832640]


@pytest.mark.parametrize("tier", list(ModelTier))
def test_debug_overfit(tier):
    spec = load_spec(ROOT / f"configs/models/{tier.value}_v1.yaml")
    seed_everything(10)
    images = torch.randn(8, 3, 30, 30)
    labels = torch.tensor([0, 1] * 4)
    result = overfit_diagnostic(spec, images, labels, torch.device("cpu"), 10, steps=70)
    assert result["passed"] and result["finite_gradients"]


def test_average_precision_ties_and_single_sigmoid():
    assert binary_metrics([0, 1, 0, 1], [0, 0, 0, 0])["average_precision"] == 0.5
    assert binary_metrics([0, 1, 0, 1], [-2, 2, -1, 1])["average_precision"] == 1
    assert binary_metrics([1, 0], [-1, 1])["average_precision"] == 0.5
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0, np.nan])
    spec = load_spec(ROOT / "configs/models/tiny_v1.yaml")
    model = ScalableRealBogusCNN(spec).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.classifier[-1].bias.fill_(-2)
    candidate = Candidate(
        run_id="test",
        trace_id="trace",
        candidate_id="sample",
        dataset_index=0,
        arrival_monotonic_ns=0,
    )
    item = CandidateInput(candidate, torch.zeros(3, 30, 30), 0, 120000)
    prediction = RealModelAdapter(model, "fixture").predict(item, ModelTier.TINY)
    assert prediction.logit == -2
    assert prediction.p_real == p_real_from_logit(-2)
    assert prediction.p_real < 0.5
