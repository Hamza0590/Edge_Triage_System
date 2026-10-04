import itertools
import math

import pytest
import torch
from torch import nn

from edge_triage.contracts import ModelTier, PredictionResult, p_real_from_logit
from edge_triage.uncertainty.artifacts import TierUncertainty, UncertaintyArtifact, load_uncertainty
from edge_triage.uncertainty.policies import CascadeEscalationPolicy
from edge_triage.uncertainty.sampling import dropout_inference, sample_probabilities, summarize


def prediction(tier=ModelTier.TINY, p=0.5):
    z = math.log(p / (1 - p))
    return PredictionResult(
        candidate_id="c1",
        tier=tier,
        checkpoint_id="test",
        logit=z,
        p_real=p_real_from_logit(z),
        inference_ms=0,
    )


def network():
    return nn.Sequential(
        nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1)
    )


@pytest.mark.parametrize("fail", [False, True])
def test_dropout_only_and_exact_state_restoration(fail):
    model = network().train()
    model[2].eval()  # Mixed original state must survive.
    before = [m.training for m in model.modules()]
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    try:
        with dropout_inference(model):
            assert model[3].training and not model[1].training and not model.training
            assert not torch.is_grad_enabled()
            model(torch.ones(1, 8))
            if fail:
                raise RuntimeError("injected")
    except RuntimeError:
        assert fail
    assert [m.training for m in model.modules()] == before
    assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())


def test_diagnostic_seed_reproducible_and_rng_restored():
    model = network().eval()
    rng = torch.random.get_rng_state().clone()
    a, cost = sample_probabilities(model, torch.ones(8), 20, seed=42)
    b, _ = sample_probabilities(model, torch.ones(8), 20, seed=42)
    c, _ = sample_probabilities(model, torch.ones(8), 20, seed=43)
    assert a == b and a != c and cost > 0 and len(set(a)) > 1
    assert torch.equal(rng, torch.random.get_rng_state())
    sample_probabilities(model, torch.ones(8), 5)
    assert not torch.equal(rng, torch.random.get_rng_state())


def test_sampling_exception_restores_rng_and_model():
    class Broken(nn.Linear):
        def forward(self, x):
            raise RuntimeError("forward failure")

    model = nn.Sequential(nn.Dropout(0.5), Broken(8, 1)).train()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        sample_probabilities(model, torch.ones(8), 5, seed=1)
    assert model.training and model[0].training
    assert torch.equal(rng, torch.random.get_rng_state())


def test_partial_sampling_failure_retains_actual_attempts():
    from edge_triage.uncertainty.sampling import SamplingError

    class ThirdFails(nn.Linear):
        calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls == 3:
                raise ValueError("third pass failed")
            return super().forward(x)

    model = nn.Sequential(nn.Dropout(0.5), ThirdFails(8, 1)).eval()
    with pytest.raises(SamplingError) as caught:
        sample_probabilities(model, torch.ones(8), 5, seed=42)
    assert caught.value.attempted == 3 and caught.value.completed == 2
    assert caught.value.duration_ms > 0 and isinstance(caught.value.__cause__, ValueError)
    assert not model.training and not model[0].training


def test_known_metrics_and_invalid_inputs():
    result = summarize(prediction(), [0, 1, 0, 1], 2)
    assert result.mean_p_real == 0.5 and result.variance == 0.25
    assert result.standard_deviation == result.variation_ratio == 0.5
    assert result.predictive_entropy == pytest.approx(math.log(2))
    assert result.additional_forward_passes == 4
    assert summarize(prediction(), [1, 1], 0).predictive_entropy == 0
    for values in ([float("nan"), 1], [1.1, 0], [0.5]):
        with pytest.raises(ValueError):
            summarize(prediction(), values, 0)
    with pytest.raises(ValueError):
        sample_probabilities(network(), torch.ones(8), 1)
    with pytest.raises(ValueError):
        sample_probabilities(nn.Linear(8, 1), torch.ones(8), 5)


PORTFOLIOS = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]


@pytest.mark.parametrize("portfolio", PORTFOLIOS)
def test_enabled_only_paths_final_fallback(portfolio):
    policy = CascadeEscalationPolicy(portfolio)
    visited = []
    for tier in portfolio:
        visited.append(tier)
        pred = prediction(tier)
        uq = summarize(pred, [0, 1], 1).model_copy(update={"unresolved": True})
        decision = policy.decide_path(pred, uq, tuple(visited))
        assert decision.visited_tiers == tuple(visited)
        if tier != portfolio[-1]:
            assert decision.to_tier == portfolio[len(visited)] and decision.escalate
        else:
            assert not decision.escalate and decision.conservative_fallback


def test_cycle_maximum_and_identity_rejection():
    pred = prediction()
    uq = summarize(pred, [0, 1], 1).model_copy(update={"unresolved": True})
    policy = CascadeEscalationPolicy(tuple(ModelTier), maximum=0)
    assert policy.decide_path(pred, uq, (ModelTier.TINY,)).conservative_fallback
    for visited in ((), (ModelTier.TINY, ModelTier.TINY), (ModelTier.LARGE, ModelTier.TINY)):
        with pytest.raises(ValueError):
            policy.decide_path(pred, uq, visited)
    with pytest.raises(ValueError):
        policy.decide_path(pred, uq.model_copy(update={"candidate_id": "wrong"}), (ModelTier.TINY,))


def artifact(passes=5, lower=0.4, upper=0.6):
    return UncertaintyArtifact(
        artifact_id="test-uq",
        split_sha256="a" * 64,
        preprocessing_sha256="b" * 64,
        family_sha256="c" * 64,
        calibration_index_sha256="d" * 64,
        objective="test only",
        passes=passes,
        tiers=(
            TierUncertainty(
                tier=ModelTier.TINY,
                checkpoint_id="test",
                calibration_artifact_id="cal",
                calibration_sha256="e" * 64,
                probability_mode="calibrated",
                temperature=2,
                lower=lower,
                upper=upper,
                std_threshold=0.1,
            ),
        ),
    )


def test_artifact_integrity_and_validation_only(tmp_path):
    from edge_triage.health import hash_file

    path = tmp_path / "uq.json"
    path.write_text(artifact().model_dump_json())
    assert load_uncertainty(path, hash_file(path)) == artifact()
    with pytest.raises(ValueError):
        load_uncertainty(path, "0" * 64)
    for change in ({"fit_partition": "test"}, {"passes": 3}):
        with pytest.raises(ValueError):
            UncertaintyArtifact.model_validate({**artifact().model_dump(), **change})
    with pytest.raises(ValueError):
        artifact(lower=0.9, upper=0.1)


@pytest.mark.parametrize("p,expected", [(0.3999, 0), (0.4, 5), (0.6, 5), (0.6001, 0)])
def test_inclusive_ambiguous_boundaries_and_no_label_input(monkeypatch, p, expected):
    from edge_triage.uncertainty import policies

    policy = object.__new__(policies.AmbiguousOnlyMCDropoutPolicy)
    policy.artifact, policy.seed = artifact(), 1
    policy.tiers = {ModelTier.TINY: policy.artifact.tiers[0]}
    from types import SimpleNamespace

    policy.models = {ModelTier.TINY: SimpleNamespace(base=SimpleNamespace(model=None))}
    calls = []

    def sample(model, image, passes, **kwargs):
        calls.append(kwargs)
        assert kwargs["temperature"] == 2
        return [p] * passes, 0.1

    monkeypatch.setattr(policies, "sample_probabilities", sample)
    pred = prediction().model_copy(
        update={"calibrated_p_real": p, "calibration_artifact_id": "cal"}
    )
    result = policy.evaluate_image(pred, torch.ones(8))
    assert result.additional_forward_passes == expected
    assert bool(calls) == bool(expected)
    assert result.mean_p_real == pytest.approx(p)
    with pytest.raises(ValueError):
        policy.evaluate(pred)


def test_observed_uq_failure_cleans_active_counters():
    from edge_triage.monitoring.estimators import RuntimeCounters
    from edge_triage.runtime.streaming import ObservedUncertainty

    class Broken:
        def evaluate(self, prediction):
            raise RuntimeError("uq failure")

    counters = RuntimeCounters()
    with pytest.raises(RuntimeError):
        ObservedUncertainty(Broken(), counters).evaluate_image(prediction(), torch.ones(8))
    assert counters.snapshot()["active_jobs_total"] == 0
    assert counters.snapshot()["latency_samples"] == 0
