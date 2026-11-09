import itertools

import numpy as np
import pytest

from edge_triage.triage.artifacts import FateOperatingPoint, TierFateDefinition
from edge_triage.triage.fitting import (
    ValidationFateRow,
    bootstrap_fates,
    evaluate_fates,
    fit_thresholds,
)
from tests.unit.test_triage import fit_fixture


@pytest.mark.parametrize("binary", [False, True])
def test_vectorized_search_matches_policy_for_uq_grid(tmp_path, binary):
    rows, args = fit_fixture(tmp_path)
    raw = rows[1].model_dump()
    raw["context"]["uncertainty"].update(
        method="mc_dropout", passes=5, standard_deviation=0.2, variance=0.04
    )
    rows[1] = ValidationFateRow.model_validate(raw)
    result = fit_thresholds(
        rows, **args, transmit_grid=(0.5, 0.9), uncertainty_grid=(0.0, 0.25), binary=binary
    )
    points = [
        FateOperatingPoint(
            max_p_real_for_discard=d, min_p_real_for_transmit=t, max_uncertainty_for_discard=u
        )
        for d, t, u in itertools.product([None, 0.1, 0.2, 0.3, 0.4], [0.5, 0.9], [0.0, 0.25])
    ]
    best = None
    for a, b in itertools.product(points, repeat=2):
        if (a.max_p_real_for_discard or -1) > (b.max_p_real_for_discard or -1):
            continue
        specs = tuple(
            TierFateDefinition.model_validate({**s.model_dump(), "operating_point": p})
            for s, p in zip(args["tiers"], [a, b], strict=True)
        )
        metrics = evaluate_fates(rows, specs, args["accounting"], binary=binary)
        if metrics["survival_recall"] >= 0.75:
            rank = (metrics["bytes_transmitted"], metrics["false_discard_count"])
            best = min(best, rank) if best is not None else rank
    assert (result.metrics["bytes_transmitted"], result.metrics["false_discard_count"]) == best


def test_vectorized_bootstrap_matches_reference_resampling(tmp_path):
    rows, args = fit_fixture(tmp_path)
    result = fit_thresholds(rows, **args)
    fast = bootstrap_fates(rows, result.tiers, args["accounting"], repeats=20)
    rng = np.random.default_rng(20260924)
    samples = {key: [] for key in fast["intervals"]}
    for _ in range(20):
        sample = [rows[int(i)] for i in rng.integers(0, len(rows), len(rows))]
        metrics = evaluate_fates(sample, result.tiers, args["accounting"])
        for key in samples:
            if metrics[key] is not None:
                samples[key].append(metrics[key])
    for key, values in samples.items():
        assert fast["intervals"][key]["defined_replicates"] == len(values)
        assert fast["intervals"][key]["lower"] == pytest.approx(np.quantile(values, 0.025))
        assert fast["intervals"][key]["upper"] == pytest.approx(np.quantile(values, 0.975))
