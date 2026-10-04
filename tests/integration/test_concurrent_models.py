"""Read-only frozen models and validation inputs; no scientific quality evaluation."""

import asyncio
from pathlib import Path

import pytest
import torch

from edge_triage.config import load_config
from edge_triage.runtime.components import DatasetSource
from edge_triage.runtime.executors import (
    DynamicConcurrencyExecutor,
    FixedConcurrencyExecutor,
    InferenceJob,
    SequentialExecutor,
)
from edge_triage.runtime.factory import assemble
from edge_triage.runtime.pipeline import minimal_snapshot
from edge_triage.scheduling.policies import MemoryContentionAwareAdmissionPolicy, PolicyContext
from edge_triage.scheduling.profiles import scheduler_profiles

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_predictions_equivalent_across_three_backends():
    config = load_config(ROOT / "configs/experiments/module08_proposed_v1.yaml")
    assert config.uncertainty.policy == config.escalation.policy == "none"
    assert config.pipeline.partition == "validation"
    threads = torch.get_num_threads()
    rng = torch.get_rng_state()
    torch.set_num_threads(1)
    parts = assemble(config)
    try:
        items = list(DatasetSource(config, "executor-equivalence", 6))
        context = PolicyContext(
            config.models.tier_portfolio, scheduler_profiles(config), config.scheduler.constraints
        )
        ledger = MemoryContentionAwareAdmissionPolicy(context)

        async def scenario():
            reference = None
            for executor in (
                SequentialExecutor(),
                FixedConcurrencyExecutor(4),
                DynamicConcurrencyExecutor(4, ledger, lambda: minimal_snapshot(0)),
            ):
                try:
                    handles = [
                        await executor.submit(
                            InferenceJob(f"{tier.value}-{i}", item, tier, model, parts.uncertainty)
                        )
                        for tier, model in parts.models.items()
                        for i, item in enumerate(items)
                    ]
                    results = await asyncio.gather(*(h.result() for h in reversed(handles)))
                    assert all(r.status == "completed" for r in results)
                    by_identity = {
                        (r.candidate_id, r.trace_id, r.tier): r.prediction for r in results
                    }
                    assert len(by_identity) == 18
                    if reference is None:
                        reference = by_identity
                    else:
                        assert by_identity.keys() == reference.keys()
                        for identity, prediction in by_identity.items():
                            expected = reference[identity]
                            assert prediction.checkpoint_id == expected.checkpoint_id
                            assert prediction.logit == pytest.approx(expected.logit, abs=1e-5)
                            assert prediction.p_real == pytest.approx(expected.p_real, abs=1e-6)
                            assert prediction.calibrated_p_real == pytest.approx(
                                expected.calibrated_p_real, abs=1e-6
                            )
                            assert (prediction.p_real >= 0.5) == (expected.p_real >= 0.5)
                    assert all(r.additional_forward_passes == 0 for r in results)
                finally:
                    await executor.shutdown()
            assert not ledger.reservations

        asyncio.run(scenario())
    finally:
        parts.executor.close()
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)
