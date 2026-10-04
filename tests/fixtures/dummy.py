import hashlib
import time

from edge_triage.contracts import ModelTier, PredictionResult, p_real_from_logit
from edge_triage.runtime.interfaces import CandidateInput
from edge_triage.runtime.reference import SMOKE_PURPOSE


class DummyModelAdapter:
    """Hash-based fake logits, restricted to tests; ignores pixels and labels."""

    def predict(self, item: CandidateInput, tier: ModelTier) -> PredictionResult:
        started = time.monotonic_ns()
        digest = hashlib.sha256(f"{item.candidate.candidate_id}:{tier.value}".encode()).digest()
        logit = (int.from_bytes(digest[:4], "big") / (2**32 - 1) - 0.5) * 8
        return PredictionResult(
            candidate_id=item.candidate.candidate_id,
            tier=tier,
            checkpoint_id=f"{SMOKE_PURPOSE}:dummy-{tier.value}-v1",
            logit=logit,
            p_real=p_real_from_logit(logit),
            inference_ms=(time.monotonic_ns() - started) / 1e6,
        )
