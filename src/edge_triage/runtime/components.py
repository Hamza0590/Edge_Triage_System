"""Concrete data source, FIFO queue, sequential executor and typed logging bridge."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator

from edge_triage.config import AppConfig
from edge_triage.contracts import Candidate, CandidateEvent, ModelTier, PredictionResult
from edge_triage.data.loaders import MeerlichtDataset
from edge_triage.event_logging import EventLogger
from edge_triage.runtime.interfaces import CandidateInput, ModelAdapter


class DatasetSource:
    def __init__(self, config: AppConfig, run_id: str, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.dataset = MeerlichtDataset(config, config.pipeline.partition, run_id=run_id)
        self.limit = min(limit, len(self.dataset))

    def __iter__(self) -> Iterator[CandidateInput]:
        for index in range(self.limit):
            sample = self.dataset[index]
            candidate = Candidate.model_validate(
                {**sample[0].model_dump(), "arrival_monotonic_ns": time.monotonic_ns()}
            )
            yield CandidateInput(candidate, sample[1], sample[2], 100 * 100 * 3 * 4)


class FifoCandidateQueue:
    def __init__(self) -> None:
        self._items: deque[CandidateInput] = deque()

    def put(self, item: CandidateInput) -> None:
        self._items.append(item)

    def get(self) -> CandidateInput:
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)


class SequentialExecutor:
    def __init__(self) -> None:
        self.closed = False
        self.active = False

    def execute(
        self, model: ModelAdapter, item: CandidateInput, tier: ModelTier
    ) -> PredictionResult:
        if self.closed or self.active:
            raise RuntimeError("sequential executor closed or already active")
        self.active = True
        try:
            return model.predict(item, tier)
        finally:
            self.active = False

    def close(self) -> None:
        self.closed = True


class LoggerEventSink:
    def __init__(self, logger: EventLogger) -> None:
        self.logger = logger

    def emit(self, event: CandidateEvent) -> None:
        self.logger.append(event)
