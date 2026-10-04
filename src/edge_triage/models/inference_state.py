"""Coordinate package inference with temporary dropout/RNG mutations.

Deterministic readers may overlap. MC sampling is a process-wide exclusive writer
because diagnostic sampling temporarily replaces process-global Torch RNG state.
External training or direct model calls must not overlap a serving session.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class InferenceStateGuard:
    """Writer-preferring guard with reentrant writes; no read-to-write upgrade."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._reader_threads: dict[int, int] = {}
        self._waiting_writers = 0
        self._writer: int | None = None
        self._depth = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        identity = threading.get_ident()
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._writer == identity
                    or identity in self._reader_threads
                    or (self._writer is None and self._waiting_writers == 0)
                )
            )
            self._readers += 1
            self._reader_threads[identity] = self._reader_threads.get(identity, 0) + 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                self._reader_threads[identity] -= 1
                if self._reader_threads[identity] == 0:
                    del self._reader_threads[identity]
                self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        identity = threading.get_ident()
        with self._condition:
            if self._writer != identity:
                if identity in self._reader_threads:
                    raise RuntimeError("inference guard does not support read-to-write upgrade")
                self._waiting_writers += 1
                try:
                    self._condition.wait_for(lambda: self._writer is None and self._readers == 0)
                    self._writer = identity
                finally:
                    self._waiting_writers -= 1
            self._depth += 1
        try:
            yield
        finally:
            with self._condition:
                self._depth -= 1
                if self._depth == 0:
                    self._writer = None
                    self._condition.notify_all()


INFERENCE_STATE = InferenceStateGuard()
