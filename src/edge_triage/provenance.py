"""Validated registry; planned source relationships do not imply implemented methods."""

from pathlib import Path
from typing import Literal

import yaml

from edge_triage.contracts import Contract, Identifier


class ProvenanceRecord(Contract):
    id: Identifier
    title: str
    authors: tuple[str, ...]
    year: int
    venue: str
    doi_or_arxiv: str | None
    url: str
    component: str
    adoption_level: Literal["reproduced", "adapted", "inspired", "infrastructure"]
    method_borrowed: str
    deviations: str
    excluded_scope: str
    code_url: str
    license_status: str
    verification_notes: str


def load_provenance(path: Path) -> tuple[ProvenanceRecord, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("provenance registry must be a list")
    records = tuple(ProvenanceRecord.model_validate(record) for record in raw)
    if len({record.id for record in records}) != len(records):
        raise ValueError("duplicate provenance IDs")
    return records
