"""write_finding — idempotent in-memory collector for confirmed findings.

The agent calls ``write_finding`` to commit a finding to the audit report. Calls
are idempotent on ``(field, category)``: a second call updates the existing
finding rather than creating a duplicate. This matches the spec contract: tool
returns the existing UUID instead of generating a new one when the same key is
revisited.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from queryargus.models.finding import Finding, FindingSeverity

logger = logging.getLogger(__name__)


class FindingsCollector:
    """Holds the set of findings for one audit run."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], Finding] = {}

    def write(
        self,
        *,
        field: str,
        category: str,
        severity: FindingSeverity,
        description: str,
        hypothesis: str,
        evidence_query: str,
        affected_count: int,
        affected_pct: float,
        sample_values: list[Any] | None = None,
        confirmed: bool = True,
    ) -> UUID:
        """Add or update a finding. Returns the finding's UUID (stable across updates)."""
        key = (field, category)
        existing = self._by_key.get(key)
        if existing is not None:
            updated = existing.model_copy(
                update={
                    "severity": severity,
                    "description": description,
                    "hypothesis": hypothesis,
                    "evidence_query": evidence_query,
                    "affected_count": affected_count,
                    "affected_pct": affected_pct,
                    "sample_values": list(sample_values or []),
                    "confirmed": confirmed,
                }
            )
            self._by_key[key] = updated
            logger.info("write_finding updated field=%s category=%s id=%s", field, category, updated.id)
            return updated.id

        finding = Finding(
            field=field,
            category=category,
            severity=severity,
            description=description,
            hypothesis=hypothesis,
            evidence_query=evidence_query,
            affected_count=affected_count,
            affected_pct=affected_pct,
            sample_values=list(sample_values or []),
            confirmed=confirmed,
        )
        self._by_key[key] = finding
        logger.info("write_finding created field=%s category=%s id=%s", field, category, finding.id)
        return finding.id

    def all(self) -> list[Finding]:
        return list(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._by_key
