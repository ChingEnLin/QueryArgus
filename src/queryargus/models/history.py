"""HistoricalContext — cross-run memory the agent sees in every prompt.

Loaded from Postgres at run start (when ``--postgres-url`` is provided) and
attached to ``AgentState.history``. The state's ``summarize()`` renders this
into the user prompt so the agent can confirm-not-rediscover persistent
findings, avoid dismissed-pattern repeats, and prioritise unexplored fields.

This module deliberately holds pure data — no SQL, no IO. The store builds it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

_STABILITY_RANGE = 0.15  # max-min of affected_pct allowed to call the finding "stable"


@dataclass(frozen=True)
class FindingHistory:
    """How a single ``(field, category)`` has behaved across recent runs.

    Lists are ordered most-recent first.
    """

    field: str
    category: str
    runs_considered: int
    runs_seen: int
    severity_history: list[str]
    affected_pct_history: list[float]
    last_seen_run_at: datetime

    @property
    def is_persistent(self) -> bool:
        """True if this finding has appeared in 2+ of the loaded runs."""
        return self.runs_seen >= 2

    @property
    def is_stable(self) -> bool:
        """True if affected_pct varies by less than ``_STABILITY_RANGE`` across runs."""
        if len(self.affected_pct_history) < 2:
            return False
        return (max(self.affected_pct_history) - min(self.affected_pct_history)) <= _STABILITY_RANGE

    @property
    def last_severity(self) -> str:
        return self.severity_history[0] if self.severity_history else "unknown"

    @property
    def affected_pct_range(self) -> tuple[float, float]:
        if not self.affected_pct_history:
            return (0.0, 0.0)
        return (min(self.affected_pct_history), max(self.affected_pct_history))


@dataclass(frozen=True)
class HistoricalContext:
    """Aggregate view of recent audit runs for one ``(collection, database)``."""

    runs_considered: int
    last_run_at: datetime | None
    finding_histories: list[FindingHistory] = field(default_factory=list)
    dismissed_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.runs_considered == 0

    @property
    def persistent_findings(self) -> list[FindingHistory]:
        return [h for h in self.finding_histories if h.is_persistent]

    @property
    def one_off_findings(self) -> list[FindingHistory]:
        return [h for h in self.finding_histories if not h.is_persistent]

    def render(self, *, max_findings: int = 40) -> str:
        """Render to a compact prompt block. Empty string when no history loaded."""
        if self.is_empty:
            return ""

        lines: list[str] = [
            f"HISTORICAL CONTEXT (from last {self.runs_considered} audits, "
            f"most recent {self.last_run_at.isoformat() if self.last_run_at else 'unknown'}):"
        ]

        persistent = self.persistent_findings[:max_findings]
        if persistent:
            lines.append(
                "\nPERSISTENT FINDINGS (seen in 2+ runs — confirm via a single targeted query"
                " and re-commit; don't rediscover):"
            )
            for h in persistent:
                lo, hi = h.affected_pct_range
                stability = "stable" if h.is_stable else "DRIFTING"
                lines.append(
                    f"  - {h.field} / {h.category}: {h.runs_seen}/{h.runs_considered} runs, "
                    f"affected_pct {lo:.3f}-{hi:.3f} ({stability}), last severity={h.last_severity}"
                )

        one_off = self.one_off_findings[:max_findings]
        if one_off:
            lines.append(
                "\nONE-OFF FINDINGS (seen once — verify whether they persist or were sample noise):"
            )
            for h in one_off:
                lines.append(
                    f"  - {h.field} / {h.category}: 1/{h.runs_considered} runs, "
                    f"affected_pct={h.affected_pct_history[0]:.3f}, last severity={h.last_severity}"
                )

        if self.dismissed_pairs:
            lines.append(
                "\nDISMISSED PATTERNS (rejected by evaluators in prior runs — only re-propose"
                " with stronger evidence than the previous attempt):"
            )
            for f_path, cat in self.dismissed_pairs[:max_findings]:
                lines.append(f"  - {f_path} / {cat}")

        return "\n".join(lines)


def empty_history() -> HistoricalContext:
    return HistoricalContext(runs_considered=0, last_run_at=None)
