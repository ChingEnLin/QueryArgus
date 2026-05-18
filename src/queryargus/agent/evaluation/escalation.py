"""EscalationFindingEvaluator — Arm B's self-escalation gate.

Wraps an inner finding evaluator (typically ``RulesFindingEvaluator``) and
reroutes findings based on the agent's self-reported ``confidence``:

==========================================  ===========================================
``finding.confidence`` band                 Outcome
==========================================  ===========================================
``< escalation_low_threshold``              FAIL — evaluator_by ``escalation:dropped``.
                                            ``rejected_finding_policy`` decides whether
                                            the finding is silently dropped, logged as
                                            dismissed, or demoted.
``in [low, high)``                          PASS — evaluator_by ``escalation:pending``.
                                            ``_commit_finding`` reads this sentinel and
                                            sets the finding's ``status`` to
                                            ``pending_review`` (excluded from the
                                            user-facing count until a human resolves).
``>= escalation_high_threshold``            Defer to the inner verdict (publish path).
``None``                                    Defer to the inner verdict. The agent did
                                            not produce a self-assessment for this
                                            proposal.
==========================================  ===========================================

If the inner evaluator returns ``FAIL`` the escalation gate respects it — a
high-confidence-but-broken finding still gets dismissed.

Adapted from jason8745/self-escalate-agent's ``Escalator`` pattern:
https://github.com/jason8745/self-escalate-agent/blob/master/agent/escalator.py
"""

from __future__ import annotations

from dataclasses import dataclass

from queryargus.agent.evaluation.base import FindingEvaluator
from queryargus.agent.state import AgentState
from queryargus.models.evaluation import EvaluationResult, EvaluationVerdict
from queryargus.models.finding import Finding

# Sentinel ``evaluated_by`` values the agent loop reads to choose a status.
ESCALATION_PENDING_SENTINEL = "escalation:pending"
ESCALATION_DROPPED_SENTINEL = "escalation:dropped"


@dataclass
class EscalationFindingEvaluator:
    """Wraps an inner finding evaluator with a confidence-based gate."""

    inner: FindingEvaluator | None
    high_threshold: float = 0.85
    low_threshold: float = 0.40
    cap: int = 10

    def __post_init__(self) -> None:
        if self.low_threshold > self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be <= high_threshold "
                f"({self.high_threshold})"
            )

    def evaluate(self, finding: Finding, state: AgentState) -> EvaluationResult:
        # Run the inner first so its rules-based sanity checks still apply.
        # A high-confidence-but-malformed finding (e.g. duplicate query, no
        # evidence) is still dismissed — confidence does not override rules.
        inner_result: EvaluationResult | None = None
        if self.inner is not None:
            inner_result = self.inner.evaluate(finding, state)
            if inner_result.verdict == EvaluationVerdict.FAIL:
                return inner_result

        confidence = finding.confidence

        if confidence is None:
            # No self-assessment — fall through. The plan prompt change asks
            # the agent for confidence on every write_finding, so this branch
            # is mostly a safety net for legacy callers.
            return inner_result or EvaluationResult(
                verdict=EvaluationVerdict.PASS,
                score=1.0,
                reason="no self-confidence provided; defaulting to publish",
                evaluated_by="escalation",
            )

        if confidence >= self.high_threshold:
            return inner_result or EvaluationResult(
                verdict=EvaluationVerdict.PASS,
                score=confidence,
                reason=(
                    f"confidence {confidence:.2f} >= publish threshold "
                    f"{self.high_threshold:.2f}"
                ),
                evaluated_by="escalation",
            )

        if confidence < self.low_threshold:
            return EvaluationResult(
                verdict=EvaluationVerdict.FAIL,
                score=confidence,
                reason=(
                    f"confidence {confidence:.2f} < drop threshold "
                    f"{self.low_threshold:.2f}; dropping without escalation"
                ),
                evaluated_by=ESCALATION_DROPPED_SENTINEL,
            )

        # Mid-confidence band — escalate for human review IF we have headroom.
        pending = sum(1 for f in state.findings.all() if f.status == "pending_review")
        if pending >= self.cap:
            return EvaluationResult(
                verdict=EvaluationVerdict.FAIL,
                score=confidence,
                reason=(
                    f"confidence {confidence:.2f} in escalation band but cap "
                    f"({self.cap}) already reached; dropping to avoid reviewer spam"
                ),
                evaluated_by=ESCALATION_DROPPED_SENTINEL,
            )

        return EvaluationResult(
            verdict=EvaluationVerdict.PASS,
            score=confidence,
            reason=(
                f"confidence {confidence:.2f} in escalation band "
                f"[{self.low_threshold:.2f}, {self.high_threshold:.2f}); "
                "queued for human review"
            ),
            evaluated_by=ESCALATION_PENDING_SENTINEL,
        )
