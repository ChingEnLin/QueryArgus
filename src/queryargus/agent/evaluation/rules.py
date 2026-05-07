"""Deterministic rule-based evaluators.

Each rule is a small function that returns one ``EvaluationResult``. The
evaluator runs every rule, picks the worst verdict, and concatenates the rule
reasons. ``FAIL`` short-circuits no further rules — but every rule still runs
so we get a full critique back to the planner. This is deliberate: cheap rules,
verbose feedback.

The three rule lists are **locked v1**:

- ``ACTION_RULES`` — gate before tool execution. Catches mechanical waste:
  repeating queries, jumping to a query before sampling, exhausted budget.
- ``FINDING_RULES`` — gate before commit. Catches over-calling: zero evidence,
  miscalibrated severity, missing evidence query.
- ``RUN_RULES`` — gate at conclude. Catches lazy audits: too few fields
  investigated, suspiciously empty findings on a non-trivial collection,
  premature termination.

If a rule looks borderline-useful, it's a WARN. If it's hard-stop wrong, FAIL.
We err on the side of WARN — the planner sees the critique and can self-correct
without the run grinding to a halt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from queryargus.agent.state import AgentState
from queryargus.models.action import AgentAction
from queryargus.models.evaluation import EvaluationResult, EvaluationVerdict
from queryargus.models.finding import Finding, FindingSeverity
from queryargus.models.report import AuditReport


@dataclass(frozen=True)
class Rule[T]:
    """One rule: a name, a description, and a check returning ``None`` for pass."""

    name: str
    description: str
    check: Callable[[T, AgentState], EvaluationResult | None]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _pass(name: str, score: float = 1.0) -> EvaluationResult:
    return EvaluationResult(
        verdict=EvaluationVerdict.PASS,
        score=score,
        reason=f"{name}: ok",
        evaluated_by="rules",
    )


def _warn(name: str, reason: str, score: float = 0.5, critique: str | None = None) -> EvaluationResult:
    return EvaluationResult(
        verdict=EvaluationVerdict.WARN,
        score=score,
        reason=f"{name}: {reason}",
        critique=critique,
        evaluated_by="rules",
    )


def _fail(name: str, reason: str, score: float = 0.0, critique: str | None = None) -> EvaluationResult:
    return EvaluationResult(
        verdict=EvaluationVerdict.FAIL,
        score=score,
        reason=f"{name}: {reason}",
        critique=critique,
        evaluated_by="rules",
    )


# ----------------------------------------------------------------------------
# ACTION RULES — gate before tool execution
# ----------------------------------------------------------------------------

def _check_no_repeat_query(action: AgentAction, state: AgentState) -> EvaluationResult | None:
    if action.action != "run_query":
        return None
    filt = action.action_input.get("filter")
    if isinstance(filt, dict) and state.has_run_query(filt):
        return _fail(
            "no_repeat_query",
            f"filter {filt!r} was already run earlier this session",
            critique="pick a different filter or move on to a new field",
        )
    return None


def _check_sample_before_query(action: AgentAction, state: AgentState) -> EvaluationResult | None:
    if action.action == "schema_sample":
        return None
    if state.schema is None and action.action != "conclude":
        return _fail(
            "sample_before_query",
            f"action {action.action!r} attempted before any schema_sample call",
            critique="call schema_sample first to learn the field shape",
        )
    return None


def _check_budget_warning(action: AgentAction, state: AgentState) -> EvaluationResult | None:
    if action.action == "conclude":
        return None
    if state.remaining_budget < 3 and action.action == "schema_sample":
        # New broad surveys late in the run waste budget.
        return _warn(
            "budget_warning",
            f"only {state.remaining_budget} iterations left — schema_sample is expensive at this stage",
        )
    return None


def _check_action_input_shape(action: AgentAction, _state: AgentState) -> EvaluationResult | None:
    """Catch obvious shape mistakes before the tool errors loudly."""
    if action.action == "run_query":
        if not isinstance(action.action_input.get("filter"), dict):
            return _fail("action_input_shape", "run_query requires action_input.filter to be a dict")
    elif action.action == "get_stats":
        if not isinstance(action.action_input.get("field"), str):
            return _fail("action_input_shape", "get_stats requires action_input.field to be a string")
        if action.action_input.get("operation") not in {"count", "min", "max", "avg", "distinct"}:
            return _fail("action_input_shape", "get_stats.operation must be one of count|min|max|avg|distinct")
    elif action.action == "write_finding":
        required = {"field", "category", "severity", "description", "hypothesis", "evidence_query", "affected_count"}
        missing = required - action.action_input.keys()
        if missing:
            return _fail(
                "action_input_shape",
                f"write_finding missing required keys: {sorted(missing)}",
            )
    return None


ACTION_RULES: list[Rule[AgentAction]] = [
    Rule("no_repeat_query", "Reject identical run_query filters within a session", _check_no_repeat_query),
    Rule("sample_before_query", "Schema must be sampled before any other read tool", _check_sample_before_query),
    Rule("budget_warning", "Warn when budget is nearly exhausted", _check_budget_warning),
    Rule("action_input_shape", "Tool inputs must have the right shape", _check_action_input_shape),
]


# ----------------------------------------------------------------------------
# FINDING RULES — gate before commit
# ----------------------------------------------------------------------------

def _check_evidence_required(finding: Finding, _state: AgentState) -> EvaluationResult | None:
    if finding.affected_count <= 0:
        return _fail(
            "evidence_required",
            "affected_count must be > 0 to commit a finding",
            critique="run a query that produces non-zero matches before writing the finding",
        )
    return None


def _check_evidence_query_present(finding: Finding, _state: AgentState) -> EvaluationResult | None:
    if not finding.evidence_query.strip():
        return _fail(
            "evidence_query_present",
            "evidence_query is empty — finding is not reproducible",
            critique="record the exact filter or aggregation that produced affected_count",
        )
    return None


def _check_severity_calibration_critical(finding: Finding, _state: AgentState) -> EvaluationResult | None:
    # CRITICAL is reserved for clear-cut, broad data corruption.
    if finding.severity == FindingSeverity.CRITICAL and finding.affected_pct < 0.01:
        return _fail(
            "severity_calibration_critical",
            f"severity=CRITICAL but affected_pct={finding.affected_pct:.4f} (<1%)",
            critique="downgrade to HIGH or MEDIUM — CRITICAL implies systemic data corruption",
        )
    return None


def _check_severity_calibration_high(finding: Finding, _state: AgentState) -> EvaluationResult | None:
    if finding.severity == FindingSeverity.HIGH and finding.affected_pct < 0.001:
        return _warn(
            "severity_calibration_high",
            f"severity=HIGH but affected_pct={finding.affected_pct:.4f} (<0.1%)",
        )
    return None


def _check_hypothesis_present(finding: Finding, _state: AgentState) -> EvaluationResult | None:
    if not finding.hypothesis.strip():
        return _warn("hypothesis_present", "hypothesis is empty — finding lacks investigative rationale")
    return None


def _check_description_present(finding: Finding, _state: AgentState) -> EvaluationResult | None:
    if len(finding.description.strip()) < 10:
        return _fail(
            "description_present",
            "description must be at least 10 chars — be specific about the issue",
        )
    return None


FINDING_RULES: list[Rule[Finding]] = [
    Rule("evidence_required", "affected_count must be positive", _check_evidence_required),
    Rule(
        "evidence_query_present",
        "Findings must include the query that confirmed them",
        _check_evidence_query_present,
    ),
    Rule(
        "severity_calibration_critical",
        "CRITICAL requires >=1% affected",
        _check_severity_calibration_critical,
    ),
    Rule(
        "severity_calibration_high",
        "HIGH on <0.1% affected is suspicious",
        _check_severity_calibration_high,
    ),
    Rule("hypothesis_present", "Findings should explain the suspected cause", _check_hypothesis_present),
    Rule("description_present", "Description must be substantive", _check_description_present),
]


# ----------------------------------------------------------------------------
# RUN RULES — gate at conclude
# ----------------------------------------------------------------------------

def _check_minimum_field_coverage(_report: AuditReport, state: AgentState) -> EvaluationResult | None:
    total = len(state.field_paths)
    if total == 0:
        return None  # nothing was sampled — caught elsewhere
    investigated = len(state.fields_investigated | state.fields_concluded)
    if investigated / total < 0.5:
        return _fail(
            "minimum_field_coverage",
            f"only {investigated}/{total} fields investigated (<50%)",
            critique="extend the run or adjust max_iterations",
        )
    return None


def _check_no_findings_on_clean_collection(report: AuditReport, state: AgentState) -> EvaluationResult | None:
    if not report.findings and state.documents_sampled > 1000:
        return _warn(
            "no_findings_on_clean_collection",
            f"zero findings on a {state.documents_sampled}-doc sample — check thresholds",
        )
    return None


def _check_early_termination(_report: AuditReport, state: AgentState) -> EvaluationResult | None:
    if state.iteration < 3:
        return _fail(
            "early_termination",
            f"conclude called after only {state.iteration} iterations",
            critique="audit ran too short — investigate at least a few fields before concluding",
        )
    return None


RUN_RULES: list[Rule[AuditReport]] = [
    Rule("minimum_field_coverage", "At least half of fields should be investigated", _check_minimum_field_coverage),
    Rule(
        "no_findings_on_clean_collection",
        "Zero findings on a large sample is suspicious",
        _check_no_findings_on_clean_collection,
    ),
    Rule("early_termination", "Avoid concluding before iteration 3", _check_early_termination),
]


# ----------------------------------------------------------------------------
# Evaluator implementations
# ----------------------------------------------------------------------------

def _aggregate(results: list[EvaluationResult], gate: str) -> EvaluationResult:
    if not results:
        return EvaluationResult(
            verdict=EvaluationVerdict.PASS,
            score=1.0,
            reason=f"{gate}: no rules triggered",
            evaluated_by="rules",
        )
    rank = {EvaluationVerdict.PASS: 0, EvaluationVerdict.WARN: 1, EvaluationVerdict.FAIL: 2}
    worst_idx = max(range(len(results)), key=lambda i: rank[results[i].verdict])
    worst_one = results[worst_idx]
    # Aggregate critique across non-pass rules so the planner sees them all.
    critiques = [r.critique for r in results if r.verdict != EvaluationVerdict.PASS and r.critique]
    return EvaluationResult(
        verdict=worst_one.verdict,
        score=min((r.score for r in results), default=1.0),
        reason="; ".join(r.reason for r in results if r.verdict != EvaluationVerdict.PASS) or worst_one.reason,
        critique=" | ".join(critiques) if critiques else worst_one.critique,
        evaluated_by="rules",
    )


@dataclass
class RulesActionEvaluator:
    """Run all ACTION_RULES and return the worst verdict."""

    rules: list[Rule[AgentAction]] = field(default_factory=lambda: list(ACTION_RULES))

    def evaluate(self, action: AgentAction, state: AgentState) -> EvaluationResult:
        results = [r for rule in self.rules if (r := rule.check(action, state)) is not None]
        return _aggregate(results, "action")


@dataclass
class RulesFindingEvaluator:
    rules: list[Rule[Finding]] = field(default_factory=lambda: list(FINDING_RULES))

    def evaluate(self, finding: Finding, state: AgentState) -> EvaluationResult:
        results = [r for rule in self.rules if (r := rule.check(finding, state)) is not None]
        return _aggregate(results, "finding")


@dataclass
class RulesRunEvaluator:
    rules: list[Rule[AuditReport]] = field(default_factory=lambda: list(RUN_RULES))

    def evaluate(self, report: AuditReport, state: AgentState) -> EvaluationResult:
        results = [r for rule in self.rules if (r := rule.check(report, state)) is not None]
        return _aggregate(results, "run")
