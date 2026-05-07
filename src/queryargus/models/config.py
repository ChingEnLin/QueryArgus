"""Configuration models — ArgusConfig and EvaluatorConfig with prebuilt profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from queryargus.models.finding import FindingSeverity

EvaluatorStrategy = Literal["none", "rules", "self", "judge", "composite"]
RejectedFindingPolicy = Literal["drop", "log_only", "demote_severity"]
RunFailPolicy = Literal["continue", "warn_only", "abort"]
OutputFormat = Literal["json", "text", "silent"]
LLMProvider = Literal["gemini"]  # v1: Gemini-only — see PLAN.md §1 D3


class EvaluatorConfig(BaseModel):
    """Per-gate evaluation configuration. See spec §5.5."""

    model_config = ConfigDict(extra="forbid")

    action_evaluator: EvaluatorStrategy = "rules"
    finding_evaluator: EvaluatorStrategy = "rules"
    run_evaluator: EvaluatorStrategy = "rules"

    judge_provider: Literal["gemini", "openai", "anthropic"] | None = None
    judge_model: str | None = None

    action_pass_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    finding_pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    run_pass_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    rejected_finding_policy: RejectedFindingPolicy = "log_only"
    run_fail_policy: RunFailPolicy = "continue"


# Prebuilt profiles from spec §5.5
PROFILE_FAST = EvaluatorConfig(
    action_evaluator="rules",
    finding_evaluator="rules",
    run_evaluator="rules",
)

PROFILE_BALANCED = EvaluatorConfig(
    action_evaluator="rules",
    finding_evaluator="composite",
    run_evaluator="self",
)

PROFILE_THOROUGH = EvaluatorConfig(
    action_evaluator="rules",
    finding_evaluator="composite",
    run_evaluator="judge",
    judge_provider="openai",
    judge_model="gpt-4o",
)


class ArgusConfig(BaseModel):
    """Top-level QueryArgus configuration."""

    model_config = ConfigDict(extra="forbid")

    # Sampling
    sample_size: int = Field(default=200, gt=0)
    max_iterations: int = Field(default=20, gt=0)

    # LLM (Gemini-only in v1)
    llm_provider: LLMProvider = "gemini"
    llm_model: str = "gemini-2.5-flash"

    # Evaluation
    evaluation: EvaluatorConfig = Field(default_factory=EvaluatorConfig)

    # Storage (optional)
    postgres_url: str | None = None

    # Output
    output_format: OutputFormat = "text"
    min_severity: FindingSeverity = FindingSeverity.LOW

    # Thresholds — fractions in [0.0, 1.0], not percentages
    null_rate_warning_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    null_rate_critical_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    type_consistency_threshold: float = Field(default=0.98, ge=0.0, le=1.0)
