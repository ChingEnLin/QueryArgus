# Evaluation Layer

QueryArgus introduces structured quality gates into the agent loop to prevent the LLM from hallucinating findings, repeating tool calls, or declaring victory prematurely. There are three gates, each composable from multiple evaluator strategies.

---

## Three Gates

```
Iteration N
  │
  ├── [1] Action Gate ─── before executing any tool
  │         FAIL → inject critique, skip execution, try again
  │         PASS/WARN → execute tool
  │
  ├── [2] Finding Gate ── before committing a write_finding result
  │         FAIL → move to dismissed_findings
  │         PASS → commit to findings
  │
  └── [3] Run Gate ────── before accepting a "conclude" action
            FAIL (policy=continue) → inject critique, loop continues
            FAIL (policy=abort)    → raise error
            PASS/WARN              → build AuditReport, exit loop
```

Each gate is independently configurable through `EvaluatorConfig`:

```python
class EvaluatorConfig(BaseModel):
    action:  Literal["none", "rules", "self", "judge", "composite"] = "rules"
    finding: Literal["none", "rules", "self", "judge", "composite"] = "rules"
    run:     Literal["none", "rules", "self", "judge", "composite"] = "self"

    # Only used when strategy == "judge"
    judge_provider: str = "gemini"
    judge_model:    str = "gemini-2.5-pro"

    run_fail_policy: Literal["continue", "abort"] = "continue"
```

### Built-in Profiles

| Profile | Action | Finding | Run | Use case |
|---------|--------|---------|-----|----------|
| `PROFILE_FAST` | rules | rules | rules | CI pipelines, quick checks |
| `PROFILE_BALANCED` | rules | composite | self | Default — good cost/quality tradeoff |
| `PROFILE_THOROUGH` | rules | composite | judge | Pre-production audits |

---

## Evaluator Strategies

### 1. Rules Evaluator

**File:** `agent/evaluation/rules.py`

Deterministic checks — no LLM call, no cost. Runs all rules in a list; aggregates worst verdict.

**Verdict aggregation:**
- Every rule runs (no short-circuit on pass)
- Worst verdict wins: `FAIL > WARN > PASS`
- Lowest score wins
- All non-PASS reasons joined with `" | "`

#### Action Rules (pre-execution)

| Rule | Verdict on violation |
|------|---------------------|
| `no_repeat_query` | FAIL — identical `run_query` filter already used this session |
| `sample_before_query` | FAIL — `run_query`/`get_stats` proposed before `schema_sample` |
| `budget_warning` | WARN — `schema_sample` proposed when < 3 iterations remain |
| `action_input_shape` | FAIL — malformed tool input (filter not dict, field not string, etc.) |

#### Finding Rules (pre-commit)

| Rule | Verdict on violation |
|------|---------------------|
| `evidence_required` | FAIL — `affected_count == 0` |
| `evidence_query_present` | FAIL — `evidence_query` empty (finding not reproducible) |
| `severity_calibration_critical` | FAIL — CRITICAL severity with `affected_pct < 0.01` (< 1%) |
| `severity_calibration_high` | WARN — HIGH severity with `affected_pct < 0.001` (< 0.1%) |
| `hypothesis_present` | WARN — `hypothesis` field empty |
| `description_present` | FAIL — `description` shorter than 10 characters |

#### Run Rules (post-run)

| Rule | Verdict on violation |
|------|---------------------|
| `minimum_field_coverage` | FAIL — fewer than 50% of schema fields investigated |
| `no_findings_on_clean_collection` | WARN — zero findings after sampling ≥ 1000 documents |
| `early_termination` | FAIL — `conclude` action proposed before iteration 3 |

---

### 2. Self-Evaluator

**File:** `agent/evaluation/self_eval.py`

The same LLM that drives the planner critiques its own proposed finding or the final run. Slower and costlier than rules, but catches semantic issues rules cannot — e.g. "the described evidence doesn't actually support this hypothesis."

**Process:**
1. Serialise the finding/report to JSON
2. Build an eval-specific prompt asking for a critique
3. Call `llm.complete_json()` → raw JSON
4. Parse as `EvaluationResult`
5. Accumulate tokens onto `state.total_usage`

**Scope in v1:** Self-eval is only applied at the Finding and Run gates. Action-gate self-eval would double LLM calls for marginal benefit.

---

### 3. Judge Evaluator

**File:** `agent/evaluation/judge.py`

An independent, typically more capable, LLM reviews the final run report. Because it uses a different model than the planner, it can catch systematic reasoning errors the self-evaluator would share.

**Use:** Run gate only (too expensive for every action or finding).

**Configuration example:**
```python
EvaluatorConfig(
    run="judge",
    judge_provider="gemini",
    judge_model="gemini-2.5-pro",   # Stronger than the planning model
)
```

---

### 4. Composite Evaluator

**File:** `agent/evaluation/composite.py`

Chains multiple evaluators in order, stopping at the first FAIL:

```
Rules → Self → Judge
  │       │      │
FAIL    FAIL   FAIL
 ↓       ↓      ↓
stop    stop   stop
```

**Merge logic:**
- Worst verdict across all completed evaluators wins
- Reasons concatenated with `" | "`
- `evaluated_by = "composite"`

**Typical stack for `PROFILE_BALANCED` finding gate:**
```
CompositeEvaluator([
    RulesFindingEvaluator(),     # fast, cheap — catches obvious issues
    SelfFindingEvaluator(llm),   # semantic validation
])
```

---

## Evaluation Result Models

```python
class EvaluationVerdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

class EvaluationResult(BaseModel):
    verdict:      EvaluationVerdict
    score:        float        # 0.0–1.0
    reason:       str          # Human-readable explanation
    critique:     str | None   # Actionable feedback injected into planner
    evaluated_by: str          # e.g. "rules", "self:gemini-2.5-flash", "judge:gemini-2.5-pro"
```

Every evaluation produces an `EvaluationRecord` stored in `AgentState.evaluation_records` and persisted to Postgres:

```python
class EvaluationRecord(BaseModel):
    id:           UUID
    gate:         Literal["action", "finding", "run"]
    evaluated_by: str
    verdict:      EvaluationVerdict
    score:        float
    reason:       str
    critique:     str | None
    target_id:    UUID | None   # ID of the action/finding/report being evaluated
    iteration:    int | None
    timestamp:    datetime
```

The full evaluation audit trail is embedded in `AuditReport.evaluation_records`.

---

## The Critique Feedback Loop

```
Planner proposes action
        │
   Action Gate → FAIL
        │
   EvaluationResult.critique = "You already ran this filter on field X.
                                 Try narrowing by date range instead."
        │
   state.last_critique = critique
        │
   Next state.summarize() includes:
     "⚠ Last action rejected: <critique>"
        │
   Planner reads critique and proposes different action
```

This closes the loop without an external retry mechanism. The planner is responsible for incorporating the feedback — the harness only guarantees it's visible.

---

## Factory

**File:** `agent/evaluation/factory.py`

Builds the correct evaluator stack from `EvaluatorConfig`, keeping conditional logic out of the loop:

```python
def build_action_evaluator(config: EvaluatorConfig, llm: LLMClient) -> ActionEvaluator:
    match config.action:
        case "none":      return NoopEvaluator()
        case "rules":     return RulesActionEvaluator()
        case "self":      return SelfActionEvaluator(llm)
        case "composite": return CompositeActionEvaluator([
                              RulesActionEvaluator(),
                              SelfActionEvaluator(llm),
                          ])
        case "judge":     return JudgeActionEvaluator(build_judge_llm(config))
```

`ArgusAgent.from_config(config, llm)` calls all three factory functions and stores the resulting evaluators.
