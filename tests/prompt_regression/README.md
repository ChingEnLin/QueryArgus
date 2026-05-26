# Prompt regression suite

Opt-in pytest suite that locks down the behaviour the deterministic rule layer
must enforce given canonical `(AgentState, AgentAction)` and `(Finding)` inputs.

It does **not** call the LLM. It exercises the same code paths the live agent
uses to validate an action / finding (`AgentAction` parsing + `RulesActionEvaluator`
/ `RulesFindingEvaluator`) against frozen fixtures so prompt edits, rule edits,
or state-rendering edits can be reviewed via a pass/fail diff.

## Run

Two layers, each gated by its own marker. Default `pytest` skips both.

```
make test-prompt-regression       # deterministic — no LLM, free
make test-prompt-regression-llm   # LLM-judged — costs tokens
```

The LLM-judge layer requires:

- `pip install -e '.[eval]'` (installs deepeval).
- `GEMINI_API_KEY` set in env.

If either is missing the suite skips cleanly rather than failing.

## Fixture layout

- `fixtures/actions/*.json` — one file per case. Schema:

  ```json
  {
    "name": "short slug used in the test id",
    "description": "what this fixture is locking down",
    "state": { ... AgentState constructor kwargs (subset) ... },
    "action": { ... AgentAction JSON ... },
    "expected_verdict": "pass" | "warn" | "fail",
    "expected_rule": "no_repeat_query" | ... (optional, asserts on rule name)
  }
  ```

- `fixtures/findings/*.json` — one file per case. Same shape but with
  `finding` instead of `action`.

- `prompts/` — golden rendered prompts (one per snapshot fixture).

## Adding a new fixture

1. Drop a JSON file in the right subdirectory.
2. Run `pytest -m prompt_regression`. New file is auto-picked.
3. If you're snapshotting a rendered prompt, set
   `UPDATE_PROMPT_SNAPSHOTS=1 pytest -m prompt_regression` once to write the
   golden file, then commit it.

## LLM-judged layer

Lives under `llm_judge/`. Built on `deepeval` with a **pinned judge model**
(`gemini-2.5-pro`, temp `0.0`) deliberately distinct from the agent's model so
the regression baseline is stable across reruns.

Two GEval metrics in v1:

- `severity_calibration` — severity matches affected_pct and the nature of the
  evidence.
- `hypothesis_groundedness` — the hypothesis follows from the evidence_query
  and sample_values, not free speculation.

Thresholds live in `llm_judge/baseline.json`. They're the floor every `pass`
finding fixture must clear. Adversarial fixtures are excluded from the judge
run — those are the deterministic layer's job.

### Refreshing the baseline

```
make refresh-prompt-baseline
```

Runs every metric on every `pass` finding fixture, takes the **minimum**
observed score per metric (not the mean — the threshold has to be a floor),
subtracts a small margin, and rewrites `baseline.json`. Always review the diff
before committing; a large downward move is a real regression, not an excuse
to lower the floor.

### Out of scope (still)

Summary-vs-findings faithfulness — needs `Report` fixtures. Not yet wired.
