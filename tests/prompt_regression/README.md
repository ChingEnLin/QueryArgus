# Prompt regression suite

Opt-in pytest suite that locks down the behaviour the deterministic rule layer
must enforce given canonical `(AgentState, AgentAction)` and `(Finding)` inputs.

It does **not** call the LLM. It exercises the same code paths the live agent
uses to validate an action / finding (`AgentAction` parsing + `RulesActionEvaluator`
/ `RulesFindingEvaluator`) against frozen fixtures so prompt edits, rule edits,
or state-rendering edits can be reviewed via a pass/fail diff.

## Run

```
pytest -m prompt_regression
```

Default `pytest` runs skip this suite (see `addopts` in `pyproject.toml`).

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

## Out of scope here

Severity-vs-evidence calibration, hypothesis groundedness, summary faithfulness
— those live in the LLM-judged layer (deepeval, not yet wired). This suite is
strictly the deterministic gate.
