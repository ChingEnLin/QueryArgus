.PHONY: test test-prompt-regression test-prompt-regression-llm refresh-prompt-baseline

# Default test run — excludes the opt-in regression suites.
test:
	pytest

# Deterministic prompt/rule regression suite. No LLM, free.
test-prompt-regression:
	pytest -m prompt_regression

# LLM-judged prompt regression suite. Costs tokens; requires GEMINI_API_KEY
# and the `eval` extra (`pip install -e '.[eval]'`).
test-prompt-regression-llm:
	pytest -m prompt_regression_llm

# Refresh the LLM-judge baseline. Review the diff in
# tests/prompt_regression/llm_judge/baseline.json before committing.
refresh-prompt-baseline:
	python scripts/refresh_prompt_baseline.py
