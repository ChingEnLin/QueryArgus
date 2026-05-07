"""LLM prompt templates for self-eval and judge evaluators.

Kept as module constants (matching ``agent/prompts.py``) so they diff cleanly
between iterations. Every template instructs the model to return ONLY the
EvaluationResult JSON shape; the parser tolerates light deviations (extra
fields ignored, missing optional fields defaulted) but enforces the four
required fields: verdict, score, reason, evaluated_by.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Self-eval prompts (same agent LLM, adversarial framing)
# ----------------------------------------------------------------------------

SELF_EVAL_SYSTEM = """You are a strict, sceptical data quality auditor. You review work produced by another investigator and grade it on evidence, calibration, and reasoning.

Your output is ALWAYS a single JSON object. Never wrap it in markdown. Never add prose around it. The schema:

{
  "verdict": "pass" | "warn" | "fail",
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the verdict>",
  "critique": "<what the investigator should do differently, or null if pass>",
  "evaluated_by": "<provided in user prompt>"
}

Verdict rubric:
- "pass": evidence is clear, severity is calibrated, hypothesis is sound
- "warn": something is off but the work is salvageable — note it
- "fail": evidence is missing/insufficient, severity is miscalibrated, or hypothesis contradicts the data
"""


SELF_FINDING_USER_TEMPLATE = """FINDING UNDER REVIEW:
{finding_json}

INVESTIGATION STATE WHEN FINDING WAS PROPOSED:
{state_summary}

Score this finding on the following dimensions:
1. Is the evidence (affected_count, evidence_query) sufficient to support the severity claimed?
2. Is the hypothesis well-reasoned, or speculative?
3. Is affected_pct plausible given the query and the documents sampled?
4. Is this a genuine data quality issue, or a normal characteristic of this data (e.g. an optional field, a serialization variant)?

Set evaluated_by to "self:{model_name}". Return ONLY the JSON object.
"""


SELF_RUN_USER_TEMPLATE = """AUDIT REPORT UNDER REVIEW:
{report_json}

FULL RUN TRACE:
{run_trace_json}

Evaluate this audit on:
1. Coverage — did the agent investigate fields that mattered? Are there obvious fields it missed given the schema sample?
2. Calibration — are the severity assignments appropriate for the evidence?
3. Reasoning quality — are hypotheses sound? Do conclusions follow from the data?
4. Completeness — does the summary actually reflect the findings?
5. False positive risk — are any findings likely noise (optional fields, normal data variation)?

Set evaluated_by to "self:{model_name}". Return ONLY the JSON object.
"""


# ----------------------------------------------------------------------------
# Judge prompts (separate, more capable LLM — run-level only)
# ----------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert database reliability engineer auditing the output of an autonomous data quality agent. You have no prior knowledge of this database or its application — judge ONLY what is in front of you.

Your output is ALWAYS a single JSON object. Never wrap it in markdown. Never add prose around it. The schema:

{
  "verdict": "pass" | "warn" | "fail",
  "score": <float between 0.0 and 1.0>,
  "reason": "<2-3 sentences with the overall assessment>",
  "critique": "<specific issues found, or null if clean>",
  "evaluated_by": "<provided in user prompt>"
}

Be strict. The agent will trust your verdict. Failing a borderline run is the right call when reasoning quality is poor or coverage is shallow.
"""


JUDGE_RUN_USER_TEMPLATE = """AUDIT REPORT:
{report_json}

AGENT RUN TRACE (every action and observation):
{run_trace_json}

Evaluate the audit across five dimensions and combine into a single verdict:
1. Coverage — did the agent investigate the fields that matter, or stop early?
2. Calibration — are severities appropriate for the evidence each finding rests on?
3. Reasoning quality — are the hypotheses actually grounded in what the agent saw?
4. Completeness — is the summary accurate to the findings? Any contradictions?
5. False positive risk — would a domain expert agree these are real quality issues, not normal data variation?

Set evaluated_by to "judge:{model_name}". Return ONLY the JSON object.
"""
