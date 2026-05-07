"""GeminiClient — google-genai backed implementation of LLMClient.

Uses ``response_mime_type="application/json"`` to coerce JSON output without
binding genai to our Pydantic class as a schema. Passing ``response_schema``
goes through genai's OpenAPI-Schema validator which rejects features Pydantic
emits by default (e.g. boolean ``additionalProperties``), so we keep it simple
and validate the JSON client-side. The system prompt is detailed enough that
the model produces well-shaped output reliably; a single retry covers the rare
malformed case.

Token usage is read from ``response.usage_metadata`` and returned alongside the
parsed action — the planner accumulates these on AgentState so the report
surfaces total cost.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pydantic

from queryargus.llm.client import JSONResponse, LLMResponse, TokenUsage
from queryargus.models.action import AgentAction

logger = logging.getLogger(__name__)

_MAX_PARSE_RETRIES = 1


def _extract_usage(response: Any) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
    )


class GeminiClient:
    """Wraps ``google-genai`` with JSON-coerced output validated into AgentAction."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; cannot build GeminiClient.")
        # Local import — google-genai is heavy; defer until first use.
        from google import genai  # noqa: PLC0415

        self._client: Any = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def propose_action(self, *, system: str, user: str) -> LLMResponse:
        from google.genai import types as genai_types  # noqa: PLC0415

        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=self._temperature,
        )

        last_error: Exception | None = None
        retry_user = user
        cumulative = TokenUsage()
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            response = self._client.models.generate_content(
                model=self._model,
                contents=retry_user,
                config=config,
            )
            cumulative = cumulative + _extract_usage(response)
            raw = getattr(response, "text", None) or ""
            try:
                action = AgentAction.model_validate_json(raw)
                return LLMResponse(action=action, usage=cumulative)
            except (pydantic.ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "gemini returned unparseable AgentAction (attempt %d/%d): %s",
                    attempt + 1, _MAX_PARSE_RETRIES + 1, exc,
                )
                retry_user = (
                    f"{user}\n\nYour previous response was not valid AgentAction JSON. "
                    f"Error: {exc}\nReturn ONLY a JSON object matching the AgentAction schema."
                )
        assert last_error is not None
        raise RuntimeError(f"Gemini failed to produce a valid AgentAction after retries: {last_error}")

    def complete_json(self, *, system: str, user: str) -> JSONResponse:
        """Free-form JSON completion. Used by self/judge evaluators.

        No client-side validation here — the caller is responsible for parsing
        the raw text into the schema it expects (typically EvaluationResult).
        """
        from google.genai import types as genai_types  # noqa: PLC0415

        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=self._temperature,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=config,
        )
        raw = getattr(response, "text", None) or ""
        return JSONResponse(raw=raw, usage=_extract_usage(response))
