"""GeminiClient — google-genai backed implementation of LLMClient.

Uses ``response_mime_type="application/json"`` to coerce JSON output without
binding genai to our Pydantic class as a schema. Passing ``response_schema``
goes through genai's OpenAPI-Schema validator which rejects features Pydantic
emits by default (e.g. boolean ``additionalProperties``), so we keep it simple
and validate the JSON client-side. The system prompt is detailed enough that
the model produces well-shaped output reliably; a single retry covers the rare
malformed case.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pydantic

from queryargus.models.action import AgentAction

logger = logging.getLogger(__name__)

_MAX_PARSE_RETRIES = 1


class GeminiClient:
    """Wraps ``google-genai`` with JSON-coerced output validated into AgentAction."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash-exp",
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

    def propose_action(self, *, system: str, user: str) -> AgentAction:
        from google.genai import types as genai_types  # noqa: PLC0415

        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=self._temperature,
        )

        last_error: Exception | None = None
        retry_user = user
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            response = self._client.models.generate_content(
                model=self._model,
                contents=retry_user,
                config=config,
            )
            raw = getattr(response, "text", None) or ""
            try:
                return AgentAction.model_validate_json(raw)
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
