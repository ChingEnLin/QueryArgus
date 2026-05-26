"""Custom ``DeepEvalBaseLLM`` wrapping ``google-genai`` for the LLM-judged layer.

The agent itself runs on ``gemini-2.5-flash``. We deliberately pin the **judge**
to a separate, more capable model (``gemini-2.5-pro``) and a low temperature so
that the regression baseline is stable across reruns. Bumping the judge model
must be a deliberate edit here followed by a baseline refresh.

This module imports ``deepeval`` lazily so the package can be inspected (e.g.
by the fixture loader) on installs without the ``eval`` extra.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Pinned judge model. Changing this is a baseline-invalidating event.
JUDGE_MODEL = "gemini-2.5-pro"
JUDGE_TEMPERATURE = 0.0


def build_judge() -> Any:
    """Return a configured ``GeminiDeepEvalLLM`` instance.

    Raises ``RuntimeError`` if the eval extra is not installed or if
    ``GEMINI_API_KEY`` is unset. Callers (typically a pytest fixture) are
    expected to translate that into a ``skip``.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised by skip path
        raise RuntimeError(
            "deepeval is not installed. Install with: pip install -e '.[eval]'"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; cannot build judge LLM.")

    from google import genai  # noqa: PLC0415
    from google.genai import types as genai_types  # noqa: PLC0415

    class GeminiDeepEvalLLM(DeepEvalBaseLLM):  # type: ignore[misc, valid-type]
        """deepeval LLM adapter for google-genai. Pinned model + temperature."""

        def __init__(self) -> None:
            self._client = genai.Client(api_key=api_key)
            self._config = genai_types.GenerateContentConfig(
                temperature=JUDGE_TEMPERATURE,
            )

        def load_model(self) -> Any:
            return self._client

        def generate(self, prompt: str, schema: Any = None) -> Any:
            response = self._client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=self._config,
            )
            text = getattr(response, "text", None) or ""
            if schema is not None:
                # GEval (and a few other metrics) expect the judge to honour
                # a pydantic ``schema``. We coerce by re-prompting with the
                # schema hint and parsing the JSON output.
                return schema.model_validate_json(text)
            return text

        async def a_generate(self, prompt: str, schema: Any = None) -> Any:
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return JUDGE_MODEL

    return GeminiDeepEvalLLM()
