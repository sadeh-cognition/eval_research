"""Build dspy.LM instances with model-aware parameter defaults."""

from __future__ import annotations

import re

import dspy
from loguru import logger

# Many GPT-5 / o-series OpenAI models only accept the default temperature (1).
# Sending 0.7 (our old default) causes 400 BadRequest and empty SafeAgent results.
_FIXED_TEMPERATURE_MODEL = re.compile(
    r"(?:^|/)("
    r"gpt-5(?:\.\d+)?(?:-[\w.]+)?|"
    r"o[1-9](?:-[\w.]+)?"
    r")$",
    re.IGNORECASE,
)


def model_requires_default_temperature(model_id: str) -> bool:
    """Return True if the model rejects non-default temperature values."""
    # model_id like "openai/gpt-5.6-terra" or "gpt-5.4-mini"
    name = model_id.split("/")[-1]
    return bool(_FIXED_TEMPERATURE_MODEL.match(name))


def build_lm(model_id: str, temperature: float | None = 0.7) -> dspy.LM:
    """Create a dspy.LM, omitting temperature when the model does not allow it."""
    kwargs: dict = {}
    effective_temp: float | None = temperature

    if model_requires_default_temperature(model_id):
        if temperature is not None and temperature != 1.0:
            logger.warning(
                "lm.temperature_omitted model={} requested={} reason="
                "model only supports default temperature (1)",
                model_id,
                temperature,
            )
        # Do not pass temperature — let the API use its default.
        effective_temp = None
    elif temperature is not None:
        kwargs["temperature"] = temperature

    logger.info(
        "lm.build model={} temperature={}",
        model_id,
        effective_temp if effective_temp is not None else "default",
    )
    return dspy.LM(model_id, **kwargs)
