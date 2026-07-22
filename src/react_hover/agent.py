"""Build the multi-hop ReAct agent from the official DSPy agents tutorial."""

from __future__ import annotations

import dspy
from loguru import logger

from react_hover.tools import lookup_wikipedia, search_wikipedia

INSTRUCTIONS = "Find all Wikipedia titles relevant to verifying (or refuting) the claim."


def build_react_agent(*, max_iters: int = 20) -> dspy.ReAct:
    signature = dspy.Signature("claim -> titles: list[str]", INSTRUCTIONS)
    return dspy.ReAct(
        signature,
        tools=[search_wikipedia, lookup_wikipedia],
        max_iters=max_iters,
    )


def safe_predict(program: dspy.Module, claim: str) -> dspy.Prediction:
    """Run the agent; return empty titles on failure (tutorial pattern).

    Failures are logged and attached as ``prediction.error`` so evals do not
    silently score 0% with no diagnostic signal.
    """
    try:
        return program(claim=claim)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "agent.safe_predict_failed claim={!r} error={}",
            (claim[:80] + "…") if len(claim) > 80 else claim,
            exc,
        )
        pred = dspy.Prediction(titles=[])
        pred.error = f"{type(exc).__name__}: {exc}"
        return pred


class SafeAgent(dspy.Module):
    """Wrapper so Evaluate can call a program that swallows failures."""

    def __init__(self, program: dspy.Module):
        super().__init__()
        self.program = program

    def forward(self, claim: str):
        return safe_predict(self.program, claim)
