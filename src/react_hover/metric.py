"""Evaluation metric from the official DSPy agents tutorial."""

from __future__ import annotations


def top5_recall(example, pred, trace=None) -> float | bool:
    """Fraction of gold pages found in the agent's top-5 predicted titles.

    During optimization (when ``trace`` is provided), returns a boolean
    requiring perfect recall so only fully correct trajectories are bootstrapped.
    At eval time, returns continuous recall in [0, 1].
    """
    gold_titles = example.titles
    pred_titles = getattr(pred, "titles", None) or []
    if not gold_titles:
        return 0.0

    recall = sum(x in pred_titles[:5] for x in gold_titles) / len(gold_titles)

    if trace is not None:
        return recall >= 1.0
    return recall
