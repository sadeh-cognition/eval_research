"""Shared baseline evaluation used by CLI and API."""

from __future__ import annotations

from typing import Any

import dspy
from loguru import logger

from react_hover.agent import SafeAgent, build_react_agent
from react_hover.data import load_hover
from react_hover.env import load_env, require_openai_api_key
from react_hover.history import build_run_record, save_run
from react_hover.lm import build_lm
from react_hover.lm_trace import capture_new_calls, snapshot_history_len
from react_hover.metric import top5_recall
from react_hover.tools import configure_backend

# OpenAI-only model ids for the UI dropdown (LiteLLM / DSPy: openai/<model>).
DEFAULT_STUDENT_LM = "openai/gpt-5.4-mini"

AVAILABLE_MODELS: list[dict[str, str]] = [
    # GPT-5.6 family (2026 frontier)
    {"id": "openai/gpt-5.6-sol", "label": "GPT-5.6 Sol (frontier)"},
    {"id": "openai/gpt-5.6-terra", "label": "GPT-5.6 Terra (balanced)"},
    {"id": "openai/gpt-5.6-luna", "label": "GPT-5.6 Luna (cost-efficient)"},
    # GPT-5.5 / 5.4
    {"id": "openai/gpt-5.5", "label": "GPT-5.5"},
    {"id": "openai/gpt-5.5-pro", "label": "GPT-5.5 Pro"},
    {"id": "openai/gpt-5.4", "label": "GPT-5.4"},
    {"id": "openai/gpt-5.4-mini", "label": "GPT-5.4 mini (default)"},
    {"id": "openai/gpt-5.4-nano", "label": "GPT-5.4 nano"},
    # Reasoning (o-series)
    {"id": "openai/o3", "label": "o3 (reasoning)"},
    {"id": "openai/o3-pro", "label": "o3-pro (reasoning)"},
    {"id": "openai/o4-mini", "label": "o4-mini (reasoning)"},
    # Still available via API
    {"id": "openai/gpt-4.1", "label": "GPT-4.1"},
    {"id": "openai/gpt-4.1-mini", "label": "GPT-4.1 mini"},
    {"id": "openai/gpt-4o", "label": "GPT-4o"},
    {"id": "openai/gpt-4o-mini", "label": "GPT-4o mini"},
]


def _jsonable_prediction_fields(prediction: Any) -> dict[str, Any]:
    from react_hover.history import _jsonable

    row: dict[str, Any] = {
        "pred_titles": list(getattr(prediction, "titles", None) or []),
    }
    reasoning = getattr(prediction, "reasoning", None)
    if reasoning:
        row["reasoning"] = reasoning
    trajectory = getattr(prediction, "trajectory", None)
    if trajectory is not None:
        row["trajectory"] = _jsonable(trajectory)
    return row


def evaluate_with_lm_traces(
    program: dspy.Module,
    *,
    devset: list,
    lm: dspy.LM,
    metric=top5_recall,
    max_errors: int = 999,
    job_id: str | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Run eval example-by-example and attach per-example LM call traces.

    Sequential on purpose so LM history can be attributed to each example
    (thread-parallel Evaluate would interleave history).
    """
    tag = f"job={job_id} " if job_id else ""
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    errors = 0

    # Ensure history is recorded.
    dspy.settings.configure(disable_history=False)

    for i, example in enumerate(devset):
        claim = getattr(example, "claim", "") or ""
        logger.info(
            "{}eval.example_start i={}/{} claim={!r}",
            tag,
            i + 1,
            len(devset),
            (claim[:60] + "…") if len(claim) > 60 else claim,
        )
        start = snapshot_history_len(lm)
        prediction = None
        score: float = 0.0
        error_msg: str | None = None
        try:
            prediction = program(**example.inputs())
            # SafeAgent attaches .error when it swallows a failure.
            swallowed = getattr(prediction, "error", None)
            if swallowed:
                error_msg = str(swallowed)
                errors += 1
                logger.warning("{}eval.example_swallowed_error i={} error={}", tag, i, error_msg)
            raw = metric(example, prediction)
            score = float(raw) if not isinstance(raw, bool) else (1.0 if raw else 0.0)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("{}eval.example_error i={} error={}", tag, i, error_msg)
            if errors > max_errors:
                raise
            prediction = dspy.Prediction(titles=[])
            score = 0.0

        llm_calls = capture_new_calls(lm, start)
        row: dict[str, Any] = {
            "claim": getattr(example, "claim", None),
            "gold_titles": list(getattr(example, "titles", None) or []),
            "score": score,
            "llm_calls": llm_calls,
            "n_llm_calls": len(llm_calls),
        }
        if prediction is not None:
            row.update(_jsonable_prediction_fields(prediction))
        else:
            row["pred_titles"] = []
        if error_msg:
            row["error"] = error_msg

        rows.append(row)
        scores.append(score)
        logger.info(
            "{}eval.example_done i={} score={} n_llm_calls={}",
            tag,
            i + 1,
            score,
            len(llm_calls),
        )

    # Match dspy.Evaluate: percentage aggregate of mean score.
    mean = sum(scores) / len(scores) if scores else 0.0
    aggregate = round(100 * mean, 2)
    logger.info(
        "{}eval.aggregate score={} n_examples={} n_errors={}",
        tag,
        aggregate,
        len(rows),
        errors,
    )
    return aggregate, rows


def run_baseline_eval(
    *,
    student_lm: str,
    backend: str = "wikipedia",
    train_size: int = 5,
    dev_size: int = 5,
    max_iters: int = 10,
    num_threads: int = 2,  # kept for API compat; traces force sequential runs
    max_errors: int = 999,
    safe: bool = True,
    temperature: float = 0.7,
    load: str | None = None,
    notes: str | None = None,
    kind: str = "baseline",
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run HoVer top5_recall eval, persist to disk (with LM traces), return record + path."""
    load_env()
    require_openai_api_key()

    tag = f"job={job_id} " if job_id else ""
    logger.info(
        "{}eval.configure model={} backend={} temperature={} max_iters={} "
        "train_size={} dev_size={} num_threads={} safe={} capture_lm_traces=True",
        tag,
        student_lm,
        backend,
        temperature,
        max_iters,
        train_size,
        dev_size,
        num_threads,
        safe,
    )
    if num_threads != 1:
        logger.info(
            "{}eval.note sequential_for_traces requested_num_threads={}",
            tag,
            num_threads,
        )

    logger.info("{}eval.backend_select name={}", tag, backend)
    configure_backend(backend)  # type: ignore[arg-type]

    logger.info("{}eval.lm_init model={}", tag, student_lm)
    lm = build_lm(student_lm, temperature=temperature)
    dspy.configure(lm=lm, disable_history=False)

    logger.info("{}eval.dataset_load source=hover train_size={} dev_size={}", tag, train_size, dev_size)
    trainset, devset, _ = load_hover(train_size=train_size, dev_size=dev_size)
    if not devset:
        logger.error("{}eval.dataset_empty", tag)
        raise ValueError("Dev set is empty; increase dev_size or check HoVer data.")
    logger.info(
        "{}eval.dataset_ready train={} dev={} example_claim={!r}",
        tag,
        len(trainset),
        len(devset),
        (devset[0].claim[:80] + "…") if len(devset[0].claim) > 80 else devset[0].claim,
    )

    logger.info("{}eval.agent_build max_iters={} load={}", tag, max_iters, load)
    react = build_react_agent(max_iters=max_iters)
    if load:
        react.load(load)
        logger.info("{}eval.agent_loaded path={}", tag, load)

    program = SafeAgent(react) if safe else react
    logger.info(
        "{}eval.scoring_start metric=top5_recall n_examples={} with_lm_traces=True",
        tag,
        len(devset),
    )
    score, results = evaluate_with_lm_traces(
        program,
        devset=devset,
        lm=lm,
        metric=top5_recall,
        max_errors=max_errors,
        job_id=job_id,
    )
    total_calls = sum(r.get("n_llm_calls") or 0 for r in results)
    logger.info(
        "{}eval.scoring_done score={} n_results={} total_llm_calls={}",
        tag,
        score,
        len(results),
        total_calls,
    )

    config = {
        "student_lm": student_lm,
        "teacher_lm": None,
        "temperature": temperature,
        "backend": backend,
        "max_iters": max_iters,
        "num_threads": 1,  # effective; sequential for traces
        "requested_num_threads": num_threads,
        "max_errors": max_errors,
        "safe": safe,
        "train_size": train_size,
        "dev_size": dev_size,
        "load": load,
        "triggered_from": "api" if notes and "UI" in notes else "runner",
        "job_id": job_id,
        "store_llm_responses": True,
        "total_llm_calls": total_calls,
    }
    record = build_run_record(
        kind=kind,
        score=float(score),
        results=results,
        config=config,
        notes=notes or "Baseline evaluation",
    )
    path = save_run(record)
    logger.info(
        "{}eval.persisted run_id={} score={} path={} n_perfect={} n_zero={} total_llm_calls={}",
        tag,
        record["id"],
        record["score"],
        path,
        record.get("n_perfect"),
        record.get("n_zero"),
        total_calls,
    )
    return {"record": record, "path": str(path), "score": float(score)}
