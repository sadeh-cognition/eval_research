"""Shared MIPROv2 optimization used by the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import dspy
from loguru import logger

from react_hover.agent import SafeAgent, build_react_agent
from react_hover.data import load_hover
from react_hover.env import load_env, require_openai_api_key
from react_hover.eval_runner import evaluate_with_lm_traces
from react_hover.history import build_run_record, save_run
from react_hover.lm import build_lm
from react_hover.metric import top5_recall
from react_hover.tools import configure_backend

AutoLevel = Literal["light", "medium", "heavy"]

DEFAULT_TEACHER_LM = "openai/gpt-5.4"
DEFAULT_PROGRAM_PATH = "artifacts/optimized_react.json"


def run_miprov2_optimize(
    *,
    student_lm: str,
    teacher_lm: str,
    backend: str = "wikipedia",
    train_size: int = 20,
    dev_size: int = 10,
    max_iters: int = 10,
    num_threads: int = 4,
    max_errors: int = 999,
    safe: bool = True,
    temperature: float = 0.7,
    auto: AutoLevel = "light",
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 0,
    save: str = DEFAULT_PROGRAM_PATH,
    notes: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Baseline eval → MIPROv2 compile → save program → post-opt eval.

    Persists two history runs (``optimize_baseline`` and ``optimize_after``)
    linked by ``parent_id``. Returns both records, scores, and program path.
    """
    load_env()
    require_openai_api_key()

    tag = f"job={job_id} " if job_id else ""
    logger.info(
        "{}opt.configure student={} teacher={} backend={} auto={} "
        "train_size={} dev_size={} max_iters={} max_bootstrapped_demos={} "
        "max_labeled_demos={}",
        tag,
        student_lm,
        teacher_lm,
        backend,
        auto,
        train_size,
        dev_size,
        max_iters,
        max_bootstrapped_demos,
        max_labeled_demos,
    )

    configure_backend(backend)  # type: ignore[arg-type]

    student = build_lm(student_lm, temperature=temperature)
    teacher = build_lm(teacher_lm, temperature=temperature)
    dspy.configure(lm=student, disable_history=False)

    trainset, devset, _ = load_hover(train_size=train_size, dev_size=dev_size)
    if not trainset:
        raise ValueError("Train set is empty; increase train_size or check HoVer data.")
    if not devset:
        raise ValueError("Dev set is empty; increase dev_size or check HoVer data.")
    logger.info(
        "{}opt.dataset_ready train={} dev={}",
        tag,
        len(trainset),
        len(devset),
    )

    react = build_react_agent(max_iters=max_iters)
    eval_program = SafeAgent(react) if safe else react

    base_config = {
        "student_lm": student_lm,
        "teacher_lm": teacher_lm,
        "temperature": temperature,
        "backend": backend,
        "max_iters": max_iters,
        "num_threads": num_threads,
        "max_errors": max_errors,
        "safe": safe,
        "train_size": train_size,
        "dev_size": dev_size,
        "auto": auto,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "triggered_from": "api" if notes and "UI" in (notes or "") else "runner",
        "job_id": job_id,
        "store_llm_responses": True,
    }

    # --- baseline ---
    logger.info("{}opt.baseline_start metric=top5_recall n_examples={}", tag, len(devset))
    baseline_score, baseline_rows = evaluate_with_lm_traces(
        eval_program,
        devset=devset,
        lm=student,
        metric=top5_recall,
        max_errors=max_errors,
        job_id=job_id,
    )
    baseline_calls = sum(r.get("n_llm_calls") or 0 for r in baseline_rows)
    baseline_record = build_run_record(
        kind="optimize_baseline",
        score=float(baseline_score),
        results=baseline_rows,
        config={**base_config, "total_llm_calls": baseline_calls},
        notes=notes or "Pre-MIPROv2 baseline",
    )
    baseline_path = save_run(baseline_record)
    baseline_id = baseline_record["id"]
    logger.info(
        "{}opt.baseline_done score={} run_id={} path={}",
        tag,
        baseline_score,
        baseline_id,
        baseline_path,
    )

    # --- MIPROv2 compile (on bare react, not SafeAgent) ---
    logger.info("{}opt.miprov2_compile_start auto={}", tag, auto)
    tp = dspy.MIPROv2(
        metric=top5_recall,
        auto=auto,
        num_threads=num_threads,
        teacher_settings=dict(lm=teacher),
        prompt_model=teacher,
        max_errors=max_errors,
    )
    optimized = tp.compile(
        react,
        trainset=trainset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    logger.info("{}opt.miprov2_compile_done", tag)

    save_path = Path(save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(save_path))
    logger.info("{}opt.program_saved path={}", tag, save_path)

    # --- post-opt eval ---
    opt_program = SafeAgent(optimized) if safe else optimized
    # Re-bind student LM after compile (MIPRO may have switched settings).
    dspy.configure(lm=student, disable_history=False)
    logger.info("{}opt.after_start metric=top5_recall n_examples={}", tag, len(devset))
    after_score, after_rows = evaluate_with_lm_traces(
        opt_program,
        devset=devset,
        lm=student,
        metric=top5_recall,
        max_errors=max_errors,
        job_id=job_id,
    )
    after_calls = sum(r.get("n_llm_calls") or 0 for r in after_rows)
    delta = float(after_score) - float(baseline_score)
    after_record = build_run_record(
        kind="optimize_after",
        score=float(after_score),
        results=after_rows,
        config={
            **base_config,
            "program_path": str(save_path),
            "total_llm_calls": after_calls,
            "baseline_score": float(baseline_score),
            "delta": delta,
        },
        parent_id=baseline_id,
        notes=(
            f"{notes + ' · ' if notes else ''}"
            f"Post-MIPROv2; delta={delta:+.2f}"
        ),
    )
    after_path = save_run(after_record)
    logger.success(
        "{}opt.after_done score={} delta={:+.2f} run_id={} path={}",
        tag,
        after_score,
        delta,
        after_record["id"],
        after_path,
    )

    return {
        "baseline": {
            "record": baseline_record,
            "path": str(baseline_path),
            "score": float(baseline_score),
        },
        "after": {
            "record": after_record,
            "path": str(after_path),
            "score": float(after_score),
        },
        "delta": delta,
        "program_path": str(save_path),
        "score": float(after_score),
        # Primary run for UI selection / job.run_id
        "run_id": after_record["id"],
        "baseline_run_id": baseline_id,
    }
