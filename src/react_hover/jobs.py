"""In-memory background job tracker for UI-triggered evals and optimizes."""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from loguru import logger

from react_hover.eval_runner import run_baseline_eval
from react_hover.opt_runner import run_miprov2_optimize

JobStatus = Literal["queued", "running", "succeeded", "failed"]
JobKind = Literal["eval", "optimize"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: JobKind
    status: JobStatus
    created_at: str
    updated_at: str
    params: dict[str, Any]
    error: str | None = None
    run_id: str | None = None
    baseline_run_id: str | None = None
    score: float | None = None
    baseline_score: float | None = None
    delta: float | None = None
    path: str | None = None
    program_path: str | None = None
    # Extra run ids produced by multi-step jobs (e.g. optimize baseline + after)
    run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Back-compat alias used by older imports / type hints.
EvalJob = Job

_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_worker_lock = threading.Lock()  # only one heavy job at a time


def list_jobs() -> list[dict[str, Any]]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def _update(job_id: str, **kwargs: Any) -> None:
    with _lock:
        job = _jobs[job_id]
        for key, value in kwargs.items():
            setattr(job, key, value)
        job.updated_at = _utc_now()
        status = job.status
    if "status" in kwargs:
        logger.info("job.status_change job_id={} status={}", job_id, status)


def _run_eval_job(job_id: str, params: dict[str, Any]) -> None:
    out = run_baseline_eval(**params, job_id=job_id)
    record = out["record"]
    _update(
        job_id,
        status="succeeded",
        run_id=record["id"],
        run_ids=[record["id"]],
        score=out["score"],
        path=out["path"],
        error=None,
    )
    logger.success(
        "job.succeeded job_id={} kind=eval run_id={} score={} path={}",
        job_id,
        record["id"],
        out["score"],
        out["path"],
    )


def _run_optimize_job(job_id: str, params: dict[str, Any]) -> None:
    out = run_miprov2_optimize(**params, job_id=job_id)
    after = out["after"]
    baseline = out["baseline"]
    _update(
        job_id,
        status="succeeded",
        run_id=out["run_id"],
        baseline_run_id=out["baseline_run_id"],
        run_ids=[out["baseline_run_id"], out["run_id"]],
        score=out["score"],
        baseline_score=baseline["score"],
        delta=out["delta"],
        path=after["path"],
        program_path=out["program_path"],
        error=None,
    )
    logger.success(
        "job.succeeded job_id={} kind=optimize run_id={} baseline_run_id={} "
        "score={} delta={:+.2f} program={}",
        job_id,
        out["run_id"],
        out["baseline_run_id"],
        out["score"],
        out["delta"],
        out["program_path"],
    )


def _run_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        logger.error("job.missing job_id={}", job_id)
        return

    params = dict(job.params)
    kind = job.kind
    logger.info(
        "job.waiting_for_worker job_id={} kind={} model={} dev_size={}",
        job_id,
        kind,
        params.get("student_lm"),
        params.get("dev_size"),
    )
    acquired = _worker_lock.acquire(blocking=True)
    try:
        logger.info("job.worker_acquired job_id={} kind={}", job_id, kind)
        _update(job_id, status="running")
        if kind == "optimize":
            _run_optimize_job(job_id, params)
        else:
            _run_eval_job(job_id, params)
    except Exception as exc:  # noqa: BLE001 — surface to UI
        tb = traceback.format_exc()
        _update(
            job_id,
            status="failed",
            error=f"{exc}\n{tb}",
        )
        logger.exception("job.failed job_id={} kind={} error={}", job_id, kind, exc)
    finally:
        if acquired:
            _worker_lock.release()
            logger.info("job.worker_released job_id={}", job_id)


def _enqueue(kind: JobKind, params: dict[str, Any]) -> Job:
    job_id = uuid.uuid4().hex[:12]
    now = _utc_now()
    job = Job(
        id=job_id,
        kind=kind,
        status="queued",
        created_at=now,
        updated_at=now,
        params=params,
    )
    with _lock:
        _jobs[job_id] = job

    logger.info(
        "job.queued job_id={} kind={} model={} backend={} train_size={} dev_size={} "
        "max_iters={} teacher={} auto={}",
        job_id,
        kind,
        params.get("student_lm"),
        params.get("backend"),
        params.get("train_size"),
        params.get("dev_size"),
        params.get("max_iters"),
        params.get("teacher_lm"),
        params.get("auto"),
    )
    thread = threading.Thread(
        target=_run_job,
        args=(job_id,),
        daemon=True,
        name=f"{kind}-{job_id}",
    )
    thread.start()
    logger.info("job.thread_started job_id={} thread={}", job_id, thread.name)
    return job


def start_eval_job(params: dict[str, Any]) -> Job:
    return _enqueue("eval", params)


def start_optimize_job(params: dict[str, Any]) -> Job:
    return _enqueue("optimize", params)
