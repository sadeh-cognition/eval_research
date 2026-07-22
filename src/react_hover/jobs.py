"""In-memory background job tracker for UI-triggered evals."""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from loguru import logger

from react_hover.eval_runner import run_baseline_eval

JobStatus = Literal["queued", "running", "succeeded", "failed"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvalJob:
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    params: dict[str, Any]
    error: str | None = None
    run_id: str | None = None
    score: float | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_lock = threading.Lock()
_jobs: dict[str, EvalJob] = {}
_worker_lock = threading.Lock()  # only one eval at a time


def list_jobs() -> list[dict[str, Any]]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]


def get_job(job_id: str) -> EvalJob | None:
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


def _run_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        logger.error("job.missing job_id={}", job_id)
        return

    params = dict(job.params)
    logger.info(
        "job.waiting_for_worker job_id={} model={} dev_size={}",
        job_id,
        params.get("student_lm"),
        params.get("dev_size"),
    )
    acquired = _worker_lock.acquire(blocking=True)
    try:
        logger.info("job.worker_acquired job_id={}", job_id)
        _update(job_id, status="running")
        out = run_baseline_eval(**params, job_id=job_id)
        record = out["record"]
        _update(
            job_id,
            status="succeeded",
            run_id=record["id"],
            score=out["score"],
            path=out["path"],
            error=None,
        )
        logger.success(
            "job.succeeded job_id={} run_id={} score={} path={}",
            job_id,
            record["id"],
            out["score"],
            out["path"],
        )
    except Exception as exc:  # noqa: BLE001 — surface to UI
        tb = traceback.format_exc()
        _update(
            job_id,
            status="failed",
            error=f"{exc}\n{tb}",
        )
        logger.exception("job.failed job_id={} error={}", job_id, exc)
    finally:
        if acquired:
            _worker_lock.release()
            logger.info("job.worker_released job_id={}", job_id)


def start_eval_job(params: dict[str, Any]) -> EvalJob:
    job_id = uuid.uuid4().hex[:12]
    now = _utc_now()
    job = EvalJob(
        id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        params=params,
    )
    with _lock:
        _jobs[job_id] = job

    logger.info(
        "job.queued job_id={} model={} backend={} train_size={} dev_size={} max_iters={}",
        job_id,
        params.get("student_lm"),
        params.get("backend"),
        params.get("train_size"),
        params.get("dev_size"),
        params.get("max_iters"),
    )
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True, name=f"eval-{job_id}")
    thread.start()
    logger.info("job.thread_started job_id={} thread={}", job_id, thread.name)
    return job
