"""HTTP API for browsing persisted eval history and triggering evals.

  uv run uvicorn react_hover.api:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from react_hover.env import load_env, log_credential_status, require_openai_api_key
from react_hover.eval_runner import AVAILABLE_MODELS, DEFAULT_STUDENT_LM
from react_hover.history import DEFAULT_EVALS_DIR, import_legacy_results, list_runs, load_run
from react_hover.jobs import get_job, list_jobs, start_eval_job
from react_hover.logging_config import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_ROOT = REPO_ROOT / DEFAULT_EVALS_DIR
LEGACY_RESULTS = REPO_ROOT / "artifacts" / "eval_results.json"

BackendName = Literal["auto", "colbert", "wikipedia"]


class StartEvalRequest(BaseModel):
    student_lm: str = Field(..., description="OpenAI model id, e.g. openai/gpt-5.4-mini")
    backend: BackendName = "wikipedia"
    train_size: int = Field(5, ge=1, le=200)
    dev_size: int = Field(5, ge=1, le=200)
    max_iters: int = Field(10, ge=1, le=30)
    num_threads: int = Field(2, ge=1, le=32)
    safe: bool = True
    temperature: float = Field(0.7, ge=0.0, le=2.0)


def _ensure_legacy_import() -> None:
    if LEGACY_RESULTS.is_file():
        path = import_legacy_results(
            LEGACY_RESULTS,
            root=EVALS_ROOT,
            kind="baseline",
            config={
                "student_lm": DEFAULT_STUDENT_LM,
                "backend": "wikipedia",
                "imported_from": str(LEGACY_RESULTS),
            },
        )
        logger.info("api.legacy_import path={}", path)


def _summarize(run: dict) -> dict:
    config = run.get("config") or {}
    return {
        "id": run.get("id"),
        "created_at": run.get("created_at"),
        "kind": run.get("kind"),
        "metric": run.get("metric", "top5_recall"),
        "score": run.get("score"),
        "n_examples": run.get("n_examples") or len(run.get("results") or []),
        "n_perfect": run.get("n_perfect"),
        "n_zero": run.get("n_zero"),
        "mean_example_score": run.get("mean_example_score"),
        "parent_id": run.get("parent_id"),
        "notes": run.get("notes"),
        "student_lm": config.get("student_lm"),
        "backend": config.get("backend"),
        "dev_size": config.get("dev_size"),
        "path": run.get("_path"),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    load_env()
    log_credential_status()
    logger.info("api.startup evals_dir={}", EVALS_ROOT)
    EVALS_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_legacy_import()
    n = len(list(EVALS_ROOT.glob("*.json")))
    logger.info("api.ready n_persisted_runs={}", n)
    yield
    logger.info("api.shutdown")


app = FastAPI(title="ReAct eval history", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip noisy polling paths at DEBUG only; always log eval triggers.
    path = request.url.path
    logger.debug("api.request method={} path={}", request.method, path)
    response = await call_next(request)
    if path.startswith("/api/evals") or path.startswith("/api/jobs"):
        logger.info(
            "api.response method={} path={} status={}",
            request.method,
            path,
            response.status_code,
        )
    return response


@app.get("/api/health")
def health() -> dict:
    from react_hover.env import openai_api_key_present

    return {
        "ok": True,
        "evals_dir": str(EVALS_ROOT),
        "openai_api_key_present": openai_api_key_present(),
    }


@app.get("/api/models")
def get_models() -> dict:
    return {"models": AVAILABLE_MODELS, "default": DEFAULT_STUDENT_LM, "provider": "openai"}


@app.get("/api/runs")
def get_runs() -> dict:
    runs = list_runs(EVALS_ROOT)
    logger.debug("api.list_runs count={}", len(runs))
    return {"runs": [_summarize(r) for r in runs], "count": len(runs)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    path = EVALS_ROOT / f"{run_id}.json"
    if not path.is_file():
        matches = list(EVALS_ROOT.glob(f"{run_id}.json")) + list(EVALS_ROOT.glob(f"{run_id}_*.json"))
        if not matches:
            logger.warning("api.run_not_found run_id={}", run_id)
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        path = sorted(matches)[0]
    try:
        record = load_run(path)
    except OSError as exc:
        logger.exception("api.run_load_error run_id={} path={}", run_id, path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    record["_path"] = str(path)
    logger.debug("api.run_loaded run_id={} score={}", run_id, record.get("score"))
    return record


@app.get("/api/jobs")
def get_jobs() -> dict:
    jobs = list_jobs()
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        logger.warning("api.job_not_found job_id={}", job_id)
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.post("/api/evals", status_code=202)
def start_eval(body: StartEvalRequest) -> dict:
    known_ids = {m["id"] for m in AVAILABLE_MODELS}
    if body.student_lm not in known_ids:
        logger.warning("api.eval_rejected_unknown_model model={}", body.student_lm)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {body.student_lm!r}. Choose from GET /api/models.",
        )

    try:
        require_openai_api_key()
    except ValueError as exc:
        logger.error("api.eval_rejected_missing_credentials")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "api.eval_request model={} backend={} train_size={} dev_size={} "
        "max_iters={} num_threads={} safe={} temperature={}",
        body.student_lm,
        body.backend,
        body.train_size,
        body.dev_size,
        body.max_iters,
        body.num_threads,
        body.safe,
        body.temperature,
    )
    params = {
        "student_lm": body.student_lm,
        "backend": body.backend,
        "train_size": body.train_size,
        "dev_size": body.dev_size,
        "max_iters": body.max_iters,
        "num_threads": body.num_threads,
        "safe": body.safe,
        "temperature": body.temperature,
        "notes": f"Triggered from UI · {body.student_lm}",
    }
    job = start_eval_job(params)
    logger.info("api.eval_accepted job_id={} model={}", job.id, body.student_lm)
    return {"job": job.to_dict()}
