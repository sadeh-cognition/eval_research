"""Persist eval runs on disk so they can be reviewed later."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default store relative to the repo / process cwd.
DEFAULT_EVALS_DIR = Path("artifacts/evals")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _slug(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-").lower()
    return (text or "run")[:max_len]


def make_run_id(*, kind: str, student_lm: str, when: datetime | None = None) -> str:
    when = when or _utc_now()
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    lm = _slug(student_lm.split("/")[-1])
    return f"{stamp}_{_slug(kind)}_{lm}"


def evals_dir(root: Path | str | None = None) -> Path:
    path = Path(root) if root else DEFAULT_EVALS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def rows_from_dspy_result(result) -> list[dict[str, Any]]:
    """Convert dspy.EvaluationResult.results into JSON-serializable rows."""
    rows: list[dict[str, Any]] = []
    for example, prediction, score in result.results:
        row: dict[str, Any] = {
            "claim": getattr(example, "claim", None),
            "gold_titles": list(getattr(example, "titles", None) or []),
            "pred_titles": list(getattr(prediction, "titles", None) or []),
            "score": float(score) if score is not None else None,
        }
        reasoning = getattr(prediction, "reasoning", None)
        if reasoning:
            row["reasoning"] = reasoning
        trajectory = getattr(prediction, "trajectory", None)
        if trajectory is not None:
            # Trajectory values may be nested; force JSON-safe strings where needed.
            row["trajectory"] = _jsonable(trajectory)
        rows.append(row)
    return rows


def _jsonable(value: Any) -> Any:
    """Recursively convert values to JSON-serializable primitives.

    Handles nested OpenAI/LiteLLM wrappers (e.g. CompletionTokensDetailsWrapper).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # pydantic v2 / openai SDK models
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    # mapping-like (but not str)
    try:
        if hasattr(value, "items") and not isinstance(value, (str, bytes)):
            return {str(k): _jsonable(v) for k, v in value.items()}  # type: ignore[union-attr]
    except Exception:
        pass
    # plain object with __dict__
    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _jsonable(v)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(value)


def build_run_record(
    *,
    kind: str,
    score: float,
    results: list[dict[str, Any]],
    config: dict[str, Any],
    run_id: str | None = None,
    metric: str = "top5_recall",
    notes: str | None = None,
    parent_id: str | None = None,
) -> dict[str, Any]:
    when = _utc_now()
    student_lm = str(config.get("student_lm") or "unknown")
    rid = run_id or make_run_id(kind=kind, student_lm=student_lm, when=when)
    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    return {
        "id": rid,
        "created_at": when.isoformat(),
        "kind": kind,
        "metric": metric,
        "score": float(score),
        "n_examples": len(results),
        "mean_example_score": (sum(scores) / len(scores)) if scores else None,
        "n_perfect": sum(1 for s in scores if s >= 1.0),
        "n_zero": sum(1 for s in scores if s == 0.0),
        "config": config,
        "parent_id": parent_id,
        "notes": notes,
        "results": results,
    }


def save_run(record: dict[str, Any], root: Path | str | None = None) -> Path:
    """Write one eval run to artifacts/evals/<id>.json. Never overwrites another id."""
    directory = evals_dir(root)
    run_id = record["id"]
    path = directory / f"{run_id}.json"
    if path.exists():
        # Extremely unlikely collision within the same second; disambiguate.
        suffix = 1
        while path.exists():
            path = directory / f"{run_id}_{suffix}.json"
            suffix += 1
        record = {**record, "id": path.stem}

    # Final pass so nested SDK wrappers never break persistence.
    safe_record = _jsonable(record)
    path.write_text(json.dumps(safe_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Keep caller's id in sync if disambiguation renamed the file.
    if isinstance(safe_record, dict):
        record["id"] = safe_record.get("id", record["id"])
    return path


def load_run(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_runs(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Load all runs, newest first. Missing/corrupt files are skipped."""
    directory = evals_dir(root)
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            record = load_run(path)
            record["_path"] = str(path)
            runs.append(record)
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return runs


def import_legacy_results(
    legacy_path: Path | str,
    *,
    root: Path | str | None = None,
    kind: str = "baseline",
    config: dict[str, Any] | None = None,
) -> Path | None:
    """Import an old flat eval_results.json into the history store if needed."""
    path = Path(legacy_path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in data:
        return None

    # Skip if this exact content was already imported (same score + n + first claim).
    results = data["results"]
    first_claim = results[0]["claim"] if results else ""
    for existing in list_runs(root):
        er = existing.get("results") or []
        if (
            existing.get("score") == data.get("score")
            and len(er) == len(results)
            and er
            and er[0].get("claim") == first_claim
            and existing.get("kind") == kind
        ):
            return Path(existing["_path"])

    record = build_run_record(
        kind=kind,
        score=float(data.get("score") or 0.0),
        results=results,
        config=config
        or {
            "student_lm": "unknown",
            "source": str(path),
            "imported": True,
        },
        notes=f"Imported from {path}",
    )
    return save_run(record, root)
