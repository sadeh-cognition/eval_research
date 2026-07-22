"""Load environment variables for OpenAI / DSPy runs."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(*, dotenv_path: Path | None = None) -> Path | None:
    """Load project ``.env`` if present. Returns the path loaded, or None."""
    from dotenv import load_dotenv

    path = dotenv_path or (REPO_ROOT / ".env")
    if not path.is_file():
        logger.info("env.dotenv_missing path={}", path)
        return None
    load_dotenv(path, override=False)
    logger.info("env.dotenv_loaded path={}", path)
    return path


def openai_api_key_present() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY"))


def require_openai_api_key() -> str:
    """Return the OpenAI API key or raise a clear ValueError."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
    if not key:
        raise ValueError(
            "Missing OpenAI credentials. Set OPENAI_API_KEY in the environment "
            f"or create {REPO_ROOT / '.env'} with OPENAI_API_KEY=sk-..."
        )
    return key


def log_credential_status() -> None:
    if openai_api_key_present():
        logger.info("env.openai_credentials present=true")
    else:
        logger.warning(
            "env.openai_credentials present=false "
            "(evals will fail until OPENAI_API_KEY is set; see .env.example)"
        )
