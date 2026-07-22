"""Capture and serialize DSPy LM call history for eval investigation."""

from __future__ import annotations

from typing import Any

from loguru import logger


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def serialize_messages(messages: Any) -> list[dict[str, str]]:
    if not messages:
        return []
    out: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append({"role": "unknown", "content": str(msg)})
            continue
        out.append(
            {
                "role": str(msg.get("role") or "unknown"),
                "content": _content_to_text(msg.get("content")),
            }
        )
    return out


def serialize_outputs(outputs: Any) -> list[str]:
    if outputs is None:
        return []
    if isinstance(outputs, str):
        return [outputs]
    if isinstance(outputs, list):
        texts: list[str] = []
        for item in outputs:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                texts.append(str(item))
        return texts
    return [str(outputs)]


def _serialize_usage(usage: Any) -> dict[str, Any]:
    """Flatten usage + nested token-detail wrappers to plain dicts/numbers."""
    from react_hover.history import _jsonable

    if usage is None:
        return {}
    converted = _jsonable(usage)
    if isinstance(converted, dict):
        return converted
    return {"raw": converted}


def serialize_lm_entry(entry: dict[str, Any], *, call_index: int) -> dict[str, Any]:
    """Reduce a DSPy LM history entry to a JSON-safe investigation record."""
    from react_hover.history import _jsonable

    messages = serialize_messages(entry.get("messages"))
    outputs = serialize_outputs(entry.get("outputs"))
    prompt = entry.get("prompt")
    usage = _serialize_usage(entry.get("usage"))
    cost = entry.get("cost")
    if cost is not None and not isinstance(cost, (int, float, str, type(None))):
        cost = _jsonable(cost)

    return {
        "call_index": call_index,
        "uuid": entry.get("uuid"),
        "timestamp": entry.get("timestamp"),
        "model": entry.get("model") or entry.get("response_model"),
        "messages": messages,
        "prompt": prompt if isinstance(prompt, str) else (str(prompt) if prompt else None),
        "outputs": outputs,
        "response_text": "\n".join(outputs) if outputs else "",
        "usage": usage if isinstance(usage, dict) else {"value": usage},
        "cost": cost,
    }


def snapshot_history_len(lm: Any) -> int:
    history = getattr(lm, "history", None) or []
    return len(history)


def capture_new_calls(lm: Any, start_len: int) -> list[dict[str, Any]]:
    """Return serialized LM calls appended since ``start_len``."""
    history = getattr(lm, "history", None) or []
    new_entries = history[start_len:]
    calls = [serialize_lm_entry(e, call_index=i) for i, e in enumerate(new_entries)]
    logger.debug("lm_trace.captured n_calls={} start_len={}", len(calls), start_len)
    return calls
