from __future__ import annotations

import hashlib
import re
from typing import Any


SENSITIVE_KEYS = {"authorization", "cookie", "x-api-key"}
REDACTED = "[REDACTED]"

_SENSITIVE_TEXT_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^\s,;]+"),
]


def sha1_truncated(value: Any, *, length: int = 12) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def preview_text(value: Any, *, limit: int = 256) -> str:
    text = "" if value is None else str(value)
    text = _redact_sensitive_text(text)
    if len(text) <= limit:
        return text
    return text[:limit]


def redact_debug_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if str(key).strip().lower() in SENSITIVE_KEYS:
                redacted.setdefault("redacted_sensitive_headers", []).append(REDACTED)
            else:
                redacted[key] = redact_debug_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_debug_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_debug_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    return value


def token_diff(before: str, after: str) -> list[dict[str, str]]:
    before_tokens = str(before or "").split()
    after_tokens = str(after or "").split()
    rows: list[dict[str, str]] = []
    max_len = max(len(before_tokens), len(after_tokens))
    for index in range(max_len):
        old = before_tokens[index] if index < len(before_tokens) else ""
        new = after_tokens[index] if index < len(after_tokens) else ""
        if old != new:
            rows.append({"before": old, "after": new})
    return rows


def _redact_sensitive_text(text: str) -> str:
    current = str(text or "")
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        current = pattern.sub(f"sensitive_header={REDACTED}", current)
    return current
