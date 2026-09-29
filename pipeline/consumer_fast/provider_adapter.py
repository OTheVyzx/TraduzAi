"""Explicit translation-provider surface for Consumer Fast V1.

The strip host may use the installed translator implementation, but it cannot
select its conventional Google/Ollama retry chain when this adapter is active.
"""

from __future__ import annotations

from translator.translate import TranslationAttemptControl
from translator.translate import translate_one_owner_attempt as _translate_one_owner_attempt


POLICY_ID = "consumer-fast-bounded-owner-v1"
_ALLOWED_CONTROLS = {
    ("google", "owner_primary"),
    ("google", "owner_contextual"),
    ("ollama", "owner_fallback"),
    ("ollama", "owner_configured"),
    ("ollama", "owner_contextual"),
    ("ocr_recovery", "owner_ocr_recovery"),
}


def translate_one_owner_attempt(page: dict, **kwargs):
    control = kwargs.get("control")
    if (
        not isinstance(control, TranslationAttemptControl)
        or (control.backend, control.variant) not in _ALLOWED_CONTROLS
    ):
        raise RuntimeError("Consumer Fast bounded translation control mismatch")
    return _translate_one_owner_attempt(page, **kwargs)


def translate_pages(*_args, **_kwargs):
    raise RuntimeError("legacy translate_pages fallback disabled for consumer-fast-v1")
