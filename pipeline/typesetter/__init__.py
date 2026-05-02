"""Typesetting package startup hooks."""

try:
    from . import safe_renderer_runtime_patch as _safe_renderer_runtime_patch

    _safe_renderer_runtime_patch.install()
except Exception:
    # The runtime guard is defensive and must never prevent importing the
    # typesetter package in constrained environments/tests.
    pass
