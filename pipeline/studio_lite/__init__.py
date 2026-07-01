"""Studio Lite worker helpers.

This package is intentionally isolated from the automatic pipeline. It provides
small, deterministic image operations for the Studio editor.
"""

from .worker import handle_request

__all__ = ["handle_request"]
