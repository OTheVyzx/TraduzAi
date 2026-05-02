"""
Startup hook for the TraduzAi Python pipeline.

Python imports ``sitecustomize`` automatically when this directory is on
``sys.path``.  The pipeline is normally executed as ``python pipeline/main.py``,
so this hook lets us install small runtime guards before lazy imports such as
``typesetter.renderer`` happen.

The hook must never break startup: failures are swallowed intentionally.
"""

try:
    import typesetter.safe_renderer_runtime_patch  # noqa: F401
except Exception:
    # Optional guard; never fail interpreter startup because of it.
    pass
