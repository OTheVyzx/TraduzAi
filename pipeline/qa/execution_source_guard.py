"""Runtime source-closure guard for authenticated acceptance runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping


class ExecutionSourceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_execution_source_ledger(bundle: Mapping[str, Any]) -> dict[str, Any]:
    repo = Path(str(bundle.get("repo_root") or "")).resolve()
    sources = bundle.get("execution_sources") if isinstance(bundle.get("execution_sources"), Mapping) else {}
    loaded: dict[str, str] = {}
    for module in list(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(str(raw)).resolve()
        try:
            relative = path.relative_to(repo).as_posix()
        except ValueError:
            continue
        if relative.endswith(".pyc"):
            candidate = repo / relative[:-1]
            if candidate.is_file():
                relative = relative[:-1]; path = candidate
        if relative.startswith("pipeline/tests/") or "/__pycache__/" in relative:
            continue
        if relative not in sources:
            raise ExecutionSourceError(f"unmanifested_loaded_source:{relative}")
        loaded[relative] = _sha256(path)
    return {"schema_version": 1, "loaded_sources": dict(sorted(loaded.items()))}


def finalize_execution_source_ledger(bundle: Mapping[str, Any], *, ledger: Mapping[str, Any] | None = None) -> dict[str, Any]:
    candidate = dict(ledger) if ledger is not None else collect_execution_source_ledger(bundle)
    sources = bundle.get("execution_sources") if isinstance(bundle.get("execution_sources"), Mapping) else {}
    repo = Path(str(bundle.get("repo_root") or "")).resolve()
    for relative, observed_hash in (candidate.get("loaded_sources") or {}).items():
        expected = sources.get(relative)
        if not isinstance(expected, Mapping):
            raise ExecutionSourceError(f"unmanifested_loaded_source:{relative}")
        if str(expected.get("content_sha256") or "") != str(observed_hash):
            raise ExecutionSourceError(f"loaded_hash_mismatch:{relative}")
    for relative, expected in sources.items():
        path = repo / relative
        if not path.is_file() or _sha256(path) != str(expected.get("content_sha256") or ""):
            raise ExecutionSourceError(f"source_changed_after_init:{relative}")
    return {**candidate, "status": "PASS"}


def finalize_acceptance_execution_ledgers(bundle: Mapping[str, Any], ledgers: Mapping[str, Mapping[str, Any]], *, required_children: set[str]) -> dict[str, Any]:
    missing = sorted(set(required_children) - set(ledgers))
    if missing:
        raise ExecutionSourceError(f"missing_child_ledger:{missing}")
    verified = {name: finalize_execution_source_ledger(bundle, ledger=ledgers[name]) for name in sorted(required_children)}
    return {"schema_version": 1, "status": "PASS", "children": verified}
