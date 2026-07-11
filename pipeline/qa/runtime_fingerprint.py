"""Stable runtime fingerprints for visual-pipeline engine decisions."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


@lru_cache(maxsize=128)
def _file_sha256_cached(resolved_path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    stat = candidate.stat()
    return _file_sha256_cached(str(candidate.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


@lru_cache(maxsize=16)
def _current_git_commit_cached(repo_root: str) -> str:
    command = ["git"]
    if repo_root:
        command.extend(["-C", repo_root])
    command.extend(["rev-parse", "HEAD"])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def current_git_commit(repo_root: str | Path | None = None) -> str:
    normalized = str(Path(repo_root).resolve()) if repo_root is not None else ""
    return _current_git_commit_cached(normalized)


def _normal_engine(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _resolved_path(value: str | Path | None) -> str:
    if value is None:
        return ""
    try:
        return str(Path(value).resolve())
    except OSError:
        return str(value)


def build_engine_fingerprint(
    *,
    stage: str,
    requested_engine: object,
    resolved_engine: object,
    executed_backend: object,
    fallback_used: bool | None = None,
    fallback_reason: str = "",
    resolution_status: str = "",
    resolution_reason: str = "",
    execution_status: str = "",
    result_status: str = "",
    execution_context: str = "chapter",
    module_file: str | Path | None = None,
    model_path: str | Path | None = None,
    model_revision: object = "",
    feature_flags: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe record without confusing a request with execution."""

    requested = _normal_engine(requested_engine) or ""
    resolved = _normal_engine(resolved_engine)
    executed = _normal_engine(executed_backend) or "none"

    used_fallback = bool(fallback_used) if fallback_used is not None else False
    resolved_status = str(resolution_status or "").strip() or (
        "resolved" if resolved is not None else "unavailable"
    )
    executed_status = str(execution_status or "").strip() or (
        "succeeded" if executed != "none" else "not_started"
    )
    output_status = str(result_status or "").strip() or (
        "accepted" if executed_status == "succeeded" else "not_produced"
    )

    module_path = _resolved_path(module_file)
    model_file = _resolved_path(model_path)
    commit = str(git_commit or "").strip() or current_git_commit(repo_root)

    return {
        "schema_version": 2,
        "stage": str(stage or "").strip(),
        "requested_engine": requested,
        "resolved_engine": resolved,
        "executed_backend": executed,
        "fallback_used": used_fallback,
        "fallback_reason": str(fallback_reason or "").strip(),
        "resolution_status": resolved_status,
        "resolution_reason": str(resolution_reason or "").strip(),
        "execution_status": executed_status,
        "result_status": output_status,
        "execution_context": str(execution_context or "chapter").strip() or "chapter",
        "module_file": module_path,
        "module_sha256": file_sha256(module_file),
        "git_commit": commit,
        "model_file": model_file,
        "model_sha256": file_sha256(model_path),
        "model_revision": str(model_revision or "").strip(),
        "feature_flags": {
            str(key): value
            for key, value in sorted((feature_flags or {}).items(), key=lambda item: str(item[0]))
        },
    }


def _env_enabled() -> bool:
    return str(os.getenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _backend_details(backend: object | None) -> tuple[str, str | Path | None, str | Path | None, object]:
    if backend is None:
        return "none", Path(__file__), None, ""
    explicit_backend = str(getattr(backend, "_backend", "") or "").strip()
    backend_type = type(backend)
    backend_name = explicit_backend or f"{backend_type.__module__}.{backend_type.__qualname__}"
    module = sys.modules.get(backend_type.__module__)
    module_file = getattr(module, "__file__", None) or Path(__file__)
    model_path = None
    for owner in (backend, getattr(backend, "_model", None)):
        if owner is None:
            continue
        for attribute in ("_model_path", "model_path", "_model_file", "model_file", "weights_path"):
            candidate = getattr(owner, attribute, None)
            if candidate:
                model_path = candidate
                break
        if model_path is not None:
            break
        paths = getattr(owner, "paths", None)
        candidate = getattr(paths, "weights", None) if paths is not None else None
        if candidate:
            model_path = candidate
            break
    revision = getattr(backend, "model_revision", getattr(backend, "_model_revision", ""))
    return backend_name, module_file, model_path, revision


def record_engine_event(
    *,
    stage: str,
    requested_engine: object,
    resolved_engine: object,
    backend: object | None,
    execution_status: str,
    result_status: str,
    fallback_used: bool = False,
    fallback_reason: str = "",
    resolution_status: str = "resolved",
    resolution_reason: str = "",
    execution_context: str = "chapter",
    model_path: str | Path | None = None,
    model_revision: object = "",
    feature_flags: Mapping[str, Any] | None = None,
    executed_backend: object | None = None,
) -> dict[str, Any]:
    if not _env_enabled():
        return {}
    from debug_tools import get_recorder

    recorder = get_recorder()
    if recorder is None or not recorder.enabled or not getattr(recorder, "_runtime_fingerprint_enabled", False):
        return {}

    backend_name, module_file, discovered_model_path, discovered_revision = _backend_details(backend)
    confirmed_execution = str(execution_status or "").strip() == "succeeded"
    actual_backend = executed_backend if executed_backend is not None else backend_name
    if not confirmed_execution:
        actual_backend = "none"
    if model_path is None:
        model_path = discovered_model_path
    if not model_revision:
        model_revision = discovered_revision
    if feature_flags is None:
        try:
            from runtime_profiles import resolve_visual_pipeline_flags
        except ImportError:
            from pipeline.runtime_profiles import resolve_visual_pipeline_flags

        feature_flags = resolve_visual_pipeline_flags()

    fingerprint = build_engine_fingerprint(
        stage=stage,
        requested_engine=requested_engine,
        resolved_engine=resolved_engine,
        executed_backend=actual_backend,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        resolution_status=resolution_status,
        resolution_reason=resolution_reason,
        execution_status=execution_status,
        result_status=result_status,
        execution_context=execution_context,
        module_file=module_file,
        model_path=model_path,
        model_revision=model_revision,
        feature_flags=feature_flags,
        repo_root=Path(__file__).resolve().parents[2],
    )
    recorder.record_runtime_fingerprint(str(stage or ""), fingerprint)
    return fingerprint


__all__ = [
    "build_engine_fingerprint",
    "current_git_commit",
    "file_sha256",
    "record_engine_event",
]
