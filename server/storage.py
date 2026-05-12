from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from server.config import Settings, load_settings


def _root(settings: Settings | None = None) -> Path:
    resolved = (settings or load_settings()).storage_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve(storage_key: str, settings: Settings | None = None) -> Path:
    root = _root(settings)
    target = (root / storage_key).resolve()
    if root != target and root not in target.parents:
        raise ValueError("storage_key fora do storage")
    return target


def resolve_path(storage_key: str, settings: Settings | None = None) -> Path:
    return _resolve(storage_key, settings)


def root_path(settings: Settings | None = None) -> Path:
    return _root(settings)


def put_file(local_path: Path, storage_key: str, content_type: str | None = None, settings: Settings | None = None) -> None:
    del content_type
    target = _resolve(storage_key, settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path, target)


def get_file(storage_key: str, dest: Path, settings: Settings | None = None) -> None:
    source = _resolve(storage_key, settings)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def open_for_read(storage_key: str, settings: Settings | None = None) -> BinaryIO:
    return _resolve(storage_key, settings).open("rb")


def delete(storage_key: str, settings: Settings | None = None) -> None:
    target = _resolve(storage_key, settings)
    if target.exists():
        target.unlink()


def exists(storage_key: str, settings: Settings | None = None) -> bool:
    return _resolve(storage_key, settings).exists()
