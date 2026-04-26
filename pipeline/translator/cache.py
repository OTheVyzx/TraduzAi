"""
Persistent cross-session translation cache.

Stores {sha256(source|src|tgt): translated} in a JSON file under
`{models_dir}/cache/translation-{src}-{tgt}.json`.

Designed to be tolerant to corrupted / missing files: reads return empty cache,
writes are atomic (tmp + rename). No TTL — translations do not expire.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _cache_key(source: str, src_lang: str, tgt_lang: str) -> str:
    digest = hashlib.sha256()
    digest.update((src_lang or "").encode("utf-8"))
    digest.update(b"|")
    digest.update((tgt_lang or "").encode("utf-8"))
    digest.update(b"|")
    digest.update((source or "").encode("utf-8"))
    return digest.hexdigest()


class PersistentTranslationCache:
    """Thread-safe JSON-backed translation cache. Flushes on demand."""

    def __init__(self, cache_path: Path):
        self._path = Path(cache_path)
        self._lock = threading.RLock()
        self._data: dict[str, str] = {}
        self._dirty = False
        self._loaded = False

    @classmethod
    def for_language_pair(cls, cache_dir: Path | str, src_lang: str, tgt_lang: str) -> "PersistentTranslationCache":
        root = Path(cache_dir)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Nao foi possivel criar diretorio de cache %s: %s", root, exc)
        safe_src = (src_lang or "xx").replace("/", "_").replace("\\", "_")
        safe_tgt = (tgt_lang or "xx").replace("/", "_").replace("\\", "_")
        return cls(root / f"translation-{safe_src}-{safe_tgt}.json")

    def _load_if_needed(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._path.exists():
                return
            try:
                with self._path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception as exc:
                logger.warning("Cache de traducao corrompido em %s (%s); iniciando vazio.", self._path, exc)
                return
            if isinstance(payload, dict):
                cleaned: dict[str, str] = {}
                for key, value in payload.items():
                    if isinstance(key, str) and isinstance(value, str):
                        cleaned[key] = value
                self._data = cleaned

    def get(self, source: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        if not source:
            return None
        self._load_if_needed()
        key = _cache_key(source, src_lang, tgt_lang)
        with self._lock:
            return self._data.get(key)

    def set(self, source: str, src_lang: str, tgt_lang: str, translated: str) -> None:
        if not source or translated is None:
            return
        self._load_if_needed()
        key = _cache_key(source, src_lang, tgt_lang)
        with self._lock:
            prior = self._data.get(key)
            if prior == translated:
                return
            self._data[key] = translated
            self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Nao foi possivel preparar diretorio de cache %s: %s", self._path.parent, exc)
                return
            try:
                fd, tmp_name = tempfile.mkstemp(
                    prefix=self._path.name + ".",
                    suffix=".tmp",
                    dir=str(self._path.parent),
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(self._data, handle, ensure_ascii=False)
                    os.replace(tmp_name, self._path)
                except Exception:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                    raise
                self._dirty = False
            except Exception as exc:
                logger.warning("Falha ao persistir cache de traducao em %s: %s", self._path, exc)

    def __len__(self) -> int:
        self._load_if_needed()
        with self._lock:
            return len(self._data)
