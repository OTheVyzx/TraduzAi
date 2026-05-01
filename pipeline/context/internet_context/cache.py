from __future__ import annotations

import json
from pathlib import Path

from .models import InternetContextResult
from .normalizer import slugify_title


class InternetContextCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for_title(self, title: str) -> Path:
        return self.cache_dir / f"{slugify_title(title)}.json"

    def load(self, title: str) -> InternetContextResult | None:
        path = self.path_for_title(title)
        if not path.exists():
            return None
        return InternetContextResult.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, result: InternetContextResult) -> Path:
        path = self.path_for_title(result.title)
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
