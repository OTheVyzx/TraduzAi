from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .cache import InternetContextCache
from .merge import best_synopsis, best_title, merge_candidates, merged_genres
from .models import InternetContextRequest, InternetContextResult, SourceResult


class InternetContextSource(Protocol):
    name: str

    def search(self, request: InternetContextRequest) -> SourceResult:
        ...


class InternetContextResolver:
    def __init__(self, cache_dir: str | Path, sources: list[InternetContextSource]):
        self.cache = InternetContextCache(cache_dir)
        self.sources = sources

    def resolve(self, request: InternetContextRequest) -> InternetContextResult:
        if request.use_cache and not request.refresh_cache:
            cached = self.cache.load(request.title)
            if cached and not self.sources:
                cached.source_results = [
                    SourceResult(source="cache", status="cached", confidence=1.0, title=cached.title)
                ]
                return cached

        enabled = set(request.enabled_sources)
        source_results: list[SourceResult] = []
        for source in self.sources:
            if enabled and source.name not in enabled:
                continue
            if source.name == "generic_web" and not request.generic_web_enabled:
                source_results.append(SourceResult(source=source.name, status="unavailable", confidence=0.0, error="generic web desligado"))
                continue
            try:
                source_results.append(source.search(request))
            except Exception as exc:
                source_results.append(SourceResult(source=source.name, status="error", confidence=0.0, error=str(exc)))

        if not source_results and request.use_cache:
            cached = self.cache.load(request.title)
            if cached:
                cached.source_results = [
                    SourceResult(source="cache", status="cached", confidence=1.0, title=cached.title)
                ]
                return cached

        candidates = merge_candidates(source_results, request.reviewed_glossary)
        title = best_title(source_results, request.title)
        result = InternetContextResult(
            title=title,
            synopsis=best_synopsis(source_results),
            genres=merged_genres(source_results),
            source_results=source_results,
            glossary_candidates=candidates,
            internet_context_loaded=any(item.status == "found" for item in source_results),
            context_quality="partial" if candidates or any(item.status == "found" for item in source_results) else "empty",
        )
        if result.internet_context_loaded:
            self.cache.save(result)
        return result
