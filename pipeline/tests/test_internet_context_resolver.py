import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context.internet_context.models import ContextCandidate, InternetContextRequest, SourceResult
from context.internet_context.resolver import InternetContextResolver


class FakeSource:
    name = "anilist"

    def search(self, request: InternetContextRequest) -> SourceResult:
        return SourceResult(
            source=self.name,
            status="found",
            confidence=0.92,
            title="The Regressed Mercenary Has a Plan",
            synopsis="A mercenary returns with memories and a plan.",
            genres=["Action", "Fantasy"],
            candidates=[
                ContextCandidate(kind="character", source="Ghislain Perdium", target="Ghislain Perdium", confidence=0.95, sources=[self.name]),
                ContextCandidate(kind="term", source="mana technique", target="técnica de mana", confidence=0.9, sources=[self.name]),
            ],
            url="https://anilist.co/manga/fixture",
        )


class UnavailableSource:
    name = "myanimelist"

    def search(self, request: InternetContextRequest) -> SourceResult:
        return SourceResult(source=self.name, status="unavailable", confidence=0.0, error="api key ausente")


def test_resolver_merges_sources_and_writes_cache(tmp_path):
    resolver = InternetContextResolver(cache_dir=tmp_path, sources=[FakeSource(), UnavailableSource()])
    result = resolver.resolve(InternetContextRequest(title="The Regressed Mercenary Has a Plan"))

    assert result.title == "The Regressed Mercenary Has a Plan"
    assert result.internet_context_loaded is True
    assert [source.source for source in result.source_results] == ["anilist", "myanimelist"]
    assert result.source_results[1].status == "unavailable"
    assert len(result.glossary_candidates) == 2
    assert result.context_quality == "partial"

    cache_files = list(tmp_path.glob("*.json"))
    assert cache_files
    cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert cached["title"] == result.title


def test_resolver_uses_cache_offline(tmp_path):
    resolver = InternetContextResolver(cache_dir=tmp_path, sources=[FakeSource()])
    first = resolver.resolve(InternetContextRequest(title="The Regressed Mercenary Has a Plan"))

    offline = InternetContextResolver(cache_dir=tmp_path, sources=[])
    second = offline.resolve(InternetContextRequest(title="The Regressed Mercenary Has a Plan", use_cache=True))

    assert second.title == first.title
    assert second.source_results[0].source == "cache"
    assert second.glossary_candidates[0].source == "Ghislain Perdium"


def test_reviewed_glossary_is_not_overwritten(tmp_path):
    resolver = InternetContextResolver(cache_dir=tmp_path, sources=[FakeSource()])
    result = resolver.resolve(
        InternetContextRequest(
            title="The Regressed Mercenary Has a Plan",
            reviewed_glossary={"mana technique": "Arte de mana"},
        )
    )

    candidate = next(item for item in result.glossary_candidates if item.source == "mana technique")
    assert candidate.target == "Arte de mana"
    assert candidate.status == "reviewed"
