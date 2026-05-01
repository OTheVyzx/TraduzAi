from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


SourceStatus = Literal["found", "not_found", "unavailable", "error", "cached"]
CandidateStatus = Literal["candidate", "auto", "reviewed", "rejected"]


@dataclass
class ContextCandidate:
    kind: str
    source: str
    target: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    status: CandidateStatus = "candidate"
    protect: bool = True
    aliases: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ContextCandidate":
        return cls(
            kind=str(data.get("kind", "term")),
            source=str(data.get("source", "")),
            target=str(data.get("target", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            sources=list(data.get("sources") or []),
            status=data.get("status", "candidate"),
            protect=bool(data.get("protect", True)),
            aliases=list(data.get("aliases") or []),
            forbidden=list(data.get("forbidden") or []),
            notes=str(data.get("notes", "")),
        )


@dataclass
class SourceResult:
    source: str
    status: SourceStatus
    confidence: float
    title: str = ""
    synopsis: str = ""
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    candidates: list[ContextCandidate] = field(default_factory=list)
    url: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "SourceResult":
        return cls(
            source=str(data.get("source", "")),
            status=data.get("status", "not_found"),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            title=str(data.get("title", "")),
            synopsis=str(data.get("synopsis", "")),
            genres=list(data.get("genres") or []),
            tags=list(data.get("tags") or []),
            candidates=[ContextCandidate.from_dict(item) for item in data.get("candidates") or []],
            url=str(data.get("url", "")),
            error=str(data.get("error", "")),
        )


@dataclass
class InternetContextRequest:
    title: str
    enabled_sources: list[str] = field(default_factory=list)
    use_cache: bool = True
    refresh_cache: bool = False
    generic_web_enabled: bool = False
    reviewed_glossary: dict[str, str] = field(default_factory=dict)


@dataclass
class InternetContextResult:
    title: str
    synopsis: str
    genres: list[str]
    source_results: list[SourceResult]
    glossary_candidates: list[ContextCandidate]
    internet_context_loaded: bool
    context_quality: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "synopsis": self.synopsis,
            "genres": self.genres,
            "source_results": [source.to_dict() for source in self.source_results],
            "glossary_candidates": [candidate.to_dict() for candidate in self.glossary_candidates],
            "internet_context_loaded": self.internet_context_loaded,
            "context_quality": self.context_quality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InternetContextResult":
        return cls(
            title=str(data.get("title", "")),
            synopsis=str(data.get("synopsis", "")),
            genres=list(data.get("genres") or []),
            source_results=[SourceResult.from_dict(item) for item in data.get("source_results") or []],
            glossary_candidates=[ContextCandidate.from_dict(item) for item in data.get("glossary_candidates") or []],
            internet_context_loaded=bool(data.get("internet_context_loaded", False)),
            context_quality=str(data.get("context_quality", "empty")),
        )
