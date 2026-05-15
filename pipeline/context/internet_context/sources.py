from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

from .models import ContextCandidate, InternetContextRequest, SourceResult


def _request_json(url: str, *, data: dict[str, Any] | None = None, timeout: int = 6) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json", "User-Agent": "TraduzAi/1.0"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _request_kitsu_json(url: str, *, timeout: int = 6) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.api+json",
            "User-Agent": "TraduzAi/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _candidate(kind: str, source: str, confidence: float, source_name: str, target: str | None = None) -> ContextCandidate:
    clean = " ".join(str(source or "").split()).strip()
    return ContextCandidate(
        kind=kind,
        source=clean,
        target=target or clean,
        confidence=confidence,
        sources=[source_name],
        protect=True,
    )


def _normalized_tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()


def _title_matches_query(query: str, title: str, aliases: list[str] | None = None) -> bool:
    query_key = " ".join(_normalized_tokens(query))
    if not query_key:
        return False
    query_tokens = query_key.split()
    for candidate in [title, *(aliases or [])]:
        candidate_key = " ".join(_normalized_tokens(candidate))
        if not candidate_key:
            continue
        if candidate_key == query_key or candidate_key in query_key or query_key in candidate_key:
            return True
        candidate_tokens = set(candidate_key.split())
        overlap = sum(1 for token in query_tokens if token in candidate_tokens)
        if overlap >= 2 and overlap * 2 >= len(query_tokens):
            return True
    return False


class AniListSource:
    name = "anilist"
    url = "https://graphql.anilist.co"

    _QUERY = """
query ($search: String) {
  Media(search: $search, type: MANGA) {
    id
    siteUrl
    title { english romaji native }
    synonyms
    description(asHtml: false)
    genres
    tags { name rank }
    characters(sort: ROLE, perPage: 20) {
      nodes { name { full native alternative } }
    }
  }
}
"""

    def search(self, request: InternetContextRequest) -> SourceResult:
        payload = _request_json(
            self.url,
            data={"query": self._QUERY, "variables": {"search": request.title}},
        )
        media = (payload.get("data") or {}).get("Media")
        if not media:
            return SourceResult(source=self.name, status="not_found", confidence=0.0)

        title_data = media.get("title") or {}
        title = title_data.get("english") or title_data.get("romaji") or title_data.get("native") or request.title
        candidates: list[ContextCandidate] = []
        seen: set[str] = set()
        for node in ((media.get("characters") or {}).get("nodes") or []):
            names = (node.get("name") or {})
            values = [names.get("full"), names.get("native"), *(names.get("alternative") or [])]
            aliases = [str(value).strip() for value in values if str(value or "").strip()]
            if not aliases:
                continue
            primary = aliases[0]
            key = primary.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidate = _candidate("character", primary, 0.95, self.name)
            candidate.aliases = aliases[1:]
            candidates.append(candidate)

        for synonym in media.get("synonyms") or []:
            synonym = str(synonym or "").strip()
            if synonym:
                candidates.append(_candidate("alias", synonym, 0.75, self.name))

        tags = [
            str(tag.get("name", "")).strip()
            for tag in media.get("tags") or []
            if int(tag.get("rank") or 0) >= 60 and str(tag.get("name", "")).strip()
        ]
        return SourceResult(
            source=self.name,
            status="found",
            confidence=0.92,
            title=title,
            synopsis=_strip_html(media.get("description") or "")[:800],
            genres=list(media.get("genres") or []),
            tags=tags,
            candidates=candidates,
            url=media.get("siteUrl") or f"https://anilist.co/manga/{media.get('id')}",
        )


class JikanSource:
    name = "myanimelist"

    def search(self, request: InternetContextRequest) -> SourceResult:
        params = urllib.parse.urlencode({"q": request.title, "limit": "1"})
        payload = _request_json(f"https://api.jikan.moe/v4/manga?{params}", timeout=8)
        data = payload.get("data") or []
        if not data:
            return SourceResult(source=self.name, status="not_found", confidence=0.0)

        manga = data[0]
        title = manga.get("title_english") or manga.get("title") or request.title
        synonyms = [str(item or "").strip() for item in manga.get("title_synonyms") or [] if str(item or "").strip()]
        if not _title_matches_query(request.title, title, synonyms):
            return SourceResult(source=self.name, status="not_found", confidence=0.0)
        mal_id = int(manga.get("mal_id") or 0)
        candidates: list[ContextCandidate] = []
        if mal_id:
            try:
                chars = _request_json(f"https://api.jikan.moe/v4/manga/{mal_id}/characters", timeout=8)
                for item in chars.get("data") or []:
                    character = item.get("character") or {}
                    name = str(character.get("name") or "").strip()
                    if name:
                        candidates.append(_candidate("character", name, 0.90, self.name))
            except Exception:
                pass

        genres = [
            str(item.get("name") or "").strip()
            for item in manga.get("genres") or []
            if str(item.get("name") or "").strip()
        ]
        return SourceResult(
            source=self.name,
            status="found",
            confidence=0.86,
            title=title,
            synopsis=_strip_html(manga.get("synopsis") or "")[:800],
            genres=genres,
            candidates=candidates,
            url=manga.get("url") or "",
        )


class MangaDexSource:
    name = "mangadex"

    def search(self, request: InternetContextRequest) -> SourceResult:
        params = urllib.parse.urlencode({"title": request.title, "limit": "1", "includes[]": "author"})
        payload = _request_json(f"https://api.mangadex.org/manga?{params}")
        data = payload.get("data") or []
        if not data:
            return SourceResult(source=self.name, status="not_found", confidence=0.0)

        manga = data[0]
        attrs = manga.get("attributes") or {}
        title_map = attrs.get("title") or {}
        title = title_map.get("en") or next(iter(title_map.values()), request.title)
        description_map = attrs.get("description") or {}
        tags = [
            ((tag.get("attributes") or {}).get("name") or {}).get("en", "")
            for tag in attrs.get("tags") or []
        ]
        tags = [tag for tag in tags if tag]
        aliases = []
        for alt in attrs.get("altTitles") or []:
            if isinstance(alt, dict):
                aliases.extend(str(value).strip() for value in alt.values() if str(value or "").strip())
        candidates = [_candidate("alias", alias, 0.72, self.name) for alias in aliases[:12]]
        manga_id = manga.get("id", "")
        return SourceResult(
            source=self.name,
            status="found",
            confidence=0.82,
            title=title,
            synopsis=(description_map.get("en") or next(iter(description_map.values()), ""))[:800],
            genres=tags,
            tags=tags,
            candidates=candidates,
            url=f"https://mangadex.org/title/{manga_id}" if manga_id else "",
        )


class KitsuSource:
    name = "kitsu"

    def search(self, request: InternetContextRequest) -> SourceResult:
        params = urllib.parse.urlencode({"filter[text]": request.title, "page[limit]": "1"})
        payload = _request_kitsu_json(f"https://kitsu.io/api/edge/manga?{params}")
        data = payload.get("data") or []
        if not data:
            return SourceResult(source=self.name, status="not_found", confidence=0.0)

        manga = data[0]
        attrs = manga.get("attributes") or {}
        titles = attrs.get("titles") or {}
        title = titles.get("en") or titles.get("en_jp") or attrs.get("canonicalTitle") or request.title
        aliases = [
            str(value).strip()
            for value in titles.values()
            if str(value or "").strip() and str(value).strip() != title
        ]
        candidates = [_candidate("alias", alias, 0.70, self.name) for alias in aliases[:12]]
        slug = attrs.get("slug") or manga.get("id")
        return SourceResult(
            source=self.name,
            status="found",
            confidence=0.78,
            title=title,
            synopsis=_strip_html(attrs.get("synopsis") or "")[:800],
            genres=[],
            tags=[],
            candidates=candidates,
            url=f"https://kitsu.io/manga/{slug}" if slug else "",
        )


def default_sources() -> list[Any]:
    return [AniListSource(), JikanSource(), MangaDexSource(), KitsuSource()]
