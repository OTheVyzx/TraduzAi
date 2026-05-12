"""
Context module — Fetches manga context from AniList GraphQL API.
Used as fallback when context is not provided by the user.
"""

import urllib.request
import json
import os
from pathlib import Path


ANILIST_URL = "https://graphql.anilist.co"

QUERY = """
query ($search: String) {
    Media(search: $search, type: MANGA) {
        title { english romaji }
        description(asHtml: false)
        genres
        characters(sort: ROLE, perPage: 15) {
            nodes { name { full } }
        }
    }
}
"""


def _default_context_cache_dir() -> Path:
    return Path(os.getenv("TRADUZAI_CONTEXT_CACHE_DIR") or Path.home() / ".traduzai" / "context_cache")


def _fetch_context_via_resolver(
    obra: str,
    cache_dir: str | Path | None,
    reviewed_glossary: dict | None,
    enabled_sources: list[str] | None,
) -> dict | None:
    try:
        from context.internet_context.models import InternetContextRequest
        from context.internet_context.resolver import InternetContextResolver
        from context.internet_context.sources import default_sources
        from glossary.builder import merge_internet_context_into_context
    except Exception:
        return None

    resolver = InternetContextResolver(cache_dir=cache_dir or _default_context_cache_dir(), sources=default_sources())
    result = resolver.resolve(
        InternetContextRequest(
            title=obra,
            enabled_sources=list(enabled_sources or []),
            use_cache=True,
            reviewed_glossary=dict(reviewed_glossary or {}),
        )
    )
    if result.context_quality == "empty":
        return None
    base = {
        "sinopse": "",
        "genero": [],
        "personagens": [],
        "aliases": [],
        "termos": [],
        "relacoes": [],
        "faccoes": [],
        "resumo_por_arco": [],
        "memoria_lexical": {},
        "fontes_usadas": [],
    }
    return merge_internet_context_into_context(base, result, reviewed_glossary or {})


def fetch_context(
    obra: str,
    cache_dir: str | Path | None = None,
    reviewed_glossary: dict | None = None,
    enabled_sources: list[str] | None = None,
) -> dict:
    """Fetch manga context from AniList API."""
    empty_context = {
        "sinopse": "",
        "genero": [],
        "personagens": [],
        "aliases": [],
        "termos": [],
        "relacoes": [],
        "faccoes": [],
        "resumo_por_arco": [],
        "memoria_lexical": {},
        "fontes_usadas": [],
    }

    resolved = _fetch_context_via_resolver(obra, cache_dir, reviewed_glossary, enabled_sources)
    if resolved:
        return merge_context(empty_context, resolved)

    payload = json.dumps({
        "query": QUERY,
        "variables": {"search": obra}
    }).encode("utf-8")

    req = urllib.request.Request(
        ANILIST_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return empty_context

    media = data.get("data", {}).get("Media")
    if not media:
        return empty_context

    title = media["title"].get("english") or media["title"].get("romaji", obra)
    synopsis = media.get("description", "") or ""

    # Clean HTML tags from synopsis
    import re
    synopsis = re.sub(r"<[^>]+>", "", synopsis).strip()

    genres = media.get("genres", [])

    characters = []
    for node in (media.get("characters", {}).get("nodes") or []):
        name = node.get("name", {}).get("full", "")
        if name:
            characters.append(name)

    return {
        **empty_context,
        "sinopse": synopsis[:500],
        "genero": genres,
        "personagens": characters,
    }


def merge_context(existing: dict, fallback: dict) -> dict:
    """Preserve enriched context and only fill missing baseline fields."""
    merged = dict(existing or {})
    for key, default in {
        "sinopse": "",
        "genero": [],
        "personagens": [],
        "aliases": [],
        "termos": [],
        "relacoes": [],
        "faccoes": [],
        "resumo_por_arco": [],
        "memoria_lexical": {},
        "fontes_usadas": [],
    }.items():
        current = merged.get(key)
        if current in (None, "", [], {}):
            merged[key] = fallback.get(key, default)
        elif key not in merged:
            merged[key] = default
    return merged
