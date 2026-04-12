"""
Context module — Fetches manga context from AniList GraphQL API.
Used as fallback when context is not provided by the user.
"""

import urllib.request
import json


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


def fetch_context(obra: str) -> dict:
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
