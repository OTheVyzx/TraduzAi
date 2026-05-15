from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.models import Setting, User


router = APIRouter(prefix="/api/setup", tags=["setup"])


class WorkSearchPayload(BaseModel):
    query: str


class WorkContextPayload(BaseModel):
    work_id: str
    title: str | None = None
    source: str | None = None
    source_url: str | None = None
    synopsis: str | None = None
    cover_url: str | None = None
    score: float | None = None
    genres: list[str] | None = None
    characters: list[str] | None = None


class GlossaryEntryPayload(BaseModel):
    source: str
    target: str
    kind: str = "termo"
    confidence: float = 1.0
    status: str = "reviewed"


class LocalMemoryPayload(BaseModel):
    data: dict[str, Any]


def _setting_key(work_id: str) -> str:
    safe = "".join(ch for ch in work_id if ch.isalnum() or ch in {"-", "_"}).strip()
    return f"work_glossary:{safe or 'default'}"


def _load_json_setting(db, key: str, fallback: Any):
    item = db.get(Setting, key)
    if item is None:
        return fallback
    try:
        return json.loads(item.value)
    except json.JSONDecodeError:
        return fallback


def _save_json_setting(db, key: str, value: Any) -> None:
    item = db.get(Setting, key)
    payload = json.dumps(value, ensure_ascii=True)
    if item is None:
        db.add(Setting(key=key, value=payload))
    else:
        item.value = payload


@router.get("/languages")
def languages(_: User = Depends(current_user)):
    return {
        "languages": [
            {"id": "en", "label": "Inglês"},
            {"id": "ko", "label": "Coreano"},
            {"id": "ja", "label": "Japonês"},
            {"id": "zh", "label": "Chinês"},
            {"id": "pt-BR", "label": "Português do Brasil"},
        ]
    }


@router.get("/presets")
def presets(_: User = Depends(current_user)):
    return {
        "presets": [
            {"id": "scan-clean", "label": "Padrão limpo", "quality": "normal", "description": "Equilibrado para mangá e manhwa."},
            {"id": "fast-review", "label": "Normal para revisão", "quality": "normal", "description": "Mais leve para conferir no editor."},
            {"id": "high-detail", "label": "Ultra", "quality": "ultra", "description": "Mais conservador para arte detalhada."},
        ]
    }


@router.post("/work-search")
def work_search(payload: WorkSearchPayload, _: User = Depends(current_user)):
    query = payload.query.strip()
    if not query:
        return {"results": []}
    results = _search_internet_works(query)
    if results:
        return {"results": results}
    slug = _slug(query)
    return {"results": [{"work_id": slug, "title": query, "source": "local", "risk_level": "high", "synopsis": ""}]}


@router.post("/work-context")
def work_context(payload: WorkContextPayload, _: User = Depends(current_user)):
    title = payload.title or payload.work_id.replace("-", " ").title()
    characters = [item for item in (payload.characters or []) if item][:10]
    terms = [title, *characters[:4], "capitulo", "scan"]
    source = payload.source or "local"
    synopsis = payload.synopsis or f"Contexto local preparado para {title}."
    source_results = []
    if source != "local":
        source_results.append(
            {
                "source": source,
                "status": "found",
                "confidence": min(1.0, max(0.0, (payload.score or 80) / 100)),
                "title": title,
                "synopsis": synopsis,
                "url": payload.source_url or "",
            }
        )
    return {
        "context": {
            "work_id": payload.work_id,
            "title": title,
            "sinopse": synopsis,
            "genero": payload.genres or [],
            "personagens": characters or [title.split()[0] if title.split() else title],
            "termos": terms,
            "faccoes": [],
            "aliases": {},
            "glossario": {},
            "memoria_lexical": {},
            "internet_context": {
                "internet_context_loaded": source != "local",
                "rejected_glossary_candidates": [],
                "source_results": source_results,
                "glossary_candidates": [
                    {"kind": "termo", "source": term, "target": term, "confidence": 0.82, "status": "pending"}
                    for term in terms[:3]
                ],
            },
        }
    }


@router.get("/glossary/{work_id}")
def load_glossary(work_id: str, _: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        entries = _load_json_setting(db, _setting_key(work_id), [])
    return {"entries": entries}


@router.post("/glossary/{work_id}/entries")
def upsert_glossary(
    work_id: str,
    payload: GlossaryEntryPayload,
    _: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
):
    if not payload.source.strip():
        raise HTTPException(status_code=422, detail="termo obrigatorio")
    with session_scope(settings) as db:
        key = _setting_key(work_id)
        entries = _load_json_setting(db, key, [])
        entry_id = payload.source.strip().lower()
        next_entry = payload.model_dump()
        next_entry["id"] = entry_id
        entries = [entry for entry in entries if entry.get("id") != entry_id and entry.get("source", "").lower() != entry_id]
        entries.append(next_entry)
        _save_json_setting(db, key, entries)
    return {"entry": next_entry}


@router.delete("/glossary/{work_id}/entries/{entry_id}")
def remove_glossary(work_id: str, entry_id: str, _: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        key = _setting_key(work_id)
        entries = _load_json_setting(db, key, [])
        entries = [entry for entry in entries if entry.get("id") != entry_id]
        _save_json_setting(db, key, entries)
    return {"ok": True}


@router.post("/local-memory/export")
def export_local_memory(payload: LocalMemoryPayload | None = None, _: User = Depends(current_user)):
    return {"memory": payload.data if payload else {"version": 1, "works": []}}


@router.post("/local-memory/import")
def import_local_memory(payload: LocalMemoryPayload, _: User = Depends(current_user)):
    return {"ok": True, "summary": {"works": len(payload.data.get("works", [])) if isinstance(payload.data, dict) else 0}}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "obra"


def _strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<br\s*/?>", " ", value)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _search_anilist(query: str) -> list[dict[str, Any]]:
    graphql = """
    query ($search: String) {
      Page(page: 1, perPage: 5) {
        media(search: $search, type: MANGA, sort: SEARCH_MATCH) {
          id
          title { english romaji native }
          description(asHtml: false)
          genres
          siteUrl
          averageScore
          coverImage { large }
          characters(sort: ROLE, perPage: 8) { nodes { name { full } } }
        }
      }
    }
    """
    data = _request_json(
        "https://graphql.anilist.co",
        data={"query": graphql, "variables": {"search": query}},
        timeout=8,
    )
    if not data:
        return []
    media = (((data.get("data") or {}).get("Page") or {}).get("media") or [])[:5]
    results = []
    for item in media:
        titles = item.get("title") or {}
        title = titles.get("english") or titles.get("romaji") or titles.get("native") or query
        score = float(item.get("averageScore") or 0)
        characters = [
            ((node.get("name") or {}).get("full") or "").strip()
            for node in (((item.get("characters") or {}).get("nodes")) or [])
        ]
        results.append(
            {
                "work_id": f"anilist:{item.get('id') or _slug(title)}",
                "title": title,
                "source": "anilist",
                "source_url": item.get("siteUrl") or "",
                "cover_url": ((item.get("coverImage") or {}).get("large")) or "",
                "synopsis": _strip_html(item.get("description")),
                "score": score or 100,
                "risk_level": "low" if score >= 70 else "medium",
                "genres": [genre for genre in (item.get("genres") or []) if isinstance(genre, str)],
                "characters": [character for character in characters if character],
            }
        )
    return results


def _search_internet_works(query: str) -> list[dict[str, Any]]:
    results = [*_search_anilist(query), *_search_jikan(query), *_search_kitsu(query)]
    deduped = []
    seen = set()
    for item in results:
        key = item["title"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return sorted(deduped, key=lambda item: item.get("score") or 0, reverse=True)[:6]


def _request_json(url: str, data: dict[str, Any] | None = None, timeout: int = 6) -> dict[str, Any] | None:
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "TraduzAI/0.2"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None


def _search_jikan(query: str) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"q": query, "limit": 3})
    data = _request_json(f"https://api.jikan.moe/v4/manga?{params}")
    if not data:
        return []
    results = []
    for item in data.get("data") or []:
        title = item.get("title_english") or item.get("title") or query
        score = float(item.get("score") or 0) * 10
        results.append(
            {
                "work_id": f"myanimelist:{item.get('mal_id') or _slug(title)}",
                "title": title,
                "source": "myanimelist",
                "source_url": item.get("url") or "",
                "cover_url": (((item.get("images") or {}).get("jpg") or {}).get("large_image_url")) or "",
                "synopsis": _strip_html(item.get("synopsis")),
                "score": score,
                "risk_level": "low" if score >= 70 else "medium",
                "genres": [genre.get("name") for genre in (item.get("genres") or []) if genre.get("name")],
                "characters": [],
            }
        )
    return results


def _search_kitsu(query: str) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"filter[text]": query, "page[limit]": 3})
    data = _request_json(f"https://kitsu.io/api/edge/manga?{params}")
    if not data:
        return []
    results = []
    for item in data.get("data") or []:
        attrs = item.get("attributes") or {}
        titles = attrs.get("titles") or {}
        title = titles.get("en") or titles.get("en_jp") or attrs.get("canonicalTitle") or query
        rating = attrs.get("averageRating")
        score = float(rating) if rating else 0
        results.append(
            {
                "work_id": f"kitsu:{item.get('id') or _slug(title)}",
                "title": title,
                "source": "kitsu",
                "source_url": f"https://kitsu.io/manga/{attrs.get('slug')}" if attrs.get("slug") else "",
                "cover_url": ((attrs.get("posterImage") or {}).get("large")) or "",
                "synopsis": _strip_html(attrs.get("synopsis")),
                "score": score,
                "risk_level": "low" if score >= 70 else "medium",
                "genres": [],
                "characters": [],
            }
        )
    return results
