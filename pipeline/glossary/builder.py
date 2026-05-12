from __future__ import annotations

from typing import Any


def _append_unique(values: list[str], value: str) -> None:
    clean = " ".join(str(value or "").split()).strip()
    if not clean:
        return
    if clean.casefold() in {item.casefold() for item in values}:
        return
    values.append(clean)


def _entry_key(source: str) -> str:
    return " ".join(str(source or "").split()).strip().casefold()


def build_glossary_entries(context: dict, glossario: dict | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(
        source: str,
        target: str | None = None,
        *,
        entry_type: str = "term",
        locked: bool = False,
        aliases: list[str] | None = None,
        forbidden: list[str] | None = None,
        confidence: float = 0.80,
        sources: list[str] | None = None,
    ) -> None:
        clean_source = " ".join(str(source or "").split()).strip()
        if not clean_source:
            return
        key = _entry_key(clean_source)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "id": key,
                "source": clean_source,
                "target": " ".join(str(target or clean_source).split()).strip(),
                "type": entry_type,
                "protect": True,
                "locked": locked,
                "aliases": list(aliases or []),
                "forbidden": list(forbidden or []),
                "confidence": float(confidence),
                "sources": list(sources or []),
            }
        )

    for source, target in (glossario or {}).items():
        push(source, target, entry_type="manual_glossary", locked=True, confidence=1.0, sources=["manual"])

    for source, target in ((context or {}).get("glossario") or {}).items():
        push(source, target, entry_type="context_glossary", locked=True, confidence=1.0, sources=["context"])

    for source, target in ((context or {}).get("memoria_lexical") or {}).items():
        push(source, target, entry_type="memory", locked=False, confidence=0.88, sources=["memory"])

    for source, target in ((context or {}).get("corpus_memoria_lexical") or {}).items():
        push(source, target, entry_type="corpus_memory", locked=False, confidence=0.86, sources=["corpus"])

    for character in (context or {}).get("personagens") or []:
        push(character, character, entry_type="character", locked=True, confidence=0.92, sources=["context"])

    for alias in (context or {}).get("aliases") or []:
        push(alias, alias, entry_type="alias", locked=False, confidence=0.74, sources=["context"])

    for term in (context or {}).get("termos") or []:
        target = ((context or {}).get("memoria_lexical") or {}).get(term) or term
        push(term, target, entry_type="term", locked=False, confidence=0.78, sources=["context"])

    for faction in (context or {}).get("faccoes") or []:
        push(faction, faction, entry_type="faction", locked=True, confidence=0.84, sources=["context"])

    for candidate in ((context or {}).get("internet_context") or {}).get("glossary_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        push(
            candidate.get("source", ""),
            candidate.get("target") or candidate.get("source", ""),
            entry_type=candidate.get("kind", "term"),
            locked=candidate.get("status") == "reviewed" or candidate.get("kind") == "character",
            aliases=list(candidate.get("aliases") or []),
            forbidden=list(candidate.get("forbidden") or []),
            confidence=float(candidate.get("confidence", 0.0) or 0.0),
            sources=list(candidate.get("sources") or []),
        )

    return entries


def merge_internet_context_into_context(context: dict, result: Any, reviewed_glossary: dict | None = None) -> dict:
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
    merged = dict(context or {})
    if payload.get("synopsis") and not merged.get("sinopse"):
        merged["sinopse"] = payload.get("synopsis", "")
    if payload.get("genres") and not merged.get("genero"):
        merged["genero"] = list(payload.get("genres") or [])

    merged.setdefault("personagens", [])
    merged.setdefault("aliases", [])
    merged.setdefault("termos", [])
    merged.setdefault("faccoes", [])
    merged.setdefault("memoria_lexical", {})
    merged.setdefault("fontes_usadas", [])

    for source in payload.get("source_results") or []:
        if source.get("status") == "found":
            _append_unique(merged["fontes_usadas"], source.get("source", ""))

    for candidate in payload.get("glossary_candidates") or []:
        kind = str(candidate.get("kind", "term"))
        source = str(candidate.get("source", "")).strip()
        target = str(candidate.get("target", "") or source).strip()
        if not source:
            continue
        if kind == "character":
            _append_unique(merged["personagens"], source)
        elif kind == "alias":
            _append_unique(merged["aliases"], source)
        elif kind == "faction":
            _append_unique(merged["faccoes"], source)
        else:
            _append_unique(merged["termos"], source)
        if candidate.get("status") == "reviewed" or source in (reviewed_glossary or {}):
            merged["memoria_lexical"][source] = (reviewed_glossary or {}).get(source, target)

    merged["internet_context"] = payload
    merged["glossary_entries"] = build_glossary_entries(merged, reviewed_glossary or {})
    return merged
