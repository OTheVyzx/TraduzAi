from __future__ import annotations

import json
import re
from pathlib import Path


def slugify_work_title(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", (title or "").lower())
    normalized = normalized.strip("-")
    return normalized or "unknown-work"


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_corpus_bundle(
    obra: str,
    models_root: str | Path | None = None,
    fallback_root: str | Path | None = None,
) -> dict:
    default_root = Path(__file__).resolve().parent.parent / "models" / "corpus"
    corpus_root = Path(models_root) if models_root else default_root
    slug = slugify_work_title(obra)
    work_dir = corpus_root / slug
    if not work_dir.exists():
        fallback_dir = Path(fallback_root) if fallback_root else default_root
        work_dir = fallback_dir / slug
    if not work_dir.exists():
        return {"slug": slug, "available": False}

    bundle = {
        "slug": slug,
        "available": True,
        "path": str(work_dir),
        "work_profile": _read_json_if_exists(work_dir / "work_profile.json"),
        "visual_benchmark_profile": _read_json_if_exists(work_dir / "visual_benchmark_profile.json"),
        "textual_benchmark_profile": _read_json_if_exists(work_dir / "textual_benchmark_profile.json"),
        "translation_memory_candidates": _read_json_if_exists(work_dir / "translation_memory_candidates.json"),
    }
    return bundle


def _is_high_trust_candidate(candidate: dict) -> bool:
    source = str(candidate.get("source_text", "")).strip()
    target = str(candidate.get("target_text", "")).strip()
    if not source or not target:
        return False
    if any(char.isdigit() for char in source + target):
        return False
    if candidate.get("occurrences", 0) >= 2:
        return True
    if float(candidate.get("mean_position_delta", 1.0) or 1.0) > 0.04:
        return False
    if int(candidate.get("source_tokens", 99) or 99) > 3:
        return False
    if int(candidate.get("target_tokens", 99) or 99) > 4:
        return False
    return True


def build_corpus_memory_map(memory_candidates: dict) -> dict[str, str]:
    memory_map: dict[str, str] = {}
    for candidate in memory_candidates.get("candidates", []):
        if not _is_high_trust_candidate(candidate):
            continue
        memory_map[candidate["source_text"]] = candidate["target_text"]
    return memory_map


def build_corpus_hint_candidates(memory_candidates: dict, limit: int = 8) -> list[dict]:
    hints = []
    for candidate in memory_candidates.get("candidates", []):
        if not _is_high_trust_candidate(candidate):
            continue
        hints.append(
            {
                "source_text": candidate["source_text"],
                "target_text": candidate["target_text"],
                "occurrences": int(candidate.get("occurrences", 0)),
            }
        )
    return hints[:limit]


def extract_expected_terms(bundle: dict, limit: int = 64) -> list[str]:
    if not bundle.get("available"):
        return []
    terms = []
    for candidate in bundle.get("translation_memory_candidates", {}).get("candidates", []):
        source = str(candidate.get("source_text", "")).strip()
        if not source or len(source) > 40:
            continue
        if any(char.isdigit() for char in source):
            continue
        terms.append(source)
    deduped = []
    seen = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
        if len(deduped) >= limit:
            break
    return deduped


def merge_corpus_into_context(context: dict, bundle: dict) -> dict:
    merged = dict(context or {})
    if not bundle.get("available"):
        return merged

    merged["corpus_slug"] = bundle["slug"]
    merged["corpus_visual_benchmark"] = bundle.get("visual_benchmark_profile", {})
    merged["corpus_textual_benchmark"] = bundle.get("textual_benchmark_profile", {})
    merged["corpus_memory_candidates"] = build_corpus_hint_candidates(
        bundle.get("translation_memory_candidates", {}),
        limit=8,
    )
    existing_memory = dict(merged.get("corpus_memoria_lexical", {}) or {})
    existing_memory.update(build_corpus_memory_map(bundle.get("translation_memory_candidates", {})))
    merged["corpus_memoria_lexical"] = existing_memory

    fontes = list(merged.get("fontes_usadas", []) or [])
    corpus_label = f"corpus:{bundle['slug']}"
    if corpus_label not in fontes:
        fontes.append(corpus_label)
    merged["fontes_usadas"] = fontes
    return merged
