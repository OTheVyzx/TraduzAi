"""
Lightweight local reviewer for OCR candidates.
Chooses between the primary reading and fallback attempts using confidence
plus a legibility heuristic tuned for manga dialogue.
"""

from __future__ import annotations

import re


def choose_best_candidate(
    primary: dict,
    fallback_candidates: list[dict],
    tipo: str = "fala",
) -> dict:
    candidates = [primary, *fallback_candidates]
    scored = [
        (score_candidate(candidate, tipo), candidate)
        for candidate in candidates
        if candidate.get("text", "").strip()
    ]
    if not scored:
        return primary

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    primary_score = score_candidate(primary, tipo)

    if best_candidate is primary:
        return primary
    if best_score >= primary_score + 0.04:
        return best_candidate
    return primary


def score_candidate(candidate: dict, tipo: str = "fala") -> float:
    confidence = float(candidate.get("confidence", 0.0))
    text = candidate.get("text", "")
    legibility = legibility_score(text, tipo)
    return (confidence * 0.72) + (legibility * 0.28)


def legibility_score(text: str, tipo: str = "fala") -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0

    total = len(stripped)
    alpha = sum(char.isalpha() for char in stripped)
    digits = sum(char.isdigit() for char in stripped)
    spaces = stripped.count(" ")
    punctuation = sum(char in "'!?.,-:;" for char in stripped)
    symbols = max(0, total - alpha - digits - spaces - punctuation)
    repeated_penalty = 0.18 if re.search(r"(.)\1{3,}", stripped) else 0.0
    mixed_digit_penalty = 0.16 if alpha > 0 and digits >= max(2, alpha // 2) else 0.0
    symbol_penalty = min(0.24, symbols * 0.08)
    uppercase_bonus = 0.06 if stripped.isupper() and alpha >= 3 and tipo in {"fala", "sfx"} else 0.0

    alpha_ratio = alpha / max(1, total - spaces)
    readable_ratio = (alpha + punctuation + spaces) / max(1, total)
    base = (alpha_ratio * 0.52) + (readable_ratio * 0.42) + uppercase_bonus
    score = base - repeated_penalty - mixed_digit_penalty - symbol_penalty
    return max(0.0, min(1.0, score))
