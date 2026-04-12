"""
Local semantic reviewer for OCR text.
Focuses on repairing common OCR confusions before translation without relying
on external services.
"""

from __future__ import annotations

import re

from .postprocess import looks_suspicious

COMMON_WORDS = {
    "A", "AM", "AN", "AND", "ARE", "AS", "AT", "BACK", "BE", "BUT", "BY",
    "CAN", "COME", "DON'T", "DOWN", "FOR", "FROM", "GET", "GO", "GOING",
    "GOOD", "HE", "HERE", "HOW", "I", "I'M", "IN", "IS", "IT", "IT'S",
    "JUST", "LET'S", "LOOK", "ME", "MOVE", "MY", "NO", "NOT", "NOW", "OF",
    "OFF", "ON", "ONE", "OR", "OUT", "OVER", "RUN", "SHE", "SO", "STOP",
    "TAKE", "THAT", "THE", "THEY", "THIS", "TO", "UP", "WAIT", "WE", "WHAT",
    "WHY", "WITH", "YOU", "YOUR",
}

SUBSTITUTIONS = {
    "0": ["O"],
    "1": ["I"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
    "|": ["I"],
}

CONTRACTIONS = {
    "IM": "I'M",
    "DONT": "DON'T",
    "CANT": "CAN'T",
    "WONT": "WON'T",
    "ITS": "IT'S",
    "ILL": "I'LL",
    "IVE": "I'VE",
    "ID": "I'D",
    "LETS": "LET'S",
    "THATS": "THAT'S",
    "THERES": "THERE'S",
}


def semantic_refine_text(text: str, tipo: str = "fala", confidence: float = 1.0) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    if confidence >= 0.88 and not looks_suspicious(stripped, confidence):
        return stripped

    parts = re.findall(r"[A-Za-z0-9|']+|[^A-Za-z0-9|']+", stripped)
    refined_parts = []
    changed = False

    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9|']+", part):
            refined = _refine_token(part, tipo)
            changed = changed or (refined != part)
            refined_parts.append(refined)
        else:
            refined_parts.append(part)

    refined_text = "".join(refined_parts)
    refined_text = re.sub(r"\s{2,}", " ", refined_text).strip()
    return refined_text if changed else stripped


def _refine_token(token: str, tipo: str) -> str:
    core, prefix, suffix = _split_affixes(token)
    if not core:
        return token

    was_upper = core.isupper()
    normalized = core.upper()
    variants = _generate_variants(normalized)
    best = max(variants, key=lambda variant: _score_token(variant, tipo))
    best = _restore_contractions(best)

    if was_upper:
        refined_core = best
    elif core[:1].isupper():
        refined_core = best.capitalize()
    else:
        refined_core = best.lower()

    return f"{prefix}{refined_core}{suffix}"


def _split_affixes(token: str) -> tuple[str, str, str]:
    prefix_match = re.match(r"^[^A-Za-z0-9|']+", token)
    suffix_match = re.search(r"[^A-Za-z0-9|']+$", token)
    prefix = prefix_match.group(0) if prefix_match else ""
    suffix = suffix_match.group(0) if suffix_match else ""
    start = len(prefix)
    end = len(token) - len(suffix) if suffix else len(token)
    return token[start:end], prefix, suffix


def _generate_variants(token: str) -> set[str]:
    variants = {token}
    queue = [token]

    while queue and len(variants) < 32:
        current = queue.pop(0)
        for index, char in enumerate(current):
            replacements = SUBSTITUTIONS.get(char)
            if not replacements:
                continue
            for replacement in replacements:
                variant = current[:index] + replacement + current[index + 1:]
                if variant not in variants:
                    variants.add(variant)
                    queue.append(variant)
                    if len(variants) >= 32:
                        break
            if len(variants) >= 32:
                break

    return variants


def _restore_contractions(token: str) -> str:
    compact = token.replace("'", "")
    return CONTRACTIONS.get(compact, token)


def _score_token(token: str, tipo: str) -> float:
    compact = token.replace("'", "")
    alpha = sum(char.isalpha() for char in token)
    digits = sum(char.isdigit() for char in token)
    symbols = sum(not char.isalnum() and char != "'" for char in token)
    vowels = sum(char in "AEIOU" for char in compact)

    score = 0.0
    if token in COMMON_WORDS or compact in CONTRACTIONS:
        score += 1.0
    if compact in COMMON_WORDS:
        score += 0.8
    if re.fullmatch(r"[A-Z]+'[A-Z]+", token):
        score += 0.35
    if vowels > 0:
        score += min(0.35, vowels * 0.08)
    if tipo == "fala" and len(token) <= 6:
        score += 0.05

    score += alpha * 0.04
    score -= digits * 0.32
    score -= symbols * 0.18
    return score
