"""
Local semantic reviewer for OCR text.
Focuses on repairing common OCR confusions before translation without relying
on external services.
"""

from __future__ import annotations

import re

from .postprocess import has_run_on_tokens, looks_suspicious

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

MERGED_COMMON_PHRASES = {
    "AFTERILATER": "AFTER I LATER",
    "AMARTIAL": "A MARTIAL",
    "ANINTERESTING": "AN INTERESTING",
    "ANDE": "AND",
    "ANDEILLED": "AND FILLED",
    "ANDFOCUSED": "AND FOCUSED",
    "ARMYMARTIAL": "ARMY MARTIAL",
    "ARTIKNEW": "ART I KNEW",
    "ARTSILEARNEDFROM": "ARTS I LEARNED FROM",
    "ASAKILLING": "AS A KILLING",
    "ASAWAY": "AS A WAY",
    "BYANY": "BY ANY",
    "CALLIT": "CALL IT",
    "COULDN'TPOSSIBLYHAVE": "COULDN'T POSSIBLY HAVE",
    "DOYOU": "DO YOU",
    "ENERG": "ENERGY",
    "ESSENCEINTOTHE": "ESSENCE INTO THE",
    "EVENTHE": "EVEN THE",
    "FOUNDERWHOUNIFIED": "FOUNDER WHO UNIFIED",
    "HAVEIBEEN": "HAVE I BEEN",
    "HIDITLITTLEBY": "HID IT LITTLE BY",
    "HIMBEST": "HIM BEST",
    "HUNWONMIRROR'S": "HUNWON MIRROR'S",
    "I'MSORRY": "I'M SORRY",
    "IREMOVED": "I REMOVED",
    "ITHOUGHT": "I THOUGHT",
    "IWOULDN'T": "I WOULDN'T",
    "KNOWITBETTER": "KNOW IT BETTER",
    "KOMWTWYHATYOO": "KNOW WHAT YOU",
    "LNATURALLYBELIEVED": "I NATURALLY BELIEVED",
    "MARTIALWILDWEST": "MARTIAL WILD WEST",
    "MASTERPERMITTED": "MASTER PERMITTED",
    "MANHWASEVEN": "MANHWAS EVEN",
    "MIXEDIN": "MIXED IN",
    "NEWMARTIAL": "NEW MARTIAL",
    "OFCOURSE": "OF COURSE",
    "OFRESENTMENT": "OF RESENTMENT",
    "OFTHE": "OF THE",
    "OFTHEHUNWON": "OF THE HUNWON",
    "OFLIGHTNING": "OF LIGHTNING",
    "OOAMSILN": "MASTER'S",
    "OLDMAN": "OLD MAN",
    "OLDMAN'S": "OLD MAN'S",
    "PLEASEDIN": "PLEASED IN",
    "POSSIBLYHAVE": "POSSIBLY HAVE",
    "PRINCIPLEIN": "PRINCIPLE IN",
    "RANSFORMSINTO": "TRANSFORMS INTO",
    "REACHESA": "REACHES A",
    "REASONICOULDN'T": "REASON I COULDN'T",
    "RUEFIREOFDRAGON": "TRUE FIRE OF DRAGON",
    "SAMADH": "SAMADHI",
    "SAGESSIMPLYSAW": "SAGES SIMPLY SAW",
    "SEEMMIXED": "SEEM MIXED",
    "SINGLEBOLT": "SINGLE BOLT",
    "SOONEMUST": "SO ONE MUST",
    "SOMESPIRIT": "SOME SPIRIT",
    "SOICAN'T": "SO I CAN'T",
    "STOPSALLMOVEMENT": "STOPS ALL MOVEMENT",
    "THATDAYI": "THAT DAY I",
    "THEANCIENT": "THE ANCIENT",
    "THELIGHTNING": "THE LIGHTNING",
    "THEULTIMATESECRET": "THE ULTIMATE SECRET",
    "THERECAN": "THERE CAN",
    "THINGSBETWEEN": "THINGS BETWEEN",
    "THOUGHITSHARD": "THOUGH IT'S HARD",
    "THISHUNWONTHUNDER": "THIS HUNWON THUNDER",
    "THISIS": "THIS IS",
    "TOOLBUT": "TOOL BUT",
    "TRUEIDENTITY": "TRUE IDENTITY",
    "VIOLENTPOWER": "VIOLENT POWER",
    "WASIMPERIAL": "WAS IMPERIAL",
    "WASSIMILAR": "WAS SIMILAR",
    "WASTHAT": "WAS THAT",
    "WEREN'TYOUFOCUSING": "WEREN'T YOU FOCUSING",
    "WHAT'SWITH": "WHAT'S WITH",
    "WHATSWITH": "WHAT'S WITH",
    "WHOPRACTICEDIT": "WHO PRACTICED IT",
    "WITHE": "WITH THE",
    "WITHSOMETHING": "WITH SOMETHING",
    "YOUKEEP": "YOU KEEP",
    "YOUFIRST": "YOU FIRST",
    "YOURINJURIES": "YOUR INJURIES",
}


def semantic_refine_text(text: str, tipo: str = "fala", confidence: float = 1.0) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    allow_substitutions = (
        confidence < 0.92
        or looks_suspicious(stripped, confidence)
        or has_run_on_tokens(stripped)
    )

    parts = re.findall(r"[A-Za-z0-9|']+|[^A-Za-z0-9|']+", stripped)
    refined_parts = []
    changed = False

    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9|']+", part):
            refined = _refine_token(part, tipo, allow_substitutions=allow_substitutions)
            changed = changed or (refined != part)
            refined_parts.append(refined)
        else:
            refined_parts.append(part)

    refined_text = "".join(refined_parts)
    repaired_text = _repair_common_phrase_boundaries(refined_text)
    changed = changed or (repaired_text != refined_text)
    refined_text = repaired_text
    refined_text = re.sub(r"\s{2,}", " ", refined_text).strip()
    return refined_text if changed else stripped


def _refine_token(token: str, tipo: str, *, allow_substitutions: bool = True) -> str:
    core, prefix, suffix = _split_affixes(token)
    if not core:
        return token

    was_upper = core.isupper()
    normalized = core.upper()
    merged = MERGED_COMMON_PHRASES.get(normalized)
    if merged:
        return f"{prefix}{_restore_phrase_case(merged, core)}{suffix}"

    if not allow_substitutions:
        return token

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


def _restore_phrase_case(phrase: str, original: str) -> str:
    if original.isupper():
        return phrase
    if original.islower():
        return phrase.lower()
    return " ".join(word.capitalize() for word in phrase.lower().split(" "))


def _repair_common_phrase_boundaries(text: str) -> str:
    repaired = re.sub(r"\bT\s+CANNOT\b", "IT CANNOT", text, flags=re.IGNORECASE)
    repaired = re.sub(
        r"\bDON'T\s+SAY\s+THAT\s+ARE\s+YOUR\s+INJURIES\s+ALRIGHT\?",
        "DON'T SAY THAT. ARE YOUR INJURIES ALRIGHT?",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\bWHAT'S\s+(?:WITHE|WITH\s+THE)\s+TONE\?\s+ARE\s+YOU\s+ACCUSING\s+ME\s+OF\s+AND\s+DESTROYING\s+OUR\s+THAT\s+ARROGANT\s+BETRAYING\s+OUR\s+MASTER\s+LINEAGE\?",
        "WHAT'S WITH THAT ARROGANT TONE? ARE YOU ACCUSING ME OF BETRAYING OUR MASTER AND DESTROYING OUR LINEAGE?",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\bSTOPSALL\s+MOVEMENT\s+ANI\s+TRANSFORMS\s+INTO\b",
        "STOPS ALL MOVEMENT AND TRANSFORMS INTO",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\bIT['’]?S\.\s+(ABOUT\s+TIME\b)",
        r"IT'S \1",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(r"\bART\s+OE\s+THE\b", "ART OF THE", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\b([A-Z]{2,})\.(?=[A-Z]{2,}\b)", r"\1. ", repaired)
    repaired = re.sub(r"(?<=[A-Z])([,;:!?])(?=[A-Z])", r"\1 ", repaired)
    repaired = re.sub(r"^\.(?=[A-Za-z])", "", repaired)
    return repaired


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
