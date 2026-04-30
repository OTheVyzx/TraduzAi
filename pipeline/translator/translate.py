"""
Traducao EN->PT-BR.
Primario: Google Translate via deep-translator.
Fallback: Ollama (traduzai-translator ou qualquer modelo disponivel).
Agora com consciencia de tipo de texto, contexto local e memoria curta.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from utils.decision_log import record_decision
except ImportError:
    from ..utils.decision_log import record_decision

OLLAMA_HOST = "http://localhost:11434"

GOOGLE_LANGUAGE_ALIASES = {
    "pt-br": "pt",
    "pt_br": "pt",
    "pt-pt": "pt",
    "pt_pt": "pt",
    "en-gb": "en",
    "en-us": "en",
    "en_us": "en",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh_cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh_hans": "zh-CN",
    "zh-tw": "zh-TW",
    "zh_tw": "zh-TW",
    "zh-hant": "zh-TW",
    "zh_hant": "zh-TW",
}

OCR_DEDICATED_GOOGLE_CODES = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ru",
    "ja",
    "ko",
    "zh-CN",
    "zh-TW",
}

ADAPTATIONS: list[tuple[str, str, int]] = []

PRE_TRANSLATION_GLOSSARY: list[tuple[str, str, int]] = [
    # PGE: Prioridade Gramatical Explicativa — substituições para o Google traduzir corretamente
    (r"\bits useless\b", "it is futile", re.IGNORECASE),
    (r"\buseless\b", "futile", re.IGNORECASE),
]

SOURCE_OCR_REPAIRS: list[tuple[str, str, int]] = [
    (r"\bcoyld\b", "could", re.IGNORECASE),
    (r"\blightbes\b", "light be...?!", re.IGNORECASE),
    # OCR digit/letter confusions (números colados em palavras curtas em CAPS)
    (r"\bM9\b", "MY", 0),                    # 9 lido como Y
    (r"\bM0\b", "MY", 0),                    # 0 lido como Y
    (r"\bi(\d{2,4})\b", r"\1", 0),           # i999 -> 999, i99 -> 99
    (r"\bND SQUAD\b", "2ND SQUAD", 0),       # 2 inicial perdido
    (r"\bRD SQUAD\b", "3RD SQUAD", 0),       # 3 inicial perdido
    (r"\bTH SQUAD\b", "4TH SQUAD", 0),       # 4 inicial perdido
    # SFX comuns que o OCR pode ter classificado como "fala"
    (r"\bGHK\b", "Ngh", 0),                  # som estrangulado
    (r"\bWGH\b", "Ugh", 0),                  # som de esforço
    (r"\bNGH\b", "Ngh", 0),                  # já forma reconhecível
]

TRANSLATION_REVIEW_REPAIRS: list[tuple[str, str, int]] = [
    (r"VocÄ™|Vocę|VocÃª", "Você", re.IGNORECASE),
    (r"atravÃ©s|atraves", "através", re.IGNORECASE),
]

# Palavras inglesas curtas em CAPS que NÃO devem ser tratadas como nomes próprios.
# Tudo que sair do OCR como single-word CAPS e estiver fora desta lista é candidato
# a nome próprio (preserva-se em PT-BR ao invés de traduzir).
_COMMON_EN_CAPS_WORDS = frozenset({
    "YES", "NO", "OK", "OKAY", "YEAH", "YEP", "NAH", "HUH", "OH", "AH", "EH",
    "NONE",
    "HEY", "HI", "HELLO", "BYE", "GO", "STOP", "WAIT", "RUN", "LOOK", "SEE",
    "WHY", "HOW", "WHAT", "WHO", "WHERE", "WHEN", "WHICH", "DAMN", "GOD",
    "WELL", "WHOA", "WOW", "UGH", "ARGH", "GAH", "TSK", "HUSH", "QUIET",
    "OFF", "ON", "OUT", "IN", "UP", "DOWN", "BACK", "FORTH", "AWAY",
    "ALL", "AGAIN", "OUR", "MY", "YOUR", "HIS", "HER", "ITS", "THE",
    "THIS", "THAT", "THESE", "THOSE", "AND", "BUT", "OR", "FOR", "WITH",
    "TODO", "AGAIN", "DONE", "SAFE", "TRUE", "FALSE", "REAL", "GOOD", "BAD",
    "PLEASE", "SORRY", "THANKS", "WAKE", "SLEEP", "EAT", "DRINK", "FIGHT",
    "ATTACK", "DEFEND", "RETREAT", "CHARGE", "FIRE", "WATER", "EARTH", "AIR",
    "LIGHT", "DARK", "LIFE", "DEATH", "LOVE", "HATE", "WAR", "PEACE",
    "ENEMY", "FRIEND", "HELP", "SAVE", "KILL", "DIE", "LIVE",
    "MAGIC", "SPELL", "POWER", "FORCE", "WILL", "MIND", "SOUL", "HEART",
    "MERCY", "JUSTICE", "HONOR", "GLORY", "SHAME", "PRIDE",
    "MASTER", "LORD", "LADY", "SIR", "SIRE", "MILORD", "KING", "QUEEN",
    "PRINCE", "PRINCESS", "KNIGHT", "LIAR", "FOOL", "IDIOT",
    # Sons curtos comuns
    "HMM", "HMPH", "HRRG", "HRGH", "HRRR", "GRR", "RAWR", "ROAR",
})

# Mapeamento infinitivo PT-BR -> imperativo PT-BR (singular informal "você").
# Quando o Google traduz comandos curtos como "WAKE UP." ele tende a devolver
# infinitivo ("Acordar.") em vez de imperativo. Esta tabela corrige.
_IMPERATIVE_FIXES = {
    "acordar": "acorde",
    "parar": "pare",
    "olhar": "olhe",
    "correr": "corra",
    "fugir": "fuja",
    "atacar": "ataque",
    "lutar": "lute",
    "esperar": "espere",
    "ouvir": "ouça",
    "voltar": "volte",
    "sair": "saia",
    "entrar": "entre",
    "calar": "cale",
    "abrir": "abra",
    "fechar": "feche",
    "ler": "leia",
    "escrever": "escreva",
    "comer": "coma",
    "beber": "beba",
    "dormir": "durma",
    "andar": "ande",
    "pular": "pule",
    "subir": "suba",
    "descer": "desça",
    "matar": "mate",
    "morrer": "morra",
    "viver": "viva",
    "amar": "ame",
    "odiar": "odeie",
    "ajudar": "ajude",
    "salvar": "salve",
    "soltar": "solte",
    "segurar": "segure",
    "soltar-me": "solte-me",
    "largar": "largue",
    "pegar": "pegue",
    "agarrar": "agarre",
    "responder": "responda",
    "falar": "fale",
    "dizer": "diga",
    "contar": "conte",
    "explicar": "explique",
    "concentrar": "concentre",
    "concentrar-se": "concentre-se",
    "render": "renda",
    "render-se": "renda-se",
    "levantar": "levante",
    "levantar-se": "levante-se",
    "sentar": "sente",
    "sentar-se": "sente-se",
}

# Máximo de palavras numa source que pode ser interpretada como comando curto
# (heurística para acionar o fix de imperativo).
_IMPERATIVE_MAX_SOURCE_WORDS = 4


def _is_likely_proper_noun(token: str) -> bool:
    """Heurística: token isolado em ALL-CAPS, 4–18 letras, fora do dicionário comum.

    Usado para preservar nomes próprios (Gillion, Willow, Vanessa, Fenris, Desmond)
    que o Google Translate insiste em traduzir como substantivos comuns
    (UM BILHÃO, SALGUEIRO, etc).
    """
    cleaned = token.strip().rstrip(".,!?;:'\"")
    if not cleaned or " " in cleaned:
        return False
    # Só letras (sem dígitos, sem hífen)
    if not cleaned.isalpha():
        return False
    if not (4 <= len(cleaned) <= 18):
        return False
    # Tem que estar em CAPS (todo caps OU primeira letra maiúscula com >50% caps)
    upper_ratio = sum(1 for c in cleaned if c.isupper()) / len(cleaned)
    if upper_ratio < 0.9:
        return False
    if cleaned.upper() in _COMMON_EN_CAPS_WORDS:
        return False
    return True


def _fix_infinitive_to_imperative(translated: str, source: str, tipo: str) -> str:
    """Converte infinitivo PT-BR para imperativo quando a source é um comando curto.

    Aplica-se apenas a `tipo in {fala, narracao}` com source de até
    `_IMPERATIVE_MAX_SOURCE_WORDS` palavras e que termina em `.` ou `!`.
    """
    if tipo not in {"fala", "narracao"}:
        return translated
    src = source.strip()
    if not src:
        return translated
    word_count = len(re.findall(r"[A-Za-z']+", src))
    if word_count == 0 or word_count > _IMPERATIVE_MAX_SOURCE_WORDS:
        return translated
    if not (src.endswith("!") or src.endswith(".") or src.endswith("?")):
        return translated

    cleaned = translated.strip()
    # Match único token (possivelmente com pontuação no fim)
    m = re.fullmatch(r"([A-Za-zÀ-ÿ-]+)([!?.,;:\s]*)", cleaned)
    if not m:
        return translated
    token, suffix = m.group(1), m.group(2)
    lookup_key = token.lower()
    if lookup_key not in _IMPERATIVE_FIXES:
        return translated
    replacement = _IMPERATIVE_FIXES[lookup_key]
    # Preservar caps da forma original
    if token.isupper():
        replacement = replacement.upper()
    elif token[0].isupper():
        replacement = replacement[0].upper() + replacement[1:]
    return replacement + suffix


def normalize_google_language_code(language_code: str) -> str:
    code = (language_code or "").strip()
    if not code:
        return "en"

    normalized = code.replace("_", "-")
    lowered = normalized.lower()
    if lowered in GOOGLE_LANGUAGE_ALIASES:
        return GOOGLE_LANGUAGE_ALIASES[lowered]

    base = lowered.split("-", 1)[0]
    if base in GOOGLE_LANGUAGE_ALIASES:
        return GOOGLE_LANGUAGE_ALIASES[base]

    return base if "-" in normalized else lowered


def list_supported_google_languages() -> list[dict[str, str]]:
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target="en")
    languages = translator.get_supported_languages(as_dict=True)
    items = []
    for label, code in sorted(languages.items(), key=lambda item: item[0].lower()):
        normalized_code = normalize_google_language_code(str(code))
        items.append(
            {
                "code": str(code),
                "label": str(label).strip().capitalize(),
                "ocr_strategy": "dedicated"
                if normalized_code in OCR_DEDICATED_GOOGLE_CODES
                else "best_effort",
            }
        )
    return items


class _GoogleTranslator:
    def __init__(self, source="en", target="pt"):
        from deep_translator import GoogleTranslator

        source_code = normalize_google_language_code(source)
        target_code = normalize_google_language_code(target)
        self._translator = GoogleTranslator(source=source_code, target=target_code)
        self._cache: dict[str, str] = {}
        self.target = target_code
        self._source_lang = source_code
        self._target_lang = target_code
        # Optional cross-session persistent cache. Assigned externally via
        # `attach_persistent_cache` once the pipeline knows the models_dir.
        self._persistent_cache = None  # type: ignore[assignment]

    def attach_persistent_cache(self, cache) -> None:
        self._persistent_cache = cache

    def _persistent_lookup(self, key: str) -> Optional[str]:
        cache = self._persistent_cache
        if cache is None:
            return None
        try:
            return cache.get(key, self._source_lang, self._target_lang)
        except Exception as exc:
            logger.debug("Persistent cache lookup falhou: %s", exc)
            return None

    def _persistent_store(self, key: str, value: str) -> None:
        cache = self._persistent_cache
        if cache is None or value is None:
            return
        try:
            cache.set(key, self._source_lang, self._target_lang, value)
        except Exception as exc:
            logger.debug("Persistent cache store falhou: %s", exc)

    def translate(self, text: str) -> Optional[str]:
        key = text.strip()
        if key in self._cache:
            return self._cache[key]
        persisted = self._persistent_lookup(key)
        if persisted is not None:
            self._cache[key] = persisted
            return persisted

        for attempt in range(3):
            try:
                result = self._translator.translate(text)
                self._cache[key] = result
                if result:
                    self._persistent_store(key, result)
                return result
            except Exception:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        return None

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        logger.info(f"Traduzindo lote de {len(texts)} textos (Google)")

        # Segregar textos em cache dos novos (memoria -> disco -> rede)
        results: list[Optional[str]] = [None] * len(texts)
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            key = text.strip()
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                continue
            persisted = self._persistent_lookup(key)
            if persisted is not None:
                self._cache[key] = persisted
                results[i] = persisted
                continue
            uncached_indices.append(i)

        if not uncached_indices:
            return [r or "" for r in results]

        # Tentativa de tradução em lote com separador robusto
        uncached_texts = [texts[i] for i in uncached_indices]
        separator = "\n===\n"
        joined = separator.join(uncached_texts)
        batch_result = self.translate(joined)

        if batch_result:
            # Dividir resultados (Google às vezes remove espaços ao redor do separador)
            parts = [p.strip() for p in batch_result.split("===") if p.strip()]
            
            if len(parts) == len(uncached_texts):
                for idx, part in zip(uncached_indices, parts):
                    cleaned = part.strip()
                    results[idx] = cleaned
                    src_key = texts[idx].strip()
                    self._cache[src_key] = cleaned
                    if cleaned:
                        self._persistent_store(src_key, cleaned)
                return [r if r is not None else texts[i] for i, r in enumerate(results)]
            else:
                logger.warning(f"Batch split mismatch: {len(parts)} vs {len(uncached_texts)}. Tentando individualmente.")

        # Fallback: tradução individual para os pendentes
        for i in uncached_indices:
            if results[i] is None:
                trans = self.translate(texts[i])
                
                # Heurística: se a tradução for idêntica ao original em texto longo, 
                # pode indicar idioma de origem incorreto. Tenta 'auto'.
                if trans == texts[i] and len(texts[i]) > 8:
                    logger.info(f"Tradução redundante detectada em {texts[i][:20]}... Tentando 'auto' detection.")
                    try:
                        from deep_translator import GoogleTranslator
                        trans = GoogleTranslator(source='auto', target=self._translator.target).translate(texts[i])
                    except:
                        pass
                
                results[i] = trans or texts[i]

        return [r if r is not None else texts[i] for i, r in enumerate(results)]


def _check_ollama(host: str) -> dict:
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        models = [model["name"] for model in data.get("models", [])]
        has_translator = any(
            ("traduzai-translator" in model) or ("mangatl-translator" in model)
            for model in models
        )
        return {"running": True, "models": models, "has_translator": has_translator}
    except Exception:
        return {"running": False, "models": [], "has_translator": False}


def _pick_ollama_model(models: list[str], preferred: str) -> str:
    preferred = (preferred or "").strip()
    if preferred:
        for model in models:
            if preferred in model:
                return model

    for candidate in ("traduzai-translator", "mangatl-translator"):
        for model in models:
            if candidate in model:
                return model

    return models[0] if models else ""


def _call_ollama(model: str, system: str, user_msg: str, host: str) -> list[dict]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.15},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())

    content = data.get("message", {}).get("content", "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
    return parsed if isinstance(parsed, list) else []


def _postprocess(
    text: str,
    was_upper: bool,
    tipo: str = "fala",
    source_text: str = "",
    lang: str = "en",
) -> str:
    result = _review_translation_grammar_semantics(source_text, text.strip(), tipo, lang=lang)
    result = result.replace("\u2026", "...")
    for pattern, replacement, flags in ADAPTATIONS:
        result = re.sub(pattern, replacement, result, flags=flags)
    result = re.sub(r"\s+([!?.,;:])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result).strip()

    # Conserta infinitivo -> imperativo quando o source \u00e9 claramente um comando curto.
    # Precisa rodar antes do upper() final para casar a tabela em min\u00fasculas.
    if source_text:
        result = _fix_infinitive_to_imperative(result, source_text, tipo)

    if tipo == "sfx":
        result = result.upper()
    elif was_upper:
        result = result.upper()
    elif tipo == "narracao" and result:
        result = result[0].upper() + result[1:]

    return result


def _prepare_source_text_for_translation(text: str, tipo: str = "fala", lang: str = "en") -> str:
    result = text.strip()
    if not result:
        return result

    if tipo == "sfx":
        return re.sub(r"\s+", " ", result)

    for pattern, replacement, flags in SOURCE_OCR_REPAIRS:
        result = re.sub(pattern, replacement, result, flags=flags)

    result = re.sub(r"\s{2,}", " ", result).strip()
    
    # Nao forca capitalizacao para idiomas CJK
    is_cjk = lang in ("ja", "ko", "zh", "zh-CN", "zh-TW")
    if not is_cjk and result and len(result) > 2:
        result = result[0].upper() + result[1:].lower()
    return result


def _review_translation_grammar_semantics(
    source_text: str,
    translated_text: str,
    tipo: str = "fala",
    lang: str = "en",
) -> str:
    del tipo

    result = translated_text.strip()
    if not result:
        return result

    for pattern, replacement, flags in TRANSLATION_REVIEW_REPAIRS:
        result = re.sub(pattern, replacement, result, flags=flags)

    for pattern, replacement, flags in ADAPTATIONS:
        result = re.sub(pattern, replacement, result, flags=flags)

    prepared_source = _prepare_source_text_for_translation(source_text, "fala", lang=lang)
    normalized_source = re.sub(r"[\W_]+", " ", prepared_source.lower()).strip()
    if re.search(r"\bcould that light be\b", normalized_source):
        if re.search(r"\bluz\b", result, re.IGNORECASE) or re.search(r"\bacende\b", result, re.IGNORECASE):
            result = "Poderia ser aquela luz...?!"
    if re.search(r"\byou mean the power he got by trading with them\b", normalized_source):
        result = "Você quer dizer o poder que ele conseguiu em um acordo com eles?"
    if re.search(r"\bhalf of a mana technique\b", normalized_source) and re.search(
        r"\binstantly surpass\b", normalized_source
    ):
        result = (
            "Pode até ser só metade de uma técnica de mana, "
            "mas seus efeitos já são mais do que suficientes. "
            "Esse poder permite ultrapassar instantaneamente os próprios limites."
        )
    if re.search(r"\bif desmond ends up using that power\b", normalized_source):
        result = "Se Desmond usar esse poder..."
    if re.search(r"\bthe rampaging vanessa also used a similar mana technique\b", normalized_source):
        result = "A Vanessa em fúria também usava um método semelhante de circulação de mana."
    if re.search(r"\bi still have many enemies\b", normalized_source) and re.search(
        r"\bfar stronger than you\b", normalized_source
    ):
        result = "E eu ainda tenho muitos inimigos. Inimigos muito mais fortes do que você."
    if re.search(r"\bwhen i killed you before\b", normalized_source) and re.search(
        r"\bsliced you in half with a single strike\b", normalized_source
    ):
        result = "Quando te matei antes, parti você ao meio com um único golpe, então não percebi..."
    if re.search(r"\bcould this light be\b", normalized_source):
        result = "Pode ser essa luz...?!"
    if re.search(r"\byou could see all of my attacks\b", normalized_source):
        result = "Que conseguia ver todos os meus ataques?"
    if re.search(r"\byou said you could see through all my attacks right\b", normalized_source):
        if re.search(r"\b(ataques|golpes|através|enxergar|ver)\b", result, re.IGNORECASE):
            result = "Você disse que podia enxergar todos os meus golpes, certo?"

    result = re.sub(r"\s+([!?.,;:])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _preprocess_text(text: str, tipo: str = "fala", lang: str = "en") -> str:
    result = text.strip()
    if tipo == "sfx":
        return re.sub(r"\s+", " ", result)

    # Nao forca capitalizacao para idiomas CJK
    is_cjk = lang in ("ja", "ko", "zh", "zh-CN", "zh-TW")
    if not is_cjk and result and len(result) > 2:
        result = result[0].upper() + result[1:].lower()
    
    # Pré-tradução: substitui expressões para o Google traduzir corretamente
    for pattern, replacement, flags in PRE_TRANSLATION_GLOSSARY:
        result = re.sub(pattern, replacement, result, flags=flags)
    result = result.replace("...", "\u2026")
    return re.sub(r"\s+", " ", result)


def _normalize_memory_key(text: str, tipo: str) -> str:
    normalized = re.sub(r"[\W_]+", "", text.lower())
    return f"{tipo}:{normalized}"


def _normalized_translation_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _normalize_entity_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _normalize_entity_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def _entity_candidates(context: dict, glossario: dict) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def _push(source: str, target: str, kind: str) -> None:
        clean_source = " ".join(str(source or "").split()).strip()
        clean_target = " ".join(str(target or "").split()).strip()
        source_norm = _normalize_entity_key(clean_source)
        target_norm = _normalize_entity_key(clean_target)
        if not clean_source or not source_norm:
            return
        fingerprint = (kind, source_norm, target_norm)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        candidates.append(
            {
                "source": clean_source,
                "target": clean_target or clean_source,
                "kind": kind,
                "source_norm": source_norm,
                "target_norm": target_norm,
                "source_word_count": len(clean_source.split()),
            }
        )

    for character in context.get("personagens", []) or []:
        _push(character, character, "character")
    for alias in context.get("aliases", []) or []:
        _push(alias, alias, "alias")
    for term in context.get("termos", []) or []:
        _push(term, (glossario or {}).get(term) or (context.get("memoria_lexical", {}) or {}).get(term) or term, "term")
    for faction in context.get("faccoes", []) or []:
        _push(faction, faction, "faction")

    for source, target in (context.get("memoria_lexical", {}) or {}).items():
        _push(source, target, "memory")
    for source, target in (context.get("corpus_memoria_lexical", {}) or {}).items():
        _push(source, target, "corpus_memory")
    for source, target in (glossario or {}).items():
        _push(source, target, "glossary")

    return candidates


def _looks_like_entity_label(text: str) -> bool:
    compact = " ".join((text or "").split()).strip(" .!?…,:;\"'()[]{}")
    if not compact or len(compact) > 48:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’-]*", compact)
    if not words or len(words) > 4:
        return False
    total_alpha = sum(1 for ch in compact if ch.isalpha())
    return total_alpha >= 4


def _repair_source_entities(text: str, context: dict, glossario: dict) -> tuple[str, list[dict], list[str]]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned or not _looks_like_entity_label(cleaned):
        return cleaned, [], []

    normalized = _normalize_entity_key(cleaned)
    if not normalized:
        return cleaned, [], []

    word_count = len(cleaned.split())
    best_candidate: dict | None = None
    best_score = 0.0

    for candidate in _entity_candidates(context, glossario):
        if abs(candidate["source_word_count"] - word_count) > 1:
            continue
        candidate_norm = candidate["source_norm"]
        score = SequenceMatcher(None, normalized, candidate_norm).ratio()
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate and best_score >= 0.84 and cleaned != best_candidate["source"]:
        repair = {
            "phase": "source",
            "kind": best_candidate["kind"],
            "from": cleaned,
            "to": best_candidate["source"],
        }
        return best_candidate["source"], [repair], ["source_entity_repaired"]

    return cleaned, [], []


def _replace_canonical_phrase(text: str, canonical_phrase: str) -> tuple[str, bool]:
    if not text or not canonical_phrase:
        return text, False

    normalized_text = _normalize_entity_search_text(text)
    normalized_phrase = _normalize_entity_search_text(canonical_phrase)
    if not normalized_phrase:
        return text, False

    position = normalized_text.find(normalized_phrase)
    if position < 0:
        return text, False

    index_map: list[int] = []
    cursor = 0
    for index, ch in enumerate(text):
        expanded = unicodedata.normalize("NFKD", ch)
        expanded = "".join(item for item in expanded if not unicodedata.combining(item))
        expanded = expanded.lower()
        if not expanded:
            continue
        for _ in expanded:
            if cursor >= len(normalized_text):
                break
            index_map.append(index)
            cursor += 1

    if not index_map or position + len(normalized_phrase) - 1 >= len(index_map):
        return text, False

    start = index_map[position]
    end = index_map[position + len(normalized_phrase) - 1] + 1
    replaced = f"{text[:start]}{canonical_phrase}{text[end:]}"
    return replaced, replaced != text


def _apply_target_entity_locks(
    source_text: str,
    translated_text: str,
    context: dict,
    glossario: dict,
) -> tuple[str, list[dict], list[str]]:
    result = translated_text
    hits: list[dict] = []
    flags: list[str] = []
    source_key = _normalize_entity_key(source_text)
    if not source_key:
        return result, hits, flags

    glossary_like_kinds = {"glossary", "memory", "corpus_memory", "term"}

    for candidate in _entity_candidates(context, glossario):
        target = candidate["target"]
        target_norm = candidate["target_norm"]
        if not target or not target_norm:
            continue
        if candidate["source_norm"] not in source_key:
            continue

        updated, changed = _replace_canonical_phrase(result, target)
        if changed:
            result = updated
        if changed and candidate["kind"] not in glossary_like_kinds:
            continue
        if (changed or target_norm in _normalize_entity_key(result)) and candidate["kind"] in glossary_like_kinds:
            hit = {
                "phase": "target",
                "source": candidate["source"],
                "target": target,
            }
            if hit not in hits:
                hits.append(hit)
            if "glossary_locked" not in flags:
                flags.append("glossary_locked")

    return result, hits, flags


def _best_entity_candidate_for_span_v2(
    span_text: str,
    context: dict,
    glossario: dict,
    *,
    expected_word_count: int | None = None,
) -> tuple[dict | None, float]:
    normalized = _normalize_entity_key(span_text)
    if not normalized:
        return None, 0.0

    best_candidate: dict | None = None
    best_score = 0.0
    span_word_count = len(span_text.split())

    for candidate in _entity_candidates(context, glossario):
        candidate_word_count = int(candidate["source_word_count"] or 0)
        reference_word_count = expected_word_count if expected_word_count is not None else span_word_count
        if abs(candidate_word_count - reference_word_count) > 1:
            continue
        score = SequenceMatcher(None, normalized, candidate["source_norm"]).ratio()
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate, best_score


def _repair_entity_phrase_inside_sentence_v2(
    text: str,
    context: dict,
    glossario: dict,
) -> tuple[str, list[dict], list[str]]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return cleaned, [], []

    token_matches = list(re.finditer(r"[^\W_][^\W_'’-]*", cleaned, flags=re.UNICODE))
    if len(token_matches) < 2 or len(token_matches) > 8:
        return cleaned, [], []

    best_payload: tuple[float, int, int, dict] | None = None
    for candidate in _entity_candidates(context, glossario):
        candidate_word_count = int(candidate["source_word_count"] or 0)
        if candidate_word_count < 2:
            continue
        for window_size in {candidate_word_count - 1, candidate_word_count, candidate_word_count + 1}:
            if window_size < 2 or window_size > len(token_matches):
                continue
            for start_index in range(0, len(token_matches) - window_size + 1):
                end_index = start_index + window_size - 1
                start = token_matches[start_index].start()
                end = token_matches[end_index].end()
                span_text = cleaned[start:end]
                if not span_text or span_text == candidate["source"]:
                    continue
                score = SequenceMatcher(None, _normalize_entity_key(span_text), candidate["source_norm"]).ratio()
                if score < 0.84:
                    continue
                if best_payload is None or score > best_payload[0]:
                    best_payload = (score, start, end, candidate)

    if best_payload is None:
        return cleaned, [], []

    _, start, end, candidate = best_payload
    repaired_text = f"{cleaned[:start]}{candidate['source']}{cleaned[end:]}"
    repair = {
        "phase": "source",
        "kind": candidate["kind"],
        "from": cleaned[start:end],
        "to": candidate["source"],
    }
    return repaired_text, [repair], ["source_entity_repaired"]


def _repair_source_entities(text: str, context: dict, glossario: dict) -> tuple[str, list[dict], list[str]]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return cleaned, [], []

    entity_label_candidate = _looks_like_entity_label(cleaned) and not re.search(r"[,.!?…:;]", cleaned)
    if entity_label_candidate:
        best_candidate, best_score = _best_entity_candidate_for_span_v2(cleaned, context, glossario)
        if best_candidate and best_score >= 0.84 and cleaned != best_candidate["source"]:
            repair = {
                "phase": "source",
                "kind": best_candidate["kind"],
                "from": cleaned,
                "to": best_candidate["source"],
            }
            return best_candidate["source"], [repair], ["source_entity_repaired"]

    repaired_text, repairs, flags = _repair_entity_phrase_inside_sentence_v2(cleaned, context, glossario)
    if repairs:
        return repaired_text, repairs, flags

    return cleaned, [], []


def _should_repair_local_translation(source_text: str, translated_text: str) -> bool:
    translated = translated_text.strip()
    if not translated:
        return True

    source_key = _normalized_translation_key(source_text)
    translated_key = _normalized_translation_key(translated)
    return bool(source_key) and source_key == translated_key


def _build_context_hints(context: dict, glossario: dict) -> str:
    aliases = ", ".join(context.get("aliases", [])[:10]) or "N/A"
    termos = ", ".join(context.get("termos", [])[:12]) or "N/A"
    faccoes = ", ".join(context.get("faccoes", [])[:10]) or "N/A"
    relacoes = ", ".join(context.get("relacoes", [])[:10]) or "N/A"
    arcos = " | ".join(context.get("resumo_por_arco", [])[:3]) or "N/A"
    memoria = dict(context.get("memoria_lexical", {}) or {})
    memoria.update(glossario or {})
    memoria_json = json.dumps(memoria, ensure_ascii=False)
    corpus_candidates = context.get("corpus_memory_candidates", [])[:6]
    corpus_memory = " | ".join(
        f"{item.get('source_text', '')} => {item.get('target_text', '')}"
        for item in corpus_candidates
        if item.get("source_text") and item.get("target_text")
    ) or "N/A"
    return (
        f"ALIASES: {aliases}\n"
        f"TERMOS: {termos}\n"
        f"FACCOES: {faccoes}\n"
        f"RELACOES: {relacoes}\n"
        f"ARCOS: {arcos}\n"
        f"MEMORIA_LEXICAL: {memoria_json}\n"
        f"MEMORIA_CORPUS: {corpus_memory}"
    )


def _lookup_memory_translation(text: str, tipo: str, context: dict, glossario: dict) -> str | None:
    del tipo
    normalized = re.sub(r"[\W_]+", "", text.lower())
    merged = dict(context.get("memoria_lexical", {}) or {})
    merged.update(context.get("corpus_memoria_lexical", {}) or {})
    merged.update(glossario or {})
    for source, target in merged.items():
        source_key = re.sub(r"[\W_]+", "", source.lower())
        if source_key and source_key == normalized:
            return target
    return None


def _lookup_special_literal_translation(text: str, tipo: str) -> str | None:
    del tipo
    stripped = str(text or "").strip()
    normalized = re.sub(r"[\W_]+", "", stripped.lower())
    if normalized != "none":
        return None
    suffix = stripped[len(stripped.rstrip("!?.")):] if stripped else ""
    punct = suffix or "."
    return f"Nenhuma{punct}"


def _build_text_payload(texts: list[dict], index: int, history_tail: list[dict]) -> dict:
    current = texts[index]
    before = texts[index - 1].get("text", "") if index > 0 else ""
    after = texts[index + 1].get("text", "") if index + 1 < len(texts) else ""
    return {
        "id": f"t{index + 1}",
        "text": current.get("text", ""),
        "tipo": current.get("tipo", "fala"),
        "context_before": before,
        "context_after": after,
        "history_tail": history_tail[-3:],
    }


_google: Optional[_GoogleTranslator] = None


def _prefer_local_translation_backend() -> bool:
    flag = (
        os.getenv("TRADUZAI_PREFER_LOCAL_TRANSLATION")
        or os.getenv("MANGATL_PREFER_LOCAL_TRANSLATION")
        or "0"
    )
    return str(flag).strip().lower() not in {"0", "false", "no", "off"}


def _resolve_translation_backend(google_ok: bool, ollama_status: dict) -> str:
    ollama_ready = bool(ollama_status.get("running")) and bool(ollama_status.get("models"))

    if _prefer_local_translation_backend() and ollama_ready:
        return "ollama"

    # Caminho padrao: Google primeiro, Ollama como fallback local.
    if google_ok:
        return "google"

    if ollama_ready:
        return "ollama"
    return "passthrough"


def translate_pages(
    ocr_results: list[dict],
    obra: str,
    context: dict,
    glossario: dict,
    idioma_destino: str = "pt-BR",
    idioma_origem: str = "en",
    qualidade: str = "normal",
    ollama_host: str = OLLAMA_HOST,
    ollama_model: str = "traduzai-translator",
    progress_callback: Callable | None = None,
    models_dir: str = "",
) -> list[dict]:
    del qualidade

    global _google
    idioma_origem = normalize_google_language_code(idioma_origem)
    idioma_destino = normalize_google_language_code(idioma_destino)

    persistent_cache = _open_persistent_cache(models_dir, idioma_origem, idioma_destino)

    google_ok = False
    try:
        if (
            _google is None
            or getattr(_google, "_source_lang", "en") != idioma_origem
            or getattr(_google, "_target_lang", "pt") != idioma_destino
        ):
            _google = _GoogleTranslator(source=idioma_origem, target=idioma_destino)
            _google._source_lang = idioma_origem
            _google._target_lang = idioma_destino
        if persistent_cache is not None:
            _google.attach_persistent_cache(persistent_cache)
        _google.translate("test")
        google_ok = True
    except Exception as exc:
        logger.warning(f"Google Translate indisponivel: {exc}")

    ollama = _check_ollama(ollama_host)
    backend = _resolve_translation_backend(google_ok=google_ok, ollama_status=ollama)

    logger.info(f"--- TRADUCAO INICIADA ---")
    logger.info(f"Backend selecionado: {backend}")
    logger.info(f"Idioma Origem: {idioma_origem}")
    logger.info(f"Idioma Destino: {idioma_destino}")
    logger.info(f"Google OK: {google_ok}")
    logger.info(f"-------------------------")
    record_decision(
        stage="translate",
        action="select_backend",
        reason=backend,
        details={
            "google_ok": google_ok,
            "ollama_running": bool(ollama.get("running")),
            "idioma_origem": idioma_origem,
            "idioma_destino": idioma_destino,
        },
    )

    try:
        if backend == "google":
            logger.info("Traducao usando Google Translate.")
            return _translate_with_google(ocr_results, context, glossario, progress_callback, idioma_origem=idioma_origem)

        if backend == "ollama":
            model = _pick_ollama_model(ollama["models"], ollama_model)
            logger.info("Traducao usando backend local Ollama: %s", model)
            return _translate_with_ollama(
                ocr_results,
                obra,
                context,
                glossario,
                idioma_destino,
                idioma_origem,
                model,
                ollama_host,
                _google if google_ok else None,
                progress_callback,
            )

        logger.warning("Nenhum backend de traducao disponivel. Retornando texto original.")
        return _passthrough(ocr_results, progress_callback)
    finally:
        if persistent_cache is not None:
            try:
                persistent_cache.flush()
            except Exception as exc:
                logger.debug("Falha ao flush do cache persistente: %s", exc)


def _open_persistent_cache(models_dir: str, src_lang: str, tgt_lang: str):
    if not models_dir:
        return None
    try:
        try:
            from translator.cache import PersistentTranslationCache
        except ImportError:
            from .cache import PersistentTranslationCache
        from pathlib import Path as _Path
        return PersistentTranslationCache.for_language_pair(
            _Path(models_dir) / "cache",
            src_lang,
            tgt_lang,
        )
    except Exception as exc:
        logger.debug("Cache persistente indisponivel: %s", exc)
        return None


def _translate_with_google(
    ocr_results: list[dict],
    context: dict,
    glossario: dict,
    progress_callback: Callable | None,
    idioma_origem: str = "en",
) -> list[dict]:
    total = len(ocr_results)
    translated_pages = []
    history_memory: dict[str, str] = {}
    history_tail: list[dict] = []

    for page_idx, ocr_page in enumerate(ocr_results):
        translated, history_tail = _translate_google_single_page(
            page_idx=page_idx,
            total=total,
            ocr_page=ocr_page,
            context=context,
            glossario=glossario,
            idioma_origem=idioma_origem,
            history_memory=history_memory,
            history_tail=history_tail,
            progress_callback=progress_callback,
        )
        translated_pages.append(translated)

    return translated_pages


def _translate_google_single_page(
    page_idx: int,
    total: int,
    ocr_page: dict,
    context: dict,
    glossario: dict,
    idioma_origem: str,
    history_memory: dict[str, str],
    history_tail: list[dict],
    progress_callback: Callable | None,
) -> tuple[dict, list[dict]]:
    """Translate a single page using Google backend with shared history state.

    Returns (translated_page_dict, updated_history_tail).
    The `history_memory` dict is mutated in-place.
    """
    is_cjk = idioma_origem in ("ja", "ko", "zh", "zh-CN", "zh-TW")

    texts = ocr_page.get("texts", [])
    if not texts:
        if progress_callback:
            progress_callback(page_idx + 1, total, f"Pagina {page_idx + 1}: sem texto")
        return {"texts": []}, history_tail

    raw_texts = [text.get("text", "") for text in texts]
    tipos = [text.get("tipo", "fala") for text in texts]
    repaired_sources: list[str] = []
    source_repairs: list[list[dict]] = []
    source_entity_flags: list[list[str]] = []

    for source in raw_texts:
        repaired_source, repairs, flags = _repair_source_entities(source, context, glossario)
        repaired_sources.append(repaired_source)
        source_repairs.append(repairs)
        source_entity_flags.append(flags)

    # Para CJK, was_upper nao faz sentido da mesma forma (seria True para tudo)
    if is_cjk:
        was_uppers = [False] * len(raw_texts)
    else:
        was_uppers = [text == text.upper() and any(c.isalpha() for c in text) for text in raw_texts]

    preprocessed: list[str] = []
    for repaired_text, tipo in zip(repaired_sources, tipos):
        prepared_text = _prepare_source_text_for_translation(repaired_text, tipo, lang=idioma_origem)
        prepared_text, _, _ = _repair_source_entities(prepared_text, context, glossario)
        normalized_text = _preprocess_text(
            prepared_text,
            tipo,
            lang=idioma_origem,
        )
        normalized_text, _, _ = _repair_source_entities(normalized_text, context, glossario)
        preprocessed.append(normalized_text)

    translations = [""] * len(texts)
    pending_indices = []
    pending_texts = []
    for index, (source, tipo, prepared) in enumerate(zip(raw_texts, tipos, preprocessed)):
        if texts[index].get("skip_processing"):
            translations[index] = source
            continue
        # Proteção de nome próprio: se o source é um único token CAPS fora do
        # vocabulário comum, preserva como estava (Title-case) para evitar
        # "GILLION" -> "UM BILHÃO", "WILLOW" -> "SALGUEIRO", etc.
        proper_noun_token = source.strip().rstrip(".,!?;:'\"")
        if _is_likely_proper_noun(source) or _is_likely_proper_noun(proper_noun_token):
            preserved = proper_noun_token.title()
            # Reanexa a pontuação final, se houver
            tail = source.strip()[len(proper_noun_token):]
            translations[index] = preserved + tail
            texts[index]["proper_noun_preserved"] = True
            continue
        special_literal = _lookup_special_literal_translation(source, tipo)
        if special_literal:
            translations[index] = special_literal
            continue
        memory_translation = _lookup_memory_translation(source, tipo, context, glossario)
        if memory_translation:
            translations[index] = memory_translation
            continue
        memory_key = _normalize_memory_key(source, tipo)
        if memory_key in history_memory:
            translations[index] = history_memory[memory_key]
        else:
            pending_indices.append(index)
            pending_texts.append(prepared)

    if pending_texts:
        try:
            pending_translations = _google.translate_batch(pending_texts)
        except Exception as exc:
            logger.warning(f"Batch falhou na pagina {page_idx + 1}: {exc}")
            pending_translations = pending_texts

        for index, translated in zip(pending_indices, pending_translations):
            translations[index] = translated

    page_texts = []
    for index, (original, translated, was_upper, tipo) in enumerate(
        zip(raw_texts, translations, was_uppers, tipos)
    ):
        final = _postprocess(
            translated or original,
            was_upper,
            tipo,
            source_text=repaired_sources[index] or original,
            lang=idioma_origem,
        )
        locked_final, glossary_hits, target_flags = _apply_target_entity_locks(
            repaired_sources[index] or original,
            final,
            context,
            glossario,
        )
        entity_flags = list(dict.fromkeys([*source_entity_flags[index], *target_flags]))
        entity_repairs = list(source_repairs[index])
        qa_flags = ["entity_suspect"] if entity_repairs else []
        texts[index]["entity_flags"] = entity_flags
        texts[index]["entity_repairs"] = entity_repairs
        texts[index]["glossary_hits"] = glossary_hits
        texts[index]["qa_flags"] = qa_flags
        if entity_repairs:
            for repair in entity_repairs:
                record_decision(
                    stage="translate",
                    action="repair_entity",
                    reason="source_entity_match",
                    page=page_idx + 1,
                    layer=texts[index].get("id") or f"t{index + 1}",
                    text=original,
                    details=repair,
                )
        if glossary_hits:
            record_decision(
                stage="translate",
                action="lock_glossary_term",
                reason="target_entity_lock",
                page=page_idx + 1,
                layer=texts[index].get("id") or f"t{index + 1}",
                text=original,
                details={"hits": glossary_hits},
            )
        memory_key = _normalize_memory_key(original, tipo)
        history_memory[memory_key] = locked_final
        payload = _build_text_payload(texts, index, history_tail)
        page_texts.append(
            {
                **texts[index],
                "original": original,
                "translated": locked_final,
                "tipo": tipo,
                "context_before": payload["context_before"],
                "context_after": payload["context_after"],
            }
        )
        history_tail.append({"source": original, "translated": locked_final, "tipo": tipo})
        history_tail = history_tail[-8:]

    if progress_callback:
        progress_callback(
            page_idx + 1,
            total,
            f"[Google] Pagina {page_idx + 1}/{total} - {len(texts)} textos",
        )

    return {"texts": page_texts}, history_tail


def _translate_with_ollama(
    ocr_results: list[dict],
    obra: str,
    context: dict,
    glossario: dict,
    idioma_destino: str,
    idioma_origem: str,
    model: str,
    host: str,
    repair_translator: Optional[_GoogleTranslator],
    progress_callback: Callable | None,
) -> list[dict]:
    total = len(ocr_results)
    system = (
        f"Voce e um tradutor de manga especializado em {idioma_origem}->{idioma_destino}. Responda SOMENTE com JSON array.\n"
        f"OBRA: {obra}\n"
        f"PERSONAGENS: {', '.join(context.get('personagens', [])[:8]) or 'N/A'}\n"
        f"GLOSSARIO: {json.dumps(glossario, ensure_ascii=False)}\n"
        f"{_build_context_hints(context, glossario)}\n"
        "Formato: [{\"id\":\"t1\",\"translated\":\"texto\"}]"
    )

    translated_pages = []
    history_tail: list[dict] = []
    for page_idx, ocr_page in enumerate(ocr_results):
        texts = ocr_page.get("texts", [])
        if not texts:
            translated_pages.append({"texts": []})
            if progress_callback:
                progress_callback(page_idx + 1, total, f"Pagina {page_idx + 1}: sem texto")
            continue

        text_list = [
            _build_text_payload(texts, i, history_tail)
            for i, t in enumerate(texts)
            if not t.get("skip_processing")
        ]
        user_msg = f"Traduza:\n{json.dumps(text_list, ensure_ascii=False)}"

        try:
            translations = _call_ollama(model, system, user_msg, host)
            translated_map = {item["id"]: item.get("translated", "") for item in translations}
        except Exception:
            translated_map = {}

        repair_indices: list[int] = []
        repair_texts: list[str] = []
        if repair_translator is not None:
            for index, text_data in enumerate(texts):
                if text_data.get("skip_processing"):
                    continue
                original = text_data.get("text", "")
                candidate = translated_map.get(f"t{index + 1}", original)
                if _should_repair_local_translation(original, candidate):
                    tipo = text_data.get("tipo", "fala")
                    repair_indices.append(index)
                    repair_source, _, _ = _repair_source_entities(original, context, glossario)
                    prepared_repair = _prepare_source_text_for_translation(repair_source or original, tipo)
                    prepared_repair, _, _ = _repair_source_entities(prepared_repair, context, glossario)
                    normalized_repair = _preprocess_text(prepared_repair, tipo)
                    normalized_repair, _, _ = _repair_source_entities(normalized_repair, context, glossario)
                    repair_texts.append(normalized_repair)

        repaired_map: dict[int, str] = {}
        if repair_indices and repair_translator is not None:
            try:
                repaired_batch = repair_translator.translate_batch(repair_texts)
            except Exception:
                repaired_batch = []

            for index, repaired in zip(repair_indices, repaired_batch):
                if repaired and repaired.strip():
                    repaired_map[index] = repaired

        page_texts = []
        for index, text_data in enumerate(texts):
            original = text_data.get("text", "")
            tipo = text_data.get("tipo", "fala")
            repaired_source, entity_repairs, source_entity_flags = _repair_source_entities(original, context, glossario)
            if text_data.get("skip_processing"):
                page_texts.append(
                    {
                        **text_data,
                        "original": original,
                        "translated": original,
                        "tipo": tipo,
                        "entity_flags": list(source_entity_flags),
                        "entity_repairs": list(entity_repairs),
                        "glossary_hits": [],
                        "qa_flags": ["entity_suspect"] if entity_repairs else [],
                    }
                )
                history_tail.append({"source": original, "translated": original, "tipo": tipo})
                continue
            translated = repaired_map.get(index) or translated_map.get(f"t{index + 1}", original)
            memory_translation = _lookup_memory_translation(original, tipo, context, glossario)
            if memory_translation:
                translated = memory_translation
            special_literal = _lookup_special_literal_translation(original, tipo)
            if special_literal:
                translated = special_literal
            is_cjk = idioma_origem in ("ja", "ko", "zh", "zh-CN", "zh-TW")
            was_upper = False if is_cjk else (original == original.upper() and any(c.isalpha() for c in original))

            final = _postprocess(
                translated,
                was_upper,
                tipo,
                source_text=repaired_source or original,
                lang=idioma_origem,
            )
            locked_final, glossary_hits, target_flags = _apply_target_entity_locks(
                repaired_source or original,
                final,
                context,
                glossario,
            )
            entity_flags = list(dict.fromkeys([*source_entity_flags, *target_flags]))
            qa_flags = ["entity_suspect"] if entity_repairs else []
            text_data["entity_flags"] = entity_flags
            text_data["entity_repairs"] = list(entity_repairs)
            text_data["glossary_hits"] = glossary_hits
            text_data["qa_flags"] = qa_flags
            if entity_repairs:
                for repair in entity_repairs:
                    record_decision(
                        stage="translate",
                        action="repair_entity",
                        reason="source_entity_match",
                        page=page_idx + 1,
                        layer=text_data.get("id") or f"t{index + 1}",
                        text=original,
                        details=repair,
                    )
            if glossary_hits:
                record_decision(
                    stage="translate",
                    action="lock_glossary_term",
                    reason="target_entity_lock",
                    page=page_idx + 1,
                    layer=text_data.get("id") or f"t{index + 1}",
                    text=original,
                    details={"hits": glossary_hits},
                )
            page_texts.append(
                {
                    **text_data,
                    "original": original,
                    "translated": locked_final,
                    "tipo": tipo,
                }
            )
            history_tail.append({"source": original, "translated": locked_final, "tipo": tipo})
        history_tail = history_tail[-8:]

        translated_pages.append({"texts": page_texts})
        if progress_callback:
            progress_callback(page_idx + 1, total, f"[{model}] Pagina {page_idx + 1}/{total}")

    return translated_pages


def _passthrough(ocr_results: list[dict], progress_callback: Callable | None) -> list[dict]:
    total = len(ocr_results)
    result = []
    for index, page in enumerate(ocr_results):
        result.append(
            {
                "texts": [
                    {
                        **text,
                        "original": text.get("text", ""),
                        "translated": text.get("text", ""),
                        "tipo": text.get("tipo", "fala"),
                    }
                    for text in page.get("texts", [])
                ]
            }
        )
        if progress_callback:
            progress_callback(index + 1, total, f"Pagina {index + 1} (sem traducao)")
    return result


def translate_single_block(block: dict, project: dict):
    """Traduz um unico bloco usando o motor configurado no projeto."""
    global _google
    
    source_lang = project.get("idioma_origem", "en")
    target_lang = project.get("idioma_destino", "pt-BR")
    
    source_lang = normalize_google_language_code(source_lang)
    target_lang = normalize_google_language_code(target_lang)
    
    text = block.get("original", "").strip()
    if not text:
        return
        
    tipo = block.get("tipo", "fala")
    
    # Setup translator if needed
    if (
        _google is None
        or getattr(_google, "_source_lang", "en") != source_lang
        or getattr(_google, "_target_lang", "pt") != target_lang
    ):
        _google = _GoogleTranslator(source=source_lang, target=target_lang)
        _google._source_lang = source_lang
        _google._target_lang = target_lang
        
    prepared = _preprocess_text(_prepare_source_text_for_translation(text, tipo, lang=source_lang), tipo, lang=source_lang)
    translated = _google.translate(prepared) or text
    
    is_cjk = source_lang in ("ja", "ko", "zh", "zh-CN", "zh-TW")
    was_upper = False if is_cjk else (text == text.upper() and any(c.isalpha() for c in text))
    
    final = _postprocess(translated, was_upper, tipo, source_text=text, lang=source_lang)
    
    block["translated"] = final
    block["traduzido"] = final
