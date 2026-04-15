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
from typing import Callable, Optional

logger = logging.getLogger(__name__)

OLLAMA_HOST = "http://localhost:11434"

ADAPTATIONS: list[tuple[str, str, int]] = []

PRE_TRANSLATION_GLOSSARY: list[tuple[str, str, int]] = [
    # PGE: Prioridade Gramatical Explicativa — substituições para o Google traduzir corretamente
    (r"\bits useless\b", "it is futile", re.IGNORECASE),
    (r"\buseless\b", "futile", re.IGNORECASE),
]

SOURCE_OCR_REPAIRS: list[tuple[str, str, int]] = [
    (r"\bcoyld\b", "could", re.IGNORECASE),
    (r"\blightbes\b", "light be...?!", re.IGNORECASE),
]

TRANSLATION_REVIEW_REPAIRS: list[tuple[str, str, int]] = [
    (r"VocÄ™|Vocę|VocÃª", "Você", re.IGNORECASE),
    (r"atravÃ©s|atraves", "através", re.IGNORECASE),
]


class _GoogleTranslator:
    def __init__(self, source="en", target="pt"):
        from deep_translator import GoogleTranslator

        # Mapeamento do TraduzAi para o GoogleTranslator
        # ja -> ja
        # ko -> ko
        # zh -> zh-CN
        # en -> en
        source_map = {
            "en": "en",
            "ja": "ja",
            "ko": "ko",
            "zh": "zh-CN",
        }.get(source, "en")
        
        self._translator = GoogleTranslator(source=source_map, target=target)
        self._cache: dict[str, str] = {}

    def translate(self, text: str) -> Optional[str]:
        key = text.strip()
        if key in self._cache:
            return self._cache[key]

        for attempt in range(3):
            try:
                result = self._translator.translate(text)
                self._cache[key] = result
                return result
            except Exception:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        return None

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        logger.info(f"Traduzindo lote de {len(texts)} textos (Google)")

        # Segregar textos em cache dos novos
        results: list[Optional[str]] = [None] * len(texts)
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(text.strip())
            if cached is not None:
                results[i] = cached
            else:
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
                    self._cache[texts[idx].strip()] = cleaned
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
    
    # Preferencia do usuario: Google se estiver OK
    if google_ok:
        return "google"
        
    if _prefer_local_translation_backend() and ollama_ready:
        return "ollama"
    
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
) -> list[dict]:
    del qualidade

    global _google

    google_ok = False
    try:
        if _google is None or getattr(_google, "_source_lang", "en") != idioma_origem:
            _google = _GoogleTranslator(source=idioma_origem, target="pt")
            _google._source_lang = idioma_origem
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

    is_cjk = idioma_origem in ("ja", "ko", "zh", "zh-CN", "zh-TW")

    for page_idx, ocr_page in enumerate(ocr_results):
        texts = ocr_page.get("texts", [])
        if not texts:
            translated_pages.append({"texts": []})
            if progress_callback:
                progress_callback(page_idx + 1, total, f"Pagina {page_idx + 1}: sem texto")
            continue

        raw_texts = [text.get("text", "") for text in texts]
        tipos = [text.get("tipo", "fala") for text in texts]
        
        # Para CJK, was_upper nao faz sentido da mesma forma (seria True para tudo)
        if is_cjk:
            was_uppers = [False] * len(raw_texts)
        else:
            was_uppers = [text == text.upper() and any(c.isalpha() for c in text) for text in raw_texts]
            
        preprocessed = [
            _preprocess_text(_prepare_source_text_for_translation(text, tipo, lang=idioma_origem), tipo, lang=idioma_origem)
            for text, tipo in zip(raw_texts, tipos)
        ]

        translations = [""] * len(texts)
        pending_indices = []
        pending_texts = []
        for index, (source, tipo, prepared) in enumerate(zip(raw_texts, tipos, preprocessed)):
            if texts[index].get("skip_processing"):
                translations[index] = source
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
                source_text=original,
                lang=idioma_origem,
            )
            memory_key = _normalize_memory_key(original, tipo)
            history_memory[memory_key] = final
            payload = _build_text_payload(texts, index, history_tail)
            page_texts.append(
                {
                    **texts[index],
                    "original": original,
                    "translated": final,
                    "tipo": tipo,
                    "context_before": payload["context_before"],
                    "context_after": payload["context_after"],
                }
            )
            history_tail.append({"source": original, "translated": final, "tipo": tipo})
            history_tail = history_tail[-8:]

        translated_pages.append({"texts": page_texts})

        if progress_callback:
            progress_callback(
                page_idx + 1,
                total,
                f"[Google] Pagina {page_idx + 1}/{total} - {len(texts)} textos",
            )

    return translated_pages


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
                    repair_texts.append(_preprocess_text(_prepare_source_text_for_translation(original, tipo), tipo))

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
            if text_data.get("skip_processing"):
                page_texts.append({**text_data, "original": original, "translated": original, "tipo": tipo})
                history_tail.append({"source": original, "translated": original, "tipo": tipo})
                continue
            translated = repaired_map.get(index) or translated_map.get(f"t{index + 1}", original)
            memory_translation = _lookup_memory_translation(original, tipo, context, glossario)
            if memory_translation:
                translated = memory_translation
            is_cjk = idioma_origem in ("ja", "ko", "zh", "zh-CN", "zh-TW")
            was_upper = False if is_cjk else (original == original.upper() and any(c.isalpha() for c in original))

            final = _postprocess(
                translated,
                was_upper,
                tipo,
                source_text=original,
                lang=idioma_origem,
            )
            page_texts.append(
                {
                    **text_data,
                    "original": original,
                    "translated": final,
                    "tipo": tipo,
                }
            )
            history_tail.append({"source": original, "translated": final, "tipo": tipo})
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
