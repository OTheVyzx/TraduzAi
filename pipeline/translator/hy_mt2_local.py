"""Explicit localhost Hy-MT2 GGUF provider using the approved source-v2 recipe."""
from __future__ import annotations

import json
import urllib.request
from urllib.parse import urlsplit

BACKEND = "hy_mt2_gguf_local"
DEFAULT_URL = "http://127.0.0.1:11438"
MODEL_SHA256 = "9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b"


def _post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("Hy-MT2: resposta HTTP invalida")
    return value


def translate(text: str, *, source_language: str, target_locale: str,
              url: str = DEFAULT_URL) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Hy-MT2 exige endpoint HTTP local")
    # Preserve the exact approved EN -> PT-BR prompt. Other pairs have an
    # explicit contract rather than silently applying the wrong language.
    if source_language.lower() not in {"en", "en-us", "en-gb"} or target_locale.lower() != "pt-br":
        raise ValueError(f"Hy-MT2 source-v2 sem receita aprovada para {source_language}->{target_locale}")
    prompt_text = ("Translate the following text into Brazilian Portuguese. "
                   "Note that you should only output the translated result without any additional explanation:\n" + text)
    base = url.rstrip("/")
    template = _post(base + "/apply-template", {"messages": [{"role": "user", "content": prompt_text}]}, 10)
    prompt = template.get("prompt")
    if not isinstance(prompt, str) or text not in prompt:
        raise RuntimeError("Hy-MT2: template perdeu o texto de entrada")
    recipe = dict(n_predict=384, seed=0, cache_prompt=False, stream=False,
                  return_tokens=True, min_p=0, typical_p=1,
                  frequency_penalty=0, presence_penalty=0, dry_multiplier=0,
                  mirostat=0, temperature=0, top_p=1, top_k=0,
                  repeat_penalty=1, repeat_last_n=0, samplers=["temperature"])
    response = _post(base + "/completion", {**recipe, "prompt": prompt}, 180)
    result = response.get("content")
    if response.get("truncated") or not isinstance(result, str) or not result.strip():
        raise RuntimeError("Hy-MT2: resposta vazia ou truncada")
    return result.strip()
