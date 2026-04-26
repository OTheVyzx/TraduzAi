"""
Post-processing helpers for OCR v2.
Keeps text cleanup, bbox normalization and lightweight classification logic
outside the main detector orchestration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_HF_BASES = [
    PROJECT_ROOT / "pk" / "huggingface",
    Path("T:/traduzai/pk/huggingface"),
    Path("T:/mangatl/pk/huggingface"),  # legado
    Path.home() / "AppData/Roaming/com.traduzai.app/huggingface",
    Path.home() / "AppData/Roaming/com.mangatl.app/huggingface",  # legado
]


def _find_hf_model(repo: str, filename: str) -> Path | None:
    """Localiza um arquivo de modelo HuggingFace nos diretórios locais."""
    for base in _HF_BASES:
        p = base / repo / filename
        if p.exists():
            return p
    return None

WATERMARK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"LAGOONSCANS?\.COM",
        r"ASURASCANS?\.COM",
        r"MEDIOCRESCAN\.COM",
        r"\b[\w.-]*(?:SCAN|SCANS|SCANLATOR|SCANLATIONS)[\w.-]*\b",
        r"\b[\w.-]*TOONS?[\w.-]*\b",
        r"mangabuddy",
        r"mangaflix",
        r"mangaball",
        r"ursaring",
        r"READ\s*ONLY\s*AT",
        r"WARNING\s*!",
        r"BETTER\s*QUALITY",
        r"MORE\s*CHAPTERS",
        r"FAST\s*UPDATES",
        r"MORE\s*CONTENT",
        r"LEIA\s*PRIMEIRO",
        r"DISCORD\.GG",
    ]
]

EDITORIAL_CREDIT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^(?:QC|TS|PR|RD|RAW)\s+[A-Z]{2,}$",
        r"\bORIGINAL\s+GOLD\s+LINE\s+ART\b",
        r"\b(?:SCAN|SCANS|SCANLATOR|SCANLATIONS|TOON|TOONS)\b",
    ]
]

KOREAN_PATTERN = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")

NON_LATIN_PATTERN = re.compile(
    r"[\u0400-\u04FF"   # Cirílico
    r"\u0600-\u06FF"    # Árabe
    r"\u0900-\u097F"    # Devanágari
    r"\u1100-\u11FF"    # Hangul Jamo
    r"\u3000-\u303F"    # Símbolos CJK
    r"\u3040-\u309F"    # Hiragana
    r"\u30A0-\u30FF"    # Katakana
    r"\u4E00-\u9FFF"    # Ideogramas CJK
    r"\uAC00-\uD7AF"    # Sílabas Hangul
    r"\uF900-\uFAFF]"   # Ideogramas CJK compat.
)
KOREAN_FIX = {
    "\uAEBC": "TH",
    "\uB300": "TCH",
    "\uC57C": "",
    "\uBC84": "B",
    "\uC0C8": "M",
    "\uB354": "TU",
    "\uC5B4": "ER",
    "\uAE4A": "",
    "\uB9AC": "LL",
    "\uD130": "T",
}


_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "3": "E",
    "4": "A",
    "5": "S",
    "7": "T",
    "8": "B",
}


def _fix_mixed_digit_word(word: str) -> str:
    """Corrige palavra que mistura dígitos e letras (erro comum de OCR em fontes estilizadas).

    Se poucos dígitos (1-2) estão grudados em letras no início/fim, remove-os
    (artefato OCR, ex: 'ELE1' → 'ELE', '7THE' → 'THE').
    Se muitos dígitos misturados, substitui por letras prováveis
    (ex: '350DDP5' → 'ESODDS').
    """
    # Separar pontuação ao redor
    prefix = ""
    suffix = ""
    core = word
    while core and not core[0].isalnum():
        prefix += core[0]
        core = core[1:]
    while core and not core[-1].isalnum():
        suffix = core[-1] + suffix
        core = core[:-1]

    if not core:
        return word

    digit_count = sum(c.isdigit() for c in core)
    letter_count = sum(c.isalpha() for c in core)

    # Só corrige se mistura dígitos e letras (não mexe em números puros)
    if digit_count == 0 or letter_count == 0:
        return word

    # Poucos dígitos (1-2) grudados em palavra com letras → remover dígitos
    # Ex: 'ELE1' → 'ELE', '1BLOQUEOU' → 'BLOQUEOU', 'TH3' → 'TH'
    if digit_count <= 2 and letter_count >= 2:
        cleaned = "".join(c for c in core if c.isalpha())
        return prefix + cleaned + suffix

    # Muitos dígitos misturados → substituir por letras prováveis
    corrected = ""
    for c in core:
        if c in _DIGIT_TO_LETTER:
            corrected += _DIGIT_TO_LETTER[c]
        else:
            corrected += c

    return prefix + corrected + suffix


def _remove_stray_digits(words: list[str]) -> list[str]:
    """Remove dígitos soltos (1-2 chars) que são artefatos de OCR em texto de mangá.

    Ex: ['7', 'ACABOU'] → ['ACABOU']
    Só remove se o texto tem outras palavras reais (letras). Não remove números
    que parecem intencionais (3+ dígitos como '100', '999').
    """
    if len(words) <= 1:
        return words

    has_real_words = any(
        any(c.isalpha() for c in w) for w in words
    )
    if not has_real_words:
        return words

    result = []
    for w in words:
        core = w.strip(".,!?;:\"'()-")
        # Dígito solto de 1-2 chars no meio de texto = artefato OCR
        if core.isdigit() and len(core) <= 2:
            continue
        result.append(w)
    return result


def fix_ocr_errors(text: str, idioma_origem: str = "en") -> str:
    if not text:
        return ""

    for korean, latin in KOREAN_FIX.items():
        text = text.replace(korean, latin)

    text = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)
    meaningful = re.sub(r"[\s\W]", "", text)
    if meaningful:
        korean_count = len(KOREAN_PATTERN.findall(text))
        # Remove coreano apenas se não for o idioma de origem e parecer ruído (menos de 50%)
        if idioma_origem != "ko" and 0 < korean_count / len(meaningful) < 0.5:
            text = KOREAN_PATTERN.sub("", text)

    # Pipe → I
    text = text.replace("|", "I")

    # Corrigir palavras com dígitos misturados a letras (OCR confuso com fontes estilizadas)
    words = text.split()
    words = [_fix_mixed_digit_word(w) for w in words]

    # Remover dígitos soltos que são artefatos de OCR (ex: "7" sozinho no meio de texto)
    words = _remove_stray_digits(words)

    text = " ".join(words)
    return re.sub(r"\s{2,}", " ", text).strip()


def is_watermark(text: str) -> bool:
    return any(pattern.search(text) for pattern in WATERMARK_PATTERNS)


def is_editorial_credit(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in EDITORIAL_CREDIT_PATTERNS)


def is_non_english(text: str) -> bool:
    """Retorna True se o texto é dominado por caracteres não-latinos (CJK, Hangul, árabe, etc.).

    Textos com mais de 30% de caracteres não-latinos são considerados não-inglês
    e devem ser ignorados no inpainting e na tradução.
    """
    stripped = re.sub(r"[\s\W_]", "", text)
    if not stripped:
        return False
    non_latin_count = len(NON_LATIN_PATTERN.findall(text))
    return non_latin_count / len(stripped) > 0.3


def is_korean_sfx(text: str) -> bool:
    meaningful = re.sub(r"[\s\W]", "", text)
    if not meaningful:
        return False
    return len(KOREAN_PATTERN.findall(text)) / len(meaningful) > 0.3


def is_structured_ocr_payload(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or stripped[0] not in "{[":
        return False
    normalized = stripped.lower()
    expected_keys = ("source_bbox", "line_polygons", "text_pixel_bbox")
    return all(key in normalized for key in expected_keys)


_HALLUCINATION_PHRASES: list[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"the quick brown fox jumps? over the lazy dog",
        r"lorem ipsum",
        r"\bhello[,\s]+world\b",
        r"\bexample text\b",
        r"\bfoo bar\b",
        r"\bthe quick brown fox\b",
        r"\bjumps? over the lazy\b",
    ]
]

# Characters-per-pixel threshold: above this, a bbox is implausibly dense
# for manhwa/manga text (30 chars in 200×22 px ≈ 0.0068 chars/px).
_HALLUCINATION_CHARS_PER_PX = 0.006


def is_hallucination(text: str, bbox: list[int], confidence: float) -> bool:
    """Return True if the OCR output is almost certainly a model hallucination.

    Two independent signals trigger the guard:

    1. Known training-corpus phrases ("quick brown fox", "lorem ipsum", etc.)
       combined with moderate confidence (< 0.85).  A real comic rarely
       contains pangrams; seeing one at less-than-high confidence is a
       strong hallucination indicator.

    2. Geometric implausibility: bbox height <= 30 px AND text length > 30
       chars AND chars-per-pixel ratio exceeds the empirical limit for
       manhwa/manga fonts.  A 200×22 balloon edge cannot physically hold a
       44-char sentence in any legible manga typeface.

    The guard is conservative: it requires *both* geometry criteria to fire
    on their own, and only needs the phrase match plus a moderate-confidence
    cap for phrase-based detection.  Real short text in small boxes is never
    discarded by this function alone.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    # 1. Known hallucination phrases at non-high confidence
    if confidence < 0.85:
        for pattern in _HALLUCINATION_PHRASES:
            if pattern.search(stripped):
                return True

    # 2. Geometric implausibility
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        bbox_h = max(1, int(y2) - int(y1))
        bbox_w = max(1, int(x2) - int(x1))
        bbox_area = bbox_h * bbox_w
        char_count = len(stripped)
        density = char_count / float(bbox_area)
        if bbox_h <= 30 and char_count > 30 and density > _HALLUCINATION_CHARS_PER_PX:
            return True

    return False


def is_punctuation_only_noise(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return not any(char.isalnum() for char in stripped)


def is_short_ornamental_text(
    text: str,
    confidence: float,
    bbox: list[int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    tipo: str,
    is_white_balloon: bool,
    page_profile: str = "standard",
) -> bool:
    if page_profile != "cover_opening":
        return False
    if tipo != "narracao" or is_white_balloon:
        return False
    if confidence >= 0.72:
        return False
    stripped = re.sub(r"\s+", "", (text or "").strip())
    if not stripped or len(stripped) > 12:
        return False

    x1, y1, x2, y2 = bbox
    box_w = max(1, int(x2) - int(x1))
    box_h = max(1, int(y2) - int(y1))
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    box_area = box_w * box_h
    page_area = max(1, image_w * image_h)

    return box_h <= 56 and box_area <= int(page_area * 0.025)


def infer_page_profile(
    page_number: int | None,
    image_shape: tuple[int, int, int] | tuple[int, int],
    block_count: int,
) -> str:
    if page_number is None:
        return "standard"

    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    aspect_ratio = max(image_h, image_w) / float(max(1, min(image_h, image_w)))

    if page_number <= 2 and aspect_ratio >= 1.2:
        return "cover_opening"
    if page_number == 3 and block_count <= 4 and aspect_ratio >= 1.2:
        return "cover_opening"
    return "standard"


def is_cover_title_logo(
    text: str,
    bbox: list[int],
    confidence: float,
    image_shape: tuple[int, int, int] | tuple[int, int],
    tipo: str,
    is_white_balloon: bool,
    page_profile: str = "standard",
) -> bool:
    if page_profile != "cover_opening":
        return False

    stripped = " ".join((text or "").split()).strip()
    if not stripped:
        return False

    words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’-]*", stripped)
    if not words:
        return False

    punctuation_marks = re.findall(r"[.!?…,:;]", stripped)
    if punctuation_marks:
        return False

    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    page_area = max(1, image_w * image_h)
    area_ratio = (box_w * box_h) / float(page_area)
    width_ratio = box_w / float(max(1, image_w))
    alpha_chars = [char for char in stripped if char.isalpha()]
    uppercase_ratio = (
        sum(1 for char in alpha_chars if char.isupper()) / float(len(alpha_chars))
        if alpha_chars
        else 0.0
    )
    title_case_ratio = sum(1 for word in words if word[:1].isupper()) / float(max(1, len(words)))
    compact_len = len(re.sub(r"\s+", "", stripped))
    word_count = len(words)

    if is_white_balloon:
        return (
            tipo in {"fala", "narracao"}
            and 1 <= word_count <= 4
            and compact_len >= 10
            and box_h >= int(image_h * 0.08)
            and width_ratio >= 0.22
            and area_ratio <= 0.22
            and confidence <= 0.84
        )

    return (
        tipo == "narracao"
        and 2 <= word_count <= 10
        and compact_len >= 14
        and width_ratio >= 0.24
        and area_ratio <= 0.20
        and (uppercase_ratio >= 0.58 or title_case_ratio >= 0.72)
    )


def infer_block_profile(
    text: str,
    bbox: list[int],
    tipo: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    page_profile: str = "standard",
    is_white_balloon: bool = False,
) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    compact = re.sub(r"\s+", "", (text or "").strip())

    if (
        page_profile == "cover_opening"
        and tipo == "narracao"
        and not is_white_balloon
        and compact
        and len(compact) <= 12
        and box_h <= 56
        and (box_w * box_h) <= int(max(1, image_w * image_h) * 0.025)
    ):
        return "decorative_noise"

    if tipo == "narracao" and y1 <= int(image_h * 0.18) and box_w >= int(image_w * 0.32):
        return "top_narration"

    if is_white_balloon:
        return "white_balloon"

    return "standard"


def suspicious_confidence_threshold(block_profile: str, page_profile: str = "standard") -> float:
    if block_profile == "decorative_noise":
        return 0.68
    if block_profile == "white_balloon":
        return 0.55
    if block_profile == "top_narration":
        return 0.58
    if page_profile == "cover_opening":
        return 0.64
    return 0.60


def looks_suspicious(text: str, confidence: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    alnum = sum(char.isalnum() for char in stripped)
    alpha = sum(char.isalpha() for char in stripped)
    digits = sum(char.isdigit() for char in stripped)
    punctuation = sum(char in "'!?.,-:;" for char in stripped)
    weird = max(0, len(stripped) - alnum - punctuation - stripped.count(" "))
    repeated = bool(re.search(r"(.)\1{3,}", stripped))

    if confidence < 0.55:
        return True
    if alnum == 0:
        return True
    if alpha == 0 and digits > 0:
        return True
    if weird >= 2:
        return True
    if digits >= max(2, alpha):
        return True
    if repeated and confidence < 0.75:
        return True
    return False


def normalize_bbox(
    bbox_pts: list[list[float]],
    scale: float,
    orig_width: int,
    orig_height: int,
) -> list[int]:
    scaled = [
        [int(round(point[0] / scale)), int(round(point[1] / scale))]
        for point in bbox_pts
    ]
    xs = [point[0] for point in scaled]
    ys = [point[1] for point in scaled]
    return [
        max(0, min(xs)),
        max(0, min(ys)),
        min(orig_width, max(xs)),
        min(orig_height, max(ys)),
    ]


def classify_text_type(text: str, bbox: list[int], page_width: int) -> str:
    x1, y1, x2, y2 = bbox
    if is_korean_sfx(text):
        return "sfx"

    center_x = (x1 + x2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Narração: Se for muito largo, ou estiver nos cantos, ou for um retângulo perfeito no topo/base
    is_edge_x = center_x < page_width * 0.15 or center_x > page_width * 0.85
    aspect = width / max(1, height)
    
    if is_edge_x and aspect > 1.8:
        return "narracao"
    
    if aspect > 2.5: # Balões muito largos costumam ser narração ou legendas
        return "narracao"
        
    return "fala"


def default_style() -> dict:
    return {
        "fonte": "KOMIKAX_.ttf",
        "tamanho": 16,
        "cor": "#FFFFFF",
        "cor_gradiente": [],
        "contorno": "#000000",
        "contorno_px": 2,
        "glow": False,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": False,
        "sombra_cor": "",
        "sombra_offset": [0, 0],
        "bold": False,
        "italico": False,
        "rotacao": 0,
        "alinhamento": "center",
    }


# ── Style detection helpers ────────────────────────────────────────────────

def _rgb_to_hex(rgb) -> str:
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def _color_distance(h1: str, h2: str) -> float:
    def p(h: str):
        h = h.lstrip("#")
        return [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    c1, c2 = p(h1), p(h2)
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _bright_pixels_color(arr_rgb: np.ndarray) -> str:
    """Extrai a cor média dos pixels mais brilhantes (estimativa para texto claro)."""
    if arr_rgb.size == 0:
        return "#FFFFFF"
    pixels = arr_rgb.reshape(-1, 3).astype(float)
    brightness = pixels.sum(axis=1)
    # Pega o topo 25% mais brilhante
    threshold = np.percentile(brightness, 75)
    bright = pixels[brightness >= threshold]
    if len(bright) == 0:
        return "#FFFFFF"
    return _rgb_to_hex(bright.mean(axis=0))


def _detect_text_color(region_rgb: np.ndarray, region_gray: np.ndarray) -> str:
    """Detecta a cor do texto separando-o do fundo via thresholding de Otsu."""
    if region_rgb.size == 0:
        return "#FFFFFF"
    
    import cv2
    # Threshold de Otsu para separar texto do fundo
    _, mask = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Decide qual lado é o texto (o menos frequente geralmente é o texto)
    count_white = np.count_nonzero(mask == 255)
    count_black = np.count_nonzero(mask == 0)
    
    # Em mangá, se o fundo for balão branco, o texto é preto (lado menor).
    # Se for texto flutuante em fundo escuro, o texto é claro (lado menor).
    text_val = 255 if count_white < count_black else 0
    text_mask = (mask == text_val).astype(np.uint8)
    
    # Se o texto for ridiculamente pequeno, fallback para cor brilhante
    if np.count_nonzero(text_mask) < 4:
        return _bright_pixels_color(region_rgb)
    
    # Calcula a cor média apenas onde a máscara de texto está ativa
    mean_color = cv2.mean(region_rgb, mask=text_mask)[:3]
    hex_color = _rgb_to_hex(mean_color)
    
    # Proteção: se a cor for quase fundo, tenta a outra polaridade
    if count_white > 0 and count_black > 0:
        bg_val = 255 - text_val
        bg_mask = (mask == bg_val).astype(np.uint8)
        bg_color = _rgb_to_hex(cv2.mean(region_rgb, mask=bg_mask)[:3])
        if _color_distance(hex_color, bg_color) < 25:
             alt_mask = (mask == bg_val).astype(np.uint8)
             alt_color = _rgb_to_hex(cv2.mean(region_rgb, mask=alt_mask)[:3])
             return alt_color

    return hex_color


def _detect_gradient(region_rgb: np.ndarray) -> list:
    """Return [top_color, bottom_color] if vertical gradient found, else []."""
    h = region_rgb.shape[0]
    if h < 12:
        return []
    band = max(2, h // 5)
    top_color = _bright_pixels_color(region_rgb[:band])
    bot_color = _bright_pixels_color(region_rgb[-band:])
    if _color_distance(top_color, bot_color) > 35:
        return [top_color, bot_color]
    return []


def _detect_outline(region_rgb: np.ndarray, region_gray: np.ndarray) -> tuple:
    """Return (outline_color_hex, outline_px). Empty string + 0 if no outline."""
    h, w = region_gray.shape
    if h < 6 or w < 6:
        return "", 0

    margin = max(2, min(4, h // 6))
    inner = region_gray[margin:-margin, margin:-margin]
    edge_vals = np.concatenate([
        region_gray[:margin, :].ravel(),
        region_gray[-margin:, :].ravel(),
        region_gray[:, :margin].ravel(),
        region_gray[:, -margin:].ravel(),
    ])
    if inner.size == 0 or edge_vals.size == 0:
        return "", 0

    diff = abs(float(edge_vals.mean()) - float(inner.mean()))
    if diff < 45:
        return "", 0

    edge_rgb = np.concatenate([
        region_rgb[:margin, :].reshape(-1, 3),
        region_rgb[-margin:, :].reshape(-1, 3),
        region_rgb[:, :margin].reshape(-1, 3),
        region_rgb[:, -margin:].reshape(-1, 3),
    ])
    outline_color = _rgb_to_hex(edge_rgb.mean(axis=0))

    # Estimate pixel thickness: keep growing margin while contrast holds
    contorno_px = 1
    for thickness in range(2, 5):
        m = thickness
        if m * 2 >= h or m * 2 >= w:
            break
        e = np.concatenate([
            region_gray[:m, :].ravel(),
            region_gray[-m:, :].ravel(),
            region_gray[:, :m].ravel(),
            region_gray[:, -m:].ravel(),
        ])
        i = region_gray[m:-m, m:-m]
        if i.size == 0 or abs(float(e.mean()) - float(i.mean())) < 35:
            break
        contorno_px = thickness

    return outline_color, contorno_px


def _detect_glow(region_rgb: np.ndarray, region_gray: np.ndarray) -> tuple:
    """Return (has_glow, glow_color_hex, glow_px). Glow = bright soft halo outside text."""
    h, w = region_gray.shape
    if h < 8 or w < 8:
        return False, "", 0

    margin = max(2, h // 8)
    if margin * 2 >= h or margin * 2 >= w:
        return False, "", 0

    inner = region_gray[margin:-margin, margin:-margin]
    outer = np.concatenate([
        region_gray[:margin, :].ravel(),
        region_gray[-margin:, :].ravel(),
        region_gray[:, :margin].ravel(),
        region_gray[:, -margin:].ravel(),
    ])
    if inner.size == 0 or outer.size == 0:
        return False, "", 0

    outer_mean = float(outer.mean())
    inner_mean = float(inner.mean())

    # Glow: outer is notably brighter than inner AND outer pixels are bright
    if outer_mean > inner_mean + 30 and outer_mean > 150:
        outer_rgb = np.concatenate([
            region_rgb[:margin, :].reshape(-1, 3),
            region_rgb[-margin:, :].reshape(-1, 3),
            region_rgb[:, :margin].reshape(-1, 3),
            region_rgb[:, -margin:].reshape(-1, 3),
        ])
        glow_color = _rgb_to_hex(outer_rgb.mean(axis=0))
        glow_px = max(2, min(8, int((outer_mean - inner_mean) / 15)))
        return True, glow_color, glow_px

    return False, "", 0


def _detect_shadow(region_gray: np.ndarray) -> tuple:
    """Return (has_shadow, shadow_color_hex, [dx, dy])."""
    h, w = region_gray.shape
    if h < 12 or w < 12:
        return False, "", [0, 0]

    qh = max(2, h // 4)
    qw = max(2, w // 4)

    tl = region_gray[:qh, :qw]
    tl_dark = float(np.sum(tl < 70)) / tl.size

    for (region_slice, offset) in [
        (region_gray[h - qh :, w - qw :], [2, 2]),
        (region_gray[h - qh :, :qw], [-2, 2]),
    ]:
        dark_ratio = float(np.sum(region_slice < 70)) / region_slice.size
        if dark_ratio > tl_dark + 0.12 and dark_ratio > 0.08:
            return True, "#000000", offset

    return False, "", [0, 0]


def _detect_italic(region_gray: np.ndarray) -> bool:
    """Detect italic by measuring dominant near-vertical stroke angle."""
    try:
        import cv2
    except ImportError:
        return False

    h, w = region_gray.shape
    if h < 24 or w < 16 or h * w < 500:
        return False

    blurred = cv2.GaussianBlur(region_gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    min_len = max(5, h // 4)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=8, minLineLength=min_len, maxLineGap=4
    )
    if lines is None or len(lines) < 3:
        return False

    angles = []
    for line in lines:
        x1_, y1_, x2_, y2_ = line[0]
        if x2_ == x1_:
            angles.append(90.0)
        else:
            a = float(np.degrees(np.arctan2(abs(y2_ - y1_), abs(x2_ - x1_))))
            if a > 45:
                angles.append(a)

    if len(angles) < 3:
        return False

    mean_a = float(np.mean(angles))
    # Vertical strokes at ~90°; italic leans to ~70-83°
    return 60.0 < mean_a < 83.0


# ── Public interface ───────────────────────────────────────────────────────

def analyze_style(img_array: np.ndarray, bbox: list[int]) -> dict:
    try:
        import cv2
    except ImportError:
        return default_style()

    x1, y1, x2, y2 = bbox
    img_h, img_w = img_array.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    region = img_array[y1:y2, x1:x2]

    if region.size == 0:
        return default_style()

    region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    bbox_h = y2 - y1

    font_size = max(10, min(48, int(bbox_h * 0.7)))
    text_color = _detect_text_color(region, region_gray)
    cor_gradiente = _detect_gradient(region)
    contorno_color, contorno_px = _detect_outline(region, region_gray)
    has_glow, glow_color, glow_px = _detect_glow(region, region_gray)
    has_shadow, shadow_color, shadow_offset = _detect_shadow(region_gray)

    bright_ratio = float(np.sum(region_gray > 160)) / max(1, region_gray.size)
    bold = bbox_h > 28 and bright_ratio > 0.15

    italico = _detect_italic(region_gray)

    return {
        "fonte": "KOMIKAX_.ttf",
        "tamanho": font_size,
        "cor": text_color,
        "cor_gradiente": cor_gradiente,
        "contorno": contorno_color,
        "contorno_px": contorno_px,
        "glow": has_glow,
        "glow_cor": glow_color,
        "glow_px": glow_px,
        "sombra": has_shadow,
        "sombra_cor": shadow_color,
        "sombra_offset": shadow_offset,
        "bold": bold,
        "italico": italico,
        "rotacao": 0,
        "alinhamento": "center",
    }


def merge_ocr_runs(primary_runs: Iterable[dict], secondary_runs: Iterable[dict]) -> list[dict]:
    merged = list(primary_runs)
    for secondary in secondary_runs:
        secondary_text = secondary["text"].strip()
        if not secondary_text or secondary["confidence"] < 0.20:
            continue

        secondary_center = _center_from_bbox_points(secondary["bbox_pts"])
        duplicate_idx = None
        for idx, primary in enumerate(merged):
            primary_center = _center_from_bbox_points(primary["bbox_pts"])
            distance = (
                (secondary_center[0] - primary_center[0]) ** 2
                + (secondary_center[1] - primary_center[1]) ** 2
            ) ** 0.5
            if distance < 50:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            merged.append(secondary)
        elif secondary["confidence"] > merged[duplicate_idx]["confidence"]:
            merged[duplicate_idx] = secondary

    return merged


def _center_from_bbox_points(bbox_pts: list[list[float]]) -> tuple[float, float]:
    xs = [point[0] for point in bbox_pts]
    ys = [point[1] for point in bbox_pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))
