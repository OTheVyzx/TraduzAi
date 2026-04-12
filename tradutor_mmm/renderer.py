"""
Renderizacao de texto traduzido sobre imagens limpas.
Aplica estilo visual detectado (cor, contorno, tamanho).
"""
import logging
import re
from typing import List

from PIL import Image, ImageDraw, ImageFont

from . import config
from .detector import TextRegion
from .style_analyzer import TextStyle
from .text_wrapper import TextWrapper

logger = logging.getLogger(__name__)


class TextRenderer:
    """Renderiza texto traduzido com estilo visual correspondente ao original."""

    def __init__(self, font_path: str):
        """
        Args:
            font_path: Caminho da fonte .ttf a usar
        """
        self._font_path = font_path
        self._wrapper = TextWrapper()
        self._font_cache = {}

    def render_all(
        self,
        image: Image.Image,
        regions: List[TextRegion],
        translated_texts: List[str],
    ) -> Image.Image:
        """Renderiza todos os textos traduzidos na imagem.

        Args:
            image: Imagem limpa (sem texto original)
            regions: Regioes de texto com estilo
            translated_texts: Textos traduzidos correspondentes

        Returns:
            Imagem com textos traduzidos
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)

        for region, text in zip(regions, translated_texts):
            if not text or not text.strip():
                continue

            # Pular SFX e watermarks
            if region.text_type in (config.TEXT_TYPE_SFX_KOREAN, config.TEXT_TYPE_WATERMARK):
                continue

            # Pular texto que parece nao ter sido traduzido (garbled pass-through)
            if self._is_garbled(text):
                logger.debug(f"Pulando texto garbled: '{text[:40]}'")
                continue

            try:
                self._render_region(draw, text, region)
            except Exception as e:
                logger.warning(f"Erro ao renderizar texto '{text[:30]}...': {e}")
                continue

        return result

    def _is_garbled(self, text: str) -> bool:
        """Detecta texto garbled que nao foi traduzido com sucesso.

        Texto garbled tipicamente contem sequencias de consoantes sem vogais,
        mistura de maiusculas/minusculas sem padrao, ou caracteres sem sentido.
        """
        clean = re.sub(r"[^A-Za-z]", "", text)
        if len(clean) < 3:
            return False

        has_portuguese = bool(re.search(r"[àáâãéêíóôõúç]", text, re.IGNORECASE))

        # Texto em portugues nunca e garbled
        if has_portuguese:
            return False

        is_all_upper = text == text.upper()
        is_all_lower = text == text.lower()

        # Contar transicoes de case
        transitions = 0
        for i in range(1, len(clean)):
            if clean[i].isupper() != clean[i - 1].isupper():
                transitions += 1
        transition_ratio = transitions / max(1, len(clean) - 1)

        # All-caps ou all-lower sao normais
        if is_all_upper or is_all_lower:
            return False

        # Muitas transicoes de case = garbled
        if transition_ratio > 0.30 and len(clean) > 5:
            return True

        # Verificar se contem palavras que nao parecem portugues nem ingles comum
        # Texto com muitas consoantes consecutivas sem vogais e garbled
        words = re.findall(r"[A-Za-z]+", text)
        garbled_words = 0
        for word in words:
            if len(word) < 3:
                continue
            # Sequencias de 4+ consoantes sem vogal = garbled
            if re.search(r"[^aeiouAEIOU]{4,}", word):
                garbled_words += 1
            # Maiusculas erraticas dentro da palavra (ex: "ICoALD", "LIoHT")
            if len(word) >= 4 and not word.isupper() and not word.islower() and not word.istitle():
                inner = word[1:-1]  # desconsiderar primeira e ultima letra
                if any(c.isupper() for c in inner) and any(c.islower() for c in inner):
                    garbled_words += 1

        if len(words) > 0 and garbled_words / len(words) > 0.3:
            return True

        return False

    def _render_region(self, draw: ImageDraw.Draw, text: str, region: TextRegion):
        """Renderiza texto em uma regiao especifica."""
        style = region.style
        if style is None:
            style = TextStyle(
                text_color=(0, 0, 0),
                background_color=(255, 255, 255),
            )

        # Aplicar uppercase se necessario
        if style.is_uppercase:
            text = text.upper()

        # Ajustar texto ao tamanho do bbox
        wrapped, font, final_size = self._wrapper.fit_text_to_box(
            text=text,
            box_width=region.width,
            box_height=region.height,
            font_path=self._font_path,
            initial_size=style.font_size,
        )

        # Calcular posicao centralizada
        x, y = self._wrapper.calculate_text_position(
            wrapped, font, region.rect
        )

        # Renderizar baseado no tipo
        if region.text_type == config.TEXT_TYPE_NARRATION_OVERLAY:
            self._render_with_outline(draw, wrapped, x, y, font, style)
        elif region.text_type == config.TEXT_TYPE_INTENSE_BUBBLE:
            self._render_intense(draw, wrapped, x, y, font, style)
        else:
            self._render_speech(draw, wrapped, x, y, font, style)

    def _render_speech(
        self,
        draw: ImageDraw.Draw,
        text: str,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        style: TextStyle,
    ):
        """Renderiza texto de balao de fala (preto sobre branco)."""
        draw.multiline_text(
            (x, y),
            text,
            fill=style.text_color,
            font=font,
            align="center",
        )

    def _render_intense(
        self,
        draw: ImageDraw.Draw,
        text: str,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        style: TextStyle,
    ):
        """Renderiza texto de balao intenso (branco sobre escuro)."""
        draw.multiline_text(
            (x, y),
            text,
            fill=style.text_color,
            font=font,
            align="center",
        )

    def _render_with_outline(
        self,
        draw: ImageDraw.Draw,
        text: str,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        style: TextStyle,
    ):
        """Renderiza texto com contorno (narracao overlay)."""
        outline_width = style.outline_width if style.has_outline else 2
        outline_color = style.outline_color if style.has_outline else (0, 0, 0)

        draw.multiline_text(
            (x, y),
            text,
            fill=style.text_color,
            font=font,
            align="center",
            stroke_width=outline_width,
            stroke_fill=outline_color,
        )
