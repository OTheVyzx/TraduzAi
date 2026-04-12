"""
Quebra de texto e ajuste de tamanho para caber em bounding boxes.
"""
import logging
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from . import config

logger = logging.getLogger(__name__)


class TextWrapper:
    """Ajusta texto traduzido para caber dentro das bounding boxes originais."""

    def wrap_text(self, text: str, max_width: int, font: ImageFont.FreeTypeFont) -> str:
        """Quebra texto em multiplas linhas para caber na largura maxima.

        Args:
            text: Texto a ser quebrado
            max_width: Largura maxima em pixels
            font: Fonte para medir largura

        Returns:
            Texto com quebras de linha
        """
        words = text.split()
        if not words:
            return text

        lines = []
        current_line = words[0]

        for word in words[1:]:
            test_line = current_line + " " + word
            line_width = font.getlength(test_line)

            if line_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)
        return "\n".join(lines)

    def fit_text_to_box(
        self,
        text: str,
        box_width: int,
        box_height: int,
        font_path: str,
        initial_size: int,
    ) -> Tuple[str, ImageFont.FreeTypeFont, int]:
        """Ajusta texto e tamanho da fonte para caber no box.

        Usa busca binaria para encontrar o tamanho ideal.

        Args:
            text: Texto a ser ajustado
            box_width: Largura do box em pixels
            box_height: Altura do box em pixels
            font_path: Caminho da fonte .ttf
            initial_size: Tamanho inicial sugerido

        Returns:
            (texto_quebrado, fonte, tamanho_final)
        """
        if box_width <= 0 or box_height <= 0:
            font = self._load_font(font_path, initial_size)
            return text, font, initial_size

        padding = 5
        effective_width = max(20, box_width - padding * 2)
        effective_height = max(20, box_height - padding * 2)

        # Limites da busca
        min_size = config.MIN_FONT_SIZE
        max_size = min(int(initial_size * 1.5), config.MAX_FONT_SIZE)
        max_size = max(max_size, min_size + 1)

        best_size = min_size
        best_wrapped = text
        best_font = self._load_font(font_path, min_size)

        # Busca binaria pelo tamanho ideal
        low, high = min_size, max_size

        while low <= high:
            mid = (low + high) // 2
            font = self._load_font(font_path, mid)
            wrapped = self.wrap_text(text, effective_width, font)

            text_height = self._measure_text_height(wrapped, font)

            if text_height <= effective_height:
                best_size = mid
                best_wrapped = wrapped
                best_font = font
                low = mid + 1
            else:
                high = mid - 1

        return best_wrapped, best_font, best_size

    def calculate_text_position(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        bbox_rect: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        """Calcula a posicao para centralizar o texto no bbox.

        Args:
            text: Texto (pode ser multilinha)
            font: Fonte
            bbox_rect: (x_min, y_min, x_max, y_max)

        Returns:
            (x, y) para desenhar o texto centralizado
        """
        x_min, y_min, x_max, y_max = bbox_rect
        box_cx = (x_min + x_max) // 2
        box_cy = (y_min + y_max) // 2

        # Medir dimensoes do texto
        text_bbox = self._get_text_bbox(text, font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Centralizar
        x = box_cx - text_width // 2
        y = box_cy - text_height // 2

        return (x, y)

    def _measure_text_height(self, text: str, font: ImageFont.FreeTypeFont) -> int:
        """Mede a altura do texto renderizado."""
        bbox = self._get_text_bbox(text, font)
        return bbox[3] - bbox[1]

    def _get_text_bbox(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
        """Retorna o bounding box do texto renderizado."""
        # Usar uma imagem dummy para medir
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
        return bbox

    def _load_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """Carrega uma fonte com tratamento de erro."""
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            try:
                return ImageFont.truetype(config.FALLBACK_FONT_PATH, size)
            except Exception:
                logger.warning(f"Impossivel carregar fonte, usando default (size={size})")
                return ImageFont.load_default(size)
