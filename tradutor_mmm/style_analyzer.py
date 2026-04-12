"""
Analise de estilo visual de regioes de texto em imagens de manga/manhwa.
Detecta cor do texto, cor do fundo, presenca de contorno, tamanho da fonte, etc.
"""
import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from . import config
from .detector import TextRegion

logger = logging.getLogger(__name__)


@dataclass
class TextStyle:
    """Propriedades visuais de uma regiao de texto."""
    text_color: Tuple[int, int, int]        # RGB
    background_color: Tuple[int, int, int]  # RGB
    has_outline: bool = False
    outline_color: Tuple[int, int, int] = (0, 0, 0)
    outline_width: int = 0
    font_size: int = 24
    is_bold: bool = True
    is_uppercase: bool = True
    text_type: str = config.TEXT_TYPE_SPEECH_BUBBLE
    bg_is_uniform: bool = True


class StyleAnalyzer:
    """Analisa o estilo visual de regioes de texto detectadas."""

    def analyze(self, image: np.ndarray, region: TextRegion) -> TextStyle:
        """Analisa o estilo visual de uma regiao de texto.

        Args:
            image: Imagem como array numpy (BGR, como OpenCV carrega)
            region: Regiao de texto detectada

        Returns:
            TextStyle com as propriedades visuais
        """
        img_h, img_w = image.shape[:2]
        x_min, y_min, x_max, y_max = region.rect

        # Clampar ao tamanho da imagem
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return TextStyle(
                text_color=(0, 0, 0),
                background_color=(255, 255, 255),
                text_type=config.TEXT_TYPE_SPEECH_BUBBLE,
            )

        crop = image[y_min:y_max, x_min:x_max]

        # Converter para RGB (OpenCV usa BGR)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        bg_color, bg_std, bg_uniform = self._sample_background(image, region, img_h, img_w)
        text_color = self._sample_text_color(crop_rgb, bg_color)
        has_outline, outline_color, outline_width = self._detect_outline(crop_rgb, text_color, bg_color)
        font_size = self._estimate_font_size(region)
        text_type = self._classify_text_type(bg_color, bg_std, bg_uniform, text_color, has_outline)

        style = TextStyle(
            text_color=text_color,
            background_color=bg_color,
            has_outline=has_outline,
            outline_color=outline_color,
            outline_width=outline_width,
            font_size=font_size,
            is_bold=True,
            is_uppercase=region.text == region.text.upper(),
            text_type=text_type,
            bg_is_uniform=bg_uniform,
        )

        # Atualizar o tipo na regiao tambem
        if region.text_type not in (config.TEXT_TYPE_SFX_KOREAN, config.TEXT_TYPE_WATERMARK):
            region.text_type = text_type

        region.style = style
        return style

    def _sample_background(
        self, image: np.ndarray, region: TextRegion, img_h: int, img_w: int
    ) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], bool]:
        """Amostra a cor de fundo ao redor da regiao de texto.

        Returns:
            (cor_media_rgb, desvio_padrao_rgb, eh_uniforme)
        """
        x_min, y_min, x_max, y_max = region.rect
        expand = config.BG_SAMPLE_EXPAND_PX

        # Retangulo expandido (clamped)
        ex_min = max(0, x_min - expand)
        ey_min = max(0, y_min - expand)
        ex_max = min(img_w, x_max + expand)
        ey_max = min(img_h, y_max + expand)

        # Criar mascara do anel (area expandida menos area interna)
        mask = np.zeros((ey_max - ey_min, ex_max - ex_min), dtype=np.uint8)
        mask[:, :] = 255  # Tudo branco

        # Recortar a area interna (onde esta o texto)
        inner_y_start = y_min - ey_min
        inner_y_end = y_max - ey_min
        inner_x_start = x_min - ex_min
        inner_x_end = x_max - ex_min
        mask[inner_y_start:inner_y_end, inner_x_start:inner_x_end] = 0

        # Extrair pixels do anel
        ring_crop = image[ey_min:ey_max, ex_min:ex_max]
        ring_rgb = cv2.cvtColor(ring_crop, cv2.COLOR_BGR2RGB)

        ring_pixels = ring_rgb[mask > 0]

        if len(ring_pixels) == 0:
            return (255, 255, 255), (0.0, 0.0, 0.0), True

        mean_color = tuple(int(c) for c in np.mean(ring_pixels, axis=0))
        std_color = tuple(float(c) for c in np.std(ring_pixels, axis=0))

        avg_std = np.mean(std_color)
        is_uniform = avg_std < config.BG_STD_UNIFORM_THRESHOLD

        return mean_color, std_color, is_uniform

    def _sample_text_color(
        self, crop_rgb: np.ndarray, bg_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Amostra a cor do texto usando Otsu thresholding."""
        h, w = crop_rgb.shape[:2]

        # Usar a parte interna (80%) para evitar bordas
        margin_x = int(w * (1 - config.TEXT_SAMPLE_INNER_RATIO) / 2)
        margin_y = int(h * (1 - config.TEXT_SAMPLE_INNER_RATIO) / 2)
        inner = crop_rgb[
            max(1, margin_y):max(2, h - margin_y),
            max(1, margin_x):max(2, w - margin_x),
        ]

        # Converter para grayscale
        gray = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)

        # Otsu thresholding para separar texto do fundo
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Determinar se o texto e escuro ou claro
        bg_brightness = np.mean(bg_color)

        if bg_brightness > 128:
            # Fundo claro -> texto nos pixels escuros (binary == 0)
            text_mask = binary == 0
        else:
            # Fundo escuro -> texto nos pixels claros (binary == 255)
            text_mask = binary == 255

        text_pixels = inner[text_mask]

        if len(text_pixels) == 0:
            # Fallback: se nao encontrou pixels de texto, usar cor oposta ao fundo
            if bg_brightness > 128:
                return (0, 0, 0)
            else:
                return (255, 255, 255)

        return tuple(int(c) for c in np.median(text_pixels, axis=0))

    def _detect_outline(
        self,
        crop_rgb: np.ndarray,
        text_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
    ) -> Tuple[bool, Tuple[int, int, int], int]:
        """Detecta contorno/outline no texto.

        Returns:
            (tem_outline, cor_outline, largura_outline)
        """
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        bg_brightness = np.mean(bg_color)

        # Criar mascara binaria do texto
        if bg_brightness > 128:
            _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilatar a mascara para pegar a borda
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(text_mask, kernel, iterations=2)

        # Anel de borda = dilatado - original
        edge_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(text_mask))

        # Amostrar cores na borda
        edge_pixels = crop_rgb[edge_ring > 0]

        if len(edge_pixels) < 10:
            return False, (0, 0, 0), 0

        edge_color = tuple(int(c) for c in np.median(edge_pixels, axis=0))

        # Verificar se a cor da borda e diferente tanto do texto quanto do fundo
        text_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(edge_color, text_color)))
        bg_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(edge_color, bg_color)))

        # Outline detectado se a cor da borda e distinta de ambos
        has_outline = text_diff > 60 and bg_diff > 60

        if has_outline:
            # Estimar largura do outline
            outline_width = 2
            return True, edge_color, outline_width

        return False, (0, 0, 0), 0

    def _estimate_font_size(self, region: TextRegion) -> int:
        """Estima o tamanho da fonte baseado na altura da bbox."""
        text = region.text
        bbox_height = region.height

        # Estimar numero de linhas
        if "\n" in text:
            line_count = text.count("\n") + 1
        else:
            # Heuristica: se o texto tem muitas palavras e o bbox e alto, sao varias linhas
            words = text.split()
            if len(words) <= 2:
                line_count = 1
            elif bbox_height > region.width * 0.8:
                line_count = max(2, len(words) // 2)
            else:
                line_count = max(1, len(words) // 3)

        per_line_height = bbox_height / max(1, line_count)
        font_size = int(per_line_height * 0.7)

        return max(config.MIN_FONT_SIZE, min(config.MAX_FONT_SIZE, font_size))

    def _classify_text_type(
        self,
        bg_color: Tuple[int, int, int],
        bg_std: Tuple[float, float, float],
        bg_uniform: bool,
        text_color: Tuple[int, int, int],
        has_outline: bool,
    ) -> str:
        """Classifica o tipo de texto baseado nas propriedades visuais."""
        bg_r, bg_g, bg_b = bg_color
        bg_brightness = (bg_r + bg_g + bg_b) / 3
        text_brightness = sum(text_color) / 3
        avg_std = np.mean(bg_std)

        # Balao branco: fundo claro e uniforme, texto escuro
        if (bg_brightness > config.BUBBLE_WHITE_THRESHOLD
                and bg_uniform
                and text_brightness < 128):
            return config.TEXT_TYPE_SPEECH_BUBBLE

        # Balao intenso: fundo escuro/vermelho e uniforme, texto claro
        if bg_uniform and bg_brightness < config.BUBBLE_DARK_THRESHOLD and text_brightness > 128:
            return config.TEXT_TYPE_INTENSE_BUBBLE

        # Balao intenso (vermelho): canal R dominante
        if bg_uniform and bg_r > 120 and bg_g < 80 and text_brightness > 128:
            return config.TEXT_TYPE_INTENSE_BUBBLE

        # Narracao overlay: fundo variado, texto claro com outline
        if not bg_uniform and text_brightness > 150 and has_outline:
            return config.TEXT_TYPE_NARRATION_OVERLAY

        # Narracao overlay: fundo variado, texto claro (mesmo sem outline detectado)
        if not bg_uniform and avg_std > config.BG_STD_VARIED_THRESHOLD and text_brightness > 150:
            return config.TEXT_TYPE_NARRATION_OVERLAY

        # Default: balao de fala
        return config.TEXT_TYPE_SPEECH_BUBBLE
