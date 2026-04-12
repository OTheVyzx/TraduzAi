"""
Limpeza de texto original das imagens.
- Baloes de fala: preenchimento com cor do fundo (branco)
- Baloes intensos: preenchimento com cor do fundo (escuro/vermelho)
- Narracao overlay: inpainting com LaMa (AI)
- Watermarks: inpainting com LaMa
"""
import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from . import config
from .detector import TextRegion

logger = logging.getLogger(__name__)


class TextCleaner:
    """Remove texto original das imagens usando fill e inpainting."""

    def __init__(self):
        self._lama = None

    @property
    def lama(self):
        """Inicializacao lazy do modelo LaMa."""
        if self._lama is None:
            logger.info("Carregando modelo LaMa para inpainting...")
            try:
                from simple_lama_inpainting import SimpleLama
                self._lama = SimpleLama()
                logger.info("Modelo LaMa carregado.")
            except Exception as e:
                logger.error(f"Erro ao carregar LaMa: {e}")
                self._lama = False  # Marcar como indisponivel
        return self._lama if self._lama is not False else None

    def clean_image(self, image: Image.Image, regions: List[TextRegion]) -> Image.Image:
        """Remove texto de todas as regioes da imagem.

        Args:
            image: Imagem PIL (RGB)
            regions: Lista de regioes com estilo ja analisado

        Returns:
            Imagem limpa
        """
        result = image.copy()

        # Separar regioes por metodo de limpeza
        fill_regions = []
        inpaint_regions = []

        for region in regions:
            if region.text_type == config.TEXT_TYPE_SFX_KOREAN:
                continue  # Nao limpar SFX

            if region.text_type == config.TEXT_TYPE_NARRATION_OVERLAY:
                inpaint_regions.append(region)
            elif region.text_type == config.TEXT_TYPE_INTENSE_BUBBLE:
                # Baloes intensos usam fill com cor amostrada do balao
                fill_regions.append(region)
            elif region.text_type == config.TEXT_TYPE_SPEECH_BUBBLE:
                fill_regions.append(region)
            else:
                fill_regions.append(region)

        # 1. Fill nas regioes de balao (rapido, sem modelo)
        if fill_regions:
            result = self._fill_regions(result, fill_regions)

        # 2. Inpainting nas regioes de overlay/watermark
        if inpaint_regions:
            result = self._inpaint_regions(result, inpaint_regions)

        return result

    def _fill_regions(self, image: Image.Image, regions: List[TextRegion]) -> Image.Image:
        """Preenche regioes de balao com a cor de fundo amostrada."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        for region in regions:
            x_min, y_min, x_max, y_max = region.rect

            # Validar coordenadas
            if x_max <= x_min or y_max <= y_min:
                continue

            margin = config.BUBBLE_FILL_MARGIN + 2

            # Expandir a area de preenchimento (clamped)
            fx_min = max(0, x_min - margin)
            fy_min = max(0, y_min - margin)
            fx_max = min(w, x_max + margin)
            fy_max = min(h, y_max + margin)

            if fx_max <= fx_min or fy_max <= fy_min:
                continue

            # Para baloes intensos (gradiente), usar clone de bordas
            if region.text_type == config.TEXT_TYPE_INTENSE_BUBBLE:
                self._fill_with_border_clone(img_array, fx_min, fy_min, fx_max, fy_max)
            else:
                # Amostrar cor de fundo do anel ao redor do texto
                fill_color = self._sample_fill_color(img_array, region)
                img_array[fy_min:fy_max, fx_min:fx_max] = fill_color

        return Image.fromarray(img_array)

    def _fill_with_border_clone(
        self, img: np.ndarray, x0: int, y0: int, x1: int, y1: int
    ):
        """Preenche uma area clonando as bordas e aplicando blur.

        Melhor para baloes com gradiente/textura (baloes intensos).
        """
        region_h = y1 - y0
        region_w = x1 - x0

        if region_h <= 0 or region_w <= 0:
            return

        # Criar uma copia da regiao
        fill = img[y0:y1, x0:x1].copy()

        # Preencher de fora para dentro usando interpolacao das bordas
        # Borda superior e inferior
        top_row = img[max(0, y0 - 1):y0, x0:x1]
        bottom_row = img[y1:min(img.shape[0], y1 + 1), x0:x1]

        if top_row.size > 0 and bottom_row.size > 0:
            top_color = top_row.mean(axis=(0, 1)).astype(np.uint8)
            bottom_color = bottom_row.mean(axis=(0, 1)).astype(np.uint8)

            # Criar gradiente vertical entre topo e fundo
            for i in range(region_h):
                alpha = i / max(1, region_h - 1)
                fill[i, :] = (
                    top_color * (1 - alpha) + bottom_color * alpha
                ).astype(np.uint8)
        else:
            # Fallback: usar cor mediana das bordas
            fill_color = self._sample_fill_color(img, None, (x0, y0, x1, y1))
            fill[:, :] = fill_color

        # Aplicar gaussian blur para suavizar
        fill = cv2.GaussianBlur(fill, (5, 5), 0)

        img[y0:y1, x0:x1] = fill

    def _sample_fill_color(
        self, img_array: np.ndarray, region: TextRegion = None, rect: tuple = None
    ) -> Tuple[int, int, int]:
        """Amostra a cor de preenchimento do entorno do balao."""
        h, w = img_array.shape[:2]
        if rect:
            x_min, y_min, x_max, y_max = rect
        elif region:
            x_min, y_min, x_max, y_max = region.rect
        else:
            return (255, 255, 255)
        expand = 20

        # Area expandida
        ex_min = max(0, x_min - expand)
        ey_min = max(0, y_min - expand)
        ex_max = min(w, x_max + expand)
        ey_max = min(h, y_max + expand)

        # Criar mascara do anel
        ring = np.zeros((ey_max - ey_min, ex_max - ex_min), dtype=bool)
        ring[:, :] = True
        inner_y0 = y_min - ey_min
        inner_y1 = y_max - ey_min
        inner_x0 = x_min - ex_min
        inner_x1 = x_max - ex_min
        ring[inner_y0:inner_y1, inner_x0:inner_x1] = False

        crop = img_array[ey_min:ey_max, ex_min:ex_max]
        ring_pixels = crop[ring]

        if len(ring_pixels) > 0:
            return tuple(int(c) for c in np.median(ring_pixels, axis=0))

        if region.style and hasattr(region.style, "background_color"):
            return region.style.background_color

        return (255, 255, 255)

    def _inpaint_regions(self, image: Image.Image, regions: List[TextRegion]) -> Image.Image:
        """Remove texto de regioes usando LaMa inpainting."""
        lama = self.lama

        if lama is None:
            logger.warning("LaMa indisponivel, usando fill simples como fallback")
            return self._fill_regions(image, regions)

        # Construir mascara combinada
        mask = self._build_inpaint_mask(image.size, regions)

        try:
            # Executar inpainting
            result = lama(image, mask)

            # Limpar memoria GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM durante inpainting, tentando com CPU...")
                torch.cuda.empty_cache()
                # Tentar com tamanho reduzido ou fill simples
                return self._fill_regions(image, regions)
            raise
        except Exception as e:
            logger.error(f"Erro no inpainting: {e}")
            return self._fill_regions(image, regions)

    def _build_inpaint_mask(
        self, image_size: Tuple[int, int], regions: List[TextRegion]
    ) -> Image.Image:
        """Constroi mascara para inpainting (branco = areas para inpaintar)."""
        width, height = image_size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for region in regions:
            x_min, y_min, x_max, y_max = region.rect
            if x_max <= x_min or y_max <= y_min:
                continue

            dilation = config.INPAINT_MASK_DILATION

            # Area expandida
            rx0 = max(0, x_min - dilation)
            ry0 = max(0, y_min - dilation)
            rx1 = min(width, x_max + dilation)
            ry1 = min(height, y_max + dilation)

            if rx1 <= rx0 or ry1 <= ry0:
                continue

            draw.rectangle([rx0, ry0, rx1, ry1], fill=255)

        # Dilatar a mascara para bordas mais suaves
        mask_np = np.array(mask)
        kernel = np.ones(
            (config.INPAINT_KERNEL_SIZE, config.INPAINT_KERNEL_SIZE), np.uint8
        )
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

        return Image.fromarray(mask_np)
