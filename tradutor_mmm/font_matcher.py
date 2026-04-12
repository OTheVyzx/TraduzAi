"""
Auto-deteccao e download da fonte mais parecida com o texto original.
Analisa caracteristicas visuais do texto e compara com um catalogo de fontes comic.
"""
import os
import logging
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import requests

from . import config

logger = logging.getLogger(__name__)


class FontMatcher:
    """Detecta a fonte mais parecida e faz download automatico."""

    def __init__(self):
        self._matched_font_path: Optional[str] = None
        self._font_profiles_cache: Dict[str, dict] = {}

    def get_best_font(self, text_crops: list = None) -> str:
        """Retorna o caminho da melhor fonte para o texto detectado.

        Args:
            text_crops: Lista de crops numpy (RGB) das regioes de texto.
                       Se None, usa o perfil padrao.

        Returns:
            Caminho absoluto para o arquivo .ttf
        """
        # Se ja encontrou a fonte, retornar cache
        if self._matched_font_path and os.path.exists(self._matched_font_path):
            return self._matched_font_path

        if text_crops and len(text_crops) > 0:
            # Analisar caracteristicas visuais dos textos
            profile = self._analyze_text_profile(text_crops)
            best_key = self._find_best_match(profile)
        else:
            # Padrao: Anime Ace 2 Bold (mais comum em scanlation)
            best_key = "anime_ace_2_bold"

        # Tentar baixar a fonte selecionada
        font_path = self._download_font(best_key)

        if font_path:
            self._matched_font_path = font_path
            return font_path

        # Tentar cada fonte do catalogo ate uma funcionar
        for key in config.FONT_CATALOG:
            if key == best_key:
                continue
            font_path = self._download_font(key)
            if font_path:
                self._matched_font_path = font_path
                return font_path

        # Fallback final: Comic Sans Bold do Windows
        if os.path.exists(config.FALLBACK_FONT_PATH):
            logger.warning("Usando fonte fallback: Comic Sans MS Bold")
            self._matched_font_path = config.FALLBACK_FONT_PATH
            return config.FALLBACK_FONT_PATH

        raise RuntimeError("Nenhuma fonte disponivel! Instale uma fonte .ttf em tradutor_mmm/fonts/")

    def _analyze_text_profile(self, crops: list) -> dict:
        """Analisa caracteristicas visuais dos crops de texto.

        Retorna um perfil com:
        - weight: ratio de pixels de texto (bold vs regular)
        - width: aspect ratio medio dos caracteres
        - serif: presenca de serifas
        - style: estilo geral
        """
        weights = []
        widths = []

        for crop in crops[:10]:  # Limitar a 10 amostras
            if crop is None or crop.size == 0:
                continue

            h, w = crop.shape[:2]
            if h < 5 or w < 5:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop

            # Binarizar
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Peso: ratio de pixels de texto
            text_ratio = np.sum(binary > 0) / binary.size
            weights.append(text_ratio)

            # Largura: buscar contornos e medir aspect ratio
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if ch > h * 0.3:  # So caracteres significativos
                        widths.append(cw / max(1, ch))

        profile = {
            "weight": float(np.mean(weights)) if weights else 0.5,
            "width": float(np.mean(widths)) if widths else 0.5,
            "serif": 0.0,  # Dificil detectar em baixa resolucao, assumir sans-serif
            "style": "comic",
        }

        logger.debug(f"Perfil de fonte detectado: {profile}")
        return profile

    def _find_best_match(self, profile: dict) -> str:
        """Encontra a fonte do catalogo mais proxima do perfil detectado."""
        best_key = None
        best_distance = float("inf")

        for key, font_info in config.FONT_CATALOG.items():
            ref_profile = font_info["profile"]

            # Distancia euclidiana ponderada entre perfis
            d_weight = (profile["weight"] - ref_profile["weight"]) ** 2 * 2.0
            d_width = (profile["width"] - ref_profile["width"]) ** 2 * 1.5
            d_serif = (profile["serif"] - ref_profile["serif"]) ** 2 * 1.0

            # Bonus para mesmo estilo
            style_bonus = 0 if profile["style"] == ref_profile["style"] else 0.1

            distance = d_weight + d_width + d_serif + style_bonus

            if distance < best_distance:
                best_distance = distance
                best_key = key

        logger.info(f"Fonte mais parecida: {config.FONT_CATALOG[best_key]['name']} (dist={best_distance:.3f})")
        return best_key

    def _download_font(self, font_key: str) -> Optional[str]:
        """Faz download de uma fonte do catalogo.

        Returns:
            Caminho do arquivo baixado, ou None em caso de erro.
        """
        if font_key not in config.FONT_CATALOG:
            return None

        font_info = config.FONT_CATALOG[font_key]
        filename = font_info["filename"]
        font_path = os.path.join(config.FONT_DIR, filename)

        # Ja existe em cache?
        if os.path.exists(font_path) and os.path.getsize(font_path) > 1000:
            logger.debug(f"Fonte ja em cache: {filename}")
            return font_path

        url = font_info["url"]
        logger.info(f"Baixando fonte: {font_info['name']} de {url}")

        try:
            response = requests.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()

            with open(font_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Fonte baixada: {filename} ({len(response.content):,} bytes)")
            return font_path

        except Exception as e:
            logger.warning(f"Erro ao baixar fonte {font_info['name']}: {e}")
            # Limpar arquivo parcial
            if os.path.exists(font_path):
                try:
                    os.remove(font_path)
                except OSError:
                    pass
            return None
