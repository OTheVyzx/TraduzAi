"""
Regression: texto cortado nas bordas — página 001 do export traduzido2.

Caso real documentado (2026-05-03):
  Layer 0 — balloon_bbox [0,29,800,257], bbox OCR [164,45,653,241]
  Layer 1 — balloon_bbox [0,525,800,750], bbox OCR [160,541,636,734]

O bug: _apply_copy_back_outside_balloons usava strip_bbox (≈ OCR bbox) como
máscara. Texto renderizado na área expandida do balão (balloon_bbox) era
sobrescrito pelo original → caracteres nas bordas sumiam.

Os testes abaixo garantem:
1. plan_text_layout retorna safe_text_box dentro de balloon_bbox
2. Nenhuma letra é cortada: render_bbox não deve tocar as bordas de safe_text_box
3. _apply_copy_back_outside_balloons preserva pixels na área de balloon_bbox
   mesmo quando strip_bbox é menor
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ---------------------------------------------------------------------------
# Dados de referência — page 001, Layer 0 e Layer 1
# ---------------------------------------------------------------------------

LAYER_0 = {
    "bbox": [164, 45, 653, 241],
    "balloon_bbox": [0, 29, 800, 257],
    "text_pixel_bbox": [169, 50, 650, 236],
    "source_bbox": [164, 45, 653, 241],
    "translated": (
        "EM TEMPOS DE PAZ, É APENAS UM TÍTULO VAZIO. "
        "MAS QUANDO A GUERRA ECLODE, CONCEDE UMA ENORME AUTORIDADE"
    ),
    "tipo": "fala",
    "balloon_type": "white",
    "layout_profile": "white_balloon",
    "layout_shape": "wide",
    "layout_group_size": 1,
    "_is_lobe_subregion": False,
    "estilo": {
        "fonte": "ComicNeue-Bold.ttf",
        "tamanho": 33,
        "cor": "#171717",
        "cor_gradiente": [],
        "contorno": "",
        "contorno_px": 0,
        "glow": False,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": True,
        "sombra_cor": "#000000",
        "sombra_offset": [-2, 2],
        "bold": True,
        "italico": False,
        "rotacao": 0,
        "alinhamento": "center",
        "force_upper": True,
    },
}

LAYER_1 = {
    "bbox": [160, 541, 636, 734],
    "balloon_bbox": [0, 525, 800, 750],
    "text_pixel_bbox": [169, 545, 635, 727],
    "source_bbox": [160, 541, 636, 734],
    "translated": "VOCÊ É DIFÍCIL DE PREVER E AGORA DESEJA QUE EU O ARME TAMBÉM?",
    "tipo": "fala",
    "balloon_type": "white",
    "layout_profile": "white_balloon",
    "layout_shape": "wide",
    "layout_group_size": 1,
    "_is_lobe_subregion": False,
    "estilo": {
        "fonte": "ComicNeue-Bold.ttf",
        "tamanho": 24,
        "cor": "#171717",
        "cor_gradiente": [],
        "contorno": "",
        "contorno_px": 0,
        "glow": False,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": True,
        "sombra_cor": "#000000",
        "sombra_offset": [-2, 2],
        "bold": True,
        "italico": False,
        "rotacao": 0,
        "alinhamento": "center",
        "force_upper": True,
    },
}


# ---------------------------------------------------------------------------
# 1. plan_text_layout deve retornar safe_text_box dentro de balloon_bbox
# ---------------------------------------------------------------------------

class TestSafeTextBoxDerived:
    @pytest.mark.parametrize("layer", [LAYER_0, LAYER_1], ids=["layer0", "layer1"])
    def test_safe_text_box_is_within_balloon_bbox(self, layer):
        from typesetter.renderer import plan_text_layout

        td = dict(layer)
        td["translated"] = td["translated"].upper()
        plan = plan_text_layout(td)

        stb = plan.get("safe_text_box")
        assert stb is not None, "plan deve conter safe_text_box"
        assert len(stb) == 4

        bb = layer["balloon_bbox"]
        assert stb[0] >= bb[0], f"safe_text_box x1={stb[0]} < balloon x1={bb[0]}"
        assert stb[1] >= bb[1], f"safe_text_box y1={stb[1]} < balloon y1={bb[1]}"
        assert stb[2] <= bb[2], f"safe_text_box x2={stb[2]} > balloon x2={bb[2]}"
        assert stb[3] <= bb[3], f"safe_text_box y2={stb[3]} > balloon y2={bb[3]}"

    @pytest.mark.parametrize("layer", [LAYER_0, LAYER_1], ids=["layer0", "layer1"])
    def test_safe_text_box_is_wider_than_ocr_bbox(self, layer):
        """safe_text_box deve ser mais larga que o OCR bbox (usa balloon, não OCR)."""
        from typesetter.renderer import plan_text_layout

        td = dict(layer)
        td["translated"] = td["translated"].upper()
        plan = plan_text_layout(td)

        stb = plan["safe_text_box"]
        stb_w = stb[2] - stb[0]
        ocr_w = layer["bbox"][2] - layer["bbox"][0]
        assert stb_w > ocr_w, (
            f"safe_text_box width {stb_w} deveria ser maior que OCR width {ocr_w}"
        )


# ---------------------------------------------------------------------------
# 2. Render completo — nenhuma letra deve ser cortada (TEXT_CLIPPED ausente)
# ---------------------------------------------------------------------------

class TestNoTextClipping:
    @pytest.mark.parametrize("layer", [LAYER_0, LAYER_1], ids=["layer0", "layer1"])
    def test_render_does_not_clip_text(self, layer):
        """Após render, render_bbox NÃO deve colidir com bordas de safe_text_box."""
        from PIL import Image
        from typesetter.renderer import (
            plan_text_layout,
            ensure_legible_plan,
            _render_single_text_block,
        )

        td = dict(layer)
        td["translated"] = td["translated"].upper()
        td["qa_flags"] = []

        bx1, by1, bx2, by2 = layer["balloon_bbox"]
        img = Image.new("RGB", (bx2, by2 + 100), (255, 255, 255))

        plan = ensure_legible_plan(img, plan_text_layout(td))
        _render_single_text_block(img, td, plan)

        assert "TEXT_CLIPPED" not in td.get("qa_flags", []), (
            f"TEXT_CLIPPED detectado — render_bbox={td.get('render_bbox')} "
            f"safe_text_box={plan.get('safe_text_box')}"
        )

    @pytest.mark.parametrize("layer", [LAYER_0, LAYER_1], ids=["layer0", "layer1"])
    def test_render_bbox_within_balloon_bbox(self, layer):
        """render_bbox deve estar completamente dentro de balloon_bbox."""
        from PIL import Image
        from typesetter.renderer import (
            plan_text_layout,
            ensure_legible_plan,
            _render_single_text_block,
        )

        td = dict(layer)
        td["translated"] = td["translated"].upper()
        td["qa_flags"] = []

        bx1, by1, bx2, by2 = layer["balloon_bbox"]
        img = Image.new("RGB", (bx2, by2 + 100), (255, 255, 255))

        plan = ensure_legible_plan(img, plan_text_layout(td))
        _render_single_text_block(img, td, plan)

        rb = td.get("render_bbox")
        assert rb is not None, "render_bbox deve ser definido após render"
        rx1, ry1, rx2, ry2 = rb
        assert rx1 >= bx1, f"texto começa antes da borda esquerda: x={rx1} < {bx1}"
        assert rx2 <= bx2, f"texto termina depois da borda direita: x={rx2} > {bx2}"


# ---------------------------------------------------------------------------
# 3. copy-back preserva balloon_bbox expandida mesmo com strip_bbox menor
# ---------------------------------------------------------------------------

class TestCopyBackExpandedBbox:
    def test_copy_back_uses_balloon_bbox_from_ocr_page(self):
        """Pixels renderizados dentro de balloon_bbox mas fora de strip_bbox
        devem ser preservados (não sobrescritos pelo original)."""
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox

        PAGE_W = 800
        BAND_H = 228  # balloon_bbox height (257-29)
        BAND_Y_TOP = 29

        original = np.full((BAND_H, PAGE_W, 3), 50, dtype=np.uint8)
        rendered = np.full((BAND_H, PAGE_W, 3), 200, dtype=np.uint8)

        # strip_bbox = detector bbox em coords absolutas (≈ OCR bbox)
        band = Band(
            y_top=BAND_Y_TOP,
            y_bottom=BAND_Y_TOP + BAND_H,
            balloons=[Balloon(
                strip_bbox=BBox(164, 29 + 16, 653, 29 + 212),  # page-abs
                confidence=0.9,
            )],
            original_slice=original.copy(),
            rendered_slice=rendered.copy(),
        )

        # ocr_page tem balloon_bbox expandida (largura total)
        ocr_page = {
            "texts": [{
                "bbox": [164, 16, 653, 212],
                "balloon_bbox": [0, 0, PAGE_W, BAND_H],  # band-local, expansão total
                "translated": "TEXTO DE TESTE",
            }]
        }

        result = _apply_copy_back_outside_balloons(band, balloon_margin=0, ocr_page=ocr_page)

        # Pixels dentro do balloon_bbox expandido devem vir do rendered (200)
        assert result[BAND_H // 2, 10, 0] == 200, (
            "pixel na borda esquerda da balloon_bbox deve ser do rendered "
            f"(got {result[BAND_H // 2, 10, 0]})"
        )
        assert result[BAND_H // 2, PAGE_W - 10, 0] == 200, (
            "pixel na borda direita da balloon_bbox deve ser do rendered "
            f"(got {result[BAND_H // 2, PAGE_W - 10, 0]})"
        )

    def test_copy_back_without_ocr_page_still_works(self):
        """Sem ocr_page, comportamento original é preservado (backward compat)."""
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox

        original = np.full((100, 300, 3), 50, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 200, dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            original_slice=original.copy(),
            rendered_slice=rendered.copy(),
        )

        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        # Dentro do balloon (com margem) → rendered
        assert result[50, 100, 0] == 200
        # Fora do balloon → original
        assert result[5, 5, 0] == 50

    def test_copy_back_preserves_render_bbox_when_balloon_bbox_collapses_to_text(self):
        """Texto renderizado fora do bbox original nao pode ser apagado pelo copyback."""
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band

        original = np.zeros((220, 420, 3), dtype=np.uint8)
        rendered = original.copy()

        # Simula texto branco renderizado que cresceu para cima/baixo do bbox
        # original, como nos baloes escuros conectados com geometria colapsada.
        rendered[58:166, 80:340] = 255

        band = Band(
            y_top=16468,
            y_bottom=16688,
            balloons=[],
            original_slice=original.copy(),
            rendered_slice=rendered.copy(),
        )
        ocr_page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [110, 90, 310, 130],
                    "balloon_bbox": [110, 90, 310, 130],
                    "text_pixel_bbox": [110, 90, 310, 130],
                    "render_bbox": [80, 58, 340, 166],
                    "safe_text_box": [70, 50, 350, 176],
                    "qa_flags": [
                        "visual_text_only_inpaint_contract",
                        "dark_bubble_visual_glyph_mask_replaced_geometry",
                    ],
                }
            ]
        }

        result = _apply_copy_back_outside_balloons(
            band,
            balloon_margin=0,
            ocr_page=ocr_page,
            rendered_slice=rendered,
        )

        assert result[60, 100, 0] == 255
        assert result[164, 320, 0] == 255
        assert result[10, 10, 0] == 0

    def test_typeset_stage_keeps_render_geometry_for_copyback(self):
        """O copyback deve receber render_bbox/safe_text_box produzidos no typeset."""
        from strip.process_bands import _run_typeset_stage

        class FakeTypesetter:
            def render_band_image(self, image, page):
                page["texts"][0]["render_bbox"] = [80, 58, 340, 166]
                page["texts"][0]["safe_text_box"] = [70, 50, 350, 176]
                return image.copy()

        translated_page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "translated": "TEXTO",
                    "bbox": [110, 90, 310, 130],
                    "balloon_bbox": [110, 90, 310, 130],
                }
            ]
        }
        image = np.zeros((220, 420, 3), dtype=np.uint8)

        _run_typeset_stage(
            image,
            typesetter=FakeTypesetter(),
            translated_page=translated_page,
        )

        assert translated_page["texts"][0]["render_bbox"] == [80, 58, 340, 166]
        assert translated_page["texts"][0]["safe_text_box"] == [70, 50, 350, 176]
