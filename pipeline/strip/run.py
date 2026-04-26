"""Entry-point do pipeline strip-based.

Chamado por `pipeline/main.py::_run_pipeline` após a Fase 6 do switchover.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from strip._diagnostics import dump_strip_debug, is_debug_enabled
from strip.bands import attach_band_slices, group_balloons_into_bands
from strip.concat import build_strip
from strip.detect_balloons import detect_strip_balloons
from strip.process_bands import process_band
from strip.reassemble import assemble_output_pages, paste_bands_into_strip
from strip.types import OutputPage


def run_chapter(
    image_files: list[Path],
    output_dir: Path,
    target_count: int = 60,
    *,
    detector,
    runtime,
    translator,
    inpainter,
    typesetter,
    context: dict | None = None,
    glossario: dict | None = None,
    idioma_origem: str = "en",
    idioma_destino: str = "pt-BR",
    obra: str = "",
    connected_reasoner_config: dict | None = None,

    progress_callback=None,
) -> list[OutputPage]:
    """Executa o pipeline strip-based ponta-a-ponta."""
    if not image_files:
        return []

    page_paths = image_files


    page_paths = image_files
    strip = build_strip(page_paths, progress_callback=progress_callback)


    if progress_callback: progress_callback("detect", 0, 1)
    balloons = detect_strip_balloons(strip, detector=detector)

    bands = group_balloons_into_bands(balloons)
    attach_band_slices(strip, bands)

    if is_debug_enabled():
        dump_strip_debug(strip, bands, output_dir.parent / "_strip_debug")

    for idx, band in enumerate(bands):
        if progress_callback: progress_callback("process", idx, len(bands))
        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=idx,
            context=context,
            glossario=glossario,
            idioma_origem=idioma_origem,
            idioma_destino=idioma_destino,
            obra=obra,
            connected_reasoner_config=connected_reasoner_config,

        )

    paste_bands_into_strip(strip, bands)
    output_pages = assemble_output_pages(strip, balloons, target_count=target_count)

    # Remapeamento de metadados para project.json
    all_texts: list[dict] = []
    all_vision_blocks: list[dict] = []
    for band in bands:
        if not band.ocr_result:
            continue
        b_y = band.y_top
        # Coleta textos e remapa para coordenadas do strip
        for txt in band.ocr_result.get("texts", []):
            new_txt = dict(txt)
            # bbox é OBRIGATÓRIO — pular texto sem bbox para evitar placeholder [0,0,32,32]
            if not new_txt.get("bbox"):
                continue
            x1, y1, x2, y2 = new_txt["bbox"]
            new_txt["bbox"] = [x1, y1 + b_y, x2, y2 + b_y]
            if "balloon_bbox" in new_txt and new_txt["balloon_bbox"]:
                bx1, by1, bx2, by2 = new_txt["balloon_bbox"]
                new_txt["balloon_bbox"] = [bx1, by1 + b_y, bx2, by2 + b_y]
            # Remapar subregions se houver
            if "balloon_subregions" in new_txt:
                new_subs = []
                for sub in new_txt["balloon_subregions"]:
                    new_subs.append([sub[0], sub[1] + b_y, sub[2], sub[3] + b_y])
                new_txt["balloon_subregions"] = new_subs
            all_texts.append(new_txt)

        for vb in band.ocr_result.get("_vision_blocks", []):
            new_vb = dict(vb)
            if not new_vb.get("bbox"):
                continue
            x1, y1, x2, y2 = new_vb["bbox"]
            new_vb["bbox"] = [x1, y1 + b_y, x2, y2 + b_y]
            all_vision_blocks.append(new_vb)

    def _assign_text_to_page(txt_y1: int, txt_y2: int, pages: list) -> int | None:
        """Retorna índice da página com maior intersecção em y (sem duplicar)."""
        best_idx = None
        best_overlap = 0
        for idx, page in enumerate(pages):
            overlap = max(0, min(txt_y2, page.y_bottom) - max(txt_y1, page.y_top))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        return best_idx

    # Inicializar listas em cada página
    for page in output_pages:
        page.ocr_result = {"_vision_blocks": []}
        page.text_layers = {"texts": []}

    # Distribuir textos para as páginas por máxima intersecção (não centro-y)
    for txt in all_texts:
        tx1, ty1, tx2, ty2 = txt["bbox"]
        pidx = _assign_text_to_page(ty1, ty2, output_pages)
        if pidx is None:
            continue
        page = output_pages[pidx]
        p_y0 = page.y_top
        local_txt = dict(txt)
        local_txt["bbox"] = [tx1, ty1 - p_y0, tx2, ty2 - p_y0]
        if "balloon_bbox" in local_txt and local_txt["balloon_bbox"]:
            bx1, by1, bx2, by2 = local_txt["balloon_bbox"]
            local_txt["balloon_bbox"] = [bx1, by1 - p_y0, bx2, by2 - p_y0]
        if "balloon_subregions" in local_txt:
            local_txt["balloon_subregions"] = [
                [s[0], s[1] - p_y0, s[2], s[3] - p_y0]
                for s in local_txt["balloon_subregions"]
            ]
        page.text_layers["texts"].append(local_txt)

    # Distribuir vision_blocks igualmente
    for vb in all_vision_blocks:
        vx1, vy1, vx2, vy2 = vb["bbox"]
        pidx = _assign_text_to_page(vy1, vy2, output_pages)
        if pidx is None:
            continue
        page = output_pages[pidx]
        p_y0 = page.y_top
        local_vb = dict(vb)
        local_vb["bbox"] = [vx1, vy1 - p_y0, vx2, vy2 - p_y0]
        page.ocr_result["_vision_blocks"].append(local_vb)

    # Preencher page_profile e inpaint_blocks em cada página
    for page in output_pages:
        page.page_profile = {
            "width": strip.width,
            "height": page.y_bottom - page.y_top,
            "y_in_strip_top": page.y_top,
            "y_in_strip_bottom": page.y_bottom,
        }
        page.inpaint_blocks = [
            {"bbox": vb["bbox"]}
            for vb in page.ocr_result.get("_vision_blocks", [])
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(output_pages):
        page.path = output_dir / f"{i + 1:03d}.jpg"
        cv2.imwrite(str(page.path), page.image, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return output_pages

