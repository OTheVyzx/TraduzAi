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
from strip.reassemble import assemble_output_pages
from strip.types import OutputPage, VerticalStrip


def _paste_band_attr_into_image(strip_image, bands: list, attr_name: str):
    result = strip_image.copy()
    strip_height = result.shape[0]
    for band in bands:
        band_slice = getattr(band, attr_name, None)
        if band_slice is None:
            continue
        y0 = max(0, band.y_top)
        y1 = min(strip_height, band.y_bottom)
        h_avail = y1 - y0
        if h_avail <= 0:
            continue
        result[y0:y1, :, :] = band_slice[:h_avail, :, :]
    return result


def _shift_bbox_y(value, delta_y: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    return [int(value[0]), int(value[1]) + delta_y, int(value[2]), int(value[3]) + delta_y]


def _shift_bbox_list_y(values, delta_y: int) -> list[list[int]]:
    shifted: list[list[int]] = []
    for value in values or []:
        bbox = _shift_bbox_y(value, delta_y)
        if bbox is not None:
            shifted.append(bbox)
    return shifted


def _shift_polygons_y(polygons, delta_y: int):
    if not isinstance(polygons, list):
        return polygons
    shifted = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            shifted.append(polygon)
            continue
        shifted_polygon = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                shifted_polygon.append([int(point[0]), int(point[1]) + delta_y])
            else:
                shifted_polygon.append(point)
        shifted.append(shifted_polygon)
    return shifted


def _shift_text_geometry_y(text: dict, delta_y: int) -> dict:
    shifted = dict(text)

    for key in ("bbox", "source_bbox", "balloon_bbox", "text_pixel_bbox"):
        bbox = _shift_bbox_y(shifted.get(key), delta_y)
        if bbox is not None:
            shifted[key] = bbox

    for key in (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_text_groups",
        "connected_position_bboxes",
        "connected_focus_bboxes",
        "_merged_source_bboxes",
    ):
        if key in shifted:
            shifted[key] = _shift_bbox_list_y(shifted.get(key), delta_y)

    for key in ("line_polygons", "connected_lobe_polygons"):
        if key in shifted:
            shifted[key] = _shift_polygons_y(shifted.get(key), delta_y)

    return shifted


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
    original_strip_image = strip.image.copy()


    if progress_callback: progress_callback("detect", 0, 1)
    balloons = detect_strip_balloons(strip, detector=detector)

    bands = group_balloons_into_bands(balloons)
    attach_band_slices(strip, bands)

    if is_debug_enabled():
        dump_strip_debug(strip, bands, output_dir.parent / "_strip_debug")

    running_glossary: dict = dict(glossario or {})
    running_history: list[dict] = []

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
            glossario=running_glossary,
            idioma_origem=idioma_origem,
            idioma_destino=idioma_destino,
            obra=obra,
            connected_reasoner_config=connected_reasoner_config,
            band_history=running_history[-20:],
        )
        # Acumular history e mesclar adições ao glossário
        if band.ocr_result:
            running_history.append(band.ocr_result)
            additions = band.ocr_result.get("_glossary_additions")
            if additions and isinstance(additions, dict):
                running_glossary.update(additions)

    clean_strip_image = _paste_band_attr_into_image(original_strip_image, bands, "cleaned_slice")
    rendered_strip_image = _paste_band_attr_into_image(original_strip_image, bands, "rendered_slice")
    strip.image[:, :, :] = rendered_strip_image

    output_pages = assemble_output_pages(strip, balloons, target_count=target_count)
    original_pages = assemble_output_pages(
        VerticalStrip(
            image=original_strip_image,
            width=strip.width,
            height=strip.height,
            source_page_breaks=list(strip.source_page_breaks),
            page_x_offsets=list(strip.page_x_offsets),
        ),
        balloons,
        target_count=target_count,
    )
    clean_pages = assemble_output_pages(
        VerticalStrip(
            image=clean_strip_image,
            width=strip.width,
            height=strip.height,
            source_page_breaks=list(strip.source_page_breaks),
            page_x_offsets=list(strip.page_x_offsets),
        ),
        balloons,
        target_count=target_count,
    )

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
            new_txt = _shift_text_geometry_y(new_txt, b_y)
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
        local_txt = _shift_text_geometry_y(txt, -p_y0)
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

    for page, original_page, clean_page in zip(output_pages, original_pages, clean_pages):
        page.original_image = original_page.image
        page.inpainted_image = clean_page.image

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(output_pages):
        page.path = output_dir / f"{i + 1:03d}.jpg"
        cv2.imwrite(str(page.path), page.image, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return output_pages
