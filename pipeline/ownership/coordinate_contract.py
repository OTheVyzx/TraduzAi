"""Convert pixel-vertex OCR geometry to half-open component boxes.

OpenCV fillPoly includes the terminal vertex pixel. The mask planner slices
component boxes as [x1:x2, y1:y2], so the source polygon's max needs +1.
This derives the endpoint from authenticated polygon vertices and leaves the
OCR observation and coverage snapshot unchanged.
"""
from __future__ import annotations

from dataclasses import replace

from ownership.model import SourceTextComponent


def pixel_polygon_component_half_open(
    component: SourceTextComponent, *, page_width: int, page_height: int
) -> tuple[SourceTextComponent, dict]:
    points = tuple(component.polygon_page)
    if len(points) < 3:
        raise ValueError('component polygon missing')
    min_x = min(int(point[0]) for point in points)
    min_y = min(int(point[1]) for point in points)
    max_x = max(int(point[0]) for point in points)
    max_y = max(int(point[1]) for point in points)
    if not (0 <= min_x <= max_x < page_width and 0 <= min_y <= max_y < page_height):
        raise ValueError('component polygon outside page')
    if component.bbox_page == (min_x, min_y, max_x + 1, max_y + 1):
        return component, {
            'component_id': component.component_id,
            'ocr_bbox_inclusive_max': None,
            'planner_bbox_half_open': list(component.bbox_page),
            'derivation': 'already_half_open',
        }
    if component.bbox_page != (min_x, min_y, max_x, max_y):
        raise ValueError('OCR bbox is neither pixel-vertex nor half-open geometry')
    new_box = (min_x, min_y, max_x + 1, max_y + 1)
    return replace(component, bbox_page=new_box), {
        'component_id': component.component_id,
        'ocr_bbox_inclusive_max': list(component.bbox_page),
        'planner_bbox_half_open': list(new_box),
        'derivation': 'polygon_pixel_vertex_extrema_plus_one',
    }
