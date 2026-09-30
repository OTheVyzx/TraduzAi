"""Fail-closed regression for OCR support that misses its source slot."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_source_glyph_outside_replacement_bbox_routes_owner_to_review(monkeypatch):
    from strip.process_bands import execute_owner_page_graph
    from test_final_pixel_qa import _graph

    graph = _graph(state="execution_planned")
    owner = graph.owners[0]
    owner.route_action = "translate_inpaint_render"
    owner.translated_payload = None
    graph.observations[0] = replace(
        graph.observations[0],
        bbox_page=(5, 5, 15, 10),
        polygons_page=(((5, 5), (15, 5), (15, 10), (5, 10)),),
        layout_bbox_page=(0, 0, 40, 24),
    )
    page = np.full((24, 40, 3), 230, dtype=np.uint8)
    misplaced_glyph = np.zeros(page.shape[:2], dtype=np.uint8)
    misplaced_glyph[12:15, 20:25] = 255

    class Translator:
        @staticmethod
        def translate_pages(_pages, **_kwargs):
            return [{"texts": [{"owner_id": owner.owner_id, "translated": "DESTINO"}]}]

    class MustNotRun:
        @staticmethod
        def inpaint_band_image(*_args, **_kwargs):
            raise AssertionError("inpaint must not run without authorized source ink")

        @staticmethod
        def render_band_image(*_args, **_kwargs):
            raise AssertionError("typeset must not run without authorized source ink")

    monkeypatch.setattr(
        "strip.process_bands._owner_component_glyph_raster",
        lambda *_args, **_kwargs: misplaced_glyph.copy(),
    )

    execution = execute_owner_page_graph(
        page,
        graph,
        translator=Translator(),
        inpainter=MustNotRun(),
        typesetter=MustNotRun(),
    )

    assert execution.commits == ()
    assert execution.graph.owners[0].disposition == "review"
    assert execution.records[0]["route_action"] == "review_required"
    assert execution.records[0]["owner_execution_rejection_reason"] == (
        "source replacement mask is empty"
    )
    assert "owner_mask_unsafe" in execution.records[0]["qa_flags"]
