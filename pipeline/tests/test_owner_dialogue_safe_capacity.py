from dataclasses import replace
from types import SimpleNamespace


def test_protected_safe_source_slot_does_not_receive_a_second_dialogue_inset():
    from test_final_pixel_qa import _graph
    from strip.process_bands import _owner_layout_regions

    graph = _graph()
    graph.components[0] = replace(
        graph.components[0],
        bbox_page=(10, 15, 230, 135),
        polygon_page=((10, 15), (230, 15), (230, 135), (10, 135)),
    )
    geometry = SimpleNamespace(
        owner_id=graph.owners[0].owner_id,
        page_id=graph.page_id,
        page_width=240,
        page_height=150,
        layout_container_bbox_page=(20, 25, 220, 125),
        layout_container_polygon_page=(
            (20, 25),
            (220, 25),
            (220, 125),
            (20, 125),
        ),
        source_replacement_bbox_page=(20, 25, 220, 125),
        layout_container_source="full_page_visual_container:protected_mask_safe",
        status="ready",
        geometry_sha256="a" * 64,
    )

    region = _owner_layout_regions(
        graph,
        page_width=240,
        page_height=150,
        owner_render_geometry=geometry,
    )[0]

    assert region["bbox_page"] == [20, 25, 220, 125]
    assert region["safe_polygon_page"] == [
        [20, 25],
        [219, 25],
        [219, 124],
        [20, 124],
    ]
    assert region["paint_safe_polygon_page"] == [
        [10, 15],
        [229, 15],
        [229, 134],
        [10, 134],
    ]
