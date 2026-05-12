from types import SimpleNamespace

from vision_stack.runtime import (
    _apply_balloon_geometry_to_text_entry,
    _apply_text_geometry_to_serialized_block,
    _serialize_block,
)


def test_serialized_block_preserves_balloon_polygon_fields():
    block = SimpleNamespace(
        xyxy=(10, 20, 80, 100),
        mask=None,
        confidence=0.91,
        balloon_polygon=[[8, 18], [84, 18], [84, 104], [8, 104]],
        connected_lobe_polygons=[[[8, 18], [44, 18], [44, 104], [8, 104]]],
    )

    serialized = _serialize_block(block, (120, 120))

    assert serialized["balloon_polygon"][0] == [8, 18]
    assert serialized["connected_lobe_polygons"][0][2] == [44, 104]


def test_text_geometry_copy_flows_into_serialized_block():
    block = SimpleNamespace(xyxy=(10, 20, 80, 100), mask=None, confidence=0.91)
    raw_record = {
        "balloonPolygon": [[8, 18], [84, 18], [84, 104], [8, 104]],
        "balloonSubregions": [[8, 18, 44, 104], [45, 18, 84, 104]],
    }
    text_entry = {"text": "테스트", "bbox": [10, 20, 80, 100]}

    _apply_balloon_geometry_to_text_entry(text_entry, raw_record, block, (120, 120))
    enriched = _apply_text_geometry_to_serialized_block(_serialize_block(block, (120, 120)), text_entry)

    assert enriched["balloon_polygon"][1] == [84, 18]
    assert enriched["balloon_subregions"][1] == [45, 18, 84, 104]
