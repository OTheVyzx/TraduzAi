from layout.reading_order import order_regions


def test_manga_order_is_top_to_bottom_right_to_left_within_row():
    regions = [
        {"id": "left", "bbox": [20, 20, 80, 80]},
        {"id": "right", "bbox": [180, 22, 240, 82]},
        {"id": "bottom", "bbox": [120, 140, 200, 200]},
    ]

    ordered = order_regions(regions)

    assert [item["id"] for item in ordered] == ["right", "left", "bottom"]
    assert [item["reading_order"] for item in ordered] == [0, 1, 2]


def test_left_to_right_order_is_available_for_non_manga_layouts():
    regions = [
        {"id": "right", "bbox": [180, 20, 240, 80]},
        {"id": "left", "bbox": [20, 20, 80, 80]},
    ]

    ordered = order_regions(regions, direction="ltr")

    assert [item["id"] for item in ordered] == ["left", "right"]

