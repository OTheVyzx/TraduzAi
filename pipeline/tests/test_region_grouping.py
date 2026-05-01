from layout.region_grouping import group_regions, write_debug_overlay


def test_simple_balloon_stays_single_separated_group():
    grouped = group_regions([
        {"id": "a", "tipo": "fala", "bbox": [50, 40, 140, 90], "balloon_bbox": [30, 20, 170, 120]},
    ])

    assert grouped[0]["grouping_status"] == "separated"
    assert grouped[0]["layout_group_size"] == 1


def test_connected_double_balloon_groups_regions_with_same_balloon_bbox():
    regions = [
        {"id": "right", "tipo": "fala", "bbox": [180, 50, 240, 90], "balloon_bbox": [30, 20, 260, 130]},
        {"id": "left", "tipo": "fala", "bbox": [50, 52, 120, 92], "balloon_bbox": [30, 20, 260, 130]},
    ]

    grouped = group_regions(regions)

    assert len({item["group_id"] for item in grouped}) == 1
    assert all(item["grouping_status"] == "grouped" for item in grouped)
    assert [item["id"] for item in grouped] == ["right", "left"]


def test_near_regions_in_different_balloons_stay_separated_when_gap_is_large():
    regions = [
        {"id": "a", "tipo": "fala", "bbox": [40, 40, 100, 90], "balloon_bbox": [20, 20, 120, 120]},
        {"id": "b", "tipo": "fala", "bbox": [190, 42, 250, 92], "balloon_bbox": [170, 20, 270, 120]},
    ]

    grouped = group_regions(regions)

    assert len({item["group_id"] for item in grouped}) == 2
    assert all(item["grouping_status"] == "separated" for item in grouped)


def test_narration_and_sfx_do_not_merge_with_dialogue():
    regions = [
        {"id": "narration", "tipo": "narracao", "bbox": [40, 20, 240, 70], "balloon_bbox": [30, 10, 250, 80]},
        {"id": "sfx", "tipo": "sfx", "bbox": [45, 90, 110, 150], "balloon_bbox": [40, 85, 120, 160]},
        {"id": "dialogue", "tipo": "fala", "bbox": [120, 92, 180, 150], "balloon_bbox": [40, 85, 190, 160]},
    ]

    grouped = group_regions(regions)

    assert len({item["group_id"] for item in grouped}) == 3


def test_debug_overlay_is_written(tmp_path):
    grouped = group_regions([
        {"id": "a", "tipo": "fala", "bbox": [20, 20, 80, 80]},
    ])
    output = write_debug_overlay((120, 120), grouped, tmp_path / "page_001_overlay.png")

    assert output.exists()
