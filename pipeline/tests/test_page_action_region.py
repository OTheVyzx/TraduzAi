from main import _bbox_in_region, _page_action_region_from_args


def test_page_action_region_parses_bbox_args():
    region = _page_action_region_from_args(["--region-bbox", "10,20,30,40"])

    assert region["bbox"] == [10, 20, 30, 40]


def test_bbox_in_region_uses_intersection():
    region = {"bbox": [10, 10, 30, 30], "mask_path": None}

    assert _bbox_in_region([20, 20, 40, 40], region) is True
    assert _bbox_in_region([31, 31, 40, 40], region) is False


def test_missing_region_keeps_global_behavior():
    assert _bbox_in_region([100, 100, 120, 120], {"bbox": None, "mask_path": None}) is True
