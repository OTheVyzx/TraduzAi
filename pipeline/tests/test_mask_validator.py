from PIL import Image

from inpainter.mask_validator import validate_export_contains_masks, validate_mask


def _mask(path, size=(20, 20), rect=(5, 5, 15, 15), alpha=255):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(rect[0], rect[2]):
        for y in range(rect[1], rect[3]):
            img.putpixel((x, y), (255, 255, 255, alpha))
    img.save(path)


def test_valid_mask(tmp_path):
    path = tmp_path / "mask.png"
    _mask(path)

    assert validate_mask(path, [0, 0, 20, 20])["valid"] is True


def test_1x1_mask_is_invalid(tmp_path):
    path = tmp_path / "mask.png"
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(path)

    assert validate_mask(path)["reason"] == "mask_too_small"


def test_transparent_mask_is_invalid(tmp_path):
    path = tmp_path / "mask.png"
    Image.new("RGBA", (10, 10), (255, 255, 255, 0)).save(path)

    assert validate_mask(path)["reason"] == "mask_transparent"


def test_bbox_mismatch_is_invalid(tmp_path):
    path = tmp_path / "mask.png"
    _mask(path, rect=(0, 0, 20, 20))

    assert validate_mask(path, [30, 30, 40, 40])["reason"] == "mask_bbox_mismatch"


def test_export_contains_masks(tmp_path):
    project = {"paginas": [{"text_layers": [{"id": "r1", "bbox": [0, 0, 20, 20]}]}]}

    flags = validate_export_contains_masks(project, tmp_path)

    assert flags[0]["type"] == "mask_missing"
