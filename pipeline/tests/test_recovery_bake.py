from pathlib import Path

from PIL import Image

from recovery import apply_recovery_layer, bake_recovery_layer


def test_bake_recovery_layer_restores_only_masked_pixels():
    rendered = Image.new("RGBA", (3, 1), (10, 10, 10, 255))
    original = Image.new("RGBA", (3, 1), (200, 20, 20, 255))
    mask = Image.new("L", (3, 1), 0)
    mask.putpixel((1, 0), 255)

    result = bake_recovery_layer(rendered, original, mask)

    assert result.getpixel((0, 0)) == (10, 10, 10, 255)
    assert result.getpixel((1, 0)) == (200, 20, 20, 255)
    assert result.getpixel((2, 0)) == (10, 10, 10, 255)


def test_apply_recovery_layer_saves_rendered_file(tmp_path: Path):
    rendered_path = tmp_path / "rendered.png"
    original_path = tmp_path / "original.png"
    mask_path = tmp_path / "recovery.png"

    Image.new("RGBA", (2, 2), (0, 0, 0, 255)).save(rendered_path)
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(original_path)
    mask = Image.new("L", (2, 2), 0)
    mask.putpixel((0, 1), 255)
    mask.save(mask_path)

    assert apply_recovery_layer(rendered_path, original_path, mask_path) is True

    with Image.open(rendered_path).convert("RGBA") as img:
        assert img.getpixel((0, 0)) == (0, 0, 0, 255)
        assert img.getpixel((0, 1)) == (255, 0, 0, 255)


def test_apply_recovery_layer_saves_valid_jpeg(tmp_path: Path):
    rendered_path = tmp_path / "rendered.jpg"
    original_path = tmp_path / "original.jpg"
    mask_path = tmp_path / "recovery.png"

    Image.new("RGB", (16, 16), (0, 0, 0)).save(rendered_path)
    Image.new("RGB", (16, 16), (255, 0, 0)).save(original_path)
    mask = Image.new("L", (16, 16), 0)
    for x in range(16):
        for y in range(8, 16):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    assert apply_recovery_layer(rendered_path, original_path, mask_path) is True

    with Image.open(rendered_path) as img:
        img.verify()
    with Image.open(rendered_path).convert("RGB") as img:
        assert img.getpixel((8, 12))[0] > 200
