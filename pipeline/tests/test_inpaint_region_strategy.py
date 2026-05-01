from PIL import Image

from inpainter.region_strategy import debug_output_paths, plan_inpaint


def _mask(path):
    Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(path)


def test_classifier_uses_region_type_strategy(tmp_path):
    path = tmp_path / "mask.png"
    _mask(path)

    plan = plan_inpaint({"bbox": [0, 0, 20, 20], "background_type": "textured_background"}, path)

    assert plan["run"] is True
    assert plan["strategy"] == "lama_required"


def test_sfx_is_skipped_without_config(tmp_path):
    path = tmp_path / "mask.png"
    _mask(path)

    plan = plan_inpaint({"bbox": [0, 0, 20, 20], "tipo": "sfx"}, path)

    assert plan["run"] is False
    assert "sfx_preserved" in plan["qa_flags"]


def test_invalid_mask_blocks_inpaint(tmp_path):
    plan = plan_inpaint({"bbox": [0, 0, 20, 20]}, tmp_path / "missing.png")

    assert plan["run"] is False
    assert "mask_missing" in plan["qa_flags"]


def test_debug_outputs_are_declared(tmp_path):
    paths = debug_output_paths(tmp_path, 1)

    assert paths["before"].name == "page_001_before.png"
    assert paths["diff"].parent.exists()
