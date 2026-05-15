from types import SimpleNamespace

from engines.registry import normalize_pipeline_quality, resolve_engines


def test_normalize_pipeline_quality_accepts_legacy_values():
    assert normalize_pipeline_quality({"qualidade": "rapida"}) == "normal"
    assert normalize_pipeline_quality({"qualidade": "alta"}) == "ultra"
    assert normalize_pipeline_quality({"pipeline_quality": "ultra"}) == "ultra"
    assert normalize_pipeline_quality("max") == "ultra"


def test_resolve_engines_defaults_to_legacy_without_changing_runtime(monkeypatch):
    monkeypatch.delenv("TRADUZAI_ENGINE_DETECTOR", raising=False)
    monkeypatch.delenv("TRADUZAI_ENGINE_REFINER", raising=False)
    monkeypatch.delenv("TRADUZAI_ENGINE_MASK", raising=False)
    monkeypatch.delenv("TRADUZAI_ENGINE_INPAINT", raising=False)
    monkeypatch.delenv("TRADUZAI_ENGINE_VALIDATION", raising=False)

    bundle = resolve_engines({"pipeline_quality": "normal"})

    assert bundle.quality == "normal"
    assert bundle.ocr_profile == "max"
    assert bundle.detector_name == "legacy_visual_stack"
    assert bundle.text_refiner_name == "legacy_ocr_stage"
    assert bundle.mask_name == "legacy_vision_blocks"
    assert bundle.inpaint_name == "legacy_inpaint_band"
    assert bundle.validator_name == "off"
    assert bundle.summary()["detector"] == "legacy_visual_stack"
    assert bundle.summary()["text_refiner"] == "legacy_ocr_stage"
    assert bundle.summary()["mask"] == "legacy_vision_blocks"
    assert bundle.summary()["inpaint"] == "legacy_inpaint_band"
    assert bundle.summary()["validation"] == "off"


def test_resolve_engines_can_select_comic_layout_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADUZAI_ENGINE_DETECTOR", "comic_layout_rtdetr")

    bundle = resolve_engines({"pipeline_quality": "ultra", "models_dir": str(tmp_path)})

    assert bundle.quality == "ultra"
    assert bundle.detector_name == "comic_layout_rtdetr"
    assert bundle.detector.fallback.name == "legacy_visual_stack"


def test_legacy_text_refiner_keeps_bbox_shape():
    bundle = resolve_engines({"pipeline_quality": "normal"})
    refined = bundle.text_refiner.refine(None, [SimpleNamespace(xyxy=(1, 2, 3, 4), confidence=0.7)])

    assert refined == [
        {
            "bbox": [1, 2, 3, 4],
            "source_bbox": [1, 2, 3, 4],
            "line_polygons": [],
            "text_pixel_bbox": [1, 2, 3, 4],
            "confidence": 0.7,
            "text_refiner": "legacy_ocr_stage",
        }
    ]


def test_resolve_engines_can_select_ppocrv5_text_refiner(monkeypatch):
    monkeypatch.setenv("TRADUZAI_ENGINE_REFINER", "ppocrv5")

    bundle = resolve_engines({"pipeline_quality": "ultra", "idioma_origem": "en"})

    assert bundle.quality == "ultra"
    assert bundle.text_refiner_name == "ppocrv5_text_refiner"
    assert bundle.summary()["text_refiner"] == "ppocrv5_text_refiner"


def test_resolve_engines_can_select_smart_mask(monkeypatch):
    monkeypatch.setenv("TRADUZAI_ENGINE_MASK", "smart")

    bundle = resolve_engines({"pipeline_quality": "normal"})

    assert bundle.mask_name == "smart_text_mask"
    assert bundle.summary()["mask"] == "smart_text_mask"


def test_resolve_engines_can_select_lama_onnx_inpaint(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADUZAI_ENGINE_INPAINT", "lama_onnx")

    bundle = resolve_engines({"pipeline_quality": "ultra", "models_dir": str(tmp_path)})

    assert bundle.inpaint_name == "lama_onnx_composite"
    assert bundle.summary()["inpaint"] == "lama_onnx_composite"


def test_resolve_engines_can_select_residual_validator(monkeypatch):
    monkeypatch.setenv("TRADUZAI_ENGINE_VALIDATION", "residual")

    bundle = resolve_engines({"pipeline_quality": "ultra"})

    assert bundle.validator_name == "residual_validator"
    assert bundle.summary()["validation"] == "residual_validator"
