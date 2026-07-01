from typesetter.backend_contract import (
    DEFAULT_FONT_FAMILY,
    TypesettingRenderRequest,
    TypesettingRenderResult,
    build_rust_render_request,
)


def _minimal_payload():
    return {
        "text": "HELLO",
        "translated": "OLA",
        "bbox": [10, 12, 110, 72],
        "safe_text_box": [18, 20, 100, 64],
        "render_bbox": [18, 20, 100, 64],
        "rotation_deg": 0,
        "line_polygons": [[[18, 20], [100, 20], [100, 64], [18, 64]]],
        "bubble_mask_path": "masks/page_001.png",
        "bubble_mask_value": 7,
        "bubble_id": "bubble-7",
        "font_family": DEFAULT_FONT_FAMILY,
        "font_weight": "bold",
        "font_size_px": 24,
        "stroke_width": 2,
        "fill_rgb": [0, 0, 0],
        "stroke_rgb": [255, 255, 255],
    }


def test_request_contract_round_trips_required_renderer_fields():
    request = TypesettingRenderRequest.from_mapping(_minimal_payload())

    assert request.to_mapping() == _minimal_payload()
    assert request.font_family == "ComicNeue-Bold.ttf"
    assert request.font_weight == "bold"


def test_request_contract_defaults_to_comic_neue_bold():
    payload = _minimal_payload()
    payload.pop("font_family")
    payload.pop("font_weight")

    request = TypesettingRenderRequest.from_mapping(payload)

    assert request.font_family == "ComicNeue-Bold.ttf"
    assert request.font_weight == "bold"


def test_request_contract_rejects_missing_bubble_mask_for_koharu():
    payload = _minimal_payload()
    payload["bubble_mask_path"] = ""

    try:
        TypesettingRenderRequest.from_mapping(payload)
    except ValueError as exc:
        assert "bubble_mask_path" in str(exc)
    else:
        raise AssertionError("request without bubble_mask_path should fail closed")


def test_build_rust_render_request_rejects_missing_bubble_mask_path():
    text_data = {
        "id": "t1",
        "translated": "OLA",
        "safe_text_box": [18, 20, 100, 64],
        "bubble_mask_value": 7,
    }

    try:
        build_rust_render_request((120, 80), text_data)
    except ValueError as exc:
        assert "bubble_mask_path" in str(exc)
    else:
        raise AssertionError("rust render request without bubble_mask_path should fail closed")


def test_result_contract_contains_required_backend_response_fields():
    result = TypesettingRenderResult.from_mapping(
        {
            "render_bbox": [18, 20, 100, 64],
            "font_size_px": 22,
            "fit_status": "ok",
            "backend": "koharu",
        }
    )

    assert result.to_mapping() == {
        "render_bbox": [18, 20, 100, 64],
        "font_size_px": 22,
        "fit_status": "ok",
        "backend": "koharu",
    }
