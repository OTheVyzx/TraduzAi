from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from editor_vision_cache import (
    EditorVisionCacheKey,
    build_detect_ocr_cache_key,
    build_detect_ocr_payload,
    build_ocr_layers_cache_key,
    build_ocr_layers_payload,
    is_detect_ocr_payload,
    is_ocr_layers_payload,
    read_cache_entry,
    write_cache_entry,
)  # noqa: E402


def test_detect_cache_key_changes_when_image_content_changes_with_same_size_and_mtime(
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.png"
    fixed_ns = 1_700_000_000_000_000_000
    image.write_bytes(b"first")
    os.utime(image, ns=(fixed_ns, fixed_ns))

    first = build_detect_ocr_cache_key(
        project_path=tmp_path / "project.json",
        page_index=0,
        image_path=image,
        idioma_origem="en",
        engine_preset_id="",
        schema_version=1,
    )

    image.write_bytes(b"third")
    os.utime(image, ns=(fixed_ns, fixed_ns))
    second = build_detect_ocr_cache_key(
        project_path=tmp_path / "project.json",
        page_index=0,
        image_path=image,
        idioma_origem="en",
        engine_preset_id="",
        schema_version=1,
    )

    assert image.stat().st_size == 5
    assert int(image.stat().st_mtime_ns) == fixed_ns
    assert first.digest != second.digest


def test_ocr_layers_cache_key_changes_when_layer_bbox_changes(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    layers = [{"id": "a", "bbox": [1, 2, 3, 4]}, {"id": "b", "bbox": [10, 20, 30, 40]}]

    first = build_ocr_layers_cache_key(
        project_path=tmp_path / "project.json",
        page_index=1,
        image_path=image,
        layers=layers,
        idioma_origem="ja",
        engine_preset_id="manga",
        schema_version=1,
    )
    layers[0]["bbox"] = [1, 2, 33, 44]
    second = build_ocr_layers_cache_key(
        project_path=tmp_path / "project.json",
        page_index=1,
        image_path=image,
        layers=layers,
        idioma_origem="ja",
        engine_preset_id="manga",
        schema_version=1,
    )

    assert first.digest != second.digest


def test_cache_round_trip_and_corrupt_file_handling(tmp_path: Path) -> None:
    key = EditorVisionCacheKey(kind="detect_ocr", digest="abc123", cache_dir=tmp_path)
    payload = {"status": "ready", "page_index": 0, "texts": [{"text": "HELLO"}]}

    write_cache_entry(key, payload)

    assert read_cache_entry(key) == payload
    key.path.write_text("{bad json", encoding="utf-8")
    assert read_cache_entry(key) is None
    key.path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    assert read_cache_entry(key) is None
    key.path.write_text(json.dumps(["ready"]), encoding="utf-8")
    assert read_cache_entry(key) is None
    key.path.write_bytes(b"\xff\xfe\xfa")
    assert read_cache_entry(key) is None


def test_detect_payload_contains_page_patch() -> None:
    payload = build_detect_ocr_payload(
        page_index=0,
        text_layers=[{"id": "t1", "original": "HELLO"}],
        inpaint_blocks=[{"bbox": [1, 2, 3, 4]}],
    )

    assert payload["status"] == "ready"
    assert payload["kind"] == "detect_ocr"
    assert payload["page_index"] == 0
    assert payload["text_layers"][0]["original"] == "HELLO"
    assert payload["inpaint_blocks"][0]["bbox"] == [1, 2, 3, 4]


def test_ocr_payload_contains_layer_text_updates() -> None:
    payload = build_ocr_layers_payload(
        page_index=3,
        layer_updates=[{"id": "a", "original": "OK", "ocr_confidence": 0.91, "confianca_ocr": 0.91}],
    )

    assert payload["status"] == "ready"
    assert payload["kind"] == "ocr_layers"
    assert payload["page_index"] == 3
    assert payload["layer_updates"][0]["id"] == "a"


def test_detect_payload_validator_rejects_wrong_contract() -> None:
    payload = build_detect_ocr_payload(
        page_index=2,
        text_layers=[{"id": "t1"}],
        inpaint_blocks=[{"bbox": [1, 2, 3, 4]}],
    )

    assert is_detect_ocr_payload(payload, page_index=2)
    assert not is_detect_ocr_payload({**payload, "kind": "ocr_layers"}, page_index=2)
    assert not is_detect_ocr_payload({**payload, "schema_version": 2}, page_index=2)
    assert not is_detect_ocr_payload({**payload, "page_index": 3}, page_index=2)
    assert not is_detect_ocr_payload({**payload, "text_layers": None}, page_index=2)
    assert not is_detect_ocr_payload({**payload, "page_index": "bad"}, page_index=2)
    assert not is_detect_ocr_payload(None, page_index=2)


def test_ocr_payload_validator_rejects_wrong_contract() -> None:
    payload = build_ocr_layers_payload(
        page_index=4,
        layer_updates=[{"id": "a"}],
    )

    assert is_ocr_layers_payload(payload, page_index=4)
    assert not is_ocr_layers_payload({**payload, "kind": "detect_ocr"}, page_index=4)
    assert not is_ocr_layers_payload({**payload, "schema_version": 2}, page_index=4)
    assert not is_ocr_layers_payload({**payload, "page_index": 5}, page_index=4)
    assert not is_ocr_layers_payload({**payload, "layer_updates": None}, page_index=4)
    assert not is_ocr_layers_payload({**payload, "page_index": "bad"}, page_index=4)
    assert not is_ocr_layers_payload(None, page_index=4)
