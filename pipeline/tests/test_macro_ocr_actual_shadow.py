import json
from pathlib import Path

import numpy as np

from pipeline.ocr.macro_ocr import classify_ocr_text_difference, compare_aligned_macro_ocr_texts
from pipeline.tools.run_macro_ocr_actual_shadow import evaluate_actual_macro_ocr_shadow


class FakeOcrEngine:
    def __init__(self, outputs, stats=None):
        self.outputs = outputs
        self.stats = stats
        self.calls = []

    def recognize_blocks_from_page(self, image_rgb, blocks, **kwargs):
        self.calls.append({"image": image_rgb, "blocks": blocks, "kwargs": kwargs})
        if self.stats is not None:
            self._last_recognize_blocks_stats = self.stats
        return self.outputs.pop(0)


def _write_project(output_dir: Path, blocks=None, texts=None) -> None:
    if blocks is None:
        blocks = [{"bbox": [10, 20, 80, 60]}]
    if texts is None:
        texts = [{"text": "HELLO", "bbox": [10, 20, 80, 60]}]
    (output_dir / "originals").mkdir(parents=True)
    (output_dir / "translated").mkdir()
    (output_dir / "originals" / "001.jpg").write_bytes(b"fake")
    (output_dir / "translated" / "001.jpg").write_bytes(b"fake")
    project = {
        "paginas": [
            {
                "numero": 1,
                "arquivo_original": "originals/001.jpg",
                "page_profile": {
                    "y_in_strip_top": 0,
                    "y_in_strip_bottom": 100,
                    "strip_perf_summary": {
                        "durations_sec": {"ocr": 10.0},
                        "entries": [
                            {"band_index": 0, "y_top": 0, "y_bottom": 100, "durations_sec": {"ocr": 10.0}}
                        ],
                    },
                },
                "inpaint_blocks": blocks,
                "text_layers": texts,
            }
        ],
        "estatisticas": {"total_paginas": 1, "total_textos": len(texts)},
    }
    (output_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")


def test_actual_macro_ocr_shadow_uses_injected_ocr_and_writes_summary(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(output_dir)
    fake_ocr = FakeOcrEngine([[{"text": "hello"}]])

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: object(),
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["pages_processed"] == 1
    assert result["gate"]["blocks_processed"] == 1
    assert result["gate"]["missing_text_rate"] == 0.0
    assert result["gate"]["exact_match_rate"] == 1.0
    assert fake_ocr.calls[0]["kwargs"]["allow_sparse_mapping"] is True
    assert fake_ocr.calls[0]["kwargs"]["crop_fallback_max"] == 0
    assert (tmp_path / "gate" / "summary.json").exists()


def test_actual_macro_ocr_shadow_passes_configured_crop_fallback_max(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(output_dir)
    fake_ocr = FakeOcrEngine([[{"text": "hello"}]])

    evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: object(),
        crop_fallback_max=2,
    )

    assert fake_ocr.calls[0]["kwargs"]["crop_fallback_max"] == 2


def test_actual_macro_ocr_shadow_fails_when_macro_output_misses_text(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(output_dir)

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=FakeOcrEngine([[{"text": ""}]]),
        image_loader=lambda _path: object(),
        max_missing_text_rate=0.02,
    )

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["missing_text_rate"] == 1.0
    assert result["gate"]["page_reports"][0]["samples"] == [
        {"index": 0, "status": "missing", "baseline": "HELLO", "macro": ""}
    ]


def test_actual_macro_ocr_shadow_fails_when_macro_output_changes_too_much_text(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(output_dir)

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=FakeOcrEngine([[{"text": "HULLO"}]]),
        image_loader=lambda _path: object(),
        max_different_text_rate=0.0,
    )

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["different_text_rate"] == 1.0
    assert result["gate"]["material_different_count"] == 1
    assert result["gate"]["material_different_text_rate"] == 1.0
    assert result["gate"]["fallback_adjusted_ocr_call_count"] == 2
    assert result["gate"]["fallback_adjusted_window_reduction_rate"] == 0.0
    assert result["gate"]["page_reports"][0]["samples"] == [
        {"index": 0, "status": "different", "baseline": "HELLO", "macro": "HULLO"}
    ]


def test_macro_ocr_difference_classifier_separates_fallback_risk_levels():
    assert (
        classify_ocr_text_difference(
            "THIS IS MAJOR DONG YOUNGSOO, CREW MEMBER OF THE ATALANTE.",
            "THIS IS MAJOR DONG YOUNGSOO, CREW MEMBER 67 OF THE ATALANTE.",
        )
        == "line_marker_artifact"
    )
    assert (
        classify_ocr_text_difference(
            "IF THEIR HIBERNATION PODS RELEASED AND THEY Reachep THE ESCAPE PODS...",
            "IF THEIR HIBERNATION PODS RELEASED AND THEY REACHED THE ESCAPE PODS...",
        )
        == "minor_ocr_variation"
    )
    assert (
        classify_ocr_text_difference(
            "TEMPERATURE DEGREES CELSIUS,",
            "TEMPERATURE 23 DEGREES CELSIUS,",
        )
        == "numeric_token_change"
    )
    assert classify_ocr_text_difference("HELLO", "HULLO") == "material"


def test_macro_ocr_difference_classifier_separates_safe_numeric_variations():
    assert (
        classify_ocr_text_difference("oo:oo:os", "00:00:05")
        == "numeric_confusable_variation"
    )
    assert (
        classify_ocr_text_difference("ESCAPEPOD ETA oo:32", "ESCAPEPOD ETA00:32")
        == "numeric_confusable_variation"
    )
    assert (
        classify_ocr_text_difference(
            "ATMOSPHERIC OXYGEN CONCENTRA TION IZ%",
            "ATMOSPHERIC OXYGEN CONCENTRA TION 1Z%",
        )
        == "numeric_confusable_variation"
    )
    assert classify_ocr_text_difference("EP Five years", "EP.1 Five years") == "episode_marker_variation"
    assert (
        classify_ocr_text_difference(
            "THE DESTINATION: KEPLER-E2GS B",
            "THE DESTINATION:S KEPLER-3265 B",
        )
        == "numeric_token_change"
    )


def test_macro_ocr_difference_classifier_treats_line_marker_with_minor_ocr_as_acceptable():
    assert (
        classify_ocr_text_difference(
            "CREW MEMBER DONG YOUNGSOO ARRINED ON PLANET ARCADIA, CONTINUING RECORD.",
            "CREW MEMBER 67 DONG YOUNGSOO ARRIVED ON PLANET ARCADIA, CONTINUING RECORD,",
        )
        == "line_marker_minor_variation"
    )


def test_actual_macro_ocr_shadow_fallback_cost_uses_required_fallback_count(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    blocks = [
        {"bbox": [10, 10, 40, 30]},
        {"bbox": [12, 45, 42, 70]},
        {"bbox": [50, 170, 90, 195]},
    ]
    texts = [
        {
            "text": "IF THEIR HIBERNATION PODS RELEASED AND THEY Reachep THE ESCAPE PODS...",
            "bbox": [10, 10, 40, 30],
        },
        {"text": "TEMPERATURE DEGREES CELSIUS,", "bbox": [12, 45, 42, 70]},
        {"text": "HELLO", "bbox": [50, 170, 90, 195]},
    ]
    _write_project(output_dir, blocks=blocks, texts=texts)
    fake_ocr = FakeOcrEngine(
        [
            [
                {"text": "IF THEIR HIBERNATION PODS RELEASED AND THEY REACHED THE ESCAPE PODS..."},
                {"text": "TEMPERATURE 23 DEGREES CELSIUS,"},
                {"text": "HULLO"},
            ]
        ]
    )

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: np.zeros((240, 120, 3), dtype=np.uint8),
        window_mode="band-groups",
        window_max_blocks=3,
        window_merge_gap=200,
        window_padding=0,
        max_different_text_rate=1.0,
    )

    gate = result["gate"]
    assert gate["minor_ocr_variation_count"] == 1
    assert gate["numeric_token_change_count"] == 1
    assert gate["material_different_count"] == 1
    assert gate["fallback_required_count"] == 2
    assert gate["fallback_adjusted_ocr_call_count"] == 3


def test_actual_macro_ocr_shadow_counts_safe_numeric_variations_as_audited(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    blocks = [
        {"bbox": [10, 10, 40, 30]},
        {"bbox": [12, 45, 42, 70]},
        {"bbox": [50, 170, 90, 195]},
    ]
    texts = [
        {"text": "oo:oo:os", "bbox": [10, 10, 40, 30]},
        {"text": "EP Five years", "bbox": [12, 45, 42, 70]},
        {"text": "THE DESTINATION: KEPLER-E2GS B", "bbox": [50, 170, 90, 195]},
    ]
    _write_project(output_dir, blocks=blocks, texts=texts)
    fake_ocr = FakeOcrEngine(
        [
            [
                {"text": "00:00:05"},
                {"text": "EP.1 Five years"},
                {"text": "THE DESTINATION:S KEPLER-3265 B"},
            ]
        ]
    )

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: np.zeros((240, 120, 3), dtype=np.uint8),
        window_mode="band-groups",
        window_max_blocks=3,
        window_merge_gap=200,
        window_padding=0,
        max_different_text_rate=1.0,
    )

    gate = result["gate"]
    assert gate["numeric_confusable_variation_count"] == 1
    assert gate["episode_marker_variation_count"] == 1
    assert gate["numeric_token_change_count"] == 1
    assert gate["acceptable_variation_count"] == 2
    assert gate["fallback_required_count"] == 1


def test_macro_ocr_compare_reports_fallback_resolved_text_rate():
    compare = compare_aligned_macro_ocr_texts(
        [
            {"text": "HELLO"},
            {"text": "TEMPERATURE DEGREES CELSIUS,"},
            {"text": "oo:oo:os"},
        ],
        [
            {"text": "HELLO"},
            {"text": "TEMPERATURE 23 DEGREES CELSIUS,"},
            {"text": "00:00:05"},
        ],
    )

    assert compare["different_count"] == 2
    assert compare["fallback_required_count"] == 1
    assert compare["fallback_resolved_different_count"] == 1
    assert compare["fallback_resolved_different_text_rate"] == 0.3333


def test_actual_macro_ocr_shadow_can_gate_on_fallback_resolved_text(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    blocks = [
        {"bbox": [10, 10, 40, 30]},
        {"bbox": [12, 45, 42, 70]},
        {"bbox": [50, 170, 90, 195]},
    ]
    texts = [
        {"text": "HELLO", "bbox": [10, 10, 40, 30]},
        {"text": "TEMPERATURE DEGREES CELSIUS,", "bbox": [12, 45, 42, 70]},
        {"text": "oo:oo:os", "bbox": [50, 170, 90, 195]},
    ]
    _write_project(output_dir, blocks=blocks, texts=texts)
    fake_ocr = FakeOcrEngine(
        [
            [
                {"text": "HELLO"},
                {"text": "TEMPERATURE 23 DEGREES CELSIUS,"},
                {"text": "00:00:05"},
            ]
        ]
    )

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: np.zeros((240, 120, 3), dtype=np.uint8),
        window_mode="band-groups",
        window_max_blocks=3,
        window_merge_gap=200,
        window_padding=0,
        max_different_text_rate=0.4,
        gate_on_fallback_resolved_text=True,
    )

    gate = result["gate"]
    assert gate["status"] == "PASS"
    assert gate["different_text_rate"] == 0.6667
    assert gate["fallback_resolved_different_text_rate"] == 0.3333
    assert gate["text_quality_gate_rate"] == 0.3333
    assert gate["gate_on_fallback_resolved_text"] is True


def test_actual_macro_ocr_shadow_fails_when_fallback_rate_is_too_high(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(output_dir)

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=FakeOcrEngine(
            [[{"text": "HELLO"}]],
            stats={"crop_fallback_attempts": 1, "crop_fallback_recovered": 1},
        ),
        image_loader=lambda _path: object(),
        max_fallback_rate=0.0,
    )

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["crop_fallback_attempts"] == 1
    assert result["gate"]["fallback_rate"] == 1.0


def test_actual_macro_ocr_shadow_groups_blocks_into_macro_windows(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    blocks = [
        {"bbox": [10, 10, 40, 30]},
        {"bbox": [12, 45, 42, 70]},
        {"bbox": [50, 170, 90, 195]},
    ]
    texts = [
        {"text": "A", "bbox": [10, 10, 40, 30]},
        {"text": "B", "bbox": [12, 45, 42, 70]},
        {"text": "C", "bbox": [50, 170, 90, 195]},
    ]
    _write_project(output_dir, blocks=blocks, texts=texts)
    fake_ocr = FakeOcrEngine([[{"text": "A"}, {"text": "B"}], [{"text": "C"}]])

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: np.zeros((240, 120, 3), dtype=np.uint8),
        window_mode="band-groups",
        window_max_blocks=2,
        window_merge_gap=20,
        window_padding=0,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["macro_window_count"] == 2
    assert len(fake_ocr.calls) == 2
    assert fake_ocr.calls[0]["image"].shape[:2] == (60, 32)
    assert fake_ocr.calls[0]["blocks"][0].xyxy == (0, 0, 30, 20)
    assert fake_ocr.calls[0]["blocks"][1].xyxy == (2, 35, 32, 60)


def test_actual_macro_ocr_shadow_fails_when_window_reduction_is_too_low(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    blocks = [
        {"bbox": [10, 10, 40, 30]},
        {"bbox": [12, 45, 42, 70]},
    ]
    texts = [
        {"text": "A", "bbox": [10, 10, 40, 30]},
        {"text": "B", "bbox": [12, 45, 42, 70]},
    ]
    _write_project(output_dir, blocks=blocks, texts=texts)
    fake_ocr = FakeOcrEngine([[{"text": "A"}], [{"text": "B"}]])

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: np.zeros((100, 100, 3), dtype=np.uint8),
        window_mode="band-groups",
        window_max_blocks=1,
        window_padding=0,
        min_window_reduction_rate=0.25,
    )

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["window_reduction_rate"] == 0.0


def test_actual_macro_ocr_shadow_fails_when_fallback_adjusted_reduction_is_too_low(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    blocks = [
        {"bbox": [10, 10, 40, 30]},
        {"bbox": [12, 45, 42, 70]},
    ]
    texts = [
        {"text": "A", "bbox": [10, 10, 40, 30]},
        {"text": "B", "bbox": [12, 45, 42, 70]},
    ]
    _write_project(output_dir, blocks=blocks, texts=texts)
    fake_ocr = FakeOcrEngine([[{"text": "X"}, {"text": "Y"}]])

    result = evaluate_actual_macro_ocr_shadow(
        output_dir,
        tmp_path / "gate",
        ocr_engine=fake_ocr,
        image_loader=lambda _path: np.zeros((100, 100, 3), dtype=np.uint8),
        window_mode="band-groups",
        window_max_blocks=2,
        window_merge_gap=100,
        window_padding=0,
        max_different_text_rate=1.0,
        min_fallback_adjusted_reduction_rate=0.25,
    )

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["fallback_adjusted_window_reduction_rate"] == 0.0
    assert "fallback-adjusted window reduction rate" in result["gate"]["reasons"][0]
