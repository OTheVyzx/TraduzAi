from __future__ import annotations

import os

from pipeline.runtime_profiles import (
    apply_runtime_profile_environment,
    build_chapter_route_shadow,
    evaluate_runtime_profile_gate,
    resolve_runtime_profile,
)


def test_runtime_profile_defaults_to_balanced_without_behavior_changes():
    decision = resolve_runtime_profile({})

    assert decision.profile == "balanced"
    assert decision.visual_stack_warmup is True
    assert decision.strip_inpainter_prewarm is True
    assert decision.semantic_review is False
    assert decision.blocked_features == []
    assert decision.env_defaults == {"TRADUZAI_SEMANTIC_REVIEW": "0"}


def test_performance_profile_keeps_blocked_accelerators_disabled_until_gates_pass():
    decision = resolve_runtime_profile({"runtime_profile": "performance"})

    assert decision.profile == "performance"
    assert decision.ready_for_default is False
    assert decision.smart_skip == "off"
    assert decision.macro_ocr == "off"
    assert "TRADUZAI_SMART_SKIP" in decision.blocked_features
    assert "TRADUZAI_MACRO_OCR" in decision.blocked_features
    assert decision.visual_stack_warmup is True
    assert decision.strip_inpainter_prewarm is True


def test_eco_profile_disables_optional_prewarm_and_caps_cpu_threads():
    decision = resolve_runtime_profile({"runtime_profile": "eco"})

    assert decision.profile == "eco"
    assert decision.ready_for_default is True
    assert decision.visual_stack_warmup is False
    assert decision.strip_inpainter_prewarm is False
    assert decision.semantic_review is False
    assert decision.cpu_thread_limit == 2
    assert decision.env_defaults["TRADUZAI_STRIP_INPAINTER_PREWARM"] == "0"
    assert decision.env_defaults["OMP_NUM_THREADS"] == "2"


def test_runtime_profile_can_be_requested_by_preset_object():
    decision = resolve_runtime_profile({"preset": {"runtime_profile": "eco"}})

    assert decision.profile == "eco"


def test_apply_runtime_profile_environment_preserves_explicit_env(monkeypatch):
    monkeypatch.setenv("TRADUZAI_STRIP_INPAINTER_PREWARM", "1")
    decision = resolve_runtime_profile({"runtime_profile": "eco"})

    applied = apply_runtime_profile_environment(decision)

    assert os.environ["TRADUZAI_STRIP_INPAINTER_PREWARM"] == "1"
    assert applied["TRADUZAI_STRIP_INPAINTER_PREWARM"] == "preserved"
    assert os.environ["TRADUZAI_SEMANTIC_REVIEW"] == "0"


def test_runtime_profile_gate_documents_performance_blockers(tmp_path):
    result = evaluate_runtime_profile_gate(tmp_path / "gate")

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["profiles"]["performance"]["ready_for_default"] is False
    assert "TRADUZAI_MACRO_OCR" in result["gate"]["profiles"]["performance"]["blocked_features"]
    assert result["gate"]["profiles"]["eco"]["ready_for_default"] is True
    assert (tmp_path / "gate" / "summary.json").exists()


def test_chapter_route_shadow_records_recommendation_without_switching_routes():
    pages = [
        {
            "numero": 1,
            "texts": [{"bbox": [10, 10, 12, 60], "qa_flags": []}],
            "_vision_blocks": [{"bbox": [10, 10, 100, 80]} for _ in range(4)],
        },
        {
            "numero": 2,
            "texts": [{"bbox": [20, 20, 80, 48], "qa_flags": ["vlm_failure_phrase"]}],
            "_vision_blocks": [{"bbox": [20, 20, 80, 48]}],
        },
    ]

    shadow = build_chapter_route_shadow(pages)

    assert shadow["mode"] == "shadow"
    assert shadow["status"] == "PASS"
    assert shadow["sample_size"] == 2
    assert shadow["recommendation"] in {"enable_bbox_expanded_reocr", "consider_page_detect_after_reocr"}
    assert all(item["route"] == "shadow_only" for item in shadow["route_history"])
