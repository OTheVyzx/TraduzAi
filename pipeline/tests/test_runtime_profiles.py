from __future__ import annotations

import os

from pipeline.runtime_profiles import (
    apply_runtime_profile_environment,
    build_chapter_route_shadow,
    evaluate_runtime_profile_gate,
    resolve_runtime_profile,
)


def test_runtime_profile_defaults_to_balanced_with_automatic_pipeline_defaults():
    decision = resolve_runtime_profile({})

    assert decision.profile == "balanced"
    assert decision.visual_stack_warmup is True
    assert decision.strip_inpainter_prewarm is True
    assert decision.semantic_review is False
    assert decision.macro_ocr == "off"
    assert decision.blocked_features == []
    assert decision.env_defaults["TRADUZAI_SEMANTIC_REVIEW"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_SCHEDULER_EXECUTOR"] == "overlap_context_release"
    assert decision.env_defaults["TRADUZAI_STRIP_PARALLEL_INPAINT_THREADS"] == "3"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_SOLID_INPAINT"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_WHITE_INPAINT"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_WHITE_NARRATION"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_DARK_PANEL_FILL"] == "0"
    assert decision.env_defaults["TRADUZAI_PAGE_CLEANUP_RERENDER"] == "0"
    assert decision.env_defaults["TRADUZAI_PADDLE_FULL_PAGE"] == "1"
    assert decision.env_defaults["TRADUZAI_STRIP_DETECT_FULL_PAGE"] == "1"
    assert "TRADUZAI_MACRO_OCR" not in decision.env_defaults
    assert decision.env_defaults["TRADUZAI_GOOGLE_PARALLEL_CHUNKS"] == "1"
    assert decision.env_defaults["TRADUZAI_GOOGLE_TRANSLATE_WORKERS"] == "3"
    assert decision.visual_pipeline_flags["runtime_fingerprint_v2"] is False
    assert decision.visual_pipeline_flags["visual_baseline_lossless_v2"] is False


def test_performance_profile_keeps_smart_skip_blocked_until_gate_passes():
    decision = resolve_runtime_profile({"runtime_profile": "performance"})

    assert decision.profile == "performance"
    assert decision.ready_for_default is False
    assert decision.smart_skip == "off"
    assert decision.macro_ocr == "off"
    assert "TRADUZAI_SMART_SKIP" in decision.blocked_features
    assert "TRADUZAI_MACRO_OCR" in decision.blocked_features
    assert decision.visual_stack_warmup is True
    assert decision.strip_inpainter_prewarm is True
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_SOLID_INPAINT"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_WHITE_INPAINT"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_WHITE_NARRATION"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_DARK_PANEL_FILL"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_PARALLEL_INPAINT_THREADS"] == "3"
    assert "TRADUZAI_MACRO_OCR" not in decision.env_defaults
    assert "TRADUZAI_STRIP_FAST_LOCAL_INPAINT" not in decision.env_defaults


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
    assert "TRADUZAI_STRIP_FAST_WHITE_INPAINT" not in decision.env_defaults
    assert "TRADUZAI_STRIP_FAST_LOCAL_INPAINT" not in decision.env_defaults


def test_fast_fill_rejects_skip_processing_and_qa_flags():
    from inpainter import _fast_local_rejection_reason, _fast_white_rejection_reason

    assert _fast_white_rejection_reason({"skip_processing": True, "tipo": "fala"}) == "skip_processing"
    assert _fast_local_rejection_reason({"skip_processing": True, "tipo": "fala"}) == "skip_processing"
    assert (
        _fast_white_rejection_reason({"tipo": "fala", "qa_flags": ["bbox_overreach"]})
        == "qa_flag:bbox_overreach"
    )
    assert (
        _fast_local_rejection_reason({"tipo": "fala", "qa_flags": ["mask_outside_balloon_critical"]})
        == "qa_flag:mask_outside_balloon_critical"
    )


def test_fast_solid_fill_is_opt_in_by_env(monkeypatch):
    from inpainter import _fast_solid_balloon_fill_enabled

    monkeypatch.delenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", raising=False)

    assert _fast_solid_balloon_fill_enabled() is False

    monkeypatch.setenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", "1")

    assert _fast_solid_balloon_fill_enabled() is True


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
    assert "TRADUZAI_SMART_SKIP" in result["gate"]["profiles"]["performance"]["blocked_features"]
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
