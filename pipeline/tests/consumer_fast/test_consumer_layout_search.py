"""The bounded search explores reflow before shrinking and fails closed."""

import numpy as np

import consumer_layout_search as search
from consumer_focal_text import wrap_lines
from pathlib import Path
import pytest


def _recipe():
    return dict(font_path="test.ttf", source_sha256="s", target="TEXTO COMPLETO",
                safe_bbox=[0, 0, 112, 100], visual_comfort_policy={"source_sha256": "s"})


def test_same_size_reflow_precedes_scale_reduction(monkeypatch):
    visited = []
    monkeypatch.setattr(search, "prepare_source_white_comfort", lambda *_: {"source_sha256": "s", "ideal_px": 9})
    monkeypatch.setattr(search, "_visual_core_height", lambda _font, size, _sample: 29 if size == 30 else 26)
    monkeypatch.setattr(search, "wrap_lines", lambda _target, _font, _size, _width: ["TEXTO", "COMPLETO"])
    monkeypatch.setattr(search, "compact_safe_advance", lambda _font, size, _lines: (size+3, {}))
    monkeypatch.setattr(search, "line_advance", lambda _font, size, _profile: size+3)
    monkeypatch.setattr(search, "rasterize", lambda _font, _size, line: np.zeros((10, len(line)*9), np.uint8))

    def render(_clean, recipe, *, prepared_comfort):
        visited.append((recipe["font_size"], recipe["max_line_width_px"]))
        if recipe["max_line_width_px"] > 80:
            raise ValueError("source contour collision")
        return object(), dict(visual_comfort=dict(ideal_px=9, minimum_actual_px=8),
                              center_error_px=[0.0, 0.0])

    monkeypatch.setattr(search, "render_reviewed_text", render)
    result = search.search_source_centered_layout(np.zeros((1, 1, 3), np.uint8), _recipe(),
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=29, maximum_font_size=30,
        width_ratios=(1.0, .75), maximum_candidates=4)
    assert result["status"] == "rendered_candidate"
    assert visited == [(30, 100), (30, 75), (29, 100), (29, 75)]
    assert result["recipe"]["font_size"] == 30
    assert result["recipe"]["max_line_width_px"] == 75


def test_no_safe_candidate_is_explicit_and_bounded(monkeypatch):
    monkeypatch.setattr(search, "prepare_source_white_comfort", lambda *_: {"source_sha256": "s", "ideal_px": 9})
    monkeypatch.setattr(search, "_visual_core_height", lambda _font, size, _sample: size)
    monkeypatch.setattr(search, "wrap_lines", lambda *_: ["TEXTO"])
    monkeypatch.setattr(search, "line_advance", lambda *_: 33)
    monkeypatch.setattr(search, "render_reviewed_text", lambda *_a, **_k:
                        (_ for _ in ()).throw(ValueError("contour collision")))
    result = search.search_source_centered_layout(np.zeros((1, 1, 3), np.uint8), _recipe(),
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=20, maximum_font_size=30,
        width_ratios=(1.0, .75), maximum_candidates=3)
    assert result["status"] == "review_required"
    assert result["candidate_count"] == 3
    assert all(row["reason"] == "contour collision" for row in result["attempts"])


def test_deadline_discards_partial_valid_group_instead_of_publishing(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(search.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(search, "prepare_source_white_comfort", lambda *_:
                        {"source_sha256": "s", "ideal_px": 9})
    monkeypatch.setattr(search, "_visual_core_height", lambda *_: 30)
    monkeypatch.setattr(search, "wrap_lines", lambda *_: ["TEXTO COMPLETO"])
    monkeypatch.setattr(search, "line_advance", lambda *_: 33)
    monkeypatch.setattr(search, "rasterize", lambda *_:
                        np.zeros((10, 20), np.uint8))

    def render(*_args, **_kwargs):
        now[0] = 2.0
        return object(), dict(
            visual_comfort=dict(minimum_actual_px=12), center_error_px=[0, 0])

    monkeypatch.setattr(search, "render_reviewed_text", render)
    result = search.search_source_centered_layout(
        np.zeros((1, 1, 3), np.uint8), _recipe(),
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=30, maximum_font_size=30,
        width_ratios=(1.0,), maximum_candidates=2, maximum_seconds=1)

    assert result["status"] == "review_required"
    assert result["reason"] == "search_time_budget_exhausted"
    assert result["accepted_candidate_count"] == 1
    assert result["discarded_due_to_deadline"] is True
    assert "recipe" not in result and "layer" not in result


def test_preferred_size_candidate_that_reaches_comfort_finishes_before_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(search.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(search, "prepare_source_white_comfort", lambda *_:
                        {"source_sha256": "s", "ideal_px": 9})
    monkeypatch.setattr(search, "_visual_core_height", lambda *_: 30)
    monkeypatch.setattr(search, "wrap_lines", lambda *_: ["TEXTO COMPLETO"])
    monkeypatch.setattr(search, "line_advance", lambda *_: 33)
    monkeypatch.setattr(search, "rasterize", lambda *_:
                        np.zeros((10, 20), np.uint8))

    def render(*_args, **_kwargs):
        now[0] = 2.0
        return object(), dict(
            visual_comfort=dict(minimum_actual_px=16), center_error_px=[0, 0])

    monkeypatch.setattr(search, "render_reviewed_text", render)
    result = search.search_source_centered_layout(
        np.zeros((1, 1, 3), np.uint8), _recipe(),
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=30, maximum_font_size=30,
        width_ratios=(1.0, .75), maximum_candidates=2, maximum_seconds=1)

    assert result["status"] == "rendered_candidate"
    assert result["candidate_count"] == 1
    assert result["recipe"]["layout_search"]["early_completion"] == \
        "preferred_size_comfort_target_reached"


def test_budget_reaches_minimum_scale_before_no_solution(monkeypatch):
    monkeypatch.setattr(search, "prepare_source_white_comfort", lambda *_: {"source_sha256": "s", "ideal_px": 9})
    monkeypatch.setattr(search, "_visual_core_height", lambda _font, size, _sample: size)
    monkeypatch.setattr(search, "wrap_lines", lambda *_: ["TEXTO"])
    monkeypatch.setattr(search, "line_advance", lambda *_: 33)
    monkeypatch.setattr(search, "render_reviewed_text", lambda *_a, **_k:
                        (_ for _ in ()).throw(ValueError("too large")))
    result = search.search_source_centered_layout(np.zeros((1, 1, 3), np.uint8), _recipe(),
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=20, maximum_font_size=30,
        width_ratios=(1.0, .75), maximum_candidates=6)
    assert result["status"] == "review_required"
    assert 24 in {row["font_size"] for row in result["attempts"]}
    assert 20 not in {row["font_size"] for row in result["attempts"]}


def test_comfort_beats_largest_safe_size_but_stops_before_tiny_text(monkeypatch):
    monkeypatch.setattr(search, "prepare_source_white_comfort", lambda *_:
                        {"source_sha256": "s", "ideal_px": 9})
    monkeypatch.setattr(search, "_visual_core_height", lambda _f, size, _s: size)
    monkeypatch.setattr(search, "wrap_lines", lambda *_: ["TEXT"])
    monkeypatch.setattr(search, "line_advance", lambda _f, size, _p: size + 3)
    monkeypatch.setattr(search, "rasterize", lambda _f, _s, _t: np.zeros((10, 20), np.uint8))

    def render(_clean, recipe, *, prepared_comfort):
        distance = {30: 5, 29: 11, 28: 15, 27: 18, 26: 22, 25: 25, 24: 28}[recipe["font_size"]]
        return object(), dict(visual_comfort=dict(ideal_px=9, minimum_actual_px=distance),
                              center_error_px=[0, 0])

    monkeypatch.setattr(search, "render_reviewed_text", render)
    result = search.search_source_centered_layout(np.zeros((1, 1, 3), np.uint8), _recipe(),
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=18, maximum_font_size=30,
        width_ratios=(1.0,), maximum_candidates=30)
    assert result["recipe"]["font_size"] == 28
    assert result["recipe"]["layout_search"]["comfort_target_reached"]
    assert result["recipe"]["layout_search"]["readable_minimum_font_size"] == 24


def test_long_word_after_a_short_word_is_rejected_before_canvas_write(monkeypatch):
    monkeypatch.setattr("consumer_focal_text.rasterize", lambda _font, _size, value:
                        np.zeros((10, len(value) * 10), np.uint8))
    with pytest.raises(ValueError, match="word exceeds local safe width"):
        wrap_lines("A LONGUNBROKENWORD", Path("test.ttf"), 20, 50)


def test_source_visual_height_uses_ink_not_ocr_box_height():
    source = np.full((50, 100, 3), 255, np.uint8)
    source[16:24, 30:70] = 0
    distance = np.full((50, 100), 20.0, np.float32)
    height, evidence = search.source_line_ink_height(
        source, dict(source_roi_bbox=[0, 0, 100, 50], distance=distance),
        [[20, 10, 80, 30]])
    assert height == 8
    assert evidence["line_heights_px"] == [8]
    assert evidence["confidence"] == "source_dark_ink_contour_gated"


def test_alternative_break_preserves_words_and_can_beat_greedy(monkeypatch):
    monkeypatch.setattr(search, "prepare_source_white_comfort", lambda *_:
                        {"source_sha256": "s", "ideal_px": 9})
    monkeypatch.setattr(search, "_visual_core_height", lambda *_: 30)
    monkeypatch.setattr(search, "rasterize", lambda _f, _s, value:
                        np.zeros((10, len(value) * 8), np.uint8))
    monkeypatch.setattr(search, "wrap_lines", lambda *_: ["UM DOIS TRES", "QUATRO CINCO"])
    monkeypatch.setattr(search, "line_advance", lambda *_: 33)
    monkeypatch.setattr(search, "contour_word_breaks", lambda *_a, **_k:
                        [["UM DOIS", "TRES QUATRO CINCO"]])

    def render(_clean, recipe, *, prepared_comfort):
        if "line_plan" not in recipe:
            raise ValueError("greedy contour collision")
        assert " ".join(recipe["line_plan"]["lines"]) == recipe["target"]
        return object(), dict(visual_comfort=dict(minimum_actual_px=12),
                              center_error_px=[0, 0])

    monkeypatch.setattr(search, "render_reviewed_text", render)
    recipe = dict(_recipe(), target="UM DOIS TRES QUATRO CINCO",
                  anchor_bbox=[30, 30, 80, 70])
    result = search.search_source_centered_layout(np.zeros((1, 1, 3), np.uint8), recipe,
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=30, maximum_font_size=30,
        width_ratios=(1.0,), maximum_candidates=5, alternative_breaks=True)
    assert result["status"] == "rendered_candidate"
    assert result["recipe"]["line_plan"]["schema"] == "explicit_word_breaks_v1"
    assert result["candidate_count"] <= 5
    without_cache = search.search_source_centered_layout(
        np.zeros((1, 1, 3), np.uint8), recipe,
        source_visual_height_px=30, preferred_font_size=30,
        minimum_font_size=30, maximum_font_size=30,
        width_ratios=(1.0,), maximum_candidates=5,
        alternative_breaks=True, use_metric_cache=False)
    assert result["recipe"] == without_cache["recipe"]
    assert result["attempts"] == without_cache["attempts"]
    assert result["metric_cache"]["enabled"] is True
    assert without_cache["metric_cache"]["enabled"] is False


def test_metric_cache_is_bounded_and_preserves_raster_arrays(monkeypatch):
    monkeypatch.setattr(search, "rasterize", lambda _font, size, value:
                        np.full((size, len(value) * 4), len(value), np.uint8))
    cached = search.LocalRasterMetricCache(Path("test.ttf"), max_entries=3, max_bytes=5000)
    plain = search.LocalRasterMetricCache(Path("test.ttf"), enabled=False)
    for _ in range(2):
        a = search.alternative_word_breaks("UM GRANDE PROBLEMA CERTO",
                                           Path("test.ttf"), 20, 120,
                                           limit=4, metric_cache=cached)
        b = search.alternative_word_breaks("UM GRANDE PROBLEMA CERTO",
                                           Path("test.ttf"), 20, 120,
                                           limit=4, metric_cache=plain)
        assert a == b
    assert np.array_equal(cached.mask(20, "GRANDE"), plain.mask(20, "GRANDE"))
    assert cached.hits > 0
    assert cached.calls < plain.calls
    assert cached.evidence()["entries"] <= 3
    assert cached.evidence()["resident_bytes"] <= 5000


def test_explicit_plan_rejects_missing_reordered_and_overwide_words(monkeypatch):
    from consumer_focal_text import _ink
    monkeypatch.setattr("consumer_focal_text.rasterize", lambda _f, _s, value:
                        np.ones((10, len(value) * 8), np.uint8))
    with pytest.raises(ValueError, match="changes target words or order"):
        _ink("A B C", Path("test.ttf"), 20, 100, 22,
             {"schema": "explicit_word_breaks_v1", "lines": ["B A", "C"]})
    with pytest.raises(ValueError, match="exceeds local safe width"):
        _ink("A B C", Path("test.ttf"), 20, 20, 22,
             {"schema": "explicit_word_breaks_v1", "lines": ["A B", "C"]})


def test_alternative_plans_do_not_isolate_function_words(monkeypatch):
    monkeypatch.setattr(search, "rasterize", lambda _f, _s, value:
                        np.zeros((10, len(value) * 8), np.uint8))
    plans = search.alternative_word_breaks("UM GRANDE PROBLEMA CERTO", Path("test.ttf"),
                                           20, 120, limit=8)
    assert plans
    assert all("UM" not in lines for lines in plans)
    assert all(" ".join(lines) == "UM GRANDE PROBLEMA CERTO" for lines in plans)
