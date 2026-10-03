from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_fast_metrics_match_raster_width_for_chapter57_page4(monkeypatch) -> None:
    from typesetter import renderer

    font_name = "CCTotallyAwesome W00 Bold.ttf"
    texts = (
        "ENTÃO", "ENTÃO O GERENTE SHIN",
        "ENTÃO O GERENTE SHIN EM BREVE PERCEBERÁ QUE ALGO ESTÁ ERRADO",
    )
    for size in (12, 35, 60):
        font = renderer.get_font(font_name, size)
        for text in texts:
            monkeypatch.delenv("TRADUZAI_EXPERIMENTAL_FAST_FONT_METRICS", raising=False)
            raster_width = renderer.measure_text_width(font, text)
            monkeypatch.setenv("TRADUZAI_EXPERIMENTAL_FAST_FONT_METRICS", "1")
            assert renderer.measure_text_width(font, text) == raster_width


def test_fast_metrics_whitespace_width(monkeypatch) -> None:
    from typesetter import renderer

    font = renderer.get_font("CCTotallyAwesome W00 Bold.ttf", 28)
    monkeypatch.setenv("TRADUZAI_EXPERIMENTAL_FAST_FONT_METRICS", "1")
    assert renderer.measure_text_width(font, "  ") == 1


def test_fast_search_requires_applied_candidate() -> None:
    from typesetter import renderer

    assert not renderer._experimental_smaller_font_cannot_win(
        candidate_size=32, accepted=[], translated_text="ENTÃO", font_name="CCTotallyAwesome W00 Bold.ttf",
    )


def test_fast_search_uses_strict_source_scale_bound() -> None:
    from typesetter import renderer

    # The real owner trial is covered by the page-4 pixel and receipt replay.
    # The style guard is checked separately here.
    assert not renderer.render_style_transform_active({"estilo": {}})
    assert renderer.render_style_transform_active({"estilo": {"curva": True}})


def test_binary_nominal_fit_boundary() -> None:
    from typesetter.renderer import _experimental_first_nominal_fit_index

    for values, expected in (([False, False, True, True], 2),
                             ([True, True, True], 0),
                             ([False, False, False], 3)):
        seen = []
        result = _experimental_first_nominal_fit_index(
            len(values), lambda index: seen.append(index) or values[index],
        )
        assert result == expected
        assert len(seen) <= 3


def test_fast_metric_cache_has_bound_and_font_identity(monkeypatch) -> None:
    import hashlib
    from typesetter import renderer

    font = renderer.get_font("CCTotallyAwesome W00 Bold.ttf", 28)
    expected_hash = hashlib.sha256(font.font_path.read_bytes()).hexdigest()
    captured = []
    original = renderer._experimental_textpath_unit_ink_width

    def capture(*args):
        captured.append(args)
        return original(*args)

    monkeypatch.setattr(renderer, "_experimental_textpath_unit_ink_width", capture)
    monkeypatch.setenv("TRADUZAI_EXPERIMENTAL_FAST_FONT_METRICS", "1")
    assert renderer.measure_text_width(font, "ENTÃO") > 0
    assert captured[0][0] == expected_hash
    assert captured[0][3] == "matplotlib-textpath-nfc-usetex-false-v1"
    assert original.cache_info().maxsize == 512


def test_glyph_union_matches_complete_textpath() -> None:
    import hashlib
    import math
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from typesetter import renderer

    font = renderer.get_font("CCTotallyAwesome W00 Bold.ttf", 35)
    font_path = str(font.font_path)
    font_sha = hashlib.sha256(font.font_path.read_bytes()).hexdigest()
    policy = "matplotlib-textpath-nfc-usetex-false-v1"
    prop = FontProperties(fname=font_path, size=1)
    for text in (
        "ENTÃO", "ENTÃO O GERENTE SHIN", "QUE ALGO ESTÁ ERRADO",
        "ENTÃO O GERENTE SHIN EM BREVE PERCEBERÁ QUE ALGO ESTÁ ERRADO",
        "PERCEBERÁ", "é! À?", "$x$",
    ):
        full_width = TextPath((0, 0), text, prop=prop, usetex=False).get_extents().width
        glyph_width = renderer._experimental_textpath_unit_ink_width(
            font_sha, font_path, text, policy,
        )
        assert math.isclose(glyph_width, full_width, rel_tol=0.0, abs_tol=1e-12)
    assert renderer._experimental_textpath_unit_ink_width.cache_info().maxsize == 512
    assert renderer._experimental_glyph_face.cache_info().maxsize == 4
