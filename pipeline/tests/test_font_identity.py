from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parent
FONTS_DIR = REPO_ROOT / "fonts"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _font_identity_module():
    spec = importlib.util.find_spec("typesetter.font_identity")
    assert spec is not None, "typesetter.font_identity must exist"
    return importlib.import_module("typesetter.font_identity")


def test_resolved_font_identity_is_bound_to_file_and_face_metadata():
    module = _font_identity_module()

    identity = module.resolve_font_identity(FONTS_DIR / "ComicNeue-Bold.ttf")

    assert identity.filename == "ComicNeue-Bold.ttf"
    assert len(identity.file_sha256) == 64
    assert identity.family
    assert identity.weight_class >= 700


def test_font_intent_and_runtime_path_compare_by_resolved_identity():
    module = _font_identity_module()
    catalog = module.load_font_identity_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")

    intent = module.canonicalize_font_intent("comicneue-bold.ttf", catalog)
    observed = module.observe_resolved_font(FONTS_DIR / "ComicNeue-Bold.ttf")

    assert module.compare_resolved_font(intent, observed).matches is True


def test_requested_weight_is_not_accepted_when_resolved_face_is_regular():
    module = _font_identity_module()
    catalog = module.load_font_identity_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")
    expected = module.canonicalize_font_intent(
        {"font_name": "ComicNeue-Bold.ttf", "font_weight": 700}, catalog
    )
    observed = module.observe_resolved_font(FONTS_DIR / "ComicNeue-Regular.ttf")

    result = module.compare_resolved_font(expected, observed)

    assert result.matches is False
    assert "file_sha256" in result.reasons


def test_catalog_order_is_invariant_to_font_map_order(tmp_path):
    module = _font_identity_module()
    original = json.loads((FONTS_DIR / "font-map.json").read_text(encoding="utf-8"))
    reversed_map = dict(original)
    reversed_map["available"] = list(reversed(original["available"]))
    map_path = tmp_path / "font-map.json"
    map_path.write_text(json.dumps(reversed_map), encoding="utf-8")

    first = module.load_font_identity_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")
    second = module.load_font_identity_catalog(FONTS_DIR, map_path)

    assert first.catalog_sha256 == second.catalog_sha256
    assert [item.identity.filename for item in first.entries] == [
        item.identity.filename for item in second.entries
    ]


def test_font_run_records_character_level_fallback_instead_of_claiming_primary_face(tmp_path):
    module = _font_identity_module()
    custom_map = {
        "available": [
            {
                "arquivo": "Newrotic.ttf",
                "detector": True,
                "license_status": "bundled-project-asset",
                "roles": ["dialogue_body"],
            },
            {
                "arquivo": "ComicNeue-Bold.ttf",
                "detector": True,
                "license_status": "OFL-1.1",
                "roles": ["dialogue_body"],
            },
        ]
    }
    map_path = tmp_path / "font-map.json"
    map_path.write_text(json.dumps(custom_map), encoding="utf-8")
    catalog = module.load_font_identity_catalog(FONTS_DIR, map_path)
    intent = module.canonicalize_font_intent("Newrotic.ttf", catalog)

    plan = module.resolve_font_run_plan(intent=intent, text="AÇÃO—AGORA", catalog=catalog)
    observed = module.observe_font_run(plan)

    assert plan.resolution_kind == "derived"
    assert plan.reason == "glyph_coverage_fallback"
    assert any(span.fallback_identity is not None for span in plan.spans)
    assert module.compare_font_run(plan, observed).matches is True


def test_unplanned_font_fallback_is_a_mismatch(tmp_path):
    module = _font_identity_module()
    catalog = module.load_font_identity_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")
    intent = module.canonicalize_font_intent("ComicNeue-Bold.ttf", catalog)
    plan = module.resolve_font_run_plan(intent=intent, text="AÇÃO", catalog=catalog)
    observed = module.observe_font_run(plan)
    wrong = module.replace_font_run_observation_primary(
        observed, module.observe_resolved_font(FONTS_DIR / "ComicNeue-Regular.ttf")
    )

    assert module.compare_font_run(plan, wrong).matches is False


def test_variable_font_observes_configured_axis_and_static_instance(tmp_path):
    module = _font_identity_module()
    catalog = module.load_font_identity_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")
    intent = module.canonicalize_font_intent(
        {
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
            "font_width": 85,
        },
        catalog,
    )

    narrow_plan = module.resolve_font_run_plan(
        intent=intent,
        text="MATERIAL",
        catalog=catalog,
        requested_axes={"wdth": 85.0},
        cache_root=tmp_path / "font-cache",
    )
    default_plan = module.resolve_font_run_plan(
        intent=module.canonicalize_font_intent(
            "LeagueGothic-Regular-VariableFont_wdth.ttf", catalog
        ),
        text="MATERIAL",
        catalog=catalog,
        requested_axes={"wdth": 100.0},
        cache_root=tmp_path / "font-cache",
    )

    assert narrow_plan.configured_axes == (("wdth", 85.0),)
    assert narrow_plan.instance_file_sha256 != narrow_plan.primary_identity.file_sha256
    assert narrow_plan.instance_path.is_file()
    assert module.glyph_advance(narrow_plan, "M") != module.glyph_advance(default_plan, "M")
