from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from typesetter import fixed_font_family as fixed
from typesetter import renderer
from qa.style_fidelity import _attribute_results
from typesetter.style_materialization import build_resolved_style_intent

def test_real_approved_file_and_portuguese_glyph_coverage():
    decision = fixed.fixed_font_decision("AÇÃO, coração, ótimo — 123!")
    assert decision["family"] == fixed.FIXED_FONT_NAME
    assert decision["identity"]["file_sha256"] == fixed.FIXED_FONT_SHA256
    assert decision["automatic_family_selection"] is False
    assert fixed.validate_fixed_font_decision("AÇÃO, coração, ótimo — 123!",decision) == decision

def test_global_default_family_changes_without_base_size_change():
    import main
    from typesetter.style_policy import CANONICAL_AUTO_FONT
    from typesetter.backend_contract import DEFAULT_FONT_FAMILY
    assert main._default_text_style()["fonte"] == fixed.FIXED_FONT_NAME
    assert main._default_text_style()["tamanho"] == 28
    assert CANONICAL_AUTO_FONT == DEFAULT_FONT_FAMILY == renderer.CANONICAL_FONT_FILE == fixed.FIXED_FONT_NAME

def test_renderer_boundary_changes_only_family_and_provenance(monkeypatch):
    style = {"fonte":"ComicNeue-Bold.ttf","tamanho":28,"tracking":1.2,"rotation_deg":-6,
             "cor":"#112233","contorno_px":2,"sombra":True,"alignment":"right"}
    layer={"translated":"Amanhã, nós voltamos!","estilo":deepcopy(style),"style":deepcopy(style),
           "bbox":[12,24,190,87],"font_size_final":28,"owner_id":"owner-independent"}
    before=deepcopy(layer)
    monkeypatch.setattr(renderer,"_apply_auto_style_policy_pre_font",lambda *_:None)
    monkeypatch.setattr(renderer,"_quality_closed_font_map",lambda:pytest.fail("Automatic family matcher used"))
    renderer._apply_auto_style_policy_if_needed(None,layer)
    for key in ("estilo","style"):
        assert layer[key] == {**style,"fonte":fixed.FIXED_FONT_NAME}
    for key in ("bbox","font_size_final","owner_id","translated"):
        assert layer[key] == before[key]
    assert layer["font_family_policy_v1"]["reason"] == fixed.POLICY_REASON
    assert Path(renderer.find_font(fixed.FIXED_FONT_NAME)) == fixed.fixed_font_path()

def test_missing_asset_and_glyph_are_errors_without_comic_fallback(tmp_path):
    with pytest.raises(ValueError,match="FixedFontUnavailable"):
        fixed.fixed_font_decision("bom",fonts_root=tmp_path)
    with pytest.raises(ValueError,match="FixedFontMissingGlyph"):
        fixed.fixed_font_decision("𠀀")
    assert renderer._find_fallback_font_path("𠀀",str(fixed.fixed_font_path())) is None

def test_policy_cannot_be_rebound_to_other_text_or_family():
    decision=fixed.fixed_font_decision("Bom dia")
    with pytest.raises(ValueError,match="BindingMismatch"):
        fixed.validate_fixed_font_decision("Boa noite",decision)
    decision["family"]="ComicNeue-Bold.ttf"
    with pytest.raises(ValueError,match="BindingMismatch"):
        fixed.validate_fixed_font_decision("Bom dia",decision)

def test_policy_provenance_survives_renderer_projection_aliases_and_actual_save(tmp_path):
    import json
    import main
    raw={"id":"font-provenance","translated":"Novo texto","original":"New text","text":"New text",
         "bbox":[10,20,180,70],"style":{"tamanho":28,"cor":"#123456"}}
    fixed.apply_fixed_font_family(raw)
    expected=deepcopy(raw["font_family_policy_v1"])
    layer=main._normalize_text_layer_for_renderer(raw,1,0)
    page={"numero":1,"text_layers":[layer]}
    main._sync_page_legacy_aliases(page)
    assert page["text_layers"][0]["font_family_policy_v1"]==expected
    assert page["textos"][0]["font_family_policy_v1"]==expected
    project={"paginas":[page],"qa":{}}
    main._save_project_json(tmp_path/"project.json",project)
    saved=json.loads((tmp_path/"project.json").read_text(encoding="utf-8"))
    for key in ("text_layers","textos"):
        assert saved["paginas"][0][key][0]["font_family_policy_v1"]==expected

def test_fixed_family_check_preserves_fill_and_requires_actual_file_identity():
    identity=fixed.fixed_font_decision("Teste")["identity"]
    profile={"style_evidence_v2":{"attributes":{"font_name":{"confidence":.99},"fill":{"confidence":.99}}},
             "style_application_decision_v2":{"applied_attributes":{"font_name":{"file_sha256":"a"*64},"fill":"#FFFFFF"}}}
    contract={"applied_attributes":{"font_name":identity,"fill":"#000000"}}
    results,bad=_attribute_results(profile,contract,fixed_font_identity=identity)
    assert results["font_name"]["status"]=="applied"
    assert results["fill"]["status"]=="mismatch" and bad==["fill"]
    contract["applied_attributes"]["font_name"]={"file_sha256":"b"*64}
    results,bad=_attribute_results(profile,contract,fixed_font_identity=identity)
    assert results["font_name"]["status"]=="mismatch" and "font_name" in bad

@pytest.mark.parametrize("abstained",[False,True])
def test_owner_materialization_seals_family_adjustment_preserving_other_intent(monkeypatch,abstained):
    profile_sha="a"*64
    approved={"fill":"#123456","font_size_px":28}
    if not abstained:
        approved["font_name"]="OriginalFont.ttf"
    intent=build_resolved_style_intent(owner_id="owner-fixed",page_id="page_001",visual_profile_sha256=profile_sha,
         decision_sha256="b"*64,group_resolution_sha256="c"*64,approved=approved,
         approved_abstentions={"font_name":"unknown_source_family"} if abstained else {})
    block={"owner_id":"owner-fixed","page_id":"page_001","translated":"Teste",
           "visual_profile_sha256":profile_sha,"visual_profile_v2":{"visual_profile_sha256":profile_sha},
           "style_resolved_intent_v1":intent.to_dict(),"font_size_final":28,"render_layout_contract":{"font_size":28}}
    fixed.apply_fixed_font_family(block)
    font=SimpleNamespace(font_path=fixed.fixed_font_path(),size=28,_font_run_observation_cache={})
    monkeypatch.setattr(renderer,"validate_owner_visual_profile",lambda *a,**kw:{"visual_profile_sha256":profile_sha})
    monkeypatch.setattr(renderer,"_build_linear_rendered_glyph_run",lambda *a,**kw:None)
    renderer._seal_owner_materialization_plan(block,{"text_color":"#123456"},font,["Teste"],[(0,0)])
    rows=block["_sealed_materialization_plan_v1"]["attribute_plans"]
    assert rows["font_name"]["resolution_kind"]=="policy_adjusted"
    assert rows["font_name"]["reason"]==fixed.POLICY_REASON
    assert rows["font_name"]["target_value"]["file_sha256"]==fixed.FIXED_FONT_SHA256
    assert rows["fill"]["target_value"]=="#123456"
    assert rows["font_size_px"]["target_value"]==28
    assert intent.to_dict()["approved_attributes"]==approved
