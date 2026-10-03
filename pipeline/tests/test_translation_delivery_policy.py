"""General classification and explicit warning-delivery regressions."""
from __future__ import annotations
import pytest
from translator.language_policy import build_page_language_evidence, validate_target_language
from ownership.translation import OwnerTranslationRequest, translate_owner, TranslationBinding
from test_owner_translation import _graph

def _evidence(source):
    return build_page_language_evidence(texts=(source,),coverage_complete=True)

@pytest.mark.parametrize("source",["DANGER", "CABINET.", "MYSTERY!", "What's?", "What’s?", "DANGER ZONE", "Narrow Passage", "Xyzzra", "“MYSTERY!”"])
def test_capitalization_and_oov_are_not_name_evidence(source):
    verdict=validate_target_language(source=source,target=source,role="dialogue_body",page_language_evidence=_evidence(source))
    assert not verdict.accepted
    assert verdict.policy_id != "source_neutral_proper_name"

@pytest.mark.parametrize("name",["Ana Maria", "Min Jae", "O'Connor", "IC-Power"])
def test_explicit_preserved_entity_survives_punctuation(name):
    source=name+"!"
    verdict=validate_target_language(source=source,target=source,role="dialogue_body",explicit_entities=(name,),page_language_evidence=_evidence(source))
    assert verdict.accepted and verdict.policy_id=="source_neutral_proper_name"

def test_known_name_does_not_preserve_adjacent_dialogue():
    source="ANA MARIA DANGER ZONE"
    verdict=validate_target_language(source=source,target=source,role="dialogue_body",explicit_entities=("Ana Maria",),page_language_evidence=_evidence(source))
    assert not verdict.accepted

def test_common_uncertain_source_calls_provider_and_rejected_quality_is_a_warning():
    graph=_graph([("a","DANGER ZONE")])
    request=OwnerTranslationRequest.from_graph(graph,"owner_a")
    calls=[]
    def backend(req,variant):
        calls.append((req.source_text,variant))
        return "DANGER ZONE"
    binding,attempts=translate_owner(request,backends=(backend,),max_attempts_per_backend=1,
         page_language_evidence=_evidence(request.source_text),allow_quality_warnings=True)
    assert calls
    assert not binding.language_verdict.accepted
    assert binding.quality_warning_reason=="unchanged_source_dialogue"
    assert not binding.preserves_original_pixels
    assert attempts[-1].status=="accepted_with_warnings"
    assert TranslationBinding.from_dict(binding.to_dict())==binding


def test_explicit_name_boundary_does_not_preserve_larger_word():
    verdict=validate_target_language(source="Anaconda",target="Anaconda",role="dialogue_body",
                                    explicit_entities=("Ana",),page_language_evidence=_evidence("Anaconda"))
    assert not verdict.accepted


def test_approved_name_handles_typographic_apostrophe():
    source="O’Connor!"
    verdict=validate_target_language(source=source,target=source,role="dialogue_body",
                                    explicit_entities=("O'Connor",),page_language_evidence=_evidence(source))
    assert verdict.accepted and verdict.policy_id=="source_neutral_proper_name"


def test_approved_terms_exclude_unreviewed_internet_guesses_and_translated_glossary():
    from translator.translate import approved_preserved_entities
    context={"personagens":["Ana Maria", "Unknown Candidate"],
             "internet_context":{"glossary_candidates":[{"source":"Unknown Candidate","kind":"character","status":"candidate"}]}}
    assert approved_preserved_entities(context,{"IC-Power":"IC-Power", "Danger":"Perigo"}) == ("Ana Maria", "IC-Power")
    assert "Unknown Candidate" in approved_preserved_entities(context,{"Unknown Candidate":"Unknown Candidate"})


@pytest.mark.parametrize("backend", [lambda req, variant: "", lambda req, variant: (_ for _ in ()).throw(OSError("provider unavailable"))])
def test_unavailable_provider_never_creates_usable_binding(backend):
    from ownership.translation import TranslationValidationExhausted, TranslationInfrastructureError
    request=OwnerTranslationRequest.from_graph(_graph([("a","DANGER")]),"owner_a")
    with pytest.raises((TranslationValidationExhausted, TranslationInfrastructureError)) as error:
        translate_owner(request,backends=(backend,),max_attempts_per_backend=1,
                        page_language_evidence=_evidence(request.source_text),allow_quality_warnings=True)
    assert error.value.attempts
    assert not any(attempt.usable for attempt in error.value.attempts)


def test_warning_binding_target_and_policy_cannot_be_tampered():
    from ownership.translation import TranslationIdentityError
    request=OwnerTranslationRequest.from_graph(_graph([("a","DANGER")]),"owner_a")
    binding,_=translate_owner(request,backends=(lambda req, variant:"DANGER",),
                            max_attempts_per_backend=1,page_language_evidence=_evidence(request.source_text),allow_quality_warnings=True)
    for field,value in [("target_text","invented"),("quality_usage_policy_id","fake"),("quality_warning_reason","fake")]:
        data=binding.to_dict(); data[field]=value
        with pytest.raises(TranslationIdentityError):
            TranslationBinding.from_dict(data)


def test_default_page_policy_keeps_available_translation_and_unsafe_geometry_not_applied():
    from strip.page_pipeline import run_page_owner_pipeline, adapt_page_execution_result_to_output_page
    from test_page_owner_pipeline import _request
    from test_owner_lifecycle_e2e import _detached_execution_rejection_services
    from dataclasses import replace
    services=replace(_detached_execution_rejection_services(),translation_backends=(lambda req,variant:req.source_text,))
    result=run_page_owner_pipeline(_request(),services)
    assert result.translations and result.translation_attempts[-1].status=="accepted_with_warnings"
    output=adapt_page_execution_result_to_output_page(result)
    layer=output.text_layers["texts"][0]
    notice=layer["translation_delivery_notice"]
    assert notice["target_produced"]==result.translations[0].target_text
    assert notice["target_used"] is None and not notice["insertion_applied"]
    assert notice["status"]=="not_applied"
    assert layer["owner_id"] is None and layer["write_authority"]=="revoked"
    assert (output.image==output.original_image).all()


def test_mixed_delivery_preserves_quality_findings_and_integrity_blocks():
    from types import SimpleNamespace
    from qa.partial_delivery import notice_for_layer, apply_partial_delivery_policy
    request=OwnerTranslationRequest.from_graph(_graph([("a","DANGER")]),"owner_a")
    binding,attempts=translate_owner(request,backends=(lambda req,variant:"DANGER",),
             max_attempts_per_backend=1,page_language_evidence=_evidence(request.source_text),allow_quality_warnings=True)
    commit=SimpleNamespace(owner_id=binding.owner_id,translation_binding_sha256=binding.translation_binding_sha256,
                          source_payload_sha256=binding.source_payload_sha256,target_payload_sha256=binding.target_payload_sha256,commit_id="mock-verified-commit")
    applied={"owner_id":binding.owner_id,"visible":True}
    applied["translation_delivery_notice"]=notice_for_layer(applied,binding=binding,attempts=attempts,commits=(commit,))
    rejected={"candidate_owner_id":"other","owner_id":None,"visible":False,"original":"Unknown text","owner_execution_rejection_reason":"geometry_unavailable"}
    rejected["translation_delivery_notice"]=notice_for_layer(rejected)
    project={"paginas":[{"text_layers":[applied,rejected]}]}
    original={"status":"BLOCK","issues":[{"reason":"uncertain_source_component_requires_review","flags":["uncertain_source_component_requires_review"],"severity":"critical","blocks_export":True}]}
    gate=apply_partial_delivery_policy(project,original)
    assert gate["status"]=="REVIEW" and gate["allowed"] and not gate["override"]
    assert project["qa"]["original_export_gate"]==original
    report=project["qa"]["translation_delivery"]
    assert report["applied_count"]==1 and report["not_applied_count"]==1
    assert report["status"]=="partial_with_warnings" and not report["quality_approved"]
    assert report["items"][0]["target_used"]==binding.target_text
    original["issues"].append({"reason":"qa_integrity_failure","severity":"critical","blocks_export":True})
    gate=apply_partial_delivery_policy(project,original)
    assert gate["status"]=="BLOCK" and not gate["allowed"]


def test_partial_notices_survive_official_aliases_atomic_save_and_readback(tmp_path):
    from qa.partial_delivery import apply_partial_delivery_policy
    from test_owner_project_contract_roundtrip import _fixture_project, _exercise
    project=_fixture_project()
    original_gate={"status":"BLOCK","allowed":False,"issues":[{
        "reason":"uncertain_source_component_requires_review", "severity":"critical",
        "blocks_export":True, "flags":["uncertain_source_component_requires_review"]}]}
    gate=apply_partial_delivery_policy(project,original_gate)
    project["qa"]["export_gate"]=gate
    for page in project["paginas"]:
        for layer in page["text_layers"]:
            from qa.partial_delivery import notice_for_layer
            layer["translation_delivery_notice"]=notice_for_layer(layer)
    saved=_exercise(project,tmp_path/"project.json")
    assert saved["translation_delivery_status"]=="partial_with_warnings"
    assert saved["qa"]["export_gate"]["status"]=="REVIEW"
    assert saved["needs_review"]
    assert saved["qa"]["original_export_gate"]==original_gate
    assert not saved["qa"]["translation_delivery"]["quality_approved"]
    assert saved["qa"]["export_gate"]["issues"][0]["original_severity"]=="critical"
    assert saved["qa"]["export_gate"]["issues"][0]["severity"]=="warning"
    assert all(l["translation_delivery_notice"]["status"]=="not_applied" for p in saved["paginas"] for l in p["text_layers"])
    for page in saved["paginas"]:
        for layer in page["text_layers"]:
            notice=layer["translation_delivery_notice"]
            assert "source_recorded" in notice["steps_done"]
            assert notice["steps_not_done"]==["cleanup","insertion"]
            assert notice["target_used"] is None
            assert notice["pending_work"]==["safe_owner_geometry_and_insertion"]


def test_official_runtime_gate_runs_quality_checks_then_authorized_warning_delivery(tmp_path):
    import main
    project={"paginas":[{"text_layers":[{"original":"Uncertain text","visible":False}]}],"qa":{}}
    functional={"status":"BLOCK","allowed":False,"issues":[{
        "reason":"uncertain_source_component_requires_review","severity":"critical",
        "blocks_export":True,"flags":["uncertain_source_component_requires_review"]}]}
    gate=main._compose_runtime_export_gate(project,tmp_path,{"style_copy_mode":"off"},functional)
    assert gate["status"]=="REVIEW" and gate["allowed"]
    assert project["qa"]["functional_export_gate"]==functional
    assert project["qa"]["original_export_gate"]["status"]=="BLOCK"
    assert project["qa"]["translation_delivery"]["items"][0]["status"]=="not_applied"
    assert not project["qa"]["translation_delivery"]["quality_approved"]
