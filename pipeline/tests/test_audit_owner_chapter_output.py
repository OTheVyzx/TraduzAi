from __future__ import annotations

from types import SimpleNamespace

from tools import audit_owner_chapter_output as auditor


def test_unowned_external_english_is_classified_without_stored_binding():
    assert auditor._looks_like_probable_source("THE ARENA WILL BEGIN", "en") is True
    assert auditor._looks_like_probable_source("FAILURE WILL RESULT IN PENALTIES", "en") is True
    assert auditor._looks_like_probable_source("A ARENA VAI COMEÇAR", "en") is False


def test_bound_source_matching_is_accent_insensitive_for_shared_target_tokens():
    bindings = (
        SimpleNamespace(
            owner_id="owner-1",
            source_text="WELL, ACTUALLY I PLACED THE SAME BET COMO NOSSO JUNIOR",
            target_text="BEM, NA VERDADE EU FIZ A MESMA APOSTA QUE NOSSO JÚNIOR",
        ),
    )

    visible_target = "BEM, NA VERDADE EU FIZ A MESMA APOSTA QUE NOSSO JUNIOR"
    visible_source = "ACTUALLY"

    assert auditor._looks_like_bound_source(visible_target, bindings) == (False, None)
    assert auditor._looks_like_bound_source(visible_source, bindings) == (True, "owner-1")


def test_source_manifest_paths_use_canonical_relative_source_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    page = source / "001.png"
    page.write_bytes(b"source-page")
    manifest = SimpleNamespace(
        pages=(SimpleNamespace(relative_source_path="001.png"),),
    )

    assert auditor._resolve_source_manifest_paths(source, manifest) == (page.resolve(),)


def test_external_ocr_stdout_ignores_provider_logs_before_json():
    payload = auditor._parse_external_ocr_stdout(
        "[paddle] provider warning\n"
        '{"auditor_execution_id":"external-execution:1","observations":[]}\n'
    )

    assert payload["auditor_execution_id"] == "external-execution:1"


def test_external_page_audit_serializes_observation_tuples_as_json_lists():
    values = {
        field: (
            ({"text": "THE ARENA"},)
            if field in {"observations", "source_only_residuals", "unowned_source_residuals"}
            else 0
            if field in {"auditor_pid", "physical_inference_count", "cache_hits", "english_dialogue_residual_count"}
            else "PASS"
            if field in {"language_verdict", "status"}
            else f"value:{field}"
        )
        for field in auditor.ExternalPageAudit.__dataclass_fields__
        if field != "audit_sha256"
    }

    payload = auditor.ExternalPageAudit.build(**values).to_dict()

    assert payload["observations"] == [{"text": "THE ARENA"}]
    assert payload["source_only_residuals"] == [{"text": "THE ARENA"}]
    assert payload["unowned_source_residuals"] == [{"text": "THE ARENA"}]


def test_audit_cli_forwards_task20_paths_and_returns_gate_code(tmp_path, monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(external_audit_gate_status="PASS")

    monkeypatch.setattr(auditor, "audit_chapter_from_paths", fake)
    source, run = tmp_path / "source", tmp_path / "run"
    report, review = tmp_path / "audit.json", tmp_path / "review"
    assert auditor.main([
        "--source", str(source), "--run", str(run), "--report", str(report),
        "--review-dir", str(review), "--source-lang", "en", "--target-lang", "pt-BR",
        "--require-final-verified",
    ]) == 0
    assert calls == [{
        "source": source.resolve(), "run": run.resolve(), "report": report.resolve(),
        "review_dir": review.resolve(), "source_lang": "en", "target_lang": "pt-BR",
        "compare_content_run": None, "require_final_verified": True,
    }]


def test_audit_cli_returns_nonzero_on_controlled_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        auditor, "audit_chapter_from_paths", lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad publication"))
    )
    assert auditor.main([
        "--source", str(tmp_path / "source"), "--run", str(tmp_path / "run"),
        "--report", str(tmp_path / "audit.json"), "--review-dir", str(tmp_path / "review"),
        "--require-final-verified",
    ]) == 1


def test_metrics_treat_verified_no_repaint_owner_as_terminal_without_fake_commit():
    owner = SimpleNamespace(
        owner_id="owner-already-ptbr",
        disposition="owned",
        route_action="translate_inpaint_render",
        component_ids=("component-already-ptbr",),
    )
    graph = SimpleNamespace(owners=(owner,))
    result = SimpleNamespace(
        owner_graph=SimpleNamespace(read=lambda: graph),
        terminal_proof=None,
        qa_probes=(),
        language_residual_issues=(),
        coverage=SimpleNamespace(
            entries=(
                SimpleNamespace(
                    component_id="component-already-ptbr",
                    materiality="material",
                    ocr_attempt_ids=("attempt-1",),
                ),
            )
        ),
        translations=(
            SimpleNamespace(
                owner_id="owner-already-ptbr",
                target_text="COMO ESPERADO DE KIM SIHYEOK!",
                target_locale="pt-BR",
                preserves_original_pixels=True,
            ),
        ),
        page_commits=(),
        owner_target_materializations=(),
    )

    metrics = auditor.compute_page_acceptance_metrics(result)

    assert metrics.owners_without_valid_pt_br == 0
    assert metrics.owners_without_atomic_cleanup_render == 0
    assert metrics.owners_without_target_materialization == 0
    assert metrics.material_components_without_terminal_lifecycle == 0
