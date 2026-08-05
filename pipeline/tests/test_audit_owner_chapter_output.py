from __future__ import annotations

from types import SimpleNamespace

from tools import audit_owner_chapter_output as auditor


def test_unowned_external_english_is_classified_without_stored_binding():
    assert auditor._looks_like_probable_source("THE ARENA WILL BEGIN", "en") is True
    assert auditor._looks_like_probable_source("FAILURE WILL RESULT IN PENALTIES", "en") is True
    assert auditor._looks_like_probable_source("A ARENA VAI COMEÇAR", "en") is False


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
