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
