from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _write_fixture_project(work_dir: Path, *, missing_final: bool = False) -> Path:
    originals = work_dir / "originals"
    translated = work_dir / "translated"
    originals.mkdir(parents=True)
    translated.mkdir(parents=True)
    (originals / "001.webp").write_bytes(b"original-page-pixels")
    if not missing_final:
        (translated / "001.webp").write_bytes(b"translated-page-pixels")
    project = {
        "versao": "1.0",
        "obra": "Fixture",
        "capitulo": 57,
        "paginas": [
            {
                "numero": 1,
                "arquivo_original": "originals/001.webp",
                "arquivo_traduzido": "translated/001.webp",
                "text_layers": [
                    {
                        "id": "owner-001",
                        "owner_id": "owner-001",
                        "route_action": "review_required",
                        "review_reason": "model_limit",
                    }
                ],
            }
        ],
        "completion_status": "complete_with_review",
        "output_review_state": "awaiting_review",
        "verified": False,
        "qa": {
            "export_gate": {
                "status": "BLOCK",
                "allowed": False,
                "critical_issue_count": 0,
                "critical_flag_count": 0,
            }
        },
    }
    project_path = work_dir / "project.json"
    project_path.write_text(json.dumps(project), encoding="utf-8")
    return project_path


def _config(tmp_path: Path) -> dict:
    source = tmp_path / "chapter.cbz"
    source.write_bytes(b"chapter-source")
    models = tmp_path / "models"
    models.mkdir()
    return {
        "runtime_id": "consumer-fast-v1",
        "source_path": str(source),
        "work_dir": str(tmp_path / "output"),
        "models_dir": str(models),
        "obra": "Fixture",
        "capitulo": 57,
        "idioma_origem": "en",
        "idioma_destino": "pt-BR",
        "glossario": {},
        "contexto": {},
        "owner_graph_mode": "enforce",
        "style_copy_mode": "shadow",
    }


def test_preflight_loads_hash_bound_consumer_fast_plan(tmp_path: Path) -> None:
    from consumer_fast.chapter_runner import build_preflight

    receipt = build_preflight(_config(tmp_path))

    assert receipt["status"] == "PASS"
    assert receipt["runtime_id"] == "consumer-fast-v1"
    assert receipt["execution_plan"]["schema"] == "traduzai.consumer-fast-plan.v1"
    assert receipt["execution_plan"]["stages"] == [
        "import",
        "analysis",
        "ocr",
        "logical_units",
        "translate",
        "restore",
        "layout",
        "rasterize",
        "review",
        "persist",
        "export_decision",
    ]
    modules = {row["name"]: row for row in receipt["resolved_modules"]}
    assert {
        "consumer_fast_core",
        "consumer_operational_recovery",
        "consumer_operational_publish",
        "consumer_fast_project",
        "integration_v1.contracts",
        "integration_v1.providers",
        "typesetter.renderer",
    } <= set(modules)
    assert all(row["inside_worktree"] is True for row in modules.values())
    assert receipt["base_sources"]["mismatches"] == []
    assert receipt["legacy_fallback_allowed"] is False
    assert Path(receipt["worktree"]).resolve() == PIPELINE_ROOT.parent.resolve()
    assert len(receipt["git"]["head"]) == 40
    assert receipt["contract_origins"]["vision"]["module"] == "vision_runtime.analysis_payload"
    assert receipt["contract_origins"]["renderer"]["module"] == "typesetter.renderer"
    assert receipt["experimental_absolute_imports"] == []


def test_preflight_rejects_missing_runtime_identity(tmp_path: Path) -> None:
    from consumer_fast.chapter_runner import ConsumerFastPreflightError, build_preflight

    config = _config(tmp_path)
    config.pop("runtime_id")

    with pytest.raises(ConsumerFastPreflightError, match="runtime_id"):
        build_preflight(config)


def test_finalize_project_is_round_trip_safe_and_fail_closed(tmp_path: Path) -> None:
    from consumer_fast.chapter_runner import build_preflight, finalize_project

    config = _config(tmp_path)
    work_dir = Path(config["work_dir"])
    project_path = _write_fixture_project(work_dir)
    preflight = build_preflight(config)

    receipt = finalize_project(project_path, preflight)
    reopened = json.loads(project_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "awaiting_review"
    assert receipt["round_trip"] is True
    assert receipt["project_sha256"] == hashlib.sha256(project_path.read_bytes()).hexdigest()
    assert reopened["runtime_id"] == "consumer-fast-v1"
    assert reopened["runtime_profile"] == "consumer_fast"
    assert reopened["consumer_fast"]["execution_plan"]["schema"] == "traduzai.consumer-fast-plan.v1"
    assert reopened["completion_status"] == "complete_with_review"
    assert reopened["output_review_state"] == "awaiting_review"
    assert reopened["qa"]["export_gate"]["status"] == "BLOCK"
    assert reopened["qa"]["export_gate"]["allowed"] is False
    assert reopened["blocker_count"] == 1
    assert receipt["originals"][0]["sha256"] == hashlib.sha256(
        (work_dir / "originals/001.webp").read_bytes()
    ).hexdigest()


def test_finalize_project_rejects_missing_referenced_raster(tmp_path: Path) -> None:
    from consumer_fast.chapter_runner import ConsumerFastIntegrityError, build_preflight, finalize_project

    config = _config(tmp_path)
    project_path = _write_fixture_project(Path(config["work_dir"]), missing_final=True)

    with pytest.raises(ConsumerFastIntegrityError, match="arquivo_traduzido"):
        finalize_project(project_path, build_preflight(config))


def test_runner_executes_explicit_physical_stage_then_publishes_receipt(tmp_path: Path) -> None:
    from consumer_fast.chapter_runner import run_chapter

    config = _config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls: list[Path] = []

    def physical_executor(path: Path) -> Path:
        calls.append(path)
        runtime_config = json.loads(path.read_text(encoding="utf-8"))
        assert runtime_config["consumer_fast_execution_plan"]["schema"] == "traduzai.consumer-fast-plan.v1"
        assert runtime_config["translation_provider_policy"] == "consumer-fast-bounded-owner-v1"
        assert runtime_config["legacy_translation_fallback_allowed"] is False
        return _write_fixture_project(Path(config["work_dir"]))

    events: list[dict] = []
    result = run_chapter(
        config_path,
        physical_executor=physical_executor,
        emit_event=events.append,
    )

    assert calls == [config_path]
    assert result["status"] == "awaiting_review"
    assert result["runtime_id"] == "consumer-fast-v1"
    assert Path(result["project_path"]).is_file()
    assert Path(result["receipt_path"]).is_file()
    assert events[-1]["type"] == "complete"
    assert events[-1]["runtime_id"] == "consumer-fast-v1"


def test_default_physical_executor_never_delegates_to_general_pipeline(monkeypatch) -> None:
    import inspect
    import sys
    from types import SimpleNamespace

    from consumer_fast import physical_executor
    from consumer_fast.chapter_runner import _physical_pipeline

    source = inspect.getsource(_physical_pipeline)

    assert "_run_pipeline" not in source
    assert "consumer_fast.physical_executor" in source

    expected = Path("direct-consumer-fast-project.json")
    calls: list[Path] = []
    monkeypatch.setattr(physical_executor, "execute_config", lambda path: calls.append(path) or expected)
    forbidden_calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "main",
        SimpleNamespace(_run_pipeline=lambda *_args: forbidden_calls.append("legacy")),
    )
    assert _physical_pipeline(Path("config.json")) == expected
    assert calls == [Path("config.json")]
    assert forbidden_calls == []


def test_physical_executor_honors_explicit_inpaint_skip_without_mutating_pixels() -> None:
    from consumer_fast.physical_executor import _build_inpainter

    source_pixels = [1, 2, 3]
    page: dict = {}
    result = _build_inpainter({"skip_inpaint": True}).inpaint_band_image(source_pixels, page)

    assert result == source_pixels
    assert result is not source_pixels
    assert page["_skip_inpaint_honored"] is True
    assert page["_strip_used_real_inpaint"] is False


def test_consumer_fast_provider_adapter_disables_page_fallback_without_changing_retry_policy() -> None:
    from consumer_fast import provider_adapter

    assert provider_adapter.POLICY_ID == "consumer-fast-bounded-owner-v1"
    assert not hasattr(provider_adapter, "consumer_fast_attempt_controls")
    with pytest.raises(RuntimeError, match="legacy translate_pages fallback disabled"):
        provider_adapter.translate_pages([])


def test_main_host_selects_consumer_fast_provider_only_with_bound_plan(tmp_path: Path) -> None:
    import main

    config = _config(tmp_path)
    config.update(
        consumer_fast_execution_plan={"schema": "traduzai.consumer-fast-plan.v1"},
        translation_provider_policy="consumer-fast-bounded-owner-v1",
        legacy_translation_fallback_allowed=False,
    )

    provider = main._resolve_chapter_translation_provider(config)

    assert provider.__name__ == "consumer_fast.provider_adapter"
    config["legacy_translation_fallback_allowed"] = True
    with pytest.raises(ValueError, match="binding is incomplete"):
        main._resolve_chapter_translation_provider(config)
