from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
RUNTIME_ROOT = PIPELINE_ROOT / "consumer_fast" / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_analysis_record_uses_hash_bound_vision_page_journal_and_completes(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _write_analysis_record

    source = tmp_path / "source.webp"
    source.write_bytes(b"source-page")
    journal = {
        "run_id": "owner-run-test",
        "execution_id": "owner-execution-test",
        "page_id": "page_001",
        "page_source_sha256": _sha(b"source-pixels"),
        "coverage_sha256": _sha(b"coverage"),
        "owner_graph_sha256": _sha(b"owner-graph"),
        "coverage": {"observations": [{"observation_id": "ocr-1", "text": "hello"}]},
    }
    selected_owner = SimpleNamespace(selected_observation_ids=("ocr-1",))
    page_result = SimpleNamespace(
        page_id=journal["page_id"],
        request=SimpleNamespace(run_id=journal["run_id"], execution_id=journal["execution_id"]),
        to_canonical_dict=lambda: journal,
        coverage=SimpleNamespace(
            sha256=journal["coverage_sha256"],
            canonical_json_bytes=json.dumps(journal["coverage"], sort_keys=True).encode(),
            observations=(SimpleNamespace(
                observation_id="ocr-1", text="hello", bbox_page=(1, 2, 20, 12),
                page_id="page_001", provider="fixture-ocr", provider_family="vision-v6",
                run_id="owner-run-test", origin_execution_id="owner-execution-test",
                attempt_id="attempt-1", invocation_id="invocation-1",
                page_source_sha256=journal["page_source_sha256"], payload_sha256=_sha(b"ocr-payload"),
                rejection_reason=None,
            ),),
            components=(),
            ledger=SimpleNamespace(sha256=_sha(b"ledger")),
        ),
        owner_graph=SimpleNamespace(read=lambda: SimpleNamespace(owners=(selected_owner,))),
        lifecycle=SimpleNamespace(sha256=_sha(b"ledger")),
    )
    page = SimpleNamespace(
        original_image=SimpleNamespace(shape=(32, 48, 3)),
        text_layers={},
        owner_page_result=page_result,
    )

    result = _write_analysis_record(
        tmp_path,
        source,
        {"idioma_origem": "en", "engine_preset_id": "max"},
        [page],
    )

    record = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    evidence_path = tmp_path / "consumer_fast" / "vision" / "page_execution_evidence.json"
    assert record["status"] == "complete"
    assert record["publishable"] is False
    assert evidence_path.is_file()
    assert _sha(evidence_path.read_bytes()) == record["dependency_hashes"]["vision_page_evidence"]
    assert record["page_id"] == journal["page_id"]
    assert record["coverage_ledger_sha256"] == _sha(b"ledger")
    assert record["selected_observation_id"] == "ocr-1"
    structural = json.loads((tmp_path / "consumer_fast" / "vision" / "structural_analysis.json").read_text())
    assert structural["logical_units"][0]["text"] == "hello"


def test_renderer_recipe_is_persisted_only_from_real_patch_recipe_evidence(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _write_renderer_recipes
    from typesetter.recipe_persistence import load_renderer_recipe

    target = "Olá mundo"
    font_path = tmp_path / "fonts" / "font.ttf"
    font_path.parent.mkdir()
    font_path.write_bytes(b"font-bytes")
    raster_path = tmp_path / "page.png"
    raster_path.write_bytes(b"raster-bytes")
    analysis_sha = _sha(b"analysis-record")
    ledger_sha = _sha(b"coverage-ledger")
    patch = SimpleNamespace(
        owner_id="owner-1",
        page_id="page_001",
        renderer_recipe_evidence={
            "target_text": target,
            "rendered_lines": ["Olá", "mundo"],
            "font_family": "Fixture Font",
            "font_path": str(font_path),
            "font_sha256": _sha(b"font-bytes"),
            "font_size_px": 18,
            "line_advance_px": 21,
            "render_bbox": [1, 2, 30, 24],
            "effects": {"force_upper": False},
            "anchors": {"mode": "source_center"},
            "geometry": {"coordinate_space": "logical_page"},
            "policy_versions": {"layout": "owner-render-layout-v1"},
            "rasterizer_runtime_id": "typesetter.renderer",
            "rasterizer_runtime_sha256": _sha(b"renderer"),
            "rasterizer_config": {"profile": "owner"},
            "rasterizer_config_sha256": _sha(b"renderer-config"),
            "dependency_hashes": {"analysis_record": analysis_sha, "coverage_ledger": ledger_sha},
            "source_pixel_sha256": _sha(b"source-pixels"),
        },
    )
    page = SimpleNamespace(
        path=raster_path,
        owner_page_result=SimpleNamespace(
            request=SimpleNamespace(run_id="run-1", execution_id="execution-1"),
                coverage=SimpleNamespace(ledger=SimpleNamespace(sha256=ledger_sha)),
            page_id="page_001",
            page_commits=(SimpleNamespace(glyph_patch=patch),),
        ),
    )

    result = _write_renderer_recipes(
        tmp_path,
        [page],
        analysis_record_sha256=analysis_sha,
        project_id="project-1",
        project_revision=1,
        ledger_sha256=ledger_sha,
    )

    assert result["recipe_count"] == 1
    assert result["project_id"] == "project-1"
    assert result["project_revision"] == 1
    recipe = load_renderer_recipe(tmp_path / result["recipes"][0]["path"])
    assert recipe.owner_id == "owner-1"
    assert recipe.dependency_hashes["analysis_record"] == analysis_sha
    assert recipe.dependency_hashes["coverage_ledger"] == ledger_sha
    assert recipe.line_plan.reconstruct() == target


def test_renderer_recipe_rejects_missing_exact_line_or_font_identity(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _write_renderer_recipes

    raster_path = tmp_path / "page.png"
    raster_path.write_bytes(b"raster")
    page = SimpleNamespace(
        path=raster_path,
        owner_page_result=SimpleNamespace(
            coverage=SimpleNamespace(ledger=SimpleNamespace(sha256=_sha(b"ledger"))),
            page_id="page_001",
            page_commits=(SimpleNamespace(glyph_patch=SimpleNamespace(
                owner_id="owner-1", page_id="page_001", renderer_recipe_evidence={"target_text": "Olá"},
            )),),
        ),
    )

    with pytest.raises(ValueError, match="rendered_lines|font_path|font_family"):
        _write_renderer_recipes(
            tmp_path,
            [page],
            analysis_record_sha256=_sha(b"analysis"),
            project_id="project-1",
            project_revision=1,
            ledger_sha256=_sha(b"ledger"),
        )


def test_renderer_recipe_blocker_names_page_owner_and_no_render_commit(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _write_renderer_recipes

    raster_path = tmp_path / "page.png"
    raster_path.write_bytes(b"unchanged-raster")
    owner = SimpleNamespace(
        owner_id="owner-neutral", state="target_ready", route_action="translate_inpaint_render",
        source_payload="BURNED", translated_payload="BURNED",
    )
    binding = SimpleNamespace(owner_id="owner-neutral", source_payload="BURNED", target_payload="BURNED")
    page = SimpleNamespace(
        path=raster_path,
        owner_page_result=SimpleNamespace(
            coverage=SimpleNamespace(ledger=SimpleNamespace(sha256=_sha(b"ledger"))),
            page_id="page_001",
            page_commits=(),
            owner_graph=SimpleNamespace(read=lambda: SimpleNamespace(owners=(owner,))),
            translations=(binding,),
        ),
    )

    with pytest.raises(ValueError, match="page_001.*page_commits=0.*owner-neutral.*unchanged"):
        _write_renderer_recipes(
            tmp_path,
            [page],
            analysis_record_sha256=_sha(b"analysis"),
            project_id="project-1",
            project_revision=1,
            ledger_sha256=_sha(b"ledger"),
        )


def test_verified_source_preserved_page_has_no_renderer_recipe(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _write_renderer_recipes

    raster_path = tmp_path / "page.png"
    raster_path.write_bytes(b"unchanged-raster")
    source_pixel_sha = _sha(b"unchanged-pixels")
    ledger_sha = _sha(b"ledger")
    page = SimpleNamespace(
        path=raster_path,
        owner_page_result=SimpleNamespace(
            status="final_verified",
            terminal_proof=SimpleNamespace(proof_sha256=_sha(b"proof")),
            request=SimpleNamespace(original_page=SimpleNamespace(page_source_sha256=source_pixel_sha)),
            final_page=SimpleNamespace(page_output_pixel_sha256=source_pixel_sha),
            coverage=SimpleNamespace(ledger=SimpleNamespace(sha256=ledger_sha)),
            page_id="page_001",
            page_commits=(),
        ),
    )

    result = _write_renderer_recipes(
        tmp_path, [page], analysis_record_sha256=_sha(b"analysis"),
        project_id="project-1", project_revision=1, ledger_sha256=ledger_sha,
    )

    assert result["status"] == "not_applicable_source_preserved"
    assert result["recipe_count"] == 0
    assert not list((tmp_path / "consumer_fast").glob("renderer_recipe_lineage.json"))


def test_changed_final_page_without_recipe_still_blocks(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _write_renderer_recipes

    raster_path = tmp_path / "page.png"
    raster_path.write_bytes(b"changed-raster")
    ledger_sha = _sha(b"ledger")
    page = SimpleNamespace(
        path=raster_path,
        owner_page_result=SimpleNamespace(
            status="final_verified",
            terminal_proof=SimpleNamespace(proof_sha256=_sha(b"proof")),
            request=SimpleNamespace(original_page=SimpleNamespace(page_source_sha256=_sha(b"original"))),
            final_page=SimpleNamespace(page_output_pixel_sha256=_sha(b"changed")),
            coverage=SimpleNamespace(ledger=SimpleNamespace(sha256=ledger_sha)),
            page_id="page_001",
            page_commits=(),
        ),
    )

    with pytest.raises(ValueError, match="no committed OwnerGlyphPatch"):
        _write_renderer_recipes(
            tmp_path, [page], analysis_record_sha256=_sha(b"analysis"),
            project_id="project-1", project_revision=1, ledger_sha256=ledger_sha,
        )
