import copy
import json
from importlib.util import find_spec
from pathlib import Path

import pytest

from debug_tools import style_runtime_probe


NO_GO_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "style_copy_v2_remediation"
    / "no_go_summary.json"
)


def test_style_runtime_probe_module_exists():
    assert find_spec("debug_tools.style_runtime_probe") is not None


def test_style_runtime_probe_exposes_read_only_probe_api():
    assert callable(getattr(style_runtime_probe, "probe_style_runtime", None))


def test_style_runtime_probe_exposes_cli_api():
    assert callable(getattr(style_runtime_probe, "main", None))


def _probe_style_runtime(*args, **kwargs):
    probe = getattr(style_runtime_probe, "probe_style_runtime", None)
    assert callable(probe)
    return probe(*args, **kwargs)


def _run_probe_cli(args):
    main = getattr(style_runtime_probe, "main", None)
    assert callable(main)
    return main(args)


def _project_with_one_rendered_owner(
    *,
    categories: tuple[str, ...] = (),
    profile_status: str = "fallback",
    applied_attributes: dict | None = None,
    raster_contract: dict | None = None,
) -> dict:
    owner_id = "owner_p001_fixture"
    decision = {
        "status": profile_status,
        "applied_attributes": dict(applied_attributes or {}),
        "abstained_attributes": {
            "font_name": "candidate_confidence_missing_or_low",
        },
    }
    layer = {
        "owner_id": owner_id,
        "render_completed": True,
        "visual_profile_v2": {
            "owner_id": owner_id,
            "status": profile_status,
            "style_application_decision_v2": decision,
        },
    }
    if raster_contract is not None:
        layer["style_v2_raster_contract"] = raster_contract
    return {
        "page_owner_graphs": [
            {
                "page_id": "page_001",
                "owners": [
                    {
                        "owner_id": owner_id,
                        "page_id": "page_001",
                        "state": "rendered",
                        "route_action": "translate_inpaint_render",
                        "style_categories": list(categories),
                    }
                ],
            }
        ],
        "paginas": [
            {
                "numero": 1,
                "page_id": "page_001",
                "text_layers": [layer],
            }
        ],
    }


def test_probe_blocks_rendered_owner_without_applied_style_or_raster_contract():
    project = _project_with_one_rendered_owner(
        profile_status="fallback",
        applied_attributes={},
        raster_contract=None,
    )

    report = _probe_style_runtime(project)

    assert report["rendered_owner_count"] == 1
    assert report["raster_contract_coverage"] == 0.0
    assert report["status"] == "BLOCK"
    assert "missing_raster_contract" in report["contracts"]


def test_probe_does_not_claim_category_coverage_without_matching_owner():
    report = _probe_style_runtime(
        _project_with_one_rendered_owner(categories=("speech",)),
        required_categories={"sfx": 1},
    )

    assert report["categories"]["sfx"]["owner_count"] == 0
    assert report["status"] == "BLOCK"


def test_probe_counts_applied_attributes_and_preserves_input():
    project = _project_with_one_rendered_owner(
        categories=("speech",),
        profile_status="applied",
        applied_attributes={"fill": "#ffffff"},
        raster_contract={
            "owner_id": "owner_p001_fixture",
            "applied_attributes": {"fill": "#ffffff"},
        },
    )
    before = copy.deepcopy(project)

    report = _probe_style_runtime(
        project,
        required_categories={"speech": 1},
    )

    assert report["status"] == "PASS"
    assert report["profile_status_counts"] == {"applied": 1}
    assert report["applied_attribute_count"] == 1
    assert report["abstained_attribute_count"] == 1
    assert report["raster_contract_coverage"] == 1.0
    assert project == before


def test_probe_blocks_raster_bound_to_different_owner():
    report = _probe_style_runtime(
        _project_with_one_rendered_owner(
            raster_contract={"owner_id": "owner_other"},
        )
    )

    assert report["status"] == "BLOCK"
    assert "owner_identity_mismatch" in report["contracts"]


def test_probe_cli_aggregates_projects_and_exits_two_for_block(tmp_path):
    first = tmp_path / "entry_first" / "project.json"
    second = tmp_path / "entry_second" / "project.json"
    output = tmp_path / "report.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(
        json.dumps(_project_with_one_rendered_owner()),
        encoding="utf-8",
    )
    second_project = _project_with_one_rendered_owner()
    second_project["page_owner_graphs"][0]["owners"][0]["owner_id"] = (
        "owner_p001_second"
    )
    second_project["paginas"][0]["text_layers"][0]["owner_id"] = (
        "owner_p001_second"
    )
    second_project["paginas"][0]["text_layers"][0]["visual_profile_v2"][
        "owner_id"
    ] = "owner_p001_second"
    second.write_text(json.dumps(second_project), encoding="utf-8")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({
        "schema_version": 3,
        "entries": [
            {"entry_id": "first", "work_dir": "entry_first", "targets": [
                {"page_id": "page_001", "owner_id": "owner_p001_fixture"}
            ]},
            {"entry_id": "second", "work_dir": "entry_second", "targets": [
                {"page_id": "page_001", "owner_id": "owner_p001_second"}
            ]},
        ],
    }), encoding="utf-8")

    exit_code = _run_probe_cli(
        ["--output", str(output), "--matrix", str(matrix), str(first), str(second)]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert len(report["projects"]) == 2
    assert report["rendered_owner_count"] == 2
    assert report["raster_contract_count"] == 0


def test_no_go_summary_fixture_is_portable_and_pins_observed_counts():
    assert NO_GO_FIXTURE.is_file()
    payload = json.loads(NO_GO_FIXTURE.read_text(encoding="utf-8"))

    assert payload["expected"]["rendered_owner_count"] == 43
    assert payload["expected"]["fallback_owner_count"] == 43
    assert payload["expected"]["applied_attribute_count"] == 0
    assert payload["expected"]["abstained_attribute_count"] == 559
    assert payload["expected"]["raster_contract_count"] == 0
    assert all(":" not in row["project_id"] for row in payload["projects"])


def _matrix_project_fixture(tmp_path: Path):
    entries = []
    projects = {}
    expected = set()
    for index in range(9):
        entry_id = f"entry_{index}"
        page_id = "page_001"
        owner_id = f"owner_{index}"
        project = _project_with_one_rendered_owner(
            raster_contract={"owner_id": owner_id, "materialization_plan": {},
                             "materialization_observation": {}, "delivery_contract": {},
                             "render_geometry": {}},
        )
        project["page_owner_graphs"][0]["owners"][0]["owner_id"] = owner_id
        project["paginas"][0]["text_layers"][0]["owner_id"] = owner_id
        project["paginas"][0]["text_layers"][0]["visual_profile_v2"]["owner_id"] = owner_id
        project["page_owner_graphs"][0]["owners"].append({
            "owner_id": f"extra_{index}", "page_id": page_id, "state": "rendered",
            "route_action": "translate_inpaint_render",
        })
        entries.append({
            "entry_id": entry_id, "work_dir": entry_id,
            "targets": [{"page_id": page_id, "owner_id": owner_id}],
        })
        projects[entry_id] = project
        expected.add(f"{entry_id}:{page_id}:{owner_id}")
    return {"schema_version": 3, "entries": entries}, projects, expected


def test_runtime_probe_uses_exact_matrix_target_keys_not_all_renderable_owners(tmp_path):
    matrix, projects, expected = _matrix_project_fixture(tmp_path)

    report = style_runtime_probe.probe_projects(projects, matrix=matrix)

    assert report["target_count"] == 9
    assert set(report["target_keys"]) == expected
    assert report["rendered_owner_count"] == 9


def test_runtime_probe_requires_matrix_hash_bound_by_acceptance_bundle(tmp_path):
    matrix, projects, _expected = _matrix_project_fixture(tmp_path)
    matrix_sha256 = style_runtime_probe._canonical_sha256(matrix)
    bundle = {
        "matrix_sha256": matrix_sha256, "acceptance_bundle_id": "b" * 64,
        "revision_sha256": "r" * 64, "source_manifest_sha256": "s" * 64,
    }

    report = style_runtime_probe.probe_projects(projects, matrix=matrix, acceptance_bundle=bundle)

    assert report["matrix_sha256"] == matrix_sha256
    assert report["acceptance_bundle_id"] == "b" * 64
    with pytest.raises(style_runtime_probe.RuntimeProbeError, match="matrix.*bundle"):
        style_runtime_probe.probe_projects(projects, matrix={**matrix, "schema_version": 4}, acceptance_bundle=bundle)


@pytest.mark.parametrize("problem", ["target_missing", "target_duplicate", "entry_project_missing"])
def test_runtime_probe_blocks_incomplete_or_ambiguous_matrix_resolution(tmp_path, problem):
    matrix, projects, _expected = _matrix_project_fixture(tmp_path)
    if problem == "target_missing":
        projects["entry_0"]["page_owner_graphs"][0]["owners"][0]["owner_id"] = "missing"
    elif problem == "target_duplicate":
        matrix["entries"][0]["targets"].append(copy.deepcopy(matrix["entries"][0]["targets"][0]))
    else:
        del projects["entry_0"]

    with pytest.raises(style_runtime_probe.RuntimeProbeError):
        style_runtime_probe.probe_projects(projects, matrix=matrix)
