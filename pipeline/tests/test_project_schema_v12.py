import json
from hashlib import sha256
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.project_schema_v12 import (  # noqa: E402
    PROJECT_SCHEMA_V12,
    SCHEMA_VERSION,
    build_empty_project_v12,
    validate_project_v12,
)
from ownership.model import OWNER_GRAPH_SCHEMA_VERSION  # noqa: E402


def _verified_owner_graph_payload() -> dict:
    return {
        "schema_version": OWNER_GRAPH_SCHEMA_VERSION,
        "page_id": "page_001",
        "run_id": "run-project-schema",
        "origin_execution_id": "execution-project-schema",
        "page_source_sha256": "a" * 64,
        "verification_status": "verified",
        "components": [
            {
                "component_id": "cmp_page_001_body",
                "page_id": "page_001",
                "bbox_page": [10, 20, 110, 80],
                "polygon_page": [[10, 20], [110, 20], [110, 80], [10, 80]],
                "detector_sources": ["fixture"],
                "confidence": 0.98,
                "script_evidence": ["latin"],
                "evidence_ids": ["evidence_001"],
                "rotation_deg": 0.0,
                "rotation_source": "fixture",
            }
        ],
        "observations": [
            {
                "observation_id": "obs_page_001_body",
                "page_id": "page_001",
                "component_ids": ["cmp_page_001_body"],
                "text": "HELLO THERE",
                "confidence": 0.97,
                "provider": "fixture",
                "bbox_page": [10, 20, 110, 80],
                "run_id": "run-project-schema",
                "origin_execution_id": "execution-project-schema",
                "invocation_id": "invocation-project-schema",
                "attempt_id": "attempt-project-schema",
                "provider_family": "fixture",
                "page_source_sha256": "a" * 64,
                "root_input_pixel_sha256": "a" * 64,
                "input_pixel_sha256": "b" * 64,
                "payload_sha256": sha256(b"HELLO THERE").hexdigest(),
            }
        ],
        "owners": [
            {
                "owner_id": "own_page_001_body",
                "page_id": "page_001",
                "component_ids": ["cmp_page_001_body"],
                "observation_ids": ["obs_page_001_body"],
                "selected_observation_ids": ["obs_page_001_body"],
                "semantic_role": "dialogue_body",
                "source_payload": "HELLO THERE",
                "translated_payload": "OLÁ",
                "disposition": "owned",
                "state": "translated",
                "route_action": "translate_inpaint_render",
                "execution_tile_id": "tile_page_001_001",
                "action_mask_ref": None,
            }
        ],
        "projections": [
            {
                "owner_id": "own_page_001_body",
                "tile_id": "tile_page_001_001",
                "role": "executor",
                "bbox_page": [10, 20, 110, 80],
                "bbox_tile": [10, 20, 110, 80],
                "offset_xy": [0, 0],
            }
        ],
        "component_dispositions": [
            {
                "component_id": "cmp_page_001_body",
                "decision": "owned",
                "owner_id": "own_page_001_body",
                "reason": "fixture",
            }
        ],
        "violations": [],
    }


class ProjectSchemaV12Tests(unittest.TestCase):
    def test_build_empty_project_v12_has_required_contract(self) -> None:
        project = build_empty_project_v12(
            input_path="fixtures/tiny_chapter",
            page_count=2,
            mode="mock",
        )

        self.assertEqual(project["schema_version"], SCHEMA_VERSION)
        self.assertEqual(project["app"], "traduzai")
        self.assertEqual(project["run"]["pipeline_version"], SCHEMA_VERSION)
        self.assertEqual(project["run"]["mode"], "mock")
        self.assertEqual(project["source"]["page_count"], 2)
        self.assertEqual(project["qa"]["summary"]["total_pages"], 2)
        self.assertEqual(project["export_report"]["status"], "not_exported")
        self.assertEqual(
            project["owner_graph_schema_version"], OWNER_GRAPH_SCHEMA_VERSION
        )
        self.assertEqual(project["owner_graph_status"], "legacy_unverified")
        self.assertEqual(project["page_owner_graphs"], [])
        self.assertIsInstance(project["owner_invariant_summary"], dict)
        self.assertEqual(validate_project_v12(project), [])

    def test_project_roundtrip_preserves_owner_graph_and_ids(self) -> None:
        project = build_empty_project_v12(page_count=1)
        graph = _verified_owner_graph_payload()
        self.assertEqual(
            project["owner_graph_schema_version"], OWNER_GRAPH_SCHEMA_VERSION
        )
        project["pages"] = [
            {
                "page": 1,
                "regions": [
                    {
                        "owner_id": "own_page_001_body",
                        "page_id": "page_001",
                        "component_ids": ["cmp_page_001_body"],
                        "observation_ids": ["obs_page_001_body"],
                        "semantic_role": "dialogue_body",
                        "route_action": "translate_inpaint_render",
                        "action_mask_ref": None,
                        "layout_region_ids": [],
                    }
                ],
            }
        ]
        project["owner_graph_status"] = "verified"
        project["page_owner_graphs"] = [graph]
        project["owner_invariant_summary"] = {
            "page_count": 1,
            "component_count": 1,
            "observation_count": 1,
            "owner_count": 1,
            "projection_count": 1,
            "violation_count": 0,
            "critical_violation_count": 0,
        }

        loaded = json.loads(json.dumps(project, ensure_ascii=False))

        self.assertEqual(loaded["page_owner_graphs"], [graph])
        self.assertEqual(
            loaded["page_owner_graphs"][0]["owners"][0]["owner_id"],
            "own_page_001_body",
        )
        self.assertEqual(validate_project_v12(loaded), [])

    def test_validation_rejects_qa_summary_that_does_not_match_flags(self) -> None:
        project = build_empty_project_v12(page_count=2)
        project["qa"]["flags"] = [
            {"page": 1, "severity": "high", "reason": "english_leak"},
            {"page": 2, "severity": "low", "reason": "layout_warning"},
        ]
        project["qa"]["summary"] = {
            "total_pages": 2,
            "pages_with_flags": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        errors = validate_project_v12(project)

        self.assertTrue(any("qa.summary" in error for error in errors), errors)

    def test_json_schema_file_matches_python_schema_version(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1] / "schema" / "project_schema_v12.json"
        )
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(
            PROJECT_SCHEMA_V12["properties"]["schema_version"]["const"], SCHEMA_VERSION
        )
        self.assertEqual(
            raw_schema["properties"]["schema_version"]["const"], SCHEMA_VERSION
        )

    def test_owner_graph_schema_contract_matches_json_schema(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1] / "schema" / "project_schema_v12.json"
        )
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))

        for field in (
            "owner_graph_schema_version",
            "owner_graph_status",
            "page_owner_graphs",
            "owner_invariant_summary",
        ):
            self.assertIn(field, PROJECT_SCHEMA_V12["required"])
            self.assertIn(field, raw_schema["required"])
            self.assertIn(field, PROJECT_SCHEMA_V12["properties"])
            self.assertIn(field, raw_schema["properties"])

    def test_owner_graph_schema_requires_internal_identity_records(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1] / "schema" / "project_schema_v12.json"
        )
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        required_by_definition = {
            "sourceTextComponent": {"component_id", "page_id", "bbox_page"},
            "textObservation": {
                "observation_id",
                "page_id",
                "component_ids",
                "text",
                "provider",
                "bbox_page",
                "run_id",
                "origin_execution_id",
                "invocation_id",
                "attempt_id",
                "provider_family",
                "page_source_sha256",
                "root_input_pixel_sha256",
                "input_pixel_sha256",
                "payload_sha256",
            },
            "textOwner": {
                "owner_id",
                "page_id",
                "component_ids",
                "observation_ids",
                "selected_observation_ids",
                "semantic_role",
                "source_payload",
                "disposition",
                "state",
                "route_action",
                "execution_tile_id",
                "action_mask_ref",
            },
            "ownerProjection": {
                "owner_id",
                "tile_id",
                "role",
                "bbox_page",
                "bbox_tile",
            },
            "componentDisposition": {"component_id", "decision", "owner_id"},
            "ownerViolation": {"code", "severity", "message", "offenders"},
        }

        self.assertEqual(PROJECT_SCHEMA_V12["$defs"], raw_schema["$defs"])
        for definition, required in required_by_definition.items():
            self.assertTrue(
                required.issubset(PROJECT_SCHEMA_V12["$defs"][definition]["required"])
            )

    def test_owner_graph_schema_encodes_verified_and_legacy_status_contracts(
        self,
    ) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1] / "schema" / "project_schema_v12.json"
        )
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(PROJECT_SCHEMA_V12["allOf"], raw_schema["allOf"])
        conditions = {
            condition["if"]["properties"]["owner_graph_status"]["const"]: condition[
                "then"
            ]
            for condition in PROJECT_SCHEMA_V12["allOf"]
        }
        self.assertEqual(
            conditions["legacy_unverified"]["properties"]["page_owner_graphs"][
                "maxItems"
            ],
            0,
        )
        self.assertEqual(
            conditions["verified"]["properties"]["page_owner_graphs"]["minItems"],
            1,
        )
        self.assertEqual(
            set(
                conditions["verified"]["properties"]["owner_invariant_summary"][
                    "required"
                ]
            ),
            {
                "page_count",
                "component_count",
                "observation_count",
                "owner_count",
                "projection_count",
                "violation_count",
                "critical_violation_count",
            },
        )
        self.assertFalse(
            PROJECT_SCHEMA_V12["properties"]["owner_invariant_summary"][
                "additionalProperties"
            ]
        )
        self.assertEqual(
            conditions["legacy_unverified"]["properties"]["owner_invariant_summary"][
                "maxProperties"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
