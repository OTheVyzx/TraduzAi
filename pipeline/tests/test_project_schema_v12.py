import json
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
        self.assertEqual(validate_project_v12(project), [])

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
        schema_path = Path(__file__).resolve().parents[1] / "schema" / "project_schema_v12.json"
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(PROJECT_SCHEMA_V12["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertEqual(raw_schema["properties"]["schema_version"]["const"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
