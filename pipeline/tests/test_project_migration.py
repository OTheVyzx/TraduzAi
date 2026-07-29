import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.migrate_project import migrate_project_to_v12  # noqa: E402
from schema.project_schema_v12 import SCHEMA_VERSION, validate_project_v12  # noqa: E402
from ownership.project import (  # noqa: E402
    OwnerProjectValidationError,
    verified_owner_graphs_from_project,
)


class ProjectMigrationTests(unittest.TestCase):
    @staticmethod
    def _verified_empty_text_page_project() -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "app": "traduzai",
            "run": {
                "run_id": "verified-existing",
                "mode": "debug",
                "pipeline_version": SCHEMA_VERSION,
            },
            "source": {"input_path": "in", "page_count": 1, "hash": ""},
            "work_context": {},
            "pages": [{"page": 1, "regions": []}],
            "glossary_hits": [],
            "entity_flags": [],
            "qa": {
                "summary": {
                    "total_pages": 1,
                    "pages_with_flags": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                },
                "flags": [],
            },
            "export_report": {"status": "not_exported", "files": []},
            "legacy": {"paginas": []},
            "owner_graph_schema_version": 1,
            "owner_graph_status": "verified",
            "page_owner_graphs": [
                {
                    "schema_version": 1,
                    "page_id": "page_001",
                    "components": [],
                    "observations": [],
                    "owners": [],
                    "projections": [],
                    "component_dispositions": [],
                    "violations": [],
                }
            ],
            "owner_invariant_summary": {
                "page_count": 1,
                "component_count": 0,
                "observation_count": 0,
                "owner_count": 0,
                "projection_count": 0,
                "violation_count": 0,
                "critical_violation_count": 0,
            },
        }

    def test_migrates_v1_paginas_textos_to_v12_pages_and_legacy(self) -> None:
        legacy_project = {
            "versao": "1.0",
            "app": "TraduzAi",
            "obra": "Fixture Tiny",
            "capitulo": 7,
            "idioma_origem": "en",
            "idioma_destino": "pt-BR",
            "paginas": [
                {
                    "numero": 1,
                    "arquivo_original": "original/page-001.png",
                    "arquivo_traduzido": "expected/page-001.png",
                    "textos": [
                        {
                            "id": "legacy-text-1",
                            "bbox": [10, 20, 110, 80],
                            "texto": "HELLO",
                            "traduzido": "OLA",
                            "tipo": "fala",
                            "confidence": 0.91,
                            "qa_flags": ["needs_review"],
                        }
                    ],
                }
            ],
        }

        migrated = migrate_project_to_v12(
            legacy_project, input_path="chapter.cbz", mode="mock"
        )

        self.assertEqual(migrated["schema_version"], SCHEMA_VERSION)
        self.assertEqual(migrated["source"]["input_path"], "chapter.cbz")
        self.assertEqual(migrated["source"]["page_count"], 1)
        self.assertEqual(migrated["legacy"]["paginas"], legacy_project["paginas"])
        self.assertEqual(migrated["pages"][0]["page"], 1)
        self.assertEqual(migrated["pages"][0]["source_path"], "original/page-001.png")
        self.assertEqual(migrated["pages"][0]["rendered_path"], "expected/page-001.png")

        region = migrated["pages"][0]["regions"][0]
        self.assertEqual(region["region_id"], "p001_r001")
        self.assertEqual(region["bbox"], [10, 20, 110, 80])
        self.assertEqual(region["raw_ocr"], "HELLO")
        self.assertEqual(region["normalized_ocr"], "HELLO")
        self.assertEqual(region["translation"]["text"], "OLA")
        self.assertEqual(region["region_type"], "speech_balloon")
        self.assertEqual(region["ocr_confidence"], 0.91)
        self.assertEqual(region["qa_flags"], ["needs_review"])
        self.assertEqual(migrated["owner_graph_schema_version"], 1)
        self.assertEqual(migrated["owner_graph_status"], "legacy_unverified")
        self.assertEqual(migrated["page_owner_graphs"], [])
        self.assertEqual(validate_project_v12(migrated), [])

    def test_legacy_project_migrates_as_legacy_unverified(self) -> None:
        migrated = migrate_project_to_v12(
            {
                "versao": "2.0",
                "paginas": [{"numero": 1, "text_layers": []}],
            }
        )

        self.assertEqual(migrated["owner_graph_status"], "legacy_unverified")
        self.assertEqual(migrated["page_owner_graphs"], [])

    def test_legacy_migration_never_synthesizes_verified_ownership(self) -> None:
        migrated = migrate_project_to_v12(
            {
                "versao": "2.0",
                "owner_graph_status": "verified",
                "page_owner_graphs": [
                    {
                        "schema_version": 1,
                        "page_id": "page_001",
                        "owners": [{"owner_id": "forged-owner"}],
                    }
                ],
                "paginas": [
                    {"numero": 1, "text_layers": [{"owner_id": "forged-owner"}]}
                ],
            }
        )

        self.assertEqual(migrated["owner_graph_status"], "legacy_unverified")
        self.assertEqual(migrated["page_owner_graphs"], [])

    def test_legacy_migration_selects_only_explicit_legacy_adapter(self) -> None:
        from ownership.legacy_adapter import LegacyUnverifiedAdapter, owner_mode_for_project

        project = {"owner_graph_status": "legacy_unverified", "paginas": []}
        self.assertEqual(owner_mode_for_project(project), "legacy")
        self.assertIs(LegacyUnverifiedAdapter(project).project, project)

    def test_v12_project_is_preserved_and_gets_legacy_alias(self) -> None:
        project = {
            "schema_version": SCHEMA_VERSION,
            "app": "traduzai",
            "run": {
                "run_id": "existing",
                "mode": "debug",
                "pipeline_version": SCHEMA_VERSION,
            },
            "source": {"input_path": "in", "page_count": 0, "hash": ""},
            "work_context": {},
            "pages": [],
            "glossary_hits": [],
            "entity_flags": [],
            "qa": {
                "summary": {
                    "total_pages": 0,
                    "pages_with_flags": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                },
                "flags": [],
            },
            "export_report": {"status": "not_exported", "files": []},
        }

        migrated = migrate_project_to_v12(project)

        self.assertEqual(migrated["run"]["run_id"], "existing")
        self.assertEqual(migrated["legacy"]["paginas"], [])
        self.assertEqual(migrated["owner_graph_status"], "legacy_unverified")
        self.assertEqual(validate_project_v12(migrated), [])

    def test_valid_verified_v12_owner_graph_is_preserved(self) -> None:
        project = self._verified_empty_text_page_project()

        migrated = migrate_project_to_v12(project)

        self.assertEqual(migrated["owner_graph_status"], "verified")
        self.assertEqual(migrated["page_owner_graphs"], project["page_owner_graphs"])
        self.assertEqual(validate_project_v12(migrated), [])

    def test_invalid_verified_v12_owner_graph_is_rejected(self) -> None:
        project = self._verified_empty_text_page_project()
        project["page_owner_graphs"] = []
        project["owner_invariant_summary"]["page_count"] = 0

        with self.assertRaises(OwnerProjectValidationError):
            migrate_project_to_v12(project)

    def test_legacy_unverified_project_cannot_load_as_verified_owner_graph(
        self,
    ) -> None:
        project = self._verified_empty_text_page_project()
        project["owner_graph_status"] = "legacy_unverified"
        project["page_owner_graphs"] = []
        project["owner_invariant_summary"] = {}
        snapshot = deepcopy(project)

        with self.assertRaisesRegex(OwnerProjectValidationError, "reprocess"):
            verified_owner_graphs_from_project(project)

        self.assertEqual(project, snapshot)


if __name__ == "__main__":
    unittest.main()
