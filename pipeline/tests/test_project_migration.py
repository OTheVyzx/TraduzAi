import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.migrate_project import migrate_project_to_v12  # noqa: E402
from schema.project_schema_v12 import SCHEMA_VERSION, validate_project_v12  # noqa: E402


class ProjectMigrationTests(unittest.TestCase):
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

        migrated = migrate_project_to_v12(legacy_project, input_path="chapter.cbz", mode="mock")

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
        self.assertEqual(validate_project_v12(migrated), [])

    def test_v12_project_is_preserved_and_gets_legacy_alias(self) -> None:
        project = {
            "schema_version": SCHEMA_VERSION,
            "app": "traduzai",
            "run": {"run_id": "existing", "mode": "debug", "pipeline_version": SCHEMA_VERSION},
            "source": {"input_path": "in", "page_count": 0, "hash": ""},
            "work_context": {},
            "pages": [],
            "glossary_hits": [],
            "entity_flags": [],
            "qa": {
                "summary": {"total_pages": 0, "pages_with_flags": 0, "critical": 0, "high": 0, "medium": 0, "low": 0},
                "flags": [],
            },
            "export_report": {"status": "not_exported", "files": []},
        }

        migrated = migrate_project_to_v12(project)

        self.assertEqual(migrated["run"]["run_id"], "existing")
        self.assertEqual(migrated["legacy"]["paginas"], [])
        self.assertEqual(validate_project_v12(migrated), [])


if __name__ == "__main__":
    unittest.main()
