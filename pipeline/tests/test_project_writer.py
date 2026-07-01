import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _save_project_json
from project_writer import validate_project_consistency, write_project_json_atomic


def _project():
    return {
        "paginas": [{"text_layers": [{"qa_flags": ["low_ocr_confidence"]}]}],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 1}},
    }


def test_atomic_write_creates_project_json(tmp_path):
    path = tmp_path / "project.json"

    write_project_json_atomic(path, _project())

    assert json.loads(path.read_text(encoding="utf-8"))["estatisticas"]["total_paginas"] == 1
    assert not (tmp_path / "project.json.tmp").exists()


def test_backup_created_before_overwrite(tmp_path):
    path = tmp_path / "project.json"
    write_project_json_atomic(path, _project())
    write_project_json_atomic(path, _project())

    assert list(tmp_path.glob("project.backup.*.json"))


def test_invalid_schema_does_not_replace_existing_file(tmp_path):
    path = tmp_path / "project.json"
    write_project_json_atomic(path, _project())

    with pytest.raises(ValueError):
        write_project_json_atomic(path, {"paginas": "bad"})

    assert json.loads(path.read_text(encoding="utf-8"))["estatisticas"]["total_paginas"] == 1


def test_summary_mismatch_fails():
    project = _project()
    project["qa"]["summary"]["total"] = 2

    with pytest.raises(ValueError, match="qa.summary"):
        validate_project_consistency(project)


def test_page_count_mismatch_fails():
    project = _project()
    project["estatisticas"]["total_paginas"] = 2

    with pytest.raises(ValueError, match="total_paginas"):
        validate_project_consistency(project)


def test_log_summary_mismatch_fails():
    project = _project()
    project["log"] = {"summary": {"actual_pages": 99}}

    with pytest.raises(ValueError, match="log.summary"):
        validate_project_consistency(project)


def test_editor_save_refreshes_stale_log_summary(tmp_path):
    path = tmp_path / "project.json"
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "layer-a",
                        "traduzido": "Ola",
                        "translated": "Ola",
                        "qa_flags": [],
                    }
                ]
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "log": {
            "summary": {
                "actual_pages": 1,
                "processed_pages": 1,
                "translated_regions": 0,
                "qa_flags": 0,
                "critical_flags": 0,
            }
        },
    }

    _save_project_json(path, project)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["log"]["summary"]["translated_regions"] == 1


def test_project_writer_normalizes_sfx_policies_without_losing_sfx_metadata(tmp_path):
    path = tmp_path / "project.json"
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "sfx-1",
                        "tipo": "text",
                        "content_class": "sfx",
                        "route_action": "translate_sfx_inpaint_render",
                        "translate_policy": "translate",
                        "render_policy": "normal",
                        "sfx": {"source_text": "\ucff5", "adapted_text": "TUM"},
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 0}},
    }

    write_project_json_atomic(path, project)

    layer = json.loads(path.read_text(encoding="utf-8"))["paginas"][0]["text_layers"][0]
    assert layer["tipo"] == "sfx"
    assert layer["content_class"] == "sfx"
    assert layer["translate_policy"] == "adapt_sfx"
    assert layer["render_policy"] == "sfx_style"
    assert layer["sfx"]["adapted_text"] == "TUM"


def test_project_writer_removes_stale_sfx_policy_from_normal_text(tmp_path):
    path = tmp_path / "project.json"
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "text-1",
                        "content_class": "text",
                        "route_action": "translate_inpaint_render",
                        "translate_policy": "adapt_sfx",
                        "render_policy": "sfx_style",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 0}},
    }

    write_project_json_atomic(path, project)

    layer = json.loads(path.read_text(encoding="utf-8"))["paginas"][0]["text_layers"][0]
    assert layer["tipo"] == "text"
    assert layer["content_class"] == "text"
    assert layer["translate_policy"] == "translate"
    assert layer["render_policy"] == "normal"
