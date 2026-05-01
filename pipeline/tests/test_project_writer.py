import json

import pytest

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
