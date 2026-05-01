import json
import time

from structured_logger import StructuredLogger, build_log_summary


def test_duration_is_positive(tmp_path):
    logger = StructuredLogger(tmp_path, "run")
    time.sleep(0.01)
    record = logger.log(stage="ocr", event="ocr_completed", page=1, region_id="p001_r001", payload={"raw_text": "TEXT"})

    assert record["duration_seconds"] > 0


def test_duplicate_event_is_written_once(tmp_path):
    logger = StructuredLogger(tmp_path, "run")
    payload = {"raw_text": "TEXT"}

    logger.log(stage="ocr", event="ocr_completed", page=1, region_id="p001_r001", payload=payload)
    logger.log(stage="ocr", event="ocr_completed", page=1, region_id="p001_r001", payload=payload)

    lines = (tmp_path / "run" / "structured_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "ocr_completed"


def test_summary_matches_project_counts():
    project = {
        "paginas": [
            {"text_layers": [
                {"translated": "A", "qa_flags": ["x"], "glossary_hits": [1], "entity_flags": [1, 2]},
                {"translated": "", "qa_flags": []},
            ]}
        ],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"critical_count": 1}},
    }

    summary = build_log_summary(project)

    assert summary["actual_pages"] == 1
    assert summary["processed_pages"] == 1
    assert summary["translated_regions"] == 1
    assert summary["qa_flags"] == 1
    assert summary["critical_flags"] == 1
    assert summary["glossary_hits"] == 1
    assert summary["entity_flags"] == 2
