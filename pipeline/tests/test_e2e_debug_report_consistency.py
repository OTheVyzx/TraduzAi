"""Tests for the consistency block produced by ``debug_tools.report`` and the
``--strict-debug-audit`` gate exposed by ``tools/analyze_e2e_debug.py``.

These guard the §5b invariants of ``docs/debug/e2e_pipeline_debug_guide.md``:

- ``debug_report.text_count == project.estatisticas.total_textos``
- ``confidence_zero_count == ocr_confidence_audit.blocks_with_confidence_zero``
- ``content_class_counts`` is not doubled when ``text_layers`` and ``textos``
  are duplicate representations of the same blocks
- ``13_report/debug_report_consistency.json`` exists and reports an explicit
  ``all_consistent`` bool
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = ROOT / "tools" / "analyze_e2e_debug.py"

# Ensure the pipeline package can be imported (mirrors test_debug_report.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools.report import generate_debug_report  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_e2e_debug", ANALYZER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# debug_tools.report — emits consistency block when project is provided
# ---------------------------------------------------------------------------


def test_report_emits_consistency_json_with_canonical_metrics(tmp_path):
    e2e_root = tmp_path / "debug" / "e2e"
    gate_dir = e2e_root / "11_qa_export_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "export_gate.json").write_text(
        json.dumps({"status": "BLOCK", "issues": []}),
        encoding="utf-8",
    )
    _write_json(
        e2e_root / "03_ocr" / "ocr_confidence_audit.json",
        {"summary": {"total_blocks": 2, "blocks_with_confidence_zero": 0}},
    )

    project_path = tmp_path / "project.json"
    # text_layers and textos both have the same content, but textos still
    # carries the legacy ``confianca_ocr=0.0`` stub — DBG2-02 regression guard.
    text_layers = [
        {"id": "a", "content_class": "dialogue", "confidence_raw": 0.82},
        {"id": "b", "content_class": "narration", "confidence_raw": 0.71},
    ]
    textos = [
        {"id": "a", "content_class": "dialogue", "confianca_ocr": 0.0, "ocr_confidence": 0.0},
        {"id": "b", "content_class": "narration", "confianca_ocr": 0.0, "ocr_confidence": 0.0},
    ]
    project_path.write_text(
        json.dumps(
            {
                "paginas": [{"text_layers": text_layers, "textos": textos}],
                "estatisticas": {"total_textos": 2},
            }
        ),
        encoding="utf-8",
    )

    result = generate_debug_report(e2e_root, project_path=project_path)

    assert result["text_count"] == 2  # not doubled
    assert result["confidence_zero_count"] == 0
    assert result["confidence_missing_count"] == 0
    assert result["content_class_counts"] == {"dialogue": 1, "narration": 1}

    consistency_path = e2e_root / "13_report" / "debug_report_consistency.json"
    consistency_payload = json.loads(consistency_path.read_text(encoding="utf-8"))
    assert consistency_payload["schema_version"] == 1
    assert isinstance(consistency_payload["all_consistent"], bool)
    assert consistency_payload["all_consistent"] is True
    names = {check["name"] for check in consistency_payload["checks"]}
    assert {"text_count_vs_estatisticas", "confidence_zero_count", "source_bbox_overreach_count"} <= names


def test_report_consistency_flags_doubled_count_mismatch(tmp_path):
    e2e_root = tmp_path / "debug" / "e2e"
    gate_dir = e2e_root / "11_qa_export_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "export_gate.json").write_text(
        json.dumps({"status": "OK", "issues": []}),
        encoding="utf-8",
    )

    project_path = tmp_path / "project.json"
    layers = [{"id": f"ocr_{i:03d}", "content_class": "dialogue"} for i in range(10)]
    project_path.write_text(
        json.dumps(
            {
                "paginas": [{"text_layers": layers}],
                "estatisticas": {"total_textos": 20},  # mismatched on purpose
            }
        ),
        encoding="utf-8",
    )
    result = generate_debug_report(e2e_root, project_path=project_path)

    assert result["text_count"] == 10
    consistency = result["debug_report_consistency"]
    text_check = next(
        c for c in consistency["checks"] if c["name"] == "text_count_vs_estatisticas"
    )
    assert text_check["stage_level"] == 20
    assert text_check["aggregator"] == 10
    assert text_check["consistent"] is False
    assert consistency["all_consistent"] is False


# ---------------------------------------------------------------------------
# Analyzer — debug_report_consistency.json is dropped per run
# ---------------------------------------------------------------------------


def test_report_consistency_cross_checks_missing_balloon_and_skip_inpaint(tmp_path):
    e2e_root = tmp_path / "debug" / "e2e"
    gate_dir = e2e_root / "11_qa_export_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "export_gate.json").write_text(
        json.dumps({"status": "OK", "issues": []}),
        encoding="utf-8",
    )

    project_path = tmp_path / "project.json"
    project_path.write_text(
        json.dumps(
            {
                "paginas": [
                    {
                        "text_layers": [
                            {"id": "a", "content_class": "dialogue", "confidence_raw": 0.8}
                        ]
                    }
                ],
                "estatisticas": {"total_textos": 1},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        e2e_root / "09_typeset" / "balloon_bbox_missing_audit.jsonl",
        [
            {"text_id": "a", "issue": "balloon_bbox_missing"},
            {"text_id": "b", "issue": "balloon_bbox_missing"},
        ],
    )
    _write_json(
        e2e_root / "00_run" / "runner_config_snapshot.json",
        {"skip_inpaint": True},
    )
    for band in ("page_001_band_001", "page_001_band_002"):
        _write_json(
            e2e_root / "08_inpaint" / band / "inpaint_decision.json",
            {"band_id": band, "skip_inpaint_honored": True},
        )

    result = generate_debug_report(e2e_root, project_path=project_path)
    consistency = result["debug_report_consistency"]

    missing_check = next(
        c for c in consistency["checks"] if c["name"] == "balloon_bbox_missing_count"
    )
    skip_check = next(c for c in consistency["checks"] if c["name"] == "skip_inpaint_honored")
    assert missing_check["stage_level"] == 2
    assert missing_check["aggregator"] == 2
    assert missing_check["consistent"] is True
    assert skip_check["stage_level"] is True
    assert skip_check["aggregator"] is True
    assert skip_check["consistent"] is True
    assert result["balloon_bbox_missing_count"] == 2
    assert result["skip_inpaint_honored_bands"] == 2
    assert result["skip_inpaint_honored"] is True


def test_analyzer_writes_per_run_consistency_json(tmp_path):
    module = _load_analyzer()
    root = tmp_path / "runs"
    run = root / "A"
    run.mkdir(parents=True)

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "a",
                            "confidence_raw": 0.7,
                            "source_bbox": [0, 0, 10, 10],
                            "balloon_bbox": [0, 0, 100, 100],
                            "content_class": "dialogue",
                        }
                    ]
                }
            ],
            "estatisticas": {"total_textos": 1},
        },
    )
    _write_jsonl(
        run / "debug" / "e2e" / "05_layout_geometry" / "source_bbox_balloon_overreach.jsonl",
        [{"text_id": "a", "issue": "source_bbox_equals_balloon_bbox"}],
    )

    module.analyze_root(root, write_report=True)

    consistency_path = run / "debug" / "e2e" / "13_report" / "debug_report_consistency.json"
    assert consistency_path.exists()
    payload = json.loads(consistency_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert isinstance(payload["all_consistent"], bool)
    overreach_check = next(
        c for c in payload["checks"] if c["name"] == "source_bbox_overreach_count"
    )
    assert overreach_check["stage_level"] == 1
    assert overreach_check["aggregator"] == 1
    assert overreach_check["consistent"] is True
