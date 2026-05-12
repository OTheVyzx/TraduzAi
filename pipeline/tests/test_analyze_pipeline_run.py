import json
from pathlib import Path

from pipeline.tools.analyze_pipeline_run import (
    load_run_metrics,
    rank_bands,
    summarize_stages,
)


def _write_fixture_project(output_dir: Path) -> None:
    (output_dir / "translated").mkdir(parents=True)
    (output_dir / "translated" / "001.jpg").write_bytes(b"fake")
    project = {
        "paginas": [
            {
                "numero": 1,
                "page_profile": {
                    "strip_perf_summary": {
                        "band_count": 3,
                        "text_count": 4,
                        "remaining_inpaint_blocks": 2,
                        "smart_skip_shadow_candidate_count": 2,
                        "durations_sec": {
                            "ocr": 6.0,
                            "inpaint": 9.0,
                            "translate": 1.5,
                            "typeset": 2.0,
                        },
                        "entries": [
                            {
                                "band_index": 0,
                                "text_count": 1,
                                "remaining_inpaint_blocks": 1,
                                "durations_sec": {"ocr": 1.0, "inpaint": 7.0},
                                "total_sec": 8.0,
                            },
                            {
                                "band_index": 1,
                                "text_count": 2,
                                "remaining_inpaint_blocks": 1,
                                "durations_sec": {"ocr": 4.0, "inpaint": 1.0},
                                "total_sec": 5.0,
                            },
                            {
                                "band_index": 2,
                                "text_count": 1,
                                "remaining_inpaint_blocks": 0,
                                "durations_sec": {"ocr": 1.0, "inpaint": 1.0},
                                "total_sec": 2.0,
                            },
                        ],
                    }
                },
                "text_layers": [{"id": "a"}, {"id": "b"}],
                "inpaint_blocks": [{"bbox": [1, 2, 3, 4]}],
            },
            {
                "numero": 2,
                "text_layers": [{"id": "c"}, {"id": "d"}],
                "inpaint_blocks": [{"bbox": [5, 6, 7, 8]}],
            },
        ],
        "estatisticas": {
            "total_paginas": 2,
            "total_textos": 4,
            "tempo_processamento_seg": 21.5,
        },
    }
    (output_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")


def test_load_run_metrics_from_project_json(tmp_path):
    _write_fixture_project(tmp_path)

    metrics = load_run_metrics(tmp_path)

    assert metrics.total_seconds == 21.5
    assert metrics.pages == 2
    assert metrics.text_count == 4
    assert metrics.inpaint_blocks_exported == 2
    assert metrics.band_count == 3
    assert metrics.skip_candidate_count == 2
    assert len(metrics.bands) == 3


def test_summarize_stages_and_rank_bands(tmp_path):
    _write_fixture_project(tmp_path)
    metrics = load_run_metrics(tmp_path)

    assert summarize_stages(metrics) == {
        "ocr": 6.0,
        "inpaint": 9.0,
        "translate": 1.5,
        "typeset": 2.0,
    }

    assert [band.band_index for band in rank_bands(metrics, "ocr", limit=2)] == [1, 0]
    assert [band.band_index for band in rank_bands(metrics, "inpaint", limit=2)] == [0, 1]
