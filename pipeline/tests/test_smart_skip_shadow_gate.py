import json
from pathlib import Path

from pipeline.tools.run_smart_skip_shadow_gate import evaluate_smart_skip_shadow_gate


def _write_shadow_project(
    output_dir: Path,
    *,
    candidate_counts: list[int],
    inpaint_times: list[float],
) -> None:
    (output_dir / "translated").mkdir(parents=True)
    (output_dir / "translated" / "001.jpg").write_bytes(b"fake")
    entries = []
    for index, (candidate_count, inpaint_time) in enumerate(
        zip(candidate_counts, inpaint_times)
    ):
        entries.append(
            {
                "band_index": index,
                "text_count": max(1, candidate_count),
                "smart_skip_shadow_candidate_count": candidate_count,
                "smart_skip_shadow_not_safe_count": 0,
                "smart_skip_shadow_category_counts": (
                    {"credit_or_watermark": candidate_count}
                    if candidate_count
                    else {}
                ),
                "durations_sec": {"ocr": 1.0, "inpaint": inpaint_time},
                "total_sec": 1.0 + inpaint_time,
            }
        )
    project = {
        "paginas": [
            {
                "numero": 1,
                "page_profile": {
                    "strip_perf_summary": {
                        "band_count": len(entries),
                        "text_count": sum(max(1, count) for count in candidate_counts),
                        "smart_skip_shadow_candidate_count": sum(candidate_counts),
                        "smart_skip_shadow_not_safe_count": 0,
                        "smart_skip_shadow_category_counts": {
                            "credit_or_watermark": sum(candidate_counts)
                        },
                        "durations_sec": {
                            "ocr": 3.0,
                            "inpaint": sum(inpaint_times),
                        },
                        "entries": entries,
                    }
                },
                "text_layers": [{"id": "a"}],
                "inpaint_blocks": [{"bbox": [1, 2, 3, 4]}],
            }
        ],
        "estatisticas": {
            "total_paginas": 1,
            "total_textos": sum(max(1, count) for count in candidate_counts),
            "tempo_processamento_seg": 30.0,
        },
    }
    (output_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")


def test_smart_skip_shadow_gate_passes_when_estimated_savings_hits_threshold(tmp_path):
    output_dir = tmp_path / "shadow"
    output_dir.mkdir()
    _write_shadow_project(output_dir, candidate_counts=[1, 1], inpaint_times=[10.0, 8.0])

    result = evaluate_smart_skip_shadow_gate(
        output_dir,
        tmp_path / "gate",
        min_savings_seconds=16.75,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["estimated_savings_seconds"] == 18.0
    assert result["gate"]["candidate_count"] == 2


def test_smart_skip_shadow_gate_fails_when_savings_are_too_low(tmp_path):
    output_dir = tmp_path / "shadow"
    output_dir.mkdir()
    _write_shadow_project(output_dir, candidate_counts=[1], inpaint_times=[4.0])

    result = evaluate_smart_skip_shadow_gate(
        output_dir,
        tmp_path / "gate",
        min_savings_seconds=16.75,
    )

    assert result["gate"]["status"] == "FAIL"
    assert "below threshold" in result["gate"]["reasons"][0]


def test_smart_skip_shadow_gate_blocks_without_shadow_summary(tmp_path):
    output_dir = tmp_path / "shadow"
    output_dir.mkdir()
    _write_shadow_project(output_dir, candidate_counts=[0], inpaint_times=[2.0])
    project_path = output_dir / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    summary = project["paginas"][0]["page_profile"]["strip_perf_summary"]
    summary.pop("smart_skip_shadow_candidate_count")
    project_path.write_text(json.dumps(project), encoding="utf-8")

    result = evaluate_smart_skip_shadow_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "BLOCK"
    assert "shadow summary" in result["gate"]["reasons"][0]
