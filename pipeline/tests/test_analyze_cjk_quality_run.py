from __future__ import annotations

import json

from tools.analyze_cjk_quality_run import analyze_cjk_quality_run, main


def test_analyze_cjk_quality_run_extracts_plan_issue_classes(tmp_path):
    run_dir = tmp_path / "translated"
    run_dir.mkdir()
    (run_dir / "project.json").write_text(
        json.dumps(
            {
                "paginas": [
                    {
                        "numero": 5,
                        "text_layers": [
                            {
                                "id": "a",
                                "text": "The image is too blurry to recognize any text content.",
                                "translated": "Nao consigo encontrar o texto original.",
                                "qa_flags": ["vlm_failure_phrase", "translation_fallback_phrase"],
                            }
                        ],
                    },
                    {
                        "numero": 18,
                        "text_layers": [
                            {
                                "id": "b",
                                "tipo": "fala",
                                "translated": "구한 것이지",
                                "ignored_reason": "cjk_sfx_preserved",
                            }
                        ],
                    },
                    {
                        "numero": 26,
                        "text_layers": [
                            {
                                "id": "c",
                                "translated": "ISSO NAO PODE SER POSSIVEL.",
                                "qa_flags": ["outline_damage_high", "text_overflow"],
                                "vertical_offset_ratio": 0.6,
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = analyze_cjk_quality_run(run_dir, tmp_path / "analysis")

    assert result["status"] == "PASS"
    assert result["selected_pages"] == [5, 18, 26]
    assert result["issue_counts"]["vlm_failure_phrase"] == 1
    assert result["issue_counts"]["translation_fallback_phrase"] == 1
    assert result["issue_counts"]["cjk_sfx_preserved"] == 1
    assert result["issue_counts"]["hangul_residual_in_speech"] == 1
    assert result["issue_counts"]["inpaint_geometry_or_residual"] == 1
    assert result["issue_counts"]["typesetting_layout"] == 1
    assert (tmp_path / "analysis" / "summary.json").exists()
    assert (tmp_path / "analysis" / "visual_sheet.html").exists()


def test_analyze_cjk_quality_run_blocks_without_project_json(tmp_path):
    result = analyze_cjk_quality_run(tmp_path / "missing", tmp_path / "analysis")

    assert result["status"] == "BLOCK"
    assert result["reasons"] == ["missing project.json"]
    assert (tmp_path / "analysis" / "summary.json").exists()


def test_analyze_cjk_quality_run_can_force_manual_review_pages(tmp_path):
    run_dir = tmp_path / "translated"
    run_dir.mkdir()
    (run_dir / "project.json").write_text(
        json.dumps({"paginas": [{"numero": 58, "text_layers": []}]}),
        encoding="utf-8",
    )

    result = analyze_cjk_quality_run(run_dir, tmp_path / "analysis", include_pages=[58])

    assert result["selected_pages"] == [58]
    assert result["issue_counts"]["manual_review"] == 1


def test_sfx_benchmark_missing_folder_exits_cleanly(tmp_path, capsys):
    exit_code = main(["--sfx-benchmark", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 0
    assert result["status"] == "SKIP"
    assert result["benchmark"] == "sfx_manhwa"
    assert result["reason"] == "folder_not_found"
