import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.export_gate import evaluate_export_gate
from qa.translation_qa import severity_for_flag, summarize_flags


def test_qa_summary_and_export_gate_count_the_same_critical_flags():
    layers = [
        {
            "id": "t1",
            "translated": "texto",
            "qa_flags": ["bbox_overreach_critical"],
        },
        {
            "id": "t2",
            "translated": "Nao consigo encontrar o texto original.",
            "qa_flags": ["translation_fallback_phrase"],
        },
        {
            "id": "t3",
            "translated": "texto revisavel",
            "qa_flags": ["TEXT_CLIPPED"],
        },
    ]
    project = {"paginas": [{"numero": 1, "text_layers": layers}]}

    summary = summarize_flags(layers)
    gate = evaluate_export_gate(project)

    assert severity_for_flag("bbox_overreach_critical") == "critical"
    assert severity_for_flag("translation_fallback_phrase") == "critical"
    assert severity_for_flag("TEXT_CLIPPED") == "high"
    assert summary["highest_severity"] == "critical"
    assert summary["critical_count"] == gate["critical_issue_count"] == 2
    assert gate["review_issue_count"] == 1
    assert gate["status"] == "BLOCK"


def test_export_gate_visual_blocker_carries_traceable_artifact_links():
    project = {
        "paginas": [
            {
                "numero": 7,
                "page_id": "page_007",
                "arquivo_traduzido": "translated/007.jpg",
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_007_band_003",
                        "trace_id": "ocr_001@page_007_band_003",
                        "text_instance_id": "ocr_001@page_007_band_003",
                        "translated": "texto",
                        "qa_flags": ["render_outside_balloon"],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    issue = gate["issues"][0]
    assert gate["status"] == "BLOCK"
    assert issue["trace_id"] == "ocr_001@page_007_band_003"
    assert issue["page_id"] == "page_007"
    assert issue["band_id"] == "page_007_band_003"
    assert issue["text_instance_id"] == "ocr_001@page_007_band_003"
    assert set(issue["artifact_links"]) >= {
        "translated/007.jpg",
        "09_typeset/render_plan_final.jsonl",
        "05_layout_geometry/layout_blocks.jsonl",
        "12_contact_sheets/page_007_band_003.jpg",
    }


def test_export_gate_residual_blocker_carries_inpaint_artifact_links():
    project = {
        "paginas": [
            {
                "numero": 6,
                "page_id": "page_006",
                "text_layers": [
                    {
                        "id": "ocr_004",
                        "band_id": "page_006_band_116",
                        "trace_id": "ocr_004@page_006_band_116",
                        "translated": "texto",
                        "qa_flags": ["text_residual_after_inpaint"],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    issue = gate["issues"][0]
    assert issue["trace_id"] == "ocr_004@page_006_band_116"
    assert set(issue["artifact_links"]) >= {
        "08_inpaint/page_006_band_116/03_inpaint_mask_overlay.jpg",
        "08_inpaint/page_006_band_116/inpaint_decision.json",
        "08_inpaint/page_006_band_116/06_band_after_inpaint.jpg",
    }


def test_export_gate_every_critical_blocker_has_traceable_artifact_links():
    project = {
        "paginas": [
            {
                "numero": 3,
                "page_id": "page_003",
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "band_id": "page_003_band_042",
                        "trace_id": "ocr_002@page_003_band_042",
                        "translated": "texto",
                        "qa_flags": ["bbox_overreach_critical"],
                    },
                    {
                        "id": "ocr_003",
                        "band_id": "page_003_band_043",
                        "trace_id": "ocr_003@page_003_band_043",
                        "translated": "texto",
                        "qa_flags": ["mask_outside_balloon_critical"],
                    },
                ],
            }
        ],
        "qa": {
            "flag_propagation_audit": {
                "missing_in_project": [
                    {
                        "identity": "ocr_004@page_003_band_044",
                        "flag": "render_outside_balloon",
                        "source": "debug",
                    }
                ]
            }
        },
    }

    gate = evaluate_export_gate(project)

    critical_issues = [issue for issue in gate["issues"] if issue["severity"] == "critical"]
    assert len(critical_issues) == 3
    for issue in critical_issues:
        assert issue["trace_id"]
        assert issue["artifact_links"]
        assert "11_qa_export_gate/qa_issues.jsonl" in issue["artifact_links"]


def test_export_gate_synthesizes_trace_id_for_legacy_critical_layer():
    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "t1",
                        "translated": "texto",
                        "qa_flags": ["bbox_overreach_critical"],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    issue = gate["issues"][0]
    assert issue["trace_id"] == "t1@page_001_layer_001"
    assert issue["page_id"] == "page_001"
    assert issue["band_id"] == "page_001_layer_001"
    assert issue["text_instance_id"] == "page_001_layer_001_t1"
    assert issue["artifact_links"]
