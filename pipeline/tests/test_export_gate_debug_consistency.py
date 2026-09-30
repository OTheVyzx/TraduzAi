import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.export_gate import evaluate_export_gate
from qa.translation_qa import severity_for_flag, summarize_flags


def test_gate_summary_and_debug_counts_match_new_contract_issues(tmp_path):
    from debug_tools import DebugRecorder
    import main

    project = {
        "owner_graph_status": "verified",
        "paginas": [
            {
                "numero": 1,
                "page_id": "page_001",
                "text_layers": [
                    {
                        "id": "owner_a",
                        "owner_id": "owner_a",
                        "render_completed": True,
                        "route_action": "preserve",
                    }
                ],
            }
        ],
        "qa": {"summary": {}, "final_pixel_reports": []},
    }
    gate = evaluate_export_gate(project)
    project["qa"]["export_gate"] = gate
    main._synchronize_qa_summary_with_export_gate(project)

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-contract-counts")
    consistency = main._write_debug_export_gate_artifacts(recorder, project)
    saved = json.loads(
        (tmp_path / "debug" / "e2e" / "11_qa_export_gate" / "qa_export_gate_consistency.json").read_text(
            encoding="utf-8"
        )
    )

    summary = project["qa"]["summary"]
    assert summary["critical_issue_count"] == gate["critical_issue_count"]
    assert summary["critical_flag_count"] == gate["critical_flag_count"]
    assert summary["blocking_issue_count"] == gate["blocking_issue_count"]
    assert consistency["consistent"] is True
    assert saved["consistent"] is True


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


def test_final_pixel_gate_counts_each_report_issue_once(tmp_path):
    import hashlib

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"final")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    project = {
        "owner_graph_status": "verified",
        "paginas": [{"numero": 1, "page_id": "page_001", "text_layers": []}],
        "qa": {
            "final_pixel_reports": [
                {
                    "page_id": "page_001",
                    "artifact_path": str(artifact),
                    "persisted_sha256": digest,
                    "observer": "DetectorOcrFinalPixelObserver",
                    "observer_available": True,
                    "observation_complete": True,
                    "contracts": {
                        "source_coverage_contract": "PASS",
                        "owner_graph_contract": "PASS",
                        "route_state_contract": "PASS",
                        "pixel_ownership_contract": "BLOCK",
                        "final_language_contract": "PASS",
                        "layout_legibility_contract": "PASS",
                        "residual_cleanup_contract": "PASS",
                        "protected_art_contract": "PASS",
                        "qa_integrity_contract": "PASS",
                    },
                    "issues": [
                        {
                            "issue_id": "pixel_a",
                            "page_id": "page_001",
                            "owner_id": "owner_a",
                            "component_ids": ["component_a"],
                            "severity": "critical",
                            "reason": "pixel_change_outside_owned_masks",
                            "offenders": ["pixel:4"],
                            "contract": "pixel_ownership_contract",
                        }
                    ],
                }
            ]
        },
    }

    gate = evaluate_export_gate(project)

    assert gate["critical_issue_count"] == 1
    assert gate["blocking_issue_count"] == 1


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
