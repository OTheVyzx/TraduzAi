from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACT_PROJECT = Path(
    r"N:/TraduzAI/DEBUGM/runs/2026-05-24_task10_real_validation_20260524_174315/chapter1_full/project.json"
)


def _bbox(value):
    if not isinstance(value, list) or len(value) < 4:
        return None
    return [int(v) for v in value[:4]]


@pytest.mark.skipif(not ARTIFACT_PROJECT.exists(), reason="local debug artifact not present")
def test_known_bad_chapter1_project_contains_mixed_coordinate_evidence():
    project = json.loads(ARTIFACT_PROJECT.read_text(encoding="utf-8"))
    offenders = []
    for page in project.get("paginas") or []:
        for text in page.get("text_layers") or []:
            if text.get("skip_processing"):
                continue
            target = _bbox(text.get("balloon_bbox") or text.get("bbox"))
            safe = _bbox(text.get("safe_text_box") or text.get("bubble_inner_bbox"))
            if not target or not safe:
                continue
            if target[1] > 1000 and safe[1] < 900:
                offenders.append((text.get("trace_id"), target, safe))
    assert offenders, "fixture should remain useful as known-bad coordinate evidence"


def test_project_route_contract_neutralizes_removed_decision_fields():
    from main import _ensure_project_route_action_contract

    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "tl_001",
                        "text": "HELLO",
                        "translated": "OLA",
                        "bbox": [10, 10, 80, 40],
                        "route_action": "translate_inpaint_render",
                        "tipo": "sfx",
                        "content_class": "noise",
                        "balloon_type": "dark",
                        "skip_processing": True,
                        "preserve_original": True,
                        "rotation_deg": 17,
                    }
                ]
            }
        ]
    }

    _ensure_project_route_action_contract(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert layer["route_action"] == "translate_inpaint_render"
    assert layer["tipo"] == "text"
    assert layer["content_class"] == "text"
    assert layer["balloon_type"] == ""
    assert layer["skip_processing"] is False
    assert layer["preserve_original"] is False
    assert layer["rotation_deg"] == 17


def test_project_writer_counts_flags_on_legacy_skip_layers():
    from project_writer import validate_project_consistency

    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "tl_001",
                        "skip_processing": True,
                        "qa_flags": ["render_missing"],
                    }
                ]
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 1}},
    }

    validate_project_consistency(project)
