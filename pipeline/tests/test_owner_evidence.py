from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _observation(
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    observation_id: str = "observation",
    component_ids: tuple[str, ...] = ("component_top", "component_bottom"),
) -> dict:
    return {
        "observation_id": observation_id,
        "page_id": "page_001",
        "component_ids": list(component_ids),
        "text": text,
        "confidence": 0.9,
        "provider": "fixture_ocr",
        "bbox_page": list(bbox),
    }


def _graph() -> dict:
    return {
        "page_id": "page_001",
        "components": [
            {"component_id": "component_top", "bbox_page": [10, 10, 90, 35]},
            {"component_id": "component_bottom", "bbox_page": [10, 35, 90, 70]},
        ],
        "observations": [
            _observation(
                "TOTAL PURCHASE AMOUNT",
                (10, 10, 90, 35),
                observation_id="observation_truncated",
                component_ids=("component_top",),
            ),
            _observation(
                "TOTAL PURCHASE AMOUNT 200MILLION",
                (10, 10, 90, 70),
                observation_id="observation_complete",
            ),
        ],
        "owners": [
            {
                "owner_id": "owner_a",
                "component_ids": ["component_top", "component_bottom"],
                "observation_ids": ["observation_truncated", "observation_complete"],
                "selected_observation_ids": ["observation_complete"],
            }
        ],
    }


def test_letter_digit_boundaries_normalize_200million():
    from ownership.evidence import normalize_evidence_tokens

    assert normalize_evidence_tokens("200MILLION") == ("200", "million")


def test_spacing_signature_ignores_only_spacing_and_apostrophe_variants():
    from ownership.evidence import normalize_evidence_spacing_signature

    assert normalize_evidence_spacing_signature("AREYOU READY?") == (
        normalize_evidence_spacing_signature("ARE YOU READY?")
    )
    assert normalize_evidence_spacing_signature("ITS READY") == (
        normalize_evidence_spacing_signature("IT'S READY")
    )
    assert normalize_evidence_spacing_signature("RATE: 4.6") != (
        normalize_evidence_spacing_signature("RATE: 46")
    )


def test_complete_observation_safely_dominates_coherent_truncation():
    from ownership.evidence import safely_dominates

    assert safely_dominates(
        complete=_observation("TOTAL PURCHASE AMOUNT 200MILLION", (10, 10, 90, 70)),
        truncated=_observation(
            "TOTAL PURCHASE AMOUNT",
            (10, 10, 90, 35),
            component_ids=("component_top",),
        ),
        same_region=True,
        corroboration_count=2,
    )


def test_complete_observation_dominates_truncation_with_collapsed_spaces():
    from ownership.evidence import safely_dominates

    assert safely_dominates(
        complete=_observation("TOTAL PURCHASE AMOUNT 200MILLION", (10, 10, 90, 70)),
        truncated=_observation("TOTALPURCHASEAMOUNT", (10, 10, 90, 35)),
        same_region=True,
        corroboration_count=2,
    )


def test_corroborated_complete_observation_dominates_minor_ocr_drift_fragment():
    from ownership.evidence import safely_dominates

    assert safely_dominates(
        complete=_observation(
            "GRADE B PERMANENTLY INCREASES AGILITY ADVANCED ALCHEMY GREATLY ENHANCED THE POTONS EFFECTIVENESS",
            (10, 10, 190, 100),
        ),
        truncated=_observation(
            "ADVANCED ALCHEMY GREATLY ENHANCED THE POTIONS EFFECTIVENESS",
            (10, 55, 190, 100),
        ),
        same_region=True,
        corroboration_count=2,
    )


def test_fuzzy_truncation_never_hides_numeric_disagreement():
    from ownership.evidence import safely_dominates

    assert not safely_dominates(
        complete=_observation(
            "GRADE A PERMANENTLY INCREASES HEALTH BY 7 UPON CONSUMPTION",
            (10, 10, 190, 100),
        ),
        truncated=_observation(
            "PERMANENTLY INCREASES HEALTH BY 2 UPON CONSUMPTION",
            (10, 40, 190, 100),
        ),
        same_region=True,
        corroboration_count=3,
    )


def test_one_character_suffix_is_not_treated_as_safe_truncation():
    from ownership.evidence import safely_dominates

    assert not safely_dominates(
        complete=_observation("WARNINGE", (10, 10, 90, 70)),
        truncated=_observation("WARNING", (10, 10, 90, 70)),
        same_region=True,
        corroboration_count=3,
    )


def test_long_adjacent_contamination_never_dominates():
    from ownership.evidence import safely_dominates

    assert not safely_dominates(
        complete=_observation(
            "TOTAL PURCHASE AMOUNT PLAYER NAME",
            (10, 10, 180, 70),
            component_ids=("component_top", "component_bottom", "foreign_component"),
        ),
        truncated=_observation(
            "TOTAL PURCHASE AMOUNT",
            (10, 10, 90, 35),
            component_ids=("component_top",),
        ),
        same_region=False,
        corroboration_count=1,
    )


def test_source_evidence_ledger_is_complete_auditable_and_immutable():
    from ownership.evidence import build_source_evidence_ledger

    rows = build_source_evidence_ledger(_graph())

    assert {row.observation_id for row in rows} == {
        "observation_truncated",
        "observation_complete",
    }
    assert all(
        {
            "observation_id",
            "owner_id",
            "component_coverage",
            "observation_precision",
            "material",
            "disposition",
            "reason",
            "dominant_observation_id",
        }
        <= row.to_dict().keys()
        for row in rows
    )
    complete = next(row for row in rows if row.observation_id == "observation_complete")
    truncated = next(row for row in rows if row.observation_id == "observation_truncated")
    assert complete.disposition == "selected"
    assert complete.component_coverage == 1.0
    assert truncated.component_coverage == 0.5
    assert truncated.disposition == "dominated"
    assert truncated.dominant_observation_id == complete.observation_id
    with pytest.raises(FrozenInstanceError):
        complete.disposition = "mutated"  # type: ignore[misc]


def test_ledger_keeps_unowned_material_observation_for_audit():
    from ownership.evidence import build_source_evidence_ledger

    graph = _graph()
    graph["observations"].append(
        _observation(
            "UNASSIGNED SOURCE",
            (200, 200, 300, 240),
            observation_id="observation_unowned",
            component_ids=("component_unowned",),
        )
    )

    row = next(
        item
        for item in build_source_evidence_ledger(graph)
        if item.observation_id == "observation_unowned"
    )
    assert row.owner_id is None
    assert row.material is True
    assert row.disposition == "unowned"
    assert row.reason == "no_owner_assignment"
