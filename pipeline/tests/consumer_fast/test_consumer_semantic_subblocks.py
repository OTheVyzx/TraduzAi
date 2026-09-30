"""The live quality overlay must keep independently translated lobes bound."""

import pytest
from typesetter import renderer


def test_independent_connected_subblocks_use_explicit_semantic_targets():
    regions = [
        dict(layout_region_id="upper", semantic_subblock=True,
             translated_payload="SEJA GRATO. É SÓ ISSO!",
             selected_observation_ids=["obs_upper"], source_anchor_bbox=[40, 30, 150, 65]),
        dict(layout_region_id="lower", semantic_subblock=True,
             translated_payload="AINDA ASSIM... VOCÊS DOIS CONSEGUEM CRIÁ-LO?",
             selected_observation_ids=["obs_lower"], source_anchor_bbox=[60, 120, 180, 190]),
    ]
    payload = "SEJA GRATO. É SÓ ISSO! AINDA ASSIM... VOCÊS DOIS CONSEGUEM CRIÁ-LO?"
    assert renderer._owner_region_chunks(payload, regions) == [
        regions[0]["translated_payload"], regions[1]["translated_payload"]
    ]


def test_independent_connected_subblocks_reject_missing_binding():
    regions = [
        dict(layout_region_id="upper", semantic_subblock=True, translated_payload="PRIMEIRO."),
        dict(layout_region_id="lower", semantic_subblock=True),
    ]
    with pytest.raises(ValueError, match="semantic subblock binding"):
        renderer._owner_region_chunks("PRIMEIRO. SEGUNDO.", regions)


def test_regular_multiline_legacy_flow_remains_available():
    regions = [
        dict(layout_region_id="top", bbox_page=[0, 0, 100, 100]),
        dict(layout_region_id="bottom", bbox_page=[0, 110, 100, 210]),
    ]
    chunks = renderer._owner_region_chunks("UMA FRASE CONTINUA EM DOIS CORPOS.", regions)
    assert len(chunks) == 2
    assert " ".join(chunks) == "UMA FRASE CONTINUA EM DOIS CORPOS."
