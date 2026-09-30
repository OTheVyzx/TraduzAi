"""Structural recovery decisions are based on evidence, not page/owner IDs."""

from copy import deepcopy

from consumer_operational_discovery import discover_recovery_proposals


def _rows(first="owner:a", second="owner:b", first_text="HELLO"):
    return [
        dict(owner_id=first, semantic_role="dialogue_body", source_payload=first_text,
             owner_render_geometry=dict(selected_observations=[dict(
                 observation_id="ocr:a", bbox_page=[10, 10, 90, 30])])),
        dict(owner_id=second, semantic_role="dialogue_body", source_payload="WORLD.",
             owner_render_geometry=dict(selected_observations=[dict(
                 observation_id="ocr:b", bbox_page=[15, 32, 88, 52])])),
    ]


def test_continuation_survives_owner_id_remapping():
    first = discover_recovery_proposals(_rows(), 120, 100)
    remapped = discover_recovery_proposals(_rows("owner:x9", "owner:y8"), 120, 100)
    assert [row["kind"] for row in first] == [row["kind"] for row in remapped]
    assert first[0]["source_anchor_bbox"] == remapped[0]["source_anchor_bbox"]
    assert first[0]["requires_local_ocr"] and first[0]["requires_same_source_white_body"]


def test_continuation_decision_survives_record_permutation():
    ordered = discover_recovery_proposals(_rows(), 120, 100)
    permuted = discover_recovery_proposals(list(reversed(_rows())), 120, 100)
    assert ordered == permuted


def test_independent_dialogue_is_not_merged():
    assert discover_recovery_proposals(_rows(first_text="HELLO."), 120, 100) == []
    distant = deepcopy(_rows())
    distant[1]["owner_render_geometry"]["selected_observations"][0]["bbox_page"] = [15, 75, 88, 95]
    assert discover_recovery_proposals(distant, 120, 100) == []


def test_missing_graph_observation_fails_closed():
    try:
        discover_recovery_proposals(_rows(), 120, 100, source_observations={})
    except ValueError as error:
        assert "absent from source graph" in str(error)
    else:
        raise AssertionError("missing graph observation was accepted")


def test_split_uses_source_structure_with_remapped_owner():
    def connected(owner):
        return [dict(owner_id=owner, semantic_role="dialogue_body",
            source_payload="FIRST. SECOND.", owner_render_geometry=dict(
                selected_observations=[
                    dict(observation_id="a", text="FIRST.", bbox_page=[20, 20, 80, 35],
                         component_ids=["one"]),
                    dict(observation_id="b", text="SECOND.", bbox_page=[150, 80, 230, 95],
                         component_ids=["two"])],
                connected_subregions=[
                    dict(subregion_id="r1", order=0, component_ids=["one"],
                         evidence_ids=["a"], bbox_page=[10, 10, 100, 60],
                         polygon_page=[[10, 10], [100, 10], [100, 60], [10, 60]]),
                    dict(subregion_id="r2", order=1, component_ids=["two"],
                         evidence_ids=["b"], bbox_page=[140, 70, 240, 120],
                         polygon_page=[[140, 70], [240, 70], [240, 120], [140, 120]])]))]
    a = discover_recovery_proposals(connected("owner:original"), 300, 200)
    b = discover_recovery_proposals(connected("owner:remapped"), 300, 200)
    assert len(a) == len(b) == 1
    assert a[0]["kind"] == b[0]["kind"] == "split_connected_bodies"
    assert [unit["source_text"] for unit in a[0]["plan"]["units"]] == [
        unit["source_text"] for unit in b[0]["plan"]["units"]]


def test_unfinished_connected_text_is_not_auto_split():
    row = dict(owner_id="owner:a", semantic_role="dialogue_body",
        source_payload="FIRST SECOND.", owner_render_geometry=dict(
            selected_observations=[
                dict(observation_id="a", text="FIRST", bbox_page=[20, 20, 80, 35],
                     component_ids=["one"]),
                dict(observation_id="b", text="SECOND.", bbox_page=[150, 80, 230, 95],
                     component_ids=["two"])],
            connected_subregions=[
                dict(subregion_id="r1", order=0, component_ids=["one"],
                     evidence_ids=["a"], bbox_page=[10, 10, 100, 60],
                     polygon_page=[[10, 10], [100, 10], [100, 60], [10, 60]]),
                dict(subregion_id="r2", order=1, component_ids=["two"],
                     evidence_ids=["b"], bbox_page=[140, 70, 240, 120],
                     polygon_page=[[140, 70], [240, 70], [240, 120], [140, 120]])]))
    assert discover_recovery_proposals([row], 300, 200) == []
