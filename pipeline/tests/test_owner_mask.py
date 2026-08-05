"""TDD contracts for owner-scoped operational inpaint masks."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.model import OWNER_GRAPH_SCHEMA_VERSION, TextOwner  # noqa: E402


def _owner() -> TextOwner:
    return TextOwner(
        owner_id="own_page_001_body",
        page_id="page_001",
        component_ids=["cmp_body_top", "cmp_body_bottom"],
        observation_ids=["obs_body"],
        selected_observation_ids=["obs_body"],
        semantic_role="dialogue_body",
        source_payload="HELLO THERE",
        translated_payload="OLÁ",
        disposition="owned",
        state="translated",
        route_action="translate_inpaint_render",
        execution_tile_id="tile_page_001_executor",
    )


def _single_component_owner(*, owner_id: str = "own_page_001_body") -> TextOwner:
    owner = _owner()
    owner.owner_id = owner_id
    owner.component_ids = ["cmp_body_top"]
    return owner


def test_operational_owner_mask_requires_verified_component_geometry(tmp_path):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
        execute_owner_inpaint,
        load_owner_action_mask,
        persist_owner_mask_plan,
    )

    class ChangingInpainter:
        def inpaint(self, image, mask, **_kwargs):
            result = image.copy()
            result[mask > 0] = 17
            return result

    image = np.full((32, 48, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[8:14, 9:21] = 255
    evidence = [
        OwnerMaskEvidence(
            evidence_id="glyph_top",
            component_id="cmp_body_top",
            glyph_mask=glyph,
        )
    ]

    draft_owner = _single_component_owner(owner_id="owner_geometry_draft")
    draft = build_owner_mask_plan(image, draft_owner, evidence)
    assert draft.component_geometry_verified is False
    with pytest.raises(UnsafeOwnerMaskError, match="geometry|component"):
        persist_owner_mask_plan(draft, tmp_path, owner=draft_owner)
    with pytest.raises(UnsafeOwnerMaskError, match="geometry|component"):
        execute_owner_inpaint(image, draft, ChangingInpainter())

    mismatched_owner = _single_component_owner(owner_id="owner_geometry_mismatch")
    with pytest.raises(UnsafeOwnerMaskError, match="geometry|component|bbox"):
        build_owner_mask_plan(
            image,
            mismatched_owner,
            evidence,
            owner_component_bboxes_page={"cmp_body_top": (28, 20, 40, 28)},
        )

    verified_owner = _single_component_owner(owner_id="owner_geometry_verified")
    verified = build_owner_mask_plan(
        image,
        verified_owner,
        evidence,
        owner_component_bboxes_page={"cmp_body_top": (7, 6, 23, 16)},
    )
    assert verified.component_geometry_verified is True
    assert verified.component_action_bboxes_page == (
        ("cmp_body_top", (7, 6, 23, 16)),
    )
    assert verified.component_bboxes_page == (
        ("cmp_body_top", (7, 6, 23, 16)),
    )
    assert verified.owner_bbox_page == (7, 6, 23, 16)
    assert len(verified.component_geometry_sha256) == 64
    persisted = persist_owner_mask_plan(verified, tmp_path, owner=verified_owner)
    assert persisted.is_file()
    loaded = load_owner_action_mask(
        tmp_path,
        verified.action_mask_ref,
        expected_owner_id=verified.owner_id,
        expected_page_id=verified.page_id,
        expected_source_sha256=verified.source_sha256,
        expected_execution_tile_id=verified.execution_tile_id,
        expected_component_geometry_sha256=verified.component_geometry_sha256,
    )
    np.testing.assert_array_equal(loaded, verified.action_mask)
    with pytest.raises(UnsafeOwnerMaskError, match="component|geometry"):
        load_owner_action_mask(
            tmp_path,
            verified.action_mask_ref,
            expected_owner_id=verified.owner_id,
            expected_page_id=verified.page_id,
            expected_source_sha256=verified.source_sha256,
            expected_execution_tile_id=verified.execution_tile_id,
            expected_component_geometry_sha256="0" * 64,
        )
    stale_owner = _single_component_owner(owner_id=verified.owner_id)
    stale_owner.component_ids = ["cmp_reassigned"]
    with pytest.raises(UnsafeOwnerMaskError, match="component|geometry"):
        persist_owner_mask_plan(
            verified,
            tmp_path / "stale_graph",
            owner=stale_owner,
        )
    stale_observation_owner = _single_component_owner(owner_id=verified.owner_id)
    stale_observation_owner.observation_ids = ["obs_body", "obs_new"]
    stale_observation_owner.selected_observation_ids = ["obs_new"]
    with pytest.raises(UnsafeOwnerMaskError, match="observation|provenance"):
        persist_owner_mask_plan(
            verified,
            tmp_path / "stale_observation",
            owner=stale_observation_owner,
        )
    mutation = execute_owner_inpaint(image, verified, ChangingInpainter())
    assert mutation.owner_bbox_page == verified.owner_bbox_page
    assert mutation.component_geometry_sha256 == verified.component_geometry_sha256


def test_owner_action_mask_is_union_of_owned_glyph_and_line_evidence():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((36, 48, 3), 240, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[5:11, 7:15] = 255
    line = np.zeros(image.shape[:2], dtype=np.uint8)
    line[20:25, 18:36] = 255
    foreign = np.zeros(image.shape[:2], dtype=np.uint8)
    foreign[2:4, 40:44] = 255

    plan = build_owner_mask_plan(
        image,
        _owner(),
        [
            OwnerMaskEvidence(
                evidence_id="glyph_top",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            ),
            OwnerMaskEvidence(
                evidence_id="line_bottom",
                component_id="cmp_body_bottom",
                line_mask=line,
            ),
            OwnerMaskEvidence(
                evidence_id="foreign",
                component_id="cmp_foreign",
                glyph_mask=foreign,
            ),
        ],
    )
    expected = np.maximum(glyph, line)
    np.testing.assert_array_equal(plan.action_mask, expected)
    assert plan.evidence_ids == ("glyph_top", "line_bottom")
    assert int(plan.action_mask[2, 41]) == 0


def test_adjacent_protected_icon_survives_complete_multiline_cleanup():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        build_owner_mask_plan,
        execute_owner_inpaint,
    )

    image = np.full((44, 64, 3), 238, dtype=np.uint8)
    first_line = np.zeros(image.shape[:2], dtype=np.uint8)
    first_line[10:14, 10:26] = 255
    second_line = np.zeros(image.shape[:2], dtype=np.uint8)
    second_line[22:26, 10:30] = 255
    icon = np.zeros(image.shape[:2], dtype=np.uint8)
    icon[14:22, 34:42] = 255
    image[icon > 0] = (16, 32, 180)
    image[first_line > 0] = 18
    image[second_line > 0] = 18

    owner = _single_component_owner(owner_id="owner_with_adjacent_icon")
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="line_0",
                component_id="cmp_body_top",
                glyph_mask=first_line,
                observation_id="obs_body",
                line_index=0,
            ),
            OwnerMaskEvidence(
                evidence_id="line_1",
                component_id="cmp_body_top",
                glyph_mask=second_line,
                observation_id="obs_body",
                line_index=1,
            ),
            OwnerMaskEvidence(
                evidence_id="icon_protection",
                component_id="cmp_icon",
                protected_art_mask=icon,
            ),
        ],
        owner_component_bboxes_page={"cmp_body_top": (6, 6, 46, 30)},
        expected_line_ids=(("obs_body", 0), ("obs_body", 1)),
    )

    class FlatInpainter:
        @staticmethod
        def inpaint(crop, mask, **_kwargs):
            result = crop.copy()
            result[mask > 0] = 238
            return result

    mutation = execute_owner_inpaint(image, plan, FlatInpainter())

    np.testing.assert_array_equal(mutation.result_rgb[icon > 0], image[icon > 0])
    assert mutation.protected_art_changed_pixels == 0


def test_connected_line_art_crossing_support_is_never_authorized():
    source = np.full((32, 48, 3), 245, dtype=np.uint8)
    source[14:17, 8:30] = 12
    source[7:25, 28:31] = 12
    source_glyph = np.zeros(source.shape[:2], dtype=np.uint8)
    source_glyph[14:17, 8:20] = 255

    protected, provenance, confidence = getattr(
        __import__("strip.process_bands", fromlist=["_owner_protected_evidence"]),
        "_owner_protected_evidence",
    )(
        source,
        owner_component_ids={"cmp_text"},
        source_glyph_mask=source_glyph,
        component_bbox_page=(6, 6, 32, 26),
        foreign_component_masks=(),
        explicit_protected_masks=(),
    )

    assert confidence < 1.0
    assert "connected_foreground_crosses_support" in provenance
    assert np.any((protected > 0) & (source_glyph > 0))


def test_changed_pixels_remain_subset_of_action_mask():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan, execute_owner_inpaint

    image = np.full((28, 40, 3), 230, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[8:13, 9:21] = 255
    owner = _single_component_owner(owner_id="owner_changed_subset")
    plan = build_owner_mask_plan(
        image,
        owner,
        [OwnerMaskEvidence("glyph", "cmp_body_top", glyph_mask=glyph)],
        owner_component_bboxes_page={"cmp_body_top": (6, 5, 25, 17)},
    )

    class WholeCropChanger:
        @staticmethod
        def inpaint(crop, _mask, **_kwargs):
            return np.zeros_like(crop)

    mutation = execute_owner_inpaint(image, plan, WholeCropChanger())
    changed = np.asarray(mutation.changed_mask) > 0
    action = np.asarray(mutation.action_mask) > 0
    assert not np.any(changed & ~action)

def test_selected_multiline_observation_authorizes_every_line_polygon():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((48, 64, 3), 240, dtype=np.uint8)
    owner = _single_component_owner(owner_id="owner_multiline_complete")
    line_top = np.zeros(image.shape[:2], dtype=np.uint8)
    line_top[10:13, 12:32] = 255
    line_bottom = np.zeros(image.shape[:2], dtype=np.uint8)
    line_bottom[24:27, 14:34] = 255

    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="line_top",
                component_id="cmp_body_top",
                glyph_mask=line_top,
                observation_id="obs_body",
                line_index=0,
            ),
            OwnerMaskEvidence(
                evidence_id="line_bottom",
                component_id="cmp_body_top",
                glyph_mask=line_bottom,
                observation_id="obs_body",
                line_index=1,
            ),
        ],
        expected_line_ids=(("obs_body", 0), ("obs_body", 1)),
        owner_component_bboxes_page={"cmp_body_top": (8, 6, 40, 32)},
    )

    assert plan.expected_line_ids == (("obs_body", 0), ("obs_body", 1))
    assert plan.covered_line_ids == plan.expected_line_ids
    assert plan.expected_line_polygon_count == 2
    assert plan.covered_line_polygon_count == 2
    assert plan.uncovered_source_ink_pixels == 0
    assert plan.coverage_complete is True


def test_expected_line_identity_missing_before_planner_still_revokes_owner():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((48, 64, 3), 240, dtype=np.uint8)
    owner = _single_component_owner(owner_id="owner_multiline_missing")
    line_top = np.zeros(image.shape[:2], dtype=np.uint8)
    line_top[10:13, 12:32] = 255

    with pytest.raises(UnsafeOwnerMaskError, match="line|coverage|identity"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="line_top",
                    component_id="cmp_body_top",
                    glyph_mask=line_top,
                    observation_id="obs_body",
                    line_index=0,
                )
            ],
            expected_line_ids=(("obs_body", 0), ("obs_body", 1)),
            owner_component_bboxes_page={"cmp_body_top": (8, 6, 40, 32)},
        )

    assert owner.state == "review_required"
    assert owner.route_action == "review_required"
    assert owner.action_mask_ref is None


def test_unselected_material_complete_evidence_is_audited_for_cleanup():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((32, 48, 3), 240, dtype=np.uint8)
    owner = _single_component_owner(owner_id="owner_unselected_material")
    owner.observation_ids.append("obs_unselected_complete")
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[8:11, 9:30] = 255

    with pytest.raises(UnsafeOwnerMaskError, match="selected observation|line|identity"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="unselected_complete",
                    component_id="cmp_body_top",
                    glyph_mask=glyph,
                    observation_id="obs_unselected_complete",
                    line_index=0,
                )
            ],
            expected_line_ids=(("obs_body", 0),),
            owner_component_bboxes_page={"cmp_body_top": (6, 5, 34, 16)},
        )

    assert owner.state == "review_required"


def test_verified_owner_mask_expands_strokes_inside_component_geometry_only():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((30, 40, 3), 240, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[12:14, 18:20] = 255
    owner = _single_component_owner()

    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="thin_glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (14, 8, 24, 18)},
    )

    assert int(plan.action_mask[11, 17]) == 255
    assert int(plan.action_mask[7, 17]) == 0
    assert int(plan.action_mask[18, 17]) == 0


def test_verified_owner_mask_expands_proportionally_for_glow_without_touching_art():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((100, 140, 3), 240, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    for x_start in (40, 60, 80, 100):
        glyph[25:69, x_start : x_start + 2] = 255
    protected = np.zeros(image.shape[:2], dtype=np.uint8)
    protected[40, 32] = 255

    plan = build_owner_mask_plan(
        image,
        _single_component_owner(),
        [
            OwnerMaskEvidence(
                evidence_id="glowing_lines",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            ),
            OwnerMaskEvidence(
                evidence_id="nearby_art",
                component_id="cmp_foreign",
                protected_art_mask=protected,
            ),
        ],
        owner_component_bboxes_page={"cmp_body_top": (30, 17, 110, 77)},
    )

    # A fixed three-pixel radius leaves colored/antialiased glow behind.
    assert int(plan.action_mask[40, 33]) == 255
    assert int(plan.action_mask[40, 30]) == 255
    assert int(plan.action_mask[40, 32]) == 0
    assert int(plan.action_mask[40, 29]) == 0


def test_owner_mask_is_persisted_and_addressable_by_owner_id(tmp_path):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        _safe_owner_ref,
        build_owner_mask_plan,
        load_owner_action_mask,
        persist_owner_mask_plan,
    )

    image = np.full((24, 32, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[8:14, 9:21] = 255
    line = np.zeros(image.shape[:2], dtype=np.uint8)
    line[16:20, 11:24] = 255
    owner = _owner()
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="glyph_top",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            ),
            OwnerMaskEvidence(
                evidence_id="line_bottom",
                component_id="cmp_body_bottom",
                line_mask=line,
            ),
        ],
        owner_component_bboxes_page={
            "cmp_body_top": (9, 8, 21, 14),
            "cmp_body_bottom": (11, 16, 24, 20),
        },
    )

    assert owner.action_mask_ref is None
    assert owner.state == "translated"

    persisted = persist_owner_mask_plan(plan, tmp_path, owner=owner)

    assert plan.owner_id in plan.action_mask_ref
    assert owner.action_mask_ref == plan.action_mask_ref
    assert owner.state == "mask_ready"
    assert persisted == tmp_path / Path(plan.action_mask_ref)
    assert persisted.is_file()
    loaded = cv2.imread(str(persisted), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, plan.action_mask)
    loaded_contract = load_owner_action_mask(
        tmp_path,
        plan.action_mask_ref,
        expected_owner_id=owner.owner_id,
        expected_page_id=owner.page_id,
        expected_source_sha256=plan.source_sha256,
        expected_execution_tile_id=plan.execution_tile_id,
        expected_component_geometry_sha256=plan.component_geometry_sha256,
    )
    np.testing.assert_array_equal(loaded_contract, plan.action_mask)
    assert not loaded_contract.flags.writeable
    with pytest.raises(UnsafeOwnerMaskError, match="source"):
        load_owner_action_mask(
            tmp_path,
            plan.action_mask_ref,
            expected_owner_id=owner.owner_id,
            expected_page_id=owner.page_id,
            expected_source_sha256="0" * 64,
            expected_execution_tile_id=plan.execution_tile_id,
            expected_component_geometry_sha256=plan.component_geometry_sha256,
        )
    with pytest.raises(UnsafeOwnerMaskError, match="executor|tile"):
        load_owner_action_mask(
            tmp_path,
            plan.action_mask_ref,
            expected_owner_id=owner.owner_id,
            expected_page_id=owner.page_id,
            expected_source_sha256=plan.source_sha256,
            expected_execution_tile_id="tile_other",
            expected_component_geometry_sha256=plan.component_geometry_sha256,
        )
    assert persisted.with_name("protected_art_mask.png").is_file()
    assert persisted.with_name("manifest.json").is_file()

    overlapping_protected = plan.protected_art_mask.copy()
    overlap_y, overlap_x = np.argwhere(plan.action_mask > 0)[0]
    overlapping_protected[overlap_y, overlap_x] = 255
    overlapping_protected.setflags(write=False)
    overlap_action_ref, overlap_protected_ref = _safe_owner_ref(
        plan.owner_id,
        plan.page_id,
        plan.source_sha256,
        plan.execution_tile_id,
        plan.action_mask,
        overlapping_protected,
        plan.evidence_ids,
        plan.protected_evidence_ids,
        plan.observation_ids,
        plan.component_action_bboxes_page,
        plan.component_bboxes_page,
        plan.owner_bbox_page,
        plan.component_geometry_sha256,
        plan.component_geometry_verified,
    )
    overlap_plan = replace(
        plan,
        action_mask_ref=overlap_action_ref,
        protected_art_mask=overlapping_protected,
        protected_art_mask_ref=overlap_protected_ref,
    )
    with patch(
        "inpainter.owner_mask._validated_plan_masks",
        return_value=(plan.action_mask, overlapping_protected),
    ):
        persist_owner_mask_plan(overlap_plan, tmp_path)

    with pytest.raises(UnsafeOwnerMaskError, match="overlap|protected"):
        load_owner_action_mask(
            tmp_path,
            overlap_action_ref,
            expected_owner_id=owner.owner_id,
            expected_page_id=owner.page_id,
            expected_source_sha256=plan.source_sha256,
            expected_execution_tile_id=plan.execution_tile_id,
            expected_component_geometry_sha256=plan.component_geometry_sha256,
        )


@pytest.mark.parametrize("invalid_schema_version", (999, True))
def test_existing_owner_mask_with_unsupported_schema_fails_persist_closed(
    tmp_path,
    invalid_schema_version,
):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
        persist_owner_mask_plan,
    )

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 8:18] = 255
    owner = _single_component_owner(owner_id="owner_schema_guard")
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (8, 6, 18, 12)},
    )
    persisted = persist_owner_mask_plan(plan, tmp_path, owner=owner)
    manifest_path = persisted.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = invalid_schema_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    owner.state = "translated"
    owner.action_mask_ref = None

    with pytest.raises(UnsafeOwnerMaskError, match="schema|version|manifest"):
        persist_owner_mask_plan(plan, tmp_path, owner=owner)

    assert owner.state == "review_required"
    assert owner.route_action == "review_required"
    assert owner.action_mask_ref is None


def test_owner_mask_loader_rejects_noncanonical_source_hash(tmp_path):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        _safe_owner_ref,
        build_owner_mask_plan,
        load_owner_action_mask,
        persist_owner_mask_plan,
    )

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 8:18] = 255
    plan = build_owner_mask_plan(
        image,
        _single_component_owner(owner_id="owner_source_hash_guard"),
        [
            OwnerMaskEvidence(
                evidence_id="glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (8, 6, 18, 12)},
    )
    invalid_source_sha256 = "not-a-sha256"
    action_ref, protected_ref = _safe_owner_ref(
        plan.owner_id,
        plan.page_id,
        invalid_source_sha256,
        plan.execution_tile_id,
        plan.action_mask,
        plan.protected_art_mask,
        plan.evidence_ids,
        plan.protected_evidence_ids,
        plan.observation_ids,
        plan.component_action_bboxes_page,
        plan.component_bboxes_page,
        plan.owner_bbox_page,
        plan.component_geometry_sha256,
        plan.component_geometry_verified,
    )
    forged_plan = replace(
        plan,
        source_sha256=invalid_source_sha256,
        action_mask_ref=action_ref,
        protected_art_mask_ref=protected_ref,
    )
    with patch(
        "inpainter.owner_mask._validated_plan_masks",
        return_value=(plan.action_mask, plan.protected_art_mask),
    ):
        persist_owner_mask_plan(forged_plan, tmp_path)

    with pytest.raises(UnsafeOwnerMaskError, match="source|hash|canonical"):
        load_owner_action_mask(
            tmp_path,
            action_ref,
            expected_owner_id=plan.owner_id,
            expected_page_id=plan.page_id,
            expected_source_sha256=invalid_source_sha256,
            expected_execution_tile_id=plan.execution_tile_id,
            expected_component_geometry_sha256=plan.component_geometry_sha256,
        )


@pytest.mark.parametrize("artifact_name", ("action_mask.png", "protected_art_mask.png"))
def test_owner_mask_loader_rejects_nonbinary_png_pixels(tmp_path, artifact_name):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
        load_owner_action_mask,
        persist_owner_mask_plan,
    )

    image = np.full((20, 30, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 7:17] = 255
    protected = np.zeros(image.shape[:2], dtype=np.uint8)
    protected[3:5, 22:25] = 255
    owner = _single_component_owner(owner_id="owner_binary_guard")
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="glyph_and_art",
                component_id="cmp_body_top",
                glyph_mask=glyph,
                protected_art_mask=protected,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (7, 6, 17, 12)},
    )
    persisted = persist_owner_mask_plan(plan, tmp_path)
    artifact_path = persisted.with_name(artifact_name)
    pixels = cv2.imread(str(artifact_path), cv2.IMREAD_GRAYSCALE)
    assert pixels is not None
    y, x = np.argwhere(pixels == 255)[0]
    pixels[y, x] = 1
    assert cv2.imwrite(str(artifact_path), pixels)

    with pytest.raises(UnsafeOwnerMaskError, match="binary|canonical|pixel"):
        persist_owner_mask_plan(plan, tmp_path, owner=owner)
    assert owner.state == "review_required"

    with pytest.raises(UnsafeOwnerMaskError, match="binary|canonical|pixel"):
        load_owner_action_mask(
            tmp_path,
            plan.action_mask_ref,
            expected_owner_id=plan.owner_id,
            expected_page_id=plan.page_id,
            expected_source_sha256=plan.source_sha256,
            expected_execution_tile_id=plan.execution_tile_id,
            expected_component_geometry_sha256=plan.component_geometry_sha256,
        )


def test_owner_action_mask_ref_roundtrips_with_owner_graph():
    from ownership.model import OwnerGraph

    owner = _owner()
    owner.action_mask_ref = "owner_masks/own_page_001_body/action_mask.png"
    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_001",
        run_id="run-owner-mask",
        origin_execution_id="execution-owner-mask",
        page_source_sha256="a" * 64,
        components=[],
        observations=[],
        owners=[owner],
        projections=[],
        component_dispositions=[],
    )

    loaded = OwnerGraph.from_dict(graph.to_dict())

    assert loaded.owners[0].action_mask_ref == owner.action_mask_ref


def test_every_owner_component_requires_safe_positive_mask_evidence():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((24, 32, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[8:14, 9:21] = 255
    owner = _owner()

    with pytest.raises(UnsafeOwnerMaskError, match="component|coverage|evidence"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="glyph_top_only",
                    component_id="cmp_body_top",
                    glyph_mask=glyph,
                )
            ],
        )

    assert owner.state == "review_required"
    assert owner.route_action == "review_required"


@pytest.mark.parametrize("foreign_overlap_pixels", (20, 1))
def test_owner_component_cannot_be_partially_authorized(foreign_overlap_pixels):
    from inpainter.owner_mask import UnsafeOwnerMaskError, build_owner_mask_plan

    image = np.full((24, 32, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:16, 8:18] = 255
    foreign = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph_pixels = np.argwhere(glyph > 0)
    for y, x in glyph_pixels[:foreign_overlap_pixels]:
        foreign[y, x] = 255
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="coverage|protected|partial"):
        build_owner_mask_plan(
            image,
            owner,
            [
                {
                    "evidence_id": "owned_glyph",
                    "component_id": "cmp_body_top",
                    "glyph_mask": glyph,
                },
                {
                    "evidence_id": "foreign_glyph",
                    "component_id": "cmp_foreign",
                    "glyph_mask": foreign,
                },
            ],
        )

    assert owner.state == "review_required"
    assert owner.route_action == "review_required"


def test_mixed_owner_evidence_is_rejected_instead_of_authorizing_foreign_pixels():
    from inpainter.owner_mask import UnsafeOwnerMaskError, build_owner_mask_plan

    image = np.full((20, 30, 3), 220, dtype=np.uint8)
    mixed = np.zeros(image.shape[:2], dtype=np.uint8)
    mixed[5:8, 5:9] = 255
    mixed[12:15, 21:25] = 255
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="mixed|foreign|owner"):
        build_owner_mask_plan(
            image,
            owner,
            [
                {
                    "evidence_id": "mixed_components",
                    "component_ids": ["cmp_body_top", "cmp_foreign"],
                    "glyph_mask": mixed,
                }
            ],
        )

    assert owner.state == "review_required"


def test_foreign_glyph_evidence_becomes_negative_protection_for_owner():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((20, 30, 3), 220, dtype=np.uint8)
    owned = np.zeros(image.shape[:2], dtype=np.uint8)
    owned[5:12, 5:16] = 255
    foreign = np.zeros(image.shape[:2], dtype=np.uint8)
    foreign[12:18, 18:26] = 255

    plan = build_owner_mask_plan(
        image,
        _single_component_owner(),
        [
            OwnerMaskEvidence(
                evidence_id="owned",
                component_id="cmp_body_top",
                glyph_mask=owned,
            ),
            OwnerMaskEvidence(
                evidence_id="foreign",
                component_id="cmp_foreign",
                glyph_mask=foreign,
            ),
        ],
    )

    assert plan.action_mask[9, 14] == 255
    assert plan.protected_art_mask[14, 20] == 255


def test_owner_mask_reference_is_collision_resistant_for_lossy_path_ids():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 8:18] = 255
    evidence = [
        OwnerMaskEvidence(
            evidence_id="glyph",
            component_id="cmp_body_top",
            glyph_mask=glyph,
        )
    ]

    slash = build_owner_mask_plan(
        image,
        _single_component_owner(owner_id="owner/a"),
        evidence,
    )
    underscore = build_owner_mask_plan(
        image,
        _single_component_owner(owner_id="owner_a"),
        evidence,
    )

    assert slash.action_mask_ref != underscore.action_mask_ref

    same_bytes = np.zeros(12, dtype=np.uint8)
    same_bytes[2] = 255
    shape_a = build_owner_mask_plan(
        np.full((2, 6, 3), 220, dtype=np.uint8),
        _single_component_owner(owner_id="owner_shape"),
        [
            OwnerMaskEvidence(
                evidence_id="shape_a",
                component_id="cmp_body_top",
                glyph_mask=same_bytes.reshape(2, 6),
            )
        ],
    )
    shape_b = build_owner_mask_plan(
        np.full((3, 4, 3), 220, dtype=np.uint8),
        _single_component_owner(owner_id="owner_shape"),
        [
            OwnerMaskEvidence(
                evidence_id="shape_b",
                component_id="cmp_body_top",
                glyph_mask=same_bytes.reshape(3, 4),
            )
        ],
    )

    assert shape_a.action_mask_ref != shape_b.action_mask_ref

    source_a = np.full((18, 26, 3), 220, dtype=np.uint8)
    source_b = source_a.copy()
    source_b[0, 0] = 219
    source_plan_a = build_owner_mask_plan(
        source_a,
        _single_component_owner(owner_id="owner_source_bound"),
        evidence,
    )
    source_plan_b = build_owner_mask_plan(
        source_b,
        _single_component_owner(owner_id="owner_source_bound"),
        evidence,
    )
    assert source_plan_a.action_mask_ref != source_plan_b.action_mask_ref


def test_invalid_mask_evidence_always_moves_owner_to_review():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    wrong_shape = np.full((17, 26), 255, dtype=np.uint8)
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="shape"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="wrong_shape",
                    component_id="cmp_body_top",
                    glyph_mask=wrong_shape,
                )
            ],
        )

    assert owner.state == "review_required"
    assert owner.route_action == "review_required"

    missing_executor = _single_component_owner()
    missing_executor.execution_tile_id = None
    valid = np.zeros(image.shape[:2], dtype=np.uint8)
    valid[6:12, 8:18] = 255
    with pytest.raises(UnsafeOwnerMaskError, match="executor|tile"):
        build_owner_mask_plan(
            image,
            missing_executor,
            [
                OwnerMaskEvidence(
                    evidence_id="valid_glyph",
                    component_id="cmp_body_top",
                    glyph_mask=valid,
                )
            ],
        )
    assert missing_executor.state == "review_required"

    malformed_owner = _single_component_owner()
    malformed = np.full(image.shape[:2], "not-a-pixel", dtype=object)
    with pytest.raises(UnsafeOwnerMaskError, match="numeric|dtype|mask"):
        build_owner_mask_plan(
            image,
            malformed_owner,
            [
                OwnerMaskEvidence(
                    evidence_id="malformed",
                    component_id="cmp_body_top",
                    glyph_mask=malformed,
                )
            ],
        )
    assert malformed_owner.state == "review_required"

    malformed_polygon_owner = _single_component_owner()
    with pytest.raises(UnsafeOwnerMaskError, match="polygon|evidence"):
        build_owner_mask_plan(
            image,
            malformed_polygon_owner,
            [
                {
                    "evidence_id": "mixed_polygon",
                    "component_id": "cmp_body_top",
                    "glyph_polygons": [
                        [[4, 4], [12, 4], [8, 12]],
                        ["malformed"],
                    ],
                }
            ],
        )
    assert malformed_polygon_owner.state == "review_required"

    unknown_record_owner = _single_component_owner()
    with pytest.raises(UnsafeOwnerMaskError, match="record|evidence"):
        build_owner_mask_plan(image, unknown_record_owner, [object()])
    assert unknown_record_owner.state == "review_required"


@pytest.mark.parametrize(
    "forbidden_route",
    (
        "do_not_inpaint",
        "review_inpaint_forbidden",
        "translate_render_only",
        " TRANSLATE_INPAINT_RENDER ",
        "InPaint_Only",
    ),
)
def test_owner_mask_requires_a_canonical_inpaint_route(tmp_path, forbidden_route):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
        persist_owner_mask_plan,
    )

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 8:18] = 255
    evidence = [
        OwnerMaskEvidence(
            evidence_id="glyph",
            component_id="cmp_body_top",
            glyph_mask=glyph,
        )
    ]

    rejected_owner = _single_component_owner(owner_id="owner_rejected_route")
    rejected_owner.route_action = forbidden_route
    with pytest.raises(UnsafeOwnerMaskError, match="route|inpaint"):
        build_owner_mask_plan(image, rejected_owner, evidence)
    assert rejected_owner.state == "review_required"
    assert rejected_owner.route_action == "review_required"

    persisted_owner = _single_component_owner(owner_id="owner_persisted_route")
    plan = build_owner_mask_plan(
        image,
        persisted_owner,
        evidence,
        owner_component_bboxes_page={"cmp_body_top": (8, 6, 18, 12)},
    )
    persisted_owner.route_action = forbidden_route
    with pytest.raises(UnsafeOwnerMaskError, match="identity|state|route"):
        persist_owner_mask_plan(plan, tmp_path, owner=persisted_owner)
    assert persisted_owner.state == "review_required"
    assert persisted_owner.route_action == "review_required"


def test_owner_mask_ref_includes_evidence_provenance():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 8:18] = 255

    plan_a = build_owner_mask_plan(
        image,
        _single_component_owner(owner_id="owner_provenance"),
        [
            OwnerMaskEvidence(
                evidence_id="glyph_detector_a",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
    )
    plan_b = build_owner_mask_plan(
        image,
        _single_component_owner(owner_id="owner_provenance"),
        [
            OwnerMaskEvidence(
                evidence_id="glyph_detector_b",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
    )

    assert plan_a.action_mask_ref != plan_b.action_mask_ref
    assert plan_a.protected_art_mask_ref != plan_b.protected_art_mask_ref


def test_page_sized_positive_mask_is_rejected_as_overbroad():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((20, 30, 3), 220, dtype=np.uint8)
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="overbroad|page"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="page_fill",
                    component_id="cmp_body_top",
                    glyph_mask=np.full(image.shape[:2], 255, dtype=np.uint8),
                )
            ],
        )

    assert owner.state == "review_required"


def test_verified_horizontal_ocr_line_may_use_dense_source_support():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((100, 320, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[40:57, 45:275] = 255
    glyph[43:47, 70:90] = 0
    owner = _single_component_owner(owner_id="owner_dense_verified_line")

    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="dense_verified_line",
                component_id="cmp_body_top",
                glyph_mask=glyph,
                observation_id="obs_body",
                line_index=0,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (45, 40, 275, 57)},
        expected_line_ids=(("obs_body", 0),),
        owner_render_geometry_sha256="a" * 64,
    )

    assert plan.coverage_complete
    assert plan.component_geometry_verified
    assert plan.component_action_bboxes_page == (
        ("cmp_body_top", (45, 40, 275, 57)),
    )


@pytest.mark.parametrize("axis", ("horizontal", "vertical"))
def test_owner_mask_rejects_opposite_edge_spanning_bands(axis):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    if axis == "horizontal":
        glyph[33:68, :] = 255
    else:
        glyph[:, 33:68] = 255
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="overbroad|band|page"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id=f"{axis}_band",
                    component_id="cmp_body_top",
                    glyph_mask=glyph,
                )
            ],
        )

    assert owner.state == "review_required"


def test_owner_mask_rejects_large_solid_bbox_disguised_as_glyph_evidence():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    bbox_fill = np.zeros(image.shape[:2], dtype=np.uint8)
    bbox_fill[25:55, 20:60] = 255
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="overbroad|solid|bbox"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="bbox_disguised_as_glyphs",
                    component_id="cmp_body_top",
                    glyph_mask=bbox_fill,
                )
            ],
        )

    assert owner.state == "review_required"


@pytest.mark.parametrize(
    "pattern",
    ("near_solid", "striped", "two_on_one_off", "checkerboard"),
)
def test_owner_mask_rejects_near_solid_bbox_like_rasters(pattern):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[20:60, 20:60] = 255
    if pattern == "near_solid":
        glyph[20:60:4, 20:60:4] = 0
    elif pattern == "striped":
        glyph[20:60:5, 20:60] = 0
    elif pattern == "two_on_one_off":
        glyph[20:60:3, 20:60] = 0
    else:
        yy, xx = np.indices((40, 40))
        glyph[20:60, 20:60] = np.where((yy + xx) % 2 == 0, 255, 0).astype(
            np.uint8
        )
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="overbroad|solid|bbox"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id=pattern,
                    component_id="cmp_body_top",
                    glyph_mask=glyph,
                )
            ],
        )

    assert owner.state == "review_required"


def test_large_sparse_stroke_raster_is_not_confused_with_a_filled_bbox():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    strokes = np.zeros(image.shape[:2], dtype=np.uint8)
    strokes[20:60:4, 20:100] = 255
    strokes[20:60, 20:100:8] = 255
    owner = _single_component_owner()

    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="sparse_strokes",
                component_id="cmp_body_top",
                line_mask=strokes,
            )
        ],
    )

    np.testing.assert_array_equal(plan.action_mask, strokes)


def test_trusted_dense_single_glyph_raster_is_not_treated_as_bbox_fallback():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[25:65, 46:54] = 255
    owner = _single_component_owner()

    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="selected_single_glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
                observation_id="obs_body",
            )
        ],
    )

    np.testing.assert_array_equal(plan.action_mask, glyph)


def test_dense_line_geometry_cannot_impersonate_a_single_glyph():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    line = np.zeros(image.shape[:2], dtype=np.uint8)
    line[25:65, 45:55] = 255
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="overbroad|line|bbox"):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="dense_line_bbox",
                    component_id="cmp_body_top",
                    line_mask=line,
                    observation_id="obs_body",
                )
            ],
        )

    assert owner.state == "review_required"


def test_line_polygon_without_stroke_raster_cannot_authorize_cleanup():
    from inpainter.owner_mask import UnsafeOwnerMaskError, build_owner_mask_plan

    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="polygon|raster|evidence"):
        build_owner_mask_plan(
            image,
            owner,
            [
                {
                    "evidence_id": "line_bbox",
                    "component_id": "cmp_body_top",
                    "line_polygon": [[20, 20], [100, 20], [100, 60], [20, 60]],
                }
            ],
        )

    assert owner.state == "review_required"


def test_owner_mask_and_mutation_bind_logical_page_render_geometry(tmp_path):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
        execute_owner_inpaint,
        persist_owner_mask_plan,
    )

    class ChangingInpainter:
        engine_name = "fixture"

        def inpaint(self, image, mask, **_kwargs):
            result = image.copy()
            result[mask > 0] = 17
            return result

    image = np.full((40, 40, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[25:30, 12:24] = 255
    owner = _single_component_owner()
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="page_glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (12, 25, 24, 30)},
        owner_render_geometry_sha256="b" * 64,
    )

    assert plan.coordinate_space == "logical_page"
    assert plan.owner_render_geometry_sha256 == "b" * 64
    persisted = persist_owner_mask_plan(plan, tmp_path)
    manifest = json.loads(persisted.with_name("manifest.json").read_text(encoding="utf-8"))
    assert manifest["coordinate_space"] == "logical_page"
    assert manifest["owner_render_geometry_sha256"] == "b" * 64
    mutation = execute_owner_inpaint(image, plan, ChangingInpainter())
    assert mutation.coordinate_space == "logical_page"
    assert mutation.owner_render_geometry_sha256 == "b" * 64
    assert mutation.result_rgb.shape == image.shape
    assert mutation.action_mask.shape == image.shape[:2]

    with pytest.raises(UnsafeOwnerMaskError, match="coordinate|page"):
        execute_owner_inpaint(
            image,
            replace(plan, coordinate_space="tile"),
            ChangingInpainter(),
        )


def test_owner_mask_and_mutation_declare_page_coordinate_space(tmp_path):
    """Preserve the historical nodeid for canonical logical-page geometry."""

    test_owner_mask_and_mutation_bind_logical_page_render_geometry(tmp_path)


def test_owner_engine_receives_bounded_crop_but_mutation_remains_page_space():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        build_owner_mask_plan,
        execute_owner_inpaint,
    )

    class CropSpyInpainter:
        engine_name = "crop_spy"

        def __init__(self):
            self.received_shape = None
            self.received_mask_shape = None

        def inpaint(self, image, mask, **_kwargs):
            self.received_shape = image.shape
            self.received_mask_shape = mask.shape
            result = image.copy()
            result[mask > 0] = 17
            return result

    image = np.full((4000, 320, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[1980:2020:3, 120:200] = 255
    glyph[1980:2020, 120:200:8] = 255
    owner = _single_component_owner()
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="long_page_glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={
            "cmp_body_top": (120, 1980, 200, 2020)
        },
    )
    spy = CropSpyInpainter()

    mutation = execute_owner_inpaint(image, plan, spy)

    assert spy.received_shape is not None
    assert spy.received_shape[0] < 400
    assert spy.received_shape[1] < image.shape[1]
    assert spy.received_mask_shape == spy.received_shape[:2]
    assert mutation.coordinate_space == "logical_page"
    assert mutation.result_rgb.shape == image.shape
    assert mutation.action_mask.shape == image.shape[:2]
    np.testing.assert_array_equal(
        mutation.result_rgb[plan.action_mask == 0],
        image[plan.action_mask == 0],
    )


def test_owner_inpaint_rejects_color_outlier_on_uniform_context():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        build_owner_mask_plan,
        execute_owner_inpaint,
    )

    class BlackArtifactInpainter:
        engine_name = "aot_fixture"

        def inpaint(self, image, mask, **_kwargs):
            result = image.copy()
            result[mask > 0] = 0
            return result

    image = np.full((96, 180, 3), 248, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.putText(
        glyph,
        "TITLE",
        (35, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        255,
        3,
        cv2.LINE_AA,
    )
    image[glyph > 0] = 12
    owner = _single_component_owner()
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="uniform_title_glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (30, 24, 150, 60)},
    )

    mutation = execute_owner_inpaint(image, plan, BlackArtifactInpainter())

    changed_pixels = mutation.result_rgb[mutation.action_mask > 0]
    assert float(changed_pixels.mean()) > 230.0
    assert mutation.engine == "aot_fixture+context_guard_median"


def test_owner_inpaint_rejects_localized_color_artifact_on_uniform_dark_context():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        build_owner_mask_plan,
        execute_owner_inpaint,
    )

    class LocalizedRedArtifactInpainter:
        engine_name = "aot_fixture"

        def inpaint(self, image, mask, **_kwargs):
            result = image.copy()
            result[mask > 0] = 0
            ys, xs = np.nonzero(mask)
            artifact_pixels = max(1, int(len(xs) * 0.05))
            result[ys[-artifact_pixels:], xs[-artifact_pixels:]] = (180, 0, 0)
            return result

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[84:105:4, 90:190] = 255
    image[glyph > 0] = 245
    # A source glow can cross the detector component boundary.  A Telea
    # fallback samples this fringe and smears it back across the cleaned text.
    image[74:112, 196:212] = (180, 0, 0)
    owner = _single_component_owner()
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="dark_burst_glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (76, 74, 204, 112)},
    )

    mutation = execute_owner_inpaint(image, plan, LocalizedRedArtifactInpainter())

    changed_pixels = mutation.result_rgb[mutation.action_mask > 0]
    assert int(changed_pixels[:, 0].max()) < 64
    assert mutation.engine == "aot_fixture+context_guard_median"


def test_union_overreach_and_nearly_fully_protected_component_fail_closed():
    from inpainter.owner_mask import UnsafeOwnerMaskError, build_owner_mask_plan

    image = np.full((20, 20, 3), 220, dtype=np.uint8)
    quadrant_records = []
    for index, (y1, y2, x1, x2) in enumerate(
        ((0, 10, 0, 10), (0, 10, 10, 20), (10, 20, 0, 10), (10, 20, 10, 20))
    ):
        quadrant = np.zeros(image.shape[:2], dtype=np.uint8)
        quadrant[y1:y2, x1:x2] = 255
        quadrant_records.append(
            {
                "evidence_id": f"quadrant_{index}",
                "component_id": "cmp_body_top",
                "glyph_mask": quadrant,
            }
        )
    union_owner = _single_component_owner()

    with pytest.raises(UnsafeOwnerMaskError, match="overbroad|page"):
        build_owner_mask_plan(image, union_owner, quadrant_records)
    assert union_owner.state == "review_required"

    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[5:15, 5:15] = 255
    protected = glyph.copy()
    protected[5, 5] = 0
    coverage_owner = _single_component_owner()
    with pytest.raises(UnsafeOwnerMaskError, match="coverage|protected"):
        build_owner_mask_plan(
            image,
            coverage_owner,
            [
                {
                    "evidence_id": "nearly_hidden",
                    "component_id": "cmp_body_top",
                    "glyph_mask": glyph,
                    "protected_art_mask": protected,
                }
            ],
        )
    assert coverage_owner.state == "review_required"


def test_mask_pair_publication_is_atomic_before_owner_reference(tmp_path):
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
        persist_owner_mask_plan,
    )

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 8:18] = 255
    owner = _single_component_owner()
    plan = build_owner_mask_plan(
        image,
        owner,
        [
            OwnerMaskEvidence(
                evidence_id="glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (8, 6, 18, 12)},
    )

    with (
        patch(
            "inpainter.owner_mask.cv2.imwrite",
            side_effect=[True, False],
        ),
        pytest.raises(OSError, match="persist"),
    ):
        persist_owner_mask_plan(plan, tmp_path, owner=owner)

    assert not (tmp_path / Path(plan.action_mask_ref)).exists()
    assert owner.action_mask_ref is None
    assert owner.state == "review_required"

    traversal_owner = _single_component_owner()
    traversal_plan = replace(
        plan,
        action_mask_ref="../escape/action_mask.png",
        protected_art_mask_ref="../escape/protected_art_mask.png",
    )
    with pytest.raises(UnsafeOwnerMaskError, match="escape|reference"):
        persist_owner_mask_plan(
            traversal_plan,
            tmp_path,
            owner=traversal_owner,
        )
    assert traversal_owner.state == "review_required"

    forged_owner = _single_component_owner()
    forged_plan = replace(
        plan,
        action_mask_ref="owner_masks/forged/action_mask.png",
        protected_art_mask_ref="owner_masks/forged/protected_art_mask.png",
    )
    with pytest.raises(UnsafeOwnerMaskError, match="reference|identity|content"):
        persist_owner_mask_plan(forged_plan, tmp_path, owner=forged_owner)
    assert forged_owner.state == "review_required"
    assert not (tmp_path / "owner_masks" / "forged").exists()


def test_noop_owner_inpaint_is_rejected_and_successful_mutation_is_immutable():
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
        execute_owner_inpaint,
    )

    class NoopInpainter:
        def inpaint(self, image, _mask, **_kwargs):
            return image.copy()

    class ChangingInpainter:
        def inpaint(self, image, mask, **_kwargs):
            result = image.copy()
            result[mask > 0] = 17
            return result

    class FloatInpainter:
        def inpaint(self, image, mask, **_kwargs):
            result = image.astype(np.float32)
            result[mask > 0] = 17.5
            return result

    image = np.full((18, 26, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:12, 8:18] = 255
    plan = build_owner_mask_plan(
        image,
        _single_component_owner(),
        [
            OwnerMaskEvidence(
                evidence_id="glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            )
        ],
        owner_component_bboxes_page={"cmp_body_top": (8, 6, 18, 12)},
    )

    with pytest.raises(UnsafeOwnerMaskError, match="no pixels|no-op|unchanged"):
        execute_owner_inpaint(image, plan, NoopInpainter())

    wrong_source = image.copy()
    wrong_source[0, 0] = 219
    with pytest.raises(UnsafeOwnerMaskError, match="source hash|input image"):
        execute_owner_inpaint(wrong_source, plan, ChangingInpainter())

    forged_plan = replace(
        plan,
        action_mask_ref="owner_masks/forged/action_mask.png",
    )
    with pytest.raises(UnsafeOwnerMaskError, match="reference|identity"):
        execute_owner_inpaint(image, forged_plan, ChangingInpainter())

    with pytest.raises(UnsafeOwnerMaskError, match="RGB uint8|dtype"):
        execute_owner_inpaint(image, plan, FloatInpainter())

    mutation = execute_owner_inpaint(image, plan, ChangingInpainter())

    assert mutation.residual_verified is True
    assert mutation.residual_score == pytest.approx(0.8)
    assert mutation.residual_score > mutation.residual_threshold
    assert mutation.residual_method == "detect_residual_text.v1"
    assert len(mutation.residual_evidence_sha256) == 64
    assert not mutation.result_rgb.flags.writeable
    assert not mutation.action_mask.flags.writeable
    assert not mutation.protected_art_mask.flags.writeable
    assert not mutation.changed_mask.flags.writeable


def test_mask_ready_owner_without_action_ref_is_invalid():
    from ownership.model import OwnerGraph

    owner = _owner()
    owner.state = "mask_ready"
    owner.action_mask_ref = None
    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_001",
        run_id="run-owner-mask",
        origin_execution_id="execution-owner-mask",
        page_source_sha256="a" * 64,
        components=[],
        observations=[],
        owners=[owner],
        projections=[],
        component_dispositions=[],
    )

    assert "owner_action_mask_missing" in {
        violation.code for violation in graph.validate()
    }

    owner.action_mask_ref = "owner_masks/other_owner--000000000000/hash/action_mask.png"
    assert "owner_action_mask_owner_mismatch" in {
        violation.code for violation in graph.validate()
    }
