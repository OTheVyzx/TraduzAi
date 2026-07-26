"""Independent page-global source text discovery contracts."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.discovery import (  # noqa: E402
    DetectorRegion,
    GlyphCandidate,
    _attached_to_large_visual_edge,
    _group_textlike_blobs,
    _stroke_primitive_records,
    discover_source_text_components,
)
from ownership.reconcile import build_page_owner_graph  # noqa: E402


def _overlap_ratio(left, right) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / float(right_area)


def _synthetic_text_surface(kind: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((220, 420, 3), 245, dtype=np.uint8)
    color = (18, 18, 18)
    if kind == "balloon":
        cv2.ellipse(image, (210, 110), (180, 80), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (210, 110), (180, 80), 0, 0, 360, (35, 35, 35), 2)
    elif kind == "card":
        image[:, :] = (238, 196, 72)
        cv2.rectangle(image, (35, 25), (385, 195), (255, 220, 105), -1)
        cv2.rectangle(image, (35, 25), (385, 195), (110, 78, 12), 3)
    elif kind == "dark_panel":
        image[:, :] = (18, 22, 35)
        color = (246, 246, 250)
    else:  # pragma: no cover - fixture guard
        raise ValueError(kind)
    cv2.putText(image, "MISSING TEXT", (92, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA)
    return image, (88, 96, 334, 130)


@pytest.mark.parametrize("surface", ["balloon", "card", "dark_panel"])
def test_discovers_textlike_component_omitted_by_primary_ocr(surface: str) -> None:
    image, expected_text_bbox = _synthetic_text_surface(surface)

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert components
    assert any(_overlap_ratio(component.bbox_page, expected_text_bbox) >= 0.45 for component in components)
    assert any("glyph_scan" in component.detector_sources for component in components)


def test_discovery_does_not_depend_on_accepted_text_layers() -> None:
    signature = inspect.signature(discover_source_text_components)
    forbidden = {"texts", "accepted_texts", "text_layers", "ocr_result"}
    assert forbidden.isdisjoint(signature.parameters)

    image, _ = _synthetic_text_surface("balloon")
    original = image.copy()
    first = discover_source_text_components(image, page_id="page_001", detector_regions=[])
    second = discover_source_text_components(image, page_id="page_001", detector_regions=[])
    assert first == second
    assert np.array_equal(image, original)


def test_plain_panel_edges_are_not_promoted_to_text_components() -> None:
    image = np.full((220, 420, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (35, 25), (385, 195), (35, 35, 35), 3)
    cv2.ellipse(image, (210, 110), (120, 60), 0, 0, 360, (70, 70, 70), 3)

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert components == []


def test_periodic_hatching_without_text_is_not_promoted() -> None:
    height, width = 1600, 1000
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for row in range(80, 1500, 35):
        for column in range(40, 940, 28):
            cv2.line(
                image,
                (column, row),
                (column + 8 + (row // 35) % 7, row + 16),
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert components == []


def test_decorative_circle_field_is_not_promoted_as_text() -> None:
    rng = np.random.default_rng(20260725)
    image = np.full((800, 600, 3), 248, dtype=np.uint8)
    for _ in range(120):
        center = (int(rng.integers(20, 580)), int(rng.integers(20, 780)))
        radius = int(rng.integers(3, 13))
        shade = int(rng.integers(15, 150))
        cv2.circle(image, center, radius, (shade, shade, shade), 2, cv2.LINE_AA)

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert components == []


def test_random_art_strokes_are_not_promoted_as_text() -> None:
    rng = np.random.default_rng(77)
    image = np.full((900, 640, 3), 245, dtype=np.uint8)
    for _ in range(500):
        x = int(rng.integers(15, 625))
        y = int(rng.integers(15, 885))
        dx = int(rng.integers(-16, 17))
        dy = int(rng.integers(-16, 17))
        cv2.line(image, (x, y), (x + dx, y + dy), (35, 35, 35), 1, cv2.LINE_AA)

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert components == []


def test_single_slender_ellipse_is_not_promoted_as_a_short_word() -> None:
    image = np.full((220, 420, 3), 245, dtype=np.uint8)
    cv2.ellipse(
        image,
        (210, 110),
        (22, 5),
        113,
        0,
        360,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert components == []


def test_unrelated_distant_art_does_not_change_a_valid_word_candidate() -> None:
    image = np.full((1200, 800, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "NO",
        (330, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (12, 12, 12),
        3,
        cv2.LINE_AA,
    )
    expected = (325, 140, 410, 190)
    baseline = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    rng = np.random.default_rng(44)
    decorated = image.copy()
    for _ in range(80):
        center = (int(rng.integers(30, 770)), int(rng.integers(320, 1160)))
        radius = int(rng.integers(4, 18))
        cv2.circle(decorated, center, radius, (45, 45, 45), 2, cv2.LINE_AA)
    with_art = discover_source_text_components(
        decorated,
        page_id="page_001",
        detector_regions=[],
    )

    assert any(_overlap_ratio(item.bbox_page, expected) >= 0.45 for item in baseline)
    assert any(_overlap_ratio(item.bbox_page, expected) >= 0.45 for item in with_art)


@pytest.mark.parametrize(
    ("background", "foreground"),
    [
        ((180, 180, 180), (165, 165, 165)),
        # OpenCV maps both RGB colours to grayscale value 76.  Discovery must
        # therefore use chroma rather than a hidden luminance assumption.
        ((255, 0, 0), (0, 129, 0)),
    ],
)
def test_local_multichannel_contrast_recovers_text_without_luma_assumptions(
    background: tuple[int, int, int],
    foreground: tuple[int, int, int],
) -> None:
    image = np.full((220, 520, 3), background, dtype=np.uint8)
    cv2.putText(
        image,
        "MISSING TEXT",
        (90, 125),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        foreground,
        2,
        cv2.LINE_AA,
    )
    expected = (84, 94, 350, 132)

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert any(_overlap_ratio(item.bbox_page, expected) >= 0.45 for item in components)


def test_large_art_shape_is_not_promoted_as_a_page_spanning_text_component() -> None:
    image = np.full((851, 800, 3), 238, dtype=np.uint8)
    body = np.array(
        [(250, 0), (540, 30), (790, 240), (710, 835), (330, 780), (225, 350)],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [body], (42, 48, 62))
    for offset in range(0, 180, 18):
        cv2.line(image, (310 + offset, 120), (260 + offset, 720), (95, 100, 112), 5)

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert all(
        (item.bbox_page[2] - item.bbox_page[0])
        * (item.bbox_page[3] - item.bbox_page[1])
        < image.shape[0] * image.shape[1] * 0.18
        for item in components
    )


def test_small_candidate_attached_to_page_spanning_art_edge_is_rejected() -> None:
    strokes = np.zeros((220, 420), dtype=np.uint8)
    cv2.line(strokes, (20, 10), (390, 205), 255, 5, cv2.LINE_8)
    cv2.rectangle(strokes, (198, 121), (202, 125), 255, -1)
    candidate = (198, 102, 220, 126)
    records, labels = _stroke_primitive_records(strokes)

    assert _attached_to_large_visual_edge(records, labels, candidate)

    isolated = np.zeros_like(strokes)
    cv2.putText(
        isolated,
        "NO",
        (185, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        255,
        2,
        cv2.LINE_AA,
    )
    isolated_records, isolated_labels = _stroke_primitive_records(isolated)
    assert not _attached_to_large_visual_edge(
        isolated_records,
        isolated_labels,
        (180, 90, 240, 135),
    )


def test_broad_art_blob_cannot_swallow_a_tight_text_line_by_containment() -> None:
    broad_art = ((28, 0, 585, 290), 3)
    tight_text = ((317, 156, 522, 211), 9)

    groups = _group_textlike_blobs([broad_art, tight_text])

    assert len(groups) == 2
    assert {tuple(group[0][0]) for group in groups} == {
        broad_art[0],
        tight_text[0],
    }


def test_multiscale_text_proposals_still_merge_when_broad_proposal_is_textlike() -> None:
    broad_text = ((20, 20, 380, 120), 18)
    tight_text = ((140, 45, 200, 85), 4)

    groups = _group_textlike_blobs([broad_text, tight_text])

    assert len(groups) == 1


def test_page_wide_source_line_is_not_clipped_or_discarded() -> None:
    image = np.full((220, 1000, 3), 252, dtype=np.uint8)
    phrase = "MISSING SOURCE TEXT MUST STAY WHOLE"
    scale = 1.35
    thickness = 2
    text_size = cv2.getTextSize(
        phrase,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )[0]
    origin = ((image.shape[1] - text_size[0]) // 2, 125)
    cv2.putText(
        image,
        phrase,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (12, 12, 12),
        thickness,
        cv2.LINE_AA,
    )
    expected = (
        origin[0] - 4,
        origin[1] - text_size[1] - 4,
        origin[0] + text_size[0] + 4,
        origin[1] + 5,
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )
    matching = [
        item
        for item in components
        if _overlap_ratio(item.bbox_page, expected) >= 0.80
    ]

    assert len(matching) == 1
    match_area = (
        (matching[0].bbox_page[2] - matching[0].bbox_page[0])
        * (matching[0].bbox_page[3] - matching[0].bbox_page[1])
    )
    expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
    assert match_area <= expected_area * 1.8


@pytest.mark.parametrize("word", ["WAIT", "NO", "HURRY!", "TH-THERE", "SOMETHING"])
def test_single_word_visual_text_is_discovered_without_ocr(word: str) -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    origin = ((image.shape[1] - text_size[0]) // 2, (image.shape[0] + text_size[1]) // 2)
    cv2.putText(
        image,
        word,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    expected = (
        origin[0] - 4,
        origin[1] - text_size[1] - 4,
        origin[0] + text_size[0] + 4,
        origin[1] + 5,
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert any(_overlap_ratio(component.bbox_page, expected) >= 0.45 for component in components)


@pytest.mark.parametrize("angle", [30, 45, 90])
def test_rotated_visual_text_is_discovered_without_ocr(angle: int) -> None:
    image, expected = _synthetic_text_surface("dark_panel")
    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(18, 22, 35),
    )
    corners = np.array(
        [
            [expected[0], expected[1], 1.0],
            [expected[2], expected[1], 1.0],
            [expected[2], expected[3], 1.0],
            [expected[0], expected[3], 1.0],
        ],
        dtype=np.float32,
    )
    projected = corners @ matrix.T
    rotated_expected = (
        int(np.floor(projected[:, 0].min())),
        int(np.floor(projected[:, 1].min())),
        int(np.ceil(projected[:, 0].max())),
        int(np.ceil(projected[:, 1].max())),
    )

    components = discover_source_text_components(
        rotated,
        page_id="page_001",
        detector_regions=[],
    )

    assert any(
        _overlap_ratio(component.bbox_page, rotated_expected) >= 0.30
        for component in components
    )


@pytest.mark.parametrize("word", ["HURRY!", "TH-THERE", "SOMETHING"])
@pytest.mark.parametrize("angle", [30, 45, 60])
def test_rotated_single_word_recall_is_orientation_invariant(word: str, angle: int) -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    origin = ((image.shape[1] - text_size[0]) // 2, (image.shape[0] + text_size[1]) // 2)
    cv2.putText(
        image,
        word,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    rotated = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((210, 110), angle, 1.0),
        (420, 220),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    components = discover_source_text_components(
        rotated,
        page_id="page_001",
        detector_regions=[],
    )

    assert components


def test_outlined_light_text_on_dark_surface_is_discovered() -> None:
    image = np.full((220, 420, 3), (20, 24, 40), dtype=np.uint8)
    origin = (98, 125)
    cv2.putText(
        image,
        "WAIT HERE",
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (2, 2, 2),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "WAIT HERE",
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert components


def test_dense_multiline_outlined_text_is_not_discarded_by_glyph_count() -> None:
    image = np.zeros((680, 800, 3), dtype=np.uint8)
    cv2.ellipse(image, (555, 455), (225, 175), 0, 0, 360, (8, 8, 8), -1)
    ink = np.zeros(image.shape[:2], dtype=np.uint8)
    for text, origin in (
        ("OUT OF COUNTLESS SOULS", (385, 405)),
        ("YOU HAVE BEEN CHOSEN AS", (355, 465)),
        ("YOU FIT THE CONDITIONS", (385, 525)),
    ):
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (110, 82, 12),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (250, 250, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            ink,
            text,
            origin,
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            255,
            2,
            cv2.LINE_AA,
        )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )
    covered = np.zeros_like(ink)
    for component in components:
        x1, y1, x2, y2 = component.bbox_page
        covered[y1:y2, x1:x2] = 255
    recall = np.count_nonzero((ink > 0) & (covered > 0)) / float(np.count_nonzero(ink))

    assert recall >= 0.90


def test_italic_text_on_repetitive_light_texture_keeps_full_visual_coverage() -> None:
    image = np.full((535, 800, 3), 245, dtype=np.uint8)
    for x in range(120, 700, 9):
        cv2.line(image, (x, 70), (x, 450), (225, 225, 225), 2)
    ink = np.zeros(image.shape[:2], dtype=np.uint8)
    for text, origin in (
        ("ALRIGHT!!!", (300, 180)),
        ("THE NATIONAL TEAM", (210, 250)),
        ("SELECTION EVENT", (225, 320)),
        ("WAITING FOR YOU", (230, 390)),
    ):
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
            1.0,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            ink,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
            1.0,
            255,
            2,
            cv2.LINE_AA,
        )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )
    covered = np.zeros_like(ink)
    for component in components:
        x1, y1, x2, y2 = component.bbox_page
        covered[y1:y2, x1:x2] = 255

    recall = np.count_nonzero((ink > 0) & (covered > 0)) / float(np.count_nonzero(ink))
    assert recall >= 0.95
    assert len(components) == 4


def test_stacked_card_roles_are_not_fused_by_transitive_proximity() -> None:
    image = np.full((300, 500, 3), 248, dtype=np.uint8)
    for text, origin, scale in (
        ("TITLE", (120, 60), 0.9),
        ("BODY LINE", (100, 125), 0.8),
        ("BODY LINE", (100, 165), 0.8),
        ("FOOTER", (140, 245), 0.7),
    ):
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[],
    )

    assert len(components) == 4
    assert max(component.bbox_page[3] - component.bbox_page[1] for component in components) < 70


def test_component_geometry_is_page_space_before_ocr() -> None:
    image = np.full((180, 320, 3), 255, dtype=np.uint8)
    detector = DetectorRegion(
        bbox_page=(30, 40, 180, 90),
        polygon_page=((30, 40), (180, 40), (180, 90), (30, 90)),
        detector_source="primary_region_detector",
    )

    components = discover_source_text_components(
        image,
        page_id="page_007",
        detector_regions=[detector],
        glyph_candidates=[],
    )

    assert len(components) == 1
    assert components[0].page_id == "page_007"
    assert components[0].bbox_page == (30, 40, 180, 90)
    assert components[0].polygon_page == detector.polygon_page


def test_detector_and_glyph_scan_candidates_are_deduplicated_without_text() -> None:
    image = np.full((180, 320, 3), 255, dtype=np.uint8)
    detector = DetectorRegion(
        bbox_page=(30, 40, 180, 90),
        detector_source="primary_region_detector",
    )
    glyph = GlyphCandidate(
        bbox_page=(32, 42, 178, 88),
        detector_source="glyph_scan",
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[detector],
        glyph_candidates=[glyph],
    )

    assert len(components) == 1
    assert components[0].detector_sources == ("glyph_scan", "primary_region_detector")
    assert "text" not in inspect.signature(DetectorRegion).parameters
    assert "text" not in inspect.signature(GlyphCandidate).parameters


def test_broad_detector_container_supports_tight_glyph_lines_without_duplication() -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    detector = DetectorRegion(
        bbox_page=(20, 20, 400, 200),
        detector_source="card_region_detector",
        evidence_id="region_card",
        confidence=0.91,
        script_evidence=("latin_likely",),
    )
    glyphs = [
        GlyphCandidate((80, 60, 320, 88), detector_source="glyph_scan", confidence=0.76),
        GlyphCandidate((80, 110, 320, 138), detector_source="glyph_scan", confidence=0.78),
    ]

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[detector],
        glyph_candidates=glyphs,
    )

    assert len(components) == 2
    assert all(
        component.detector_sources == ("card_region_detector", "glyph_scan")
        for component in components
    )
    assert all(component.evidence_ids == ("region_card",) for component in components)
    assert all(component.confidence is not None for component in components)
    assert all(component.script_evidence == ("latin_likely",) for component in components)


def test_single_tiny_glyph_does_not_replace_a_broad_detector_container() -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    detector = DetectorRegion(
        bbox_page=(20, 20, 400, 200),
        detector_source="card_region_detector",
        evidence_id="region_card",
        confidence=0.91,
    )
    tiny = GlyphCandidate(
        bbox_page=(205, 105, 214, 114),
        detector_source="glyph_scan",
        confidence=0.52,
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[detector],
        glyph_candidates=[tiny],
    )

    assert any(component.bbox_page == detector.bbox_page for component in components)
    assert any(component.bbox_page == tiny.bbox_page for component in components)


def test_broad_detector_supports_one_short_tight_line_without_duplication() -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    detector = DetectorRegion(
        bbox_page=(20, 20, 400, 200),
        detector_source="card_region_detector",
        evidence_id="region_card",
        confidence=0.91,
        script_evidence=("latin_likely",),
    )
    glyph = GlyphCandidate(
        bbox_page=(73, 97, 146, 137),
        detector_source="glyph_scan",
        confidence=0.78,
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[detector],
        glyph_candidates=[glyph],
    )

    assert len(components) == 1
    assert components[0].bbox_page == glyph.bbox_page
    assert components[0].detector_sources == (
        "card_region_detector",
        "glyph_scan",
    )
    assert components[0].evidence_ids == ("region_card",)
    assert components[0].script_evidence == ("latin_likely",)


def test_small_word_scan_supports_broad_detector_without_duplicate_owner() -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "NO",
        (70, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    detector = DetectorRegion(
        bbox_page=(20, 20, 400, 200),
        detector_source="card_region_detector",
        evidence_id="region_card",
        confidence=0.91,
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[detector],
    )

    assert len(components) == 1
    assert components[0].bbox_page != detector.bbox_page
    assert components[0].detector_sources == (
        "card_region_detector",
        "glyph_scan",
    )


def test_component_polygon_is_clamped_to_page_bounds() -> None:
    image = np.full((180, 320, 3), 255, dtype=np.uint8)
    detector = DetectorRegion(
        bbox_page=(-20, -10, 340, 200),
        polygon_page=((-20, -10), (340, -10), (340, 200), (-20, 200)),
        detector_source="region_detector",
    )

    components = discover_source_text_components(
        image,
        page_id="page_001",
        detector_regions=[detector],
        glyph_candidates=[],
    )

    assert components[0].bbox_page == (0, 0, 320, 180)
    assert components[0].polygon_page == ((0, 0), (320, 0), (320, 180), (0, 180))


def test_every_discovered_component_needs_owner_or_explicit_disposition() -> None:
    image, _ = _synthetic_text_surface("dark_panel")
    components = discover_source_text_components(image, page_id="page_001", detector_regions=[])

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=[],
        semantic_regions=[],
    )

    assert len(graph.component_dispositions) == len(components)
    assert {item.decision for item in graph.component_dispositions} == {"review"}


def test_unresolved_textlike_component_blocks_instead_of_disappearing() -> None:
    image, _ = _synthetic_text_surface("card")
    components = discover_source_text_components(image, page_id="page_001", detector_regions=[])

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=[],
        semantic_regions=[],
    )

    assert graph.validate() == []
    assert graph.owners
    assert all(owner.state == "review_required" for owner in graph.owners)
    assert all(owner.route_action == "review_required" for owner in graph.owners)
