"""TDD contracts for the page-global owner control-plane barrier."""

from __future__ import annotations

from dataclasses import replace
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.model import (
    ComponentDisposition,
    OwnerGraph,
    OwnerGraphValidationError,
    OwnerProjection,
    OwnerViolation,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)
from ownership.ocr_adapter import TileProjection
from strip.types import BBox, Balloon, Band


def _component() -> SourceTextComponent:
    return SourceTextComponent(
        component_id="component_a",
        page_id="page_001",
        bbox_page=(30, 30, 70, 70),
        polygon_page=((30, 30), (70, 30), (70, 70), (30, 70)),
        detector_sources=("independent_text_recall",),
    )


def _observation(*, tile_id: str = "tile_a") -> TextObservation:
    return TextObservation(
        observation_id=f"observation_{tile_id}",
        page_id="page_001",
        component_ids=("component_a",),
        text="COMPLETE SOURCE BODY",
        confidence=0.93,
        provider="paddle_full_page",
        bbox_page=(30, 30, 70, 70),
        tile_provenance=(tile_id,),
        coverage_score=1.0,
    )


def _graph(*, projections: list[OwnerProjection] | None = None) -> OwnerGraph:
    owner = TextOwner(
        owner_id="owner_a",
        page_id="page_001",
        component_ids=["component_a"],
        observation_ids=["observation_tile_a"],
        selected_observation_ids=["observation_tile_a"],
        semantic_role="dialogue_body",
        source_payload="COMPLETE SOURCE BODY",
        translated_payload=None,
        disposition="owned",
        state="ocr_ready",
        route_action="translate_inpaint_render",
        execution_tile_id=None,
    )
    return OwnerGraph(
        schema_version=1,
        page_id="page_001",
        components=[_component()],
        observations=[_observation()],
        owners=[owner],
        projections=list(projections or []),
        component_dispositions=[
            ComponentDisposition(
                component_id="component_a",
                decision="owned",
                owner_id="owner_a",
                reason="semantic_owner_resolved",
            )
        ],
    )


def _detector_with_two_separate_balloons() -> MagicMock:
    detector = MagicMock()
    first = SimpleNamespace(x1=10.0, y1=20.0, x2=180.0, y2=70.0, confidence=0.9)
    second = SimpleNamespace(x1=10.0, y1=420.0, x2=180.0, y2=470.0, confidence=0.9)
    detector.detect.return_value = [first, second]
    return detector


def test_run_chapter_collects_all_tile_evidence_before_first_translation() -> None:
    """The legacy execution path may start only after the owner barrier returns."""

    from strip.run import run_chapter

    events: list[str] = []
    process_calls = 0

    def owner_control_plane(bands, **_kwargs):
        for index, _band in enumerate(bands):
            events.append(f"collect:tile_{chr(ord('a') + index)}")
        events.append("resolve:page_001")
        return {"page_001": _graph()}

    def legacy_process_band(band, **_kwargs):
        nonlocal process_calls
        if process_calls == 0:
            events.extend(("translate:owner_a", "execute:owner_a"))
        process_calls += 1
        band.cleaned_slice = band.original_slice.copy()
        band.rendered_slice = band.original_slice.copy()
        band.ocr_result = {"texts": [], "_vision_blocks": []}
        return band

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_path = root / "001.jpg"
        cv2.imwrite(str(input_path), np.full((600, 200, 3), 128, dtype=np.uint8))

        with patch(
            "strip.run._run_owner_control_plane",
            side_effect=owner_control_plane,
            create=True,
        ), patch(
            "strip.run.process_band",
            side_effect=legacy_process_band,
        ), patch(
            "strip.run._build_precomputed_macro_ocr_pages",
            return_value={},
        ), patch(
            "strip.run._build_precomputed_koharu_cjk_pages",
            return_value={},
        ), patch(
            "strip.run._start_inpainter_prewarm",
            return_value=None,
        ):
            run_chapter(
                [input_path],
                root / "output",
                detector=_detector_with_two_separate_balloons(),
                runtime=MagicMock(),
                translator=MagicMock(),
                inpainter=MagicMock(),
                typesetter=MagicMock(),
                owner_graph_mode="shadow",
                skip_page_cleanup_rerender=True,
            )

    assert events == [
        "collect:tile_a",
        "collect:tile_b",
        "resolve:page_001",
        "translate:owner_a",
        "execute:owner_a",
    ]


def test_run_chapter_resolves_each_page_once_before_owner_execution() -> None:
    from strip.run import _run_owner_control_plane

    events: list[str] = []
    bands = [
        SimpleNamespace(tile_id="tile_a", page_id="page_001"),
        SimpleNamespace(tile_id="tile_b", page_id="page_001"),
    ]

    def collector(band):
        events.append(f"collect:{band.tile_id}")
        return SimpleNamespace(page_id=band.page_id, tile_id=band.tile_id)

    def resolver(page_id, evidence):
        events.append(f"resolve:{page_id}")
        assert [item.tile_id for item in evidence] == ["tile_a", "tile_b"]
        graph = _graph(
            projections=[
                OwnerProjection(
                    owner_id="owner_a",
                    tile_id="tile_a",
                    role="executor",
                    bbox_page=(30, 30, 70, 70),
                    bbox_tile=(30, 30, 70, 70),
                    offset_xy=(0, 0),
                )
            ]
        )
        graph.owners[0].execution_tile_id = "tile_a"
        return graph

    def executor(_band, *, graph, projection):
        assert graph.page_id == "page_001"
        events.append(f"execute:{projection.owner_id}")

    graphs = _run_owner_control_plane(
        bands,
        owner_graph_mode="shadow",
        collector=collector,
        resolver=resolver,
        executor=executor,
    )

    assert list(graphs) == ["page_001"]
    assert events == [
        "collect:tile_a",
        "collect:tile_b",
        "resolve:page_001",
        "execute:owner_a",
    ]


def test_executor_assignment_prefers_coverage_and_edge_distance() -> None:
    from strip.run import _assign_owner_executor_projections

    tiles = [
        # Full coverage, but the owner is only 10 px from this tile's edge.
        TileProjection(
            page_id="page_001",
            tile_id="tile_edge",
            offset_xy=(0, 0),
            page_size=(100, 100),
            tile_size=(80, 80),
        ),
        # Partial coverage, despite a locally comfortable crop.
        TileProjection(
            page_id="page_001",
            tile_id="tile_partial",
            offset_xy=(20, 20),
            page_size=(100, 100),
            tile_size=(40, 60),
        ),
        # Full coverage and 20 px of minimum edge distance: deterministic winner.
        TileProjection(
            page_id="page_001",
            tile_id="tile_centered",
            offset_xy=(10, 10),
            page_size=(100, 100),
            tile_size=(80, 80),
        ),
    ]

    assigned = _assign_owner_executor_projections(_graph(), tiles)

    assert assigned.owners[0].execution_tile_id == "tile_centered"
    assert {
        projection.tile_id: projection.role for projection in assigned.projections
    } == {
        "tile_centered": "executor",
        "tile_edge": "context_only",
        "tile_partial": "context_only",
    }


def test_context_only_projection_never_calls_translate_inpaint_or_typeset() -> None:
    from strip.process_bands import execute_owner_tile

    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    band = Band(
        y_top=0,
        y_bottom=80,
        balloons=[Balloon(BBox(20, 20, 80, 60), 0.9)],
        strip_slice=image.copy(),
        original_slice=image.copy(),
        tile_id="tile_context",
    )
    projection = OwnerProjection(
        owner_id="owner_a",
        tile_id="tile_context",
        role="context_only",
        bbox_page=(30, 30, 70, 70),
        bbox_tile=(30, 30, 70, 70),
        offset_xy=(0, 0),
    )
    translator = MagicMock()
    inpainter = MagicMock()
    typesetter = MagicMock()

    execute_owner_tile(
        band,
        runtime=MagicMock(),
        translator=translator,
        inpainter=inpainter,
        typesetter=typesetter,
        graph=_graph(projections=[projection]),
        projection=projection,
    )

    translator.translate_pages.assert_not_called()
    inpainter.inpaint_band_image.assert_not_called()
    typesetter.render_band_image.assert_not_called()
    assert band.cleaned_slice is None
    assert band.rendered_slice is None


def test_band_permutation_produces_same_owner_manifest() -> None:
    from strip.run import _resolve_page_owner_graphs_once

    component = _component()
    tile_a = SimpleNamespace(
        page_id="page_001",
        tile_id="tile_a",
        components=[component],
        observations=[_observation(tile_id="tile_a")],
    )
    tile_b = SimpleNamespace(
        page_id="page_001",
        tile_id="tile_b",
        components=[],
        observations=[_observation(tile_id="tile_b")],
    )

    def resolver(page_id, evidence):
        from ownership.reconcile import SemanticRegion, build_page_owner_graph

        components = [component for item in evidence for component in item.components]
        observations = [observation for item in evidence for observation in item.observations]
        return build_page_owner_graph(
            page_id=page_id,
            components=components,
            observations=observations,
            semantic_regions=[
                SemanticRegion(
                    region_id="region_a",
                    component_ids=("component_a",),
                    semantic_role="dialogue_body",
                )
            ],
        )

    forward = _resolve_page_owner_graphs_once([tile_a, tile_b], resolver=resolver)
    reversed_order = _resolve_page_owner_graphs_once([tile_b, tile_a], resolver=resolver)

    assert forward["page_001"].to_dict() == reversed_order["page_001"].to_dict()


def test_collect_band_evidence_is_read_only_and_disables_mutating_stages() -> None:
    from strip.process_bands import collect_band_evidence

    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    cleaned_before = np.full_like(image, 31)
    rendered_before = np.full_like(image, 63)
    band = Band(
        y_top=0,
        y_bottom=80,
        balloons=[Balloon(BBox(20, 20, 80, 60), 0.9)],
        strip_slice=image.copy(),
        original_slice=image.copy(),
        cleaned_slice=cleaned_before,
        rendered_slice=rendered_before,
        tile_id="tile_a",
    )
    calls: list[dict] = []

    def collecting_process_band(target, **kwargs):
        calls.append(kwargs)
        assert kwargs["translator"] is None
        assert kwargs["inpainter"] is None
        assert kwargs["typesetter"] is None
        assert kwargs["control_plane_only"] is True
        target.cleaned_slice = np.zeros_like(image)
        target.rendered_slice = np.zeros_like(image)
        target.ocr_result = {"texts": [], "owner_observations": []}
        return target

    projection = TileProjection(
        page_id="page_001",
        tile_id="tile_a",
        offset_xy=(0, 0),
        page_size=(100, 80),
        tile_size=(100, 80),
    )
    with patch("strip.process_bands.process_band", side_effect=collecting_process_band):
        evidence = collect_band_evidence(
            band,
            runtime=MagicMock(),
            page_idx=0,
            tile_projection=projection,
        )

    assert len(calls) == 1
    assert band.cleaned_slice is cleaned_before
    assert band.rendered_slice is rendered_before
    assert evidence.band.cleaned_slice is cleaned_before
    assert evidence.band.rendered_slice is rendered_before


def test_repeated_observation_id_merges_provenance_exactly_once() -> None:
    from strip.run import _resolve_owner_graph_from_evidence

    component = replace(_component(), evidence_ids=("container_a",))
    first = replace(
        _observation(tile_id="tile_a"),
        projection_ids=("projection_a",),
    )
    repeated = replace(
        first,
        tile_provenance=("tile_a", "tile_b", "tile_b"),
        projection_ids=("projection_a", "projection_b", "projection_b"),
    )
    graph = _resolve_owner_graph_from_evidence(
        "page_001",
        [
            SimpleNamespace(components=[component], observations=[first]),
            SimpleNamespace(components=[component], observations=[repeated]),
        ],
    )

    assert len(graph.observations) == 1
    assert graph.observations[0].tile_provenance == ("tile_a", "tile_b")
    assert graph.observations[0].projection_ids == ("projection_a", "projection_b")
    assert graph.owners[0].observation_ids == [first.observation_id]
    assert graph.owners[0].selected_observation_ids == [first.observation_id]


@pytest.mark.parametrize("accepted_first", [True, False])
def test_repeated_observation_is_accepted_when_any_tile_accepts_it(
    accepted_first: bool,
) -> None:
    from strip.run import _resolve_owner_graph_from_evidence

    component = replace(_component(), evidence_ids=("container_a",))
    accepted = replace(
        _observation(tile_id="tile_a"),
        observation_id="observation_shared",
        rejection_reason=None,
    )
    rejected = replace(
        accepted,
        tile_provenance=("tile_b",),
        confidence=0.71,
        rejection_reason="partial_tile_edge",
    )
    observations = [accepted, rejected] if accepted_first else [rejected, accepted]

    graph = _resolve_owner_graph_from_evidence(
        "page_001",
        [
            SimpleNamespace(components=[component], observations=[observations[0]]),
            SimpleNamespace(components=[component], observations=[observations[1]]),
        ],
    )

    assert len(graph.observations) == 1
    assert graph.observations[0].rejection_reason is None
    assert graph.observations[0].tile_provenance == ("tile_a", "tile_b")
    assert graph.owners[0].selected_observation_ids == ["observation_shared"]


def test_observation_without_component_ids_is_associated_by_page_geometry() -> None:
    from strip.run import _resolve_owner_graph_from_evidence

    component = replace(_component(), evidence_ids=("container_a",))
    observation = replace(
        _observation(),
        component_ids=(),
        bbox_page=(35, 35, 65, 65),
    )
    graph = _resolve_owner_graph_from_evidence(
        "page_001",
        [SimpleNamespace(components=[component], observations=[observation])],
    )

    assert graph.observations[0].component_ids == ("component_a",)
    assert len(graph.owners) == 1
    assert graph.owners[0].component_ids == ["component_a"]
    assert graph.owners[0].source_payload == "COMPLETE SOURCE BODY"


def test_components_in_same_visual_container_form_one_body_owner() -> None:
    from strip.run import _resolve_owner_graph_from_evidence

    first = replace(_component(), evidence_ids=("container_shared",))
    second = SourceTextComponent(
        component_id="component_b",
        page_id="page_001",
        bbox_page=(30, 72, 70, 102),
        polygon_page=((30, 72), (70, 72), (70, 102), (30, 102)),
        detector_sources=("independent_text_recall",),
        evidence_ids=("container_shared",),
    )
    observation = replace(
        _observation(),
        component_ids=("component_a", "component_b"),
        bbox_page=(30, 30, 70, 102),
        text="COMPLETE TWO LINE SOURCE BODY",
    )
    graph = _resolve_owner_graph_from_evidence(
        "page_001",
        [SimpleNamespace(components=[second, first], observations=[observation])],
    )

    assert len(graph.owners) == 1
    assert graph.owners[0].semantic_role == "dialogue_body"
    assert graph.owners[0].component_ids == ["component_a", "component_b"]
    assert graph.owners[0].source_payload == "COMPLETE TWO LINE SOURCE BODY"


def test_partial_only_tiles_are_shadow_divergence_and_enforce_blocker() -> None:
    from strip.run import _run_owner_control_plane

    partial = TileProjection(
        page_id="page_001",
        tile_id="tile_partial",
        offset_xy=(0, 0),
        page_size=(100, 100),
        tile_size=(50, 100),
    )
    band = SimpleNamespace(tile_id="tile_partial")
    evidence = SimpleNamespace(
        page_id="page_001",
        tile_id="tile_partial",
        band_index=0,
        tile_projection=partial,
        band=band,
    )
    executor = MagicMock()

    shadow_graphs = _run_owner_control_plane(
        [band],
        owner_graph_mode="shadow",
        collector=lambda _band: evidence,
        resolver=lambda _page_id, _evidence: _graph(),
        executor=executor,
    )

    assert "owner_executor_full_coverage_missing" in {
        violation.code for violation in shadow_graphs["page_001"].violations
    }
    executor.assert_not_called()

    with pytest.raises(OwnerGraphValidationError) as exc_info:
        _run_owner_control_plane(
            [band],
            owner_graph_mode="enforce",
            collector=lambda _band: evidence,
            resolver=lambda _page_id, _evidence: _graph(),
            executor=executor,
        )

    assert "owner_executor_full_coverage_missing" in {
        violation.code for violation in exc_info.value.violations
    }
    executor.assert_not_called()


def test_executor_tie_uses_lexicographically_stable_tile_id() -> None:
    from strip.run import _assign_owner_executor_projections

    tiles = [
        TileProjection(
            page_id="page_001",
            tile_id="tile_z",
            offset_xy=(0, 0),
            page_size=(100, 100),
            tile_size=(100, 100),
        ),
        TileProjection(
            page_id="page_001",
            tile_id="tile_a",
            offset_xy=(0, 0),
            page_size=(100, 100),
            tile_size=(100, 100),
        ),
    ]

    assigned = _assign_owner_executor_projections(_graph(), tiles)

    assert assigned.owners[0].execution_tile_id == "tile_a"
    assert {
        projection.tile_id: projection.role for projection in assigned.projections
    } == {"tile_a": "executor", "tile_z": "context_only"}


def test_grouping_splits_close_balloons_at_page_break_and_clamps_margin() -> None:
    from strip.bands import group_balloons_into_bands

    balloons = [
        Balloon(BBox(10, 34, 90, 48), 0.9, metadata={"page_id": "page_001"}),
        Balloon(BBox(10, 52, 90, 66), 0.9, metadata={"page_id": "page_002"}),
    ]

    bands = group_balloons_into_bands(
        balloons,
        gap_threshold=64,
        margin=16,
        page_breaks=[0, 50, 100],
    )

    assert len(bands) == 2
    assert [(band.y_top, band.y_bottom) for band in bands] == [(18, 50), (50, 82)]
    assert [
        {balloon.metadata["page_id"] for balloon in band.balloons}
        for band in bands
    ] == [{"page_001"}, {"page_002"}]


def test_corrupt_mixed_page_band_is_rejected_before_legacy_execution() -> None:
    from strip.run import run_chapter

    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    mixed_band = Band(
        y_top=0,
        y_bottom=80,
        balloons=[
            Balloon(BBox(10, 10, 45, 40), 0.9, metadata={"page_id": "page_001"}),
            Balloon(BBox(55, 10, 90, 40), 0.9, metadata={"page_id": "page_002"}),
        ],
        tile_id="tile_mixed",
    )
    legacy_process = MagicMock(side_effect=AssertionError("legacy execution started"))
    translator = MagicMock()
    inpainter = MagicMock()
    typesetter = MagicMock()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_path = root / "001.jpg"
        cv2.imwrite(str(input_path), image)

        with patch("strip.run.detect_strip_balloons", return_value=mixed_band.balloons), patch(
            "strip.run._discover_source_components_for_strip", return_value={}
        ), patch("strip.run.group_balloons_into_bands", return_value=[mixed_band]), patch(
            "strip.run._build_precomputed_macro_ocr_pages", return_value={}
        ), patch(
            "strip.run._build_precomputed_koharu_cjk_pages", return_value={}
        ), patch(
            "strip.run._start_inpainter_prewarm", return_value=None
        ), patch(
            "strip.run.process_band", legacy_process
        ):
            with pytest.raises(ValueError, match="band spanning pages"):
                run_chapter(
                    [input_path],
                    root / "output",
                    detector=MagicMock(),
                    runtime=MagicMock(),
                    translator=translator,
                    inpainter=inpainter,
                    typesetter=typesetter,
                    owner_graph_mode="shadow",
                    skip_page_cleanup_rerender=True,
                )

    legacy_process.assert_not_called()
    translator.translate_pages.assert_not_called()
    inpainter.inpaint_band_image.assert_not_called()
    typesetter.render_band_image.assert_not_called()


def test_shadow_telemetry_counts_graph_violations_by_code() -> None:
    from strip.run import _owner_graph_divergence_counts

    graph = _graph()
    graph.violations.append(
        OwnerViolation(
            code="owner_executor_full_coverage_missing",
            severity="critical",
            message="No complete tile.",
            offenders=("owner_a",),
        )
    )

    counts = _owner_graph_divergence_counts(graph, legacy_text_count=1)

    assert counts == {"owner_executor_full_coverage_missing": 1}


def test_run_chapter_cannot_execute_legacy_pixels_under_enforce_mode(
    tmp_path, monkeypatch
) -> None:
    import strip.run as run

    input_path = tmp_path / "001.jpg"
    cv2.imwrite(str(input_path), np.full((32, 40, 3), 235, dtype=np.uint8))
    band = Band(y_top=0, y_bottom=32)
    legacy_process = MagicMock(side_effect=AssertionError("legacy band execution called"))
    monkeypatch.setattr(run, "detect_strip_balloons", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(run, "group_balloons_into_bands", lambda *_args, **_kwargs: [band])
    empty_graph = OwnerGraph(
        schema_version=1,
        page_id="page_001",
        components=[],
        observations=[],
        owners=[],
        projections=[],
    )
    monkeypatch.setattr(
        run,
        "_run_owner_control_plane",
        lambda *_args, **_kwargs: {"page_001": empty_graph},
    )
    monkeypatch.setattr(run, "process_band", legacy_process)
    monkeypatch.setattr(run, "_start_inpainter_prewarm", lambda *_args, **_kwargs: None)

    pages = run.run_chapter(
        [input_path],
        tmp_path / "output",
        detector=MagicMock(),
        runtime=MagicMock(),
        translator=MagicMock(),
        inpainter=MagicMock(),
        typesetter=MagicMock(),
        owner_graph_mode="enforce",
        skip_page_cleanup_rerender=True,
    )

    assert len(pages) == 1
    legacy_process.assert_not_called()
