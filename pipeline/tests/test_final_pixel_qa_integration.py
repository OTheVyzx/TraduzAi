from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_evaluate_final_pixels_observes_persisted_path_instead_of_project_metadata(monkeypatch):
    from test_final_pixel_qa import _composition, _graph, _observation
    from qa.final_pixel_qa import evaluate_final_pixels

    persisted_path = Path("persisted-final.png")
    observation = replace(
        _observation(
            {"text": "SOURCE BODY", "bbox": [6, 6, 28, 16], "qa_flags": []}
        ),
        image_path=persisted_path,
    )

    class Observer:
        def __init__(self):
            self.calls = []

        def observe(self, image_path, *, source_language, **kwargs):
            self.calls.append((image_path, source_language, kwargs))
            return observation

    observer = Observer()
    report = evaluate_final_pixels(
        image_path=persisted_path,
        graph=_graph(),
        composition=_composition(),
        observer=observer,
        source_language="en",
    )

    assert observer.calls[0][0:2] == (persisted_path, "en")
    assert observer.calls[0][2]["page_id"] == "page_001"
    assert observer.calls[0][2]["source_challenges"][0]["component_id"] == "component_a"
    assert any(issue.reason == "source_payload_visible" for issue in report.issues)


def test_explicit_preserve_policy_excludes_source_challenge(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    import main
    from ownership.model import ComponentDisposition
    from test_final_pixel_qa import _composition, _graph, _observation

    graph = _graph(source="CHOI JIN-SOO")
    graph.owners = []
    graph.projections = []
    graph.component_dispositions = [
        ComponentDisposition(
            component_id="component_a",
            decision="preserve",
            reason="policy:character_name",
        )
    ]
    artifact = tmp_path / "translated" / "001.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"persisted")
    captured = {}

    class Observer:
        def observe(self, image_path, **kwargs):
            captured.update(kwargs)
            return replace(
                _observation(),
                image_path=image_path,
                expected_source_challenge_count=0,
                completed_source_challenge_count=0,
                coverage_complete=True,
            )

    composition = _composition(glyph=False)
    main._observe_verified_owner_final_pages(
        project_data={
            "_work_dir": str(tmp_path),
            "owner_graph_status": "verified",
            "paginas": [{
                "numero": 1,
                "page_id": "page_001",
                "image_layers": {"rendered": {"path": "translated/001.png"}},
            }],
        },
        output_pages=[SimpleNamespace(
            owner_graph=graph,
            owner_composition=composition,
            page_surface_geometry=composition.page_surface_geometry,
        )],
        observer=Observer(),
        source_language="en",
    )

    assert captured["source_challenges"] == []
