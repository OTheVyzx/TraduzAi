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

        def observe(self, image_path, *, source_language):
            self.calls.append((image_path, source_language))
            return observation

    observer = Observer()
    report = evaluate_final_pixels(
        image_path=persisted_path,
        graph=_graph(),
        composition=_composition(),
        observer=observer,
        source_language="en",
    )

    assert observer.calls == [(persisted_path, "en")]
    assert any(issue.reason == "source_payload_visible" for issue in report.issues)
