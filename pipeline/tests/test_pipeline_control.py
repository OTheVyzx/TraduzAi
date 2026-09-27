from __future__ import annotations

import pytest

from main import PipelineCancelled, wait_if_paused


def test_cancel_marker_fails_at_cooperative_boundary(tmp_path):
    cancel = tmp_path / "pipeline.cancel"
    cancel.write_text("cancelled", encoding="utf-8")
    with pytest.raises(PipelineCancelled, match="pipeline_cancelled"):
        wait_if_paused({"cancel_file": str(cancel)})


def test_absent_control_markers_continue(tmp_path):
    wait_if_paused({
        "pause_file": str(tmp_path / "pipeline.pause"),
        "cancel_file": str(tmp_path / "pipeline.cancel"),
    })
