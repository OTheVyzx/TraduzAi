from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from PIL import Image
import numpy as np

from tools.init_owner_validation_run import (
    init_validation_context,
    load_validation_context,
    main,
)


def test_validation_context_uses_new_root_and_is_atomically_reloadable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.fromarray(np.full((8, 9, 3), 220, dtype=np.uint8), "RGB").save(
        source / "001.png"
    )
    context_path = tmp_path / "current.json"

    first = init_validation_context(tmp_path / "validation", source, context_path)
    second = init_validation_context(tmp_path / "validation", source, context_path)

    assert first.validation_root != second.validation_root
    assert load_validation_context(context_path) == second
    assert second.source_path == source.resolve()
    assert second.off_out.parent == second.validation_root
    assert not context_path.with_suffix(".tmp").exists()


def test_cli_returns_nonzero_without_source_and_has_no_unhandled_exception(tmp_path):
    assert main([
        "--base", str(tmp_path / "validation"),
        "--source", str(tmp_path / "missing"),
        "--context", str(tmp_path / "current.json"),
    ]) != 0


def test_direct_cli_runs_from_repository_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.fromarray(np.full((8, 9, 3), 220, dtype=np.uint8), "RGB").save(source / "001.png")
    script = Path(__file__).resolve().parents[1] / "tools" / "init_owner_validation_run.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--base", str(tmp_path / "validation"),
         "--source", str(source), "--context", str(tmp_path / "current.json")],
        cwd=script.parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "current.json").is_file()
