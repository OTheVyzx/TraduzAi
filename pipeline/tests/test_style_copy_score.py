from __future__ import annotations

import builtins
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from debug_tools import style_copy_score
from debug_tools.run_style_copy_regression import _copy_run, _run_style_score
from debug_tools.style_copy_score import _matches_real_kind


PIPELINE_DIR = Path(__file__).resolve().parents[1]
ROOT = PIPELINE_DIR.parent
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "style_copy_atlas"
SCORE_SCRIPT = PIPELINE_DIR / "debug_tools" / "style_copy_score.py"


def _tracked_atlas_snapshot() -> dict[Path, tuple[bytes, int]]:
    paths = (
        FIXTURE_DIR / "style_copy_atlas.png",
        FIXTURE_DIR / "style_copy_manifest.json",
    )
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_cli_rejects_regenerate_atlas_and_leaves_tracked_fixture_untouched():
    before = _tracked_atlas_snapshot()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), "--regenerate-atlas"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    assert "unrecognized arguments: --regenerate-atlas" in result.stderr
    assert _tracked_atlas_snapshot() == before


def test_cli_score_output_includes_runtime_versions_without_importing_pil(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    try:
        expected_pillow_version = metadata.version("Pillow")
    except metadata.PackageNotFoundError:
        expected_pillow_version = "unavailable"
    expected_runtime = {
        "python": platform.python_version(),
        "pillow": expected_pillow_version,
        "opencv": style_copy_score.cv2.__version__,
        "freetype": "2.6.1",
        "freetype_provider": "matplotlib.ft2font",
        "freetype_provider_version": metadata.version("matplotlib"),
    }

    def fake_score_synthetic(_atlas_dir: Path):
        return {"cases": 14, "passed": 14, "failed": [], "pass_rate": 1.0}

    original_import = builtins.__import__

    def reject_pil_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise AssertionError("score metadata must not import PIL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(style_copy_score, "score_synthetic", fake_score_synthetic)
    monkeypatch.setattr(sys, "argv", ["style_copy_score.py"])
    monkeypatch.setattr(builtins, "__import__", reject_pil_import)

    assert style_copy_score.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["runtime"] == expected_runtime


def test_runtime_metadata_tolerates_missing_pillow_distribution(
    monkeypatch: pytest.MonkeyPatch,
):
    real_version = metadata.version

    def version_or_missing(distribution_name: str) -> str:
        if distribution_name == "Pillow":
            raise metadata.PackageNotFoundError(distribution_name)
        return real_version(distribution_name)

    monkeypatch.setattr(style_copy_score.importlib.metadata, "version", version_or_missing)

    assert style_copy_score._runtime_metadata()["pillow"] == "unavailable"


def test_style_score_module_import_does_not_import_pil():
    code = """
import builtins
import json
import sys

real_import = builtins.__import__

def reject_pil(name, *args, **kwargs):
    if name == "PIL" or name.startswith("PIL."):
        raise AssertionError(f"unexpected PIL import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_pil
from debug_tools import style_copy_score
before_parent = sys.modules.get("matplotlib")
before_child = sys.modules.get("matplotlib.ft2font")
runtime = style_copy_score._runtime_metadata()
if sys.modules.get("matplotlib") is not before_parent:
    raise AssertionError("matplotlib parent module leaked")
if sys.modules.get("matplotlib.ft2font") is not before_child:
    raise AssertionError("matplotlib.ft2font module leaked")
loaded = sorted(name for name in sys.modules if name == "PIL" or name.startswith("PIL."))
if loaded:
    raise AssertionError(f"PIL was imported: {loaded}")
print(json.dumps(runtime, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PIPELINE_DIR)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    runtime = json.loads(result.stdout)
    assert runtime["freetype"] == "2.6.1"
    assert runtime["freetype_provider"] == "matplotlib.ft2font"


def test_direct_script_cli_works_from_root_without_importing_pil(tmp_path: Path):
    guard_dir = tmp_path / "import-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        """
import atexit
import builtins
import sys

_real_import = builtins.__import__

def _reject_pil(name, *args, **kwargs):
    if name == "PIL" or name.startswith("PIL."):
        raise AssertionError(f"unexpected PIL import: {name}")
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _reject_pil

def _verify_no_pil():
    loaded = sorted(name for name in sys.modules if name == "PIL" or name.startswith("PIL."))
    if loaded:
        raise AssertionError(f"PIL was imported: {loaded}")

atexit.register(_verify_no_pil)
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "direct-style-score.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(guard_dir)

    result = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), "--output", str(output_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["synthetic"]["passed"] == payload["synthetic"]["cases"] == 14
    assert payload["runtime"]["freetype"] == "2.6.1"
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_style_regression_runner_invokes_score_cli_with_pipeline_imports(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    records_dir = tmp_path / "debug" / "codex_style_audit" / "visual_report"
    records_dir.mkdir(parents=True)
    (records_dir / "style_audit_records.jsonl").write_text("", encoding="utf-8")

    result = _run_style_score(tmp_path)

    assert result["returncode"] == 0, result["output_tail"]
    assert Path(result["output_path"]).is_file()
    assert result["synthetic"]["cases"] == 14


def test_style_regression_runner_resolves_windows_relative_score_paths_before_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    relative_run = Path("relative output") / "chapter run"
    report_dir = relative_run / "debug" / "codex_style_audit" / "visual_report"
    report_dir.mkdir(parents=True)
    (report_dir / "style_audit_records.jsonl").write_text("", encoding="utf-8")
    expected_records = (report_dir / "style_audit_records.jsonl").resolve()
    expected_output = (report_dir / "style_copy_score.json").resolve()
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout="{}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _run_style_score(relative_run)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert Path(cmd[cmd.index("--records") + 1]) == expected_records
    assert Path(cmd[cmd.index("--output") + 1]) == expected_output
    assert Path(str(captured["cwd"])).is_absolute()
    assert Path(result["output_path"]) == expected_output


def test_style_regression_copy_refuses_to_overwrite_an_existing_run(tmp_path: Path):
    source_run = tmp_path / "source"
    source_run.mkdir()
    (source_run / "project.json").write_text("{}", encoding="utf-8")
    dest_run = tmp_path / "existing-run"
    dest_run.mkdir()
    sentinel = dest_run / "preserve.txt"
    sentinel.write_text("existing output", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _copy_run(source_run, dest_run)

    assert sentinel.read_text(encoding="utf-8") == "existing output"


def test_real_style_score_uses_applied_gradient_when_report_has_applied_fields():
    detected_but_not_applied = {
        "text_color": "#102040",
        "gradient": True,
        "gradient_colors": ["#102040", "#284060"],
        "glow": False,
        "applied_text_color": "#000000",
        "applied_gradient": False,
        "applied_gradient_colors": [],
        "applied_glow": False,
        "bbox": [0, 0, 160, 60],
    }
    applied = {
        **detected_but_not_applied,
        "applied_gradient": True,
        "applied_gradient_colors": ["#102040", "#284060"],
    }

    assert _matches_real_kind(detected_but_not_applied, "dark_text_gradient") is False
    assert _matches_real_kind(applied, "dark_text_gradient") is True


def test_real_style_score_falls_back_to_detected_fields_for_old_reports():
    old_report_record = {
        "text_color": "#102040",
        "gradient": True,
        "gradient_colors": ["#102040", "#284060"],
        "glow": False,
        "bbox": [0, 0, 160, 60],
    }

    assert _matches_real_kind(old_report_record, "dark_text_gradient") is True


def test_real_style_score_accepts_solid_dark_text_without_false_gradient():
    record = {
        "applied_text_color": "#020202",
        "applied_gradient": False,
        "applied_gradient_colors": [],
        "applied_glow": False,
        "applied_stroke_color": "",
        "bbox": [0, 0, 260, 90],
    }

    assert _matches_real_kind(record, "solid_dark_text") is True
