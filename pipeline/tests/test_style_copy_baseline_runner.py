from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "style_copy_atlas"
ATLAS_NAME = "style_copy_atlas.png"
SOURCE_MANIFEST_NAME = "style_copy_manifest.json"
BASELINE_MANIFEST_NAME = "baseline_manifest.json"
RUNTIME_LOCK_PATH = PIPELINE_DIR / "debug_tools" / "style_copy_benchmark_runtime.lock.json"


def _runner_module():
    return importlib.import_module("debug_tools.run_style_copy_baseline")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_three_runs_are_canonical_isolated_and_leave_source_untouched(tmp_path: Path):
    runner = _runner_module()
    source_before = _tree_snapshot(FIXTURE_DIR)
    source_atlas = (FIXTURE_DIR / ATLAS_NAME).read_bytes()
    source_manifest = (FIXTURE_DIR / SOURCE_MANIFEST_NAME).read_bytes()
    output_root = tmp_path / "style-baselines"
    run_ids = ("baseline-a", "baseline-b", "baseline-c")

    run_dirs = [
        runner.run_baseline(
            source_atlas_dir=FIXTURE_DIR,
            output_root=output_root,
            run_id=run_id,
            seed=1729,
        )
        for run_id in run_ids
    ]

    assert run_dirs == [output_root / run_id for run_id in run_ids]
    assert len({run_dir.resolve() for run_dir in run_dirs}) == 3
    assert {path.name for path in output_root.iterdir()} == set(run_ids)

    manifest_bytes = [(run_dir / BASELINE_MANIFEST_NAME).read_bytes() for run_dir in run_dirs]
    manifest_hashes = {_sha256(data) for data in manifest_bytes}
    assert manifest_bytes[0] == manifest_bytes[1] == manifest_bytes[2]
    assert len(manifest_hashes) == 1

    expected_artifacts = {
        "atlas": {
            "filename": ATLAS_NAME,
            "sha256": _sha256(source_atlas),
        },
        "source_manifest": {
            "filename": SOURCE_MANIFEST_NAME,
            "sha256": _sha256(source_manifest),
        },
    }
    manifest = json.loads(manifest_bytes[0])
    assert set(manifest) == {"artifacts", "runtime", "schema_version", "score", "seed"}
    assert manifest["schema_version"] == 1
    assert manifest["seed"] == 1729
    assert manifest["artifacts"] == expected_artifacts
    assert manifest["score"] == {
        "cases": 14,
        "failed": [],
        "pass_rate": 1.0,
        "passed": 14,
    }
    assert set(manifest["runtime"]) == {
        "freetype",
        "freetype_provider",
        "freetype_provider_version",
        "opencv",
        "pillow",
        "python",
    }
    assert all(
        isinstance(value, str) and value and value != "unavailable"
        for value in manifest["runtime"].values()
    )
    assert manifest["runtime"]["freetype_provider"] == "matplotlib.ft2font"
    assert manifest["runtime"]["freetype_provider_version"] == metadata.version("matplotlib")

    canonical_text = manifest_bytes[0].decode("utf-8")
    assert str(output_root) not in canonical_text
    assert all(run_id not in canonical_text for run_id in run_ids)
    for run_dir in run_dirs:
        assert {path.name for path in run_dir.iterdir()} == {
            ATLAS_NAME,
            SOURCE_MANIFEST_NAME,
            BASELINE_MANIFEST_NAME,
        }
        assert (run_dir / ATLAS_NAME).read_bytes() == source_atlas
        assert (run_dir / SOURCE_MANIFEST_NAME).read_bytes() == source_manifest

    first_run_before = _tree_snapshot(run_dirs[0])
    with pytest.raises(FileExistsError):
        runner.run_baseline(
            source_atlas_dir=FIXTURE_DIR,
            output_root=output_root,
            run_id=run_ids[0],
            seed=1729,
        )
    assert _tree_snapshot(run_dirs[0]) == first_run_before
    assert _tree_snapshot(FIXTURE_DIR) == source_before


def test_runner_scores_existing_atlas_without_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "atlas.png").write_bytes(b"tracked atlas bytes")
    (source_dir / SOURCE_MANIFEST_NAME).write_text(
        json.dumps({"image": "atlas.png", "cases": []}),
        encoding="utf-8",
    )
    calls: list[Path] = []
    expected_score = {"cases": 0, "passed": 0, "failed": [], "pass_rate": 0.0}

    def fake_score_synthetic(atlas_dir: Path):
        calls.append(atlas_dir)
        return expected_score

    monkeypatch.setattr(runner.style_copy_score, "score_synthetic", fake_score_synthetic)

    run_dir = runner.run_baseline(
        source_atlas_dir=source_dir,
        output_root=tmp_path / "runs",
        run_id="no-regeneration",
        seed=7,
    )

    assert calls == [source_dir]
    assert json.loads((run_dir / BASELINE_MANIFEST_NAME).read_bytes())["score"] == expected_score


def test_runtime_lock_matches_the_validated_benchmark_runtime():
    runner = _runner_module()

    lock = json.loads(RUNTIME_LOCK_PATH.read_text(encoding="utf-8"))
    runtime = runner.style_copy_score._runtime_metadata()

    assert lock == {"schema_version": 1, "runtime": runtime}
    assert runtime["freetype"] != "unavailable"
    assert runtime["freetype_provider"] == "matplotlib.ft2font"


@pytest.mark.parametrize(
    ("runtime_key", "invalid_value"),
    [
        ("freetype", "unavailable"),
        ("python", "0.0.0"),
    ],
)
def test_runner_rejects_runtime_mismatch_before_creating_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_key: str,
    invalid_value: str,
):
    runner = _runner_module()
    runtime = dict(runner.style_copy_score._runtime_metadata())
    runtime[runtime_key] = invalid_value
    monkeypatch.setattr(runner.style_copy_score, "_runtime_metadata", lambda: runtime)
    output_root = tmp_path / "rejected-runs"

    with pytest.raises(
        RuntimeError,
        match=rf"runtime mismatch.*{runtime_key}.*expected.*got",
    ):
        runner.run_baseline(
            source_atlas_dir=FIXTURE_DIR,
            output_root=output_root,
            run_id="must-not-exist",
            seed=1729,
        )

    assert not output_root.exists()


def test_cli_import_and_execution_never_import_pil(tmp_path: Path):
    output_root = tmp_path / "isolated-runs"
    code = r'''
import builtins
import os
import sys

preloaded = sorted(name for name in sys.modules if name == "PIL" or name.startswith("PIL."))
if preloaded:
    raise AssertionError(f"PIL was preloaded: {preloaded}")

real_import = builtins.__import__

def reject_pil(name, *args, **kwargs):
    if name == "PIL" or name.startswith("PIL."):
        raise AssertionError(f"unexpected PIL import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_pil
from debug_tools import run_style_copy_baseline

exit_code = run_style_copy_baseline.main(
    [
        "--source-atlas-dir",
        os.environ["STYLE_BASELINE_SOURCE"],
        "--output-root",
        os.environ["STYLE_BASELINE_OUTPUT"],
        "--run-id",
        "isolated-no-pil",
        "--seed",
        "1729",
    ]
)
if exit_code != 0:
    raise AssertionError(f"unexpected exit code: {exit_code}")
loaded = sorted(name for name in sys.modules if name == "PIL" or name.startswith("PIL."))
if loaded:
    raise AssertionError(f"PIL was imported: {loaded}")
print("isolated-no-pil-ok")
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PIPELINE_DIR)
    env["STYLE_BASELINE_SOURCE"] = str(FIXTURE_DIR)
    env["STYLE_BASELINE_OUTPUT"] = str(output_root)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PIPELINE_DIR.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "isolated-no-pil-ok"
    assert (output_root / "isolated-no-pil" / BASELINE_MANIFEST_NAME).is_file()


def test_direct_script_cli_accepts_source_output_run_id_and_seed(tmp_path: Path):
    output_root = tmp_path / "direct-script-runs"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "debug_tools" / "run_style_copy_baseline.py"),
            "--source-atlas-dir",
            str(FIXTURE_DIR),
            "--output-root",
            str(output_root),
            "--run-id",
            "direct-script",
            "--seed",
            "1729",
        ],
        cwd=PIPELINE_DIR.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output_root / "direct-script")
    assert (output_root / "direct-script" / BASELINE_MANIFEST_NAME).is_file()
