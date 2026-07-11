from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from debug_tools import generate_style_benchmark_v2


FIXTURE_SPEC = Path(__file__).resolve().parent / "fixtures" / "style_benchmark_v2" / "benchmark_spec.json"


def _runtime_lock_for_current_process(tmp_path: Path) -> Path:
    lock_path = tmp_path / "runtime.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime": generate_style_benchmark_v2._runtime_metadata(),
            }
        ),
        encoding="utf-8",
    )
    return lock_path


def test_style_specs_are_seeded_and_cover_all_required_levels():
    spec = generate_style_benchmark_v2.load_benchmark_spec(FIXTURE_SPEC)

    first = generate_style_benchmark_v2.build_style_specs(spec, seed=1729)
    second = generate_style_benchmark_v2.build_style_specs(spec, seed=1729)

    assert first == second
    assert {case["level"] for case in first} == {
        "smoke",
        "combinatorial",
        "hard-negative",
        "holdout",
    }
    assert {case["font_name"] for case in first} <= {
        "ComicNeue-Bold.ttf",
        "LeagueGothic-Regular-VariableFont_wdth.ttf",
    }
    assert all(case["text_a"] != case["text_b"] for case in first)
    assert all("container" in case and "rotation_deg" in case for case in first)


def test_generation_rejects_runtime_mismatch_before_creating_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    invalid_runtime = {
        "python": "3.12.10",
        "pillow": "incompatible",
        "opencv": "4.11.0",
        "freetype": "2.6.1",
        "freetype_provider": "matplotlib.ft2font",
        "freetype_provider_version": "3.10.8",
    }
    monkeypatch.setattr(generate_style_benchmark_v2, "_runtime_metadata", lambda: invalid_runtime)

    with pytest.raises(RuntimeError, match="runtime mismatch"):
        generate_style_benchmark_v2.generate_benchmark(
            spec_path=FIXTURE_SPEC,
            level="smoke",
            output_root=tmp_path / "runs",
            run_id="rejected",
            seed=1729,
        )

    assert not (tmp_path / "runs").exists()


def test_smoke_generation_writes_two_text_images_only_inside_its_run(tmp_path: Path):
    run_dir = generate_style_benchmark_v2.generate_benchmark(
        spec_path=FIXTURE_SPEC,
        level="smoke",
        output_root=tmp_path / "runs",
        run_id="smoke-1729",
        seed=1729,
        runtime_lock_path=_runtime_lock_for_current_process(tmp_path),
    )

    manifest = json.loads((run_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))

    assert run_dir == tmp_path / "runs" / "smoke-1729"
    assert {path.name for path in (tmp_path / "runs").iterdir()} == {"smoke-1729"}
    assert manifest["level"] == "smoke"
    assert manifest["seed"] == 1729
    assert len(manifest["cases"]) == 3
    for case in manifest["cases"]:
        assert case["text_a"] != case["text_b"]
        image_a = run_dir / case["image_a"]
        image_b = run_dir / case["image_b"]
        assert image_a.is_file() and image_b.is_file()
        assert image_a.read_bytes() != image_b.read_bytes()


def test_combinatorial_gradient_contains_the_declared_end_color(tmp_path: Path):
    run_dir = generate_style_benchmark_v2.generate_benchmark(
        spec_path=FIXTURE_SPEC,
        level="combinatorial",
        output_root=tmp_path / "runs",
        run_id="gradient-1729",
        seed=1729,
        runtime_lock_path=_runtime_lock_for_current_process(tmp_path),
    )
    image = cv2.imread(str(run_dir / "images" / "combination-gradient-glow-a.png"), cv2.IMREAD_COLOR)

    assert image is not None
    # Declared gradient endpoint #755BFF is BGR (255, 91, 117), unlike the dark background/stroke.
    purple_endpoint_pixels = (image[:, :, 0] > 200) & (image[:, :, 1] < 160) & (image[:, :, 2] > 80)
    assert int(np.count_nonzero(purple_endpoint_pixels)) > 10


def test_generator_cli_requires_an_explicit_isolated_output_root_and_run_id(tmp_path: Path):
    output_root = tmp_path / "runs"

    exit_code = generate_style_benchmark_v2.main(
        [
            "--spec",
            str(FIXTURE_SPEC),
            "--level",
            "smoke",
            "--output-root",
            str(output_root),
            "--run-id",
            "cli-smoke",
            "--seed",
            "1729",
            "--runtime-lock",
            str(_runtime_lock_for_current_process(tmp_path)),
        ]
    )

    assert exit_code == 0
    assert (output_root / "cli-smoke" / "benchmark_manifest.json").is_file()


def test_child_generator_crash_never_publishes_a_partial_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_root = tmp_path / "runs"
    monkeypatch.setattr(
        generate_style_benchmark_v2.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=-1073741819, stdout="", stderr="native crash"),
    )

    with pytest.raises(RuntimeError, match="child generator failed"):
        generate_style_benchmark_v2.generate_benchmark(
            spec_path=FIXTURE_SPEC,
            level="smoke",
            output_root=output_root,
            run_id="crashed",
            seed=1729,
            runtime_lock_path=_runtime_lock_for_current_process(tmp_path),
        )

    assert not (output_root / "crashed").exists()
    warnings = sorted((output_root / ".style-benchmark-v2-debug").glob("crashed-*.json"))
    assert len(warnings) == 1
    assert json.loads(warnings[0].read_text(encoding="utf-8"))["returncode"] == -1073741819


def test_incomplete_child_output_never_publishes_a_partial_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_root = tmp_path / "runs"
    monkeypatch.setattr(
        generate_style_benchmark_v2.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="child generator failed"):
        generate_style_benchmark_v2.generate_benchmark(
            spec_path=FIXTURE_SPEC,
            level="smoke",
            output_root=output_root,
            run_id="incomplete",
            seed=1729,
            runtime_lock_path=_runtime_lock_for_current_process(tmp_path),
        )

    assert not (output_root / "incomplete").exists()
    warnings = sorted((output_root / ".style-benchmark-v2-debug").glob("incomplete-*.json"))
    assert len(warnings) == 1
    assert "benchmark_manifest.json" in json.loads(warnings[0].read_text(encoding="utf-8"))["error"]
