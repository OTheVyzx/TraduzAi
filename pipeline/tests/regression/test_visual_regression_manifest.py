import json
import os
import sys
import warnings
import hashlib
from datetime import date
from pathlib import Path

import pytest


MANIFEST_DIR = Path(__file__).with_name("manifests")
REPORT_PATH = Path(__file__).resolve().parents[3] / "docs" / "debug" / "visual_regression_report.md"
REQUIRED_MANIFEST_FIELDS = {
    "manifest_version",
    "run_id",
    "run_path",
    "schema_version",
    "current_issue_classes",
    "target_issue_classes_after_fix",
    "pages_of_interest",
    "sample_artifacts",
}


def _manifest_paths() -> list[Path]:
    if not MANIFEST_DIR.exists():
        return []
    paths = sorted(MANIFEST_DIR.glob("*.json"))
    if _running_in_ci():
        return [path for path in paths if _load_manifest(path).get("ci_fixture") is True]
    return paths


def _running_in_ci() -> bool:
    value = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or ""
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_manifest_shape(manifest: dict, manifest_path: Path) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    _require(not missing, f"{manifest_path} missing required fields: {', '.join(missing)}")
    _require(manifest["manifest_version"] == 1, f"{manifest_path} manifest_version must be 1")
    _require(isinstance(manifest["run_id"], str) and manifest["run_id"], f"{manifest_path} run_id must be non-empty")
    _require(isinstance(manifest["run_path"], str) and manifest["run_path"], f"{manifest_path} run_path must be non-empty")
    _require(manifest["schema_version"] == 12, f"{manifest_path} schema_version must be 12")
    _require(isinstance(manifest["current_issue_classes"], list), f"{manifest_path} current_issue_classes must be a list")
    _require(
        all(isinstance(item, str) and item for item in manifest["current_issue_classes"]),
        f"{manifest_path} current_issue_classes entries must be non-empty strings",
    )
    _require(
        isinstance(manifest["target_issue_classes_after_fix"], list),
        f"{manifest_path} target_issue_classes_after_fix must be a list",
    )
    _require(isinstance(manifest["pages_of_interest"], list), f"{manifest_path} pages_of_interest must be a list")
    _require(
        all(isinstance(item, int) and item > 0 for item in manifest["pages_of_interest"]),
        f"{manifest_path} pages_of_interest entries must be positive integers",
    )
    _require(isinstance(manifest["sample_artifacts"], list), f"{manifest_path} sample_artifacts must be a list")
    ci_fixture = bool(manifest.get("ci_fixture", False))
    if not ci_fixture:
        _require(
            isinstance(manifest.get("qa_report_sha256_at_record_time"), str)
            and len(manifest["qa_report_sha256_at_record_time"]) == 64,
            f"{manifest_path} qa_report_sha256_at_record_time must be a sha256 hex digest",
        )
        _require(
            isinstance(manifest.get("baseline_total_sec"), (int, float))
            and float(manifest["baseline_total_sec"]) > 0,
            f"{manifest_path} baseline_total_sec must be positive",
        )
    if "waive_known_review_flags_explicitly" in set(manifest["target_issue_classes_after_fix"]):
        waivers = manifest.get("waivers")
        _require(isinstance(waivers, list) and bool(waivers), f"{manifest_path} waivers must be explicit")
        for index, waiver in enumerate(waivers):
            _validate_waiver(waiver, manifest_path, index)
    for index, artifact in enumerate(manifest["sample_artifacts"]):
        _require(isinstance(artifact, dict), f"{manifest_path} sample_artifacts[{index}] must be an object")
        for field in ("page", "issue_class", "crop_path"):
            _require(field in artifact, f"{manifest_path} sample_artifacts[{index}] missing {field}")
        _require(isinstance(artifact["crop_path"], str) and artifact["crop_path"], f"{manifest_path} sample_artifacts[{index}].crop_path must be non-empty")


def _validate_waiver(waiver: object, manifest_path: Path, index: int) -> None:
    _require(isinstance(waiver, dict), f"{manifest_path} waivers[{index}] must be an object")
    for field in ("waiver_id", "flag", "scope", "reason", "expires", "approved_by"):
        _require(
            isinstance(waiver.get(field), str) and waiver[field],
            f"{manifest_path} waivers[{index}] missing {field}",
        )
    expires = date.fromisoformat(str(waiver["expires"]))
    _require(
        expires >= date.today(),
        f"Waiver {waiver['waiver_id']} expired {waiver['expires']}; review fixture or extend waiver",
    )


def _collect_issue_from_issue_item(issue: object) -> set[str]:
    found: set[str] = set()
    if isinstance(issue, str):
        found.add(issue)
    elif isinstance(issue, dict):
        for key in ("flag", "type", "issue_class", "class", "code"):
            value = issue.get(key)
            if isinstance(value, str) and value:
                found.add(value)
            elif isinstance(value, list):
                found.update(item for item in value if isinstance(item, str) and item)
    elif isinstance(issue, list):
        for item in issue:
            found.update(_collect_issue_from_issue_item(item))
    return found


def _collect_issue_summary_tokens(summary: object) -> set[str]:
    found: set[str] = set()
    if isinstance(summary, dict):
        for key, value in summary.items():
            lowered = key.lower()
            if isinstance(value, (int, float)) and value > 0 and lowered.endswith(("_count", "_counts")):
                found.add(key.removesuffix("_count").removesuffix("_counts"))
            elif isinstance(value, dict):
                if any(token in lowered for token in ("flag", "issue", "class", "count")):
                    found.update(str(item_key) for item_key, item_value in value.items() if item_value)
                found.update(_collect_issue_summary_tokens(value))
            elif isinstance(value, list):
                if any(token in lowered for token in ("flag", "issue", "class")):
                    found.update(item for item in value if isinstance(item, str) and item)
                for item in value:
                    found.update(_collect_issue_summary_tokens(item))
            elif isinstance(value, str) and any(token in lowered for token in ("flag", "issue", "class")):
                found.add(value)
    return found


def _iter_text_layers(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "text_layers" and isinstance(child, list):
                yield from child
            else:
                yield from _iter_text_layers(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_text_layers(item)


def _collect_qa_flag_tokens(flags: object) -> set[str]:
    found: set[str] = set()
    if isinstance(flags, str):
        found.add(flags)
    elif isinstance(flags, list):
        for item in flags:
            found.update(_collect_qa_flag_tokens(item))
    elif isinstance(flags, dict):
        for key, value in flags.items():
            if isinstance(value, bool) and value:
                found.add(str(key))
            elif isinstance(value, (int, float)) and value > 0:
                found.add(str(key))
            else:
                found.update(_collect_qa_flag_tokens(value))
    return found


def collect_observable_issue_classes(project: dict) -> set[str]:
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    found: set[str] = set()
    export_gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), dict) else {}
    found.update(_collect_issue_from_issue_item(export_gate.get("issues", [])))
    found.update(_collect_issue_from_issue_item(qa.get("issues", [])))
    found.update(_collect_issue_summary_tokens(qa.get("summary", {})))
    for layer in _iter_text_layers(project):
        if isinstance(layer, dict):
            found.update(_collect_qa_flag_tokens(layer.get("qa_flags", [])))
    return found


def _read_project_if_available(run_path: Path) -> dict:
    project_path = run_path / "project.json"
    if not project_path.exists():
        return {}
    with project_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _export_gate_status(project: dict) -> tuple[str, str]:
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    export_gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), dict) else {}
    status = str(export_gate.get("status") or "UNKNOWN")
    if status == "UNKNOWN" and project:
        status = "PASS"
    needs_review = "yes" if export_gate.get("needs_review") else "no"
    return status, needs_review


def _manifest_report_row(manifest_path: Path) -> list[str]:
    manifest = _load_manifest(manifest_path)
    run_path = _resolve_run_path(manifest["run_path"], manifest_path)
    project = _read_project_if_available(run_path)
    status, needs_review = _export_gate_status(project)
    if not run_path.exists():
        status = "MISSING_RUN"
        needs_review = "unknown"
    pages = ", ".join(str(page) for page in manifest.get("pages_of_interest") or []) or "-"
    flags_before = ", ".join(manifest.get("current_issue_classes") or []) or "-"
    target = ", ".join(manifest.get("target_issue_classes_after_fix") or []) or "-"
    evidence = ", ".join(
        str(item.get("crop_path") or "")
        for item in manifest.get("sample_artifacts") or []
        if isinstance(item, dict) and item.get("crop_path")
    ) or "-"
    return [
        str(manifest.get("run_id") or manifest_path.stem),
        pages,
        flags_before,
        target,
        status,
        needs_review,
        evidence,
    ]


def write_visual_regression_report(
    manifest_paths: list[Path] | None = None,
    *,
    output_path: Path | None = None,
) -> Path:
    manifest_paths = manifest_paths or _manifest_paths()
    report_manifest_paths = [path for path in manifest_paths if path.name != "ci_visual_smoke.json"]
    if not report_manifest_paths:
        report_manifest_paths = manifest_paths
    rows = [_manifest_report_row(path) for path in report_manifest_paths]
    report_path = Path(output_path) if output_path is not None else REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Visual Regression Report",
        "",
        "Generated from `pipeline/tests/regression/manifests/*.json`.",
        "",
        "| Chapter | Pages | Flags before/current | Flags after/target | Final status | Needs review | Evidence |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        escaped = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def validate_manifest(manifest: dict, manifest_path: Path) -> None:
    _validate_manifest_shape(manifest, manifest_path)

    run_path = _resolve_run_path(manifest["run_path"], manifest_path)
    ci_fixture = bool(manifest.get("ci_fixture", False))
    if not run_path.exists() and not ci_fixture:
        reason = f"Visual regression run_path missing for {manifest['run_id']}: {run_path}"
        warnings.warn(reason, stacklevel=2)
        pytest.skip(reason)

    project_path = run_path / "project.json"
    qa_report_path = run_path / "qa_report.json"
    performance_path = run_path / "performance_timing.json"
    debug_e2e_path = run_path / "debug" / "e2e"
    _require(project_path.exists(), f"{manifest_path} requires {project_path}")
    _require(debug_e2e_path.exists(), f"{manifest_path} requires {debug_e2e_path}")
    if not ci_fixture:
        _require(qa_report_path.exists(), f"{manifest_path} requires {qa_report_path}")
        digest = hashlib.sha256(qa_report_path.read_bytes()).hexdigest()
        _require(
            digest == manifest["qa_report_sha256_at_record_time"],
            f"{manifest_path} qa_report_sha256_at_record_time does not match {qa_report_path}",
        )
        if performance_path.exists():
            performance = json.loads(performance_path.read_text(encoding="utf-8"))
            current_total = float(performance.get("total_sec") or 0.0)
            baseline_total = float(manifest["baseline_total_sec"])
            _require(
                current_total <= baseline_total * 1.15,
                f"{manifest_path} performance budget exceeded: {current_total:.4f}s > {baseline_total * 1.15:.4f}s",
            )

    with project_path.open("r", encoding="utf-8") as handle:
        project = json.load(handle)

    project_schema_version = project.get("schema_version")
    if isinstance(project_schema_version, int):
        _require(
            project_schema_version == manifest["schema_version"],
            f"{manifest_path} schema_version {manifest['schema_version']} does not match project.json {project_schema_version}",
        )

    observable_issue_classes = collect_observable_issue_classes(project)
    for issue_class in manifest["current_issue_classes"]:
        _require(
            issue_class in observable_issue_classes,
            f"Expected issue class '{issue_class}' missing from run {manifest['run_id']}.",
        )

    for page_number in manifest["pages_of_interest"]:
        page_token = f"page_{page_number:03d}"
        matches = [
            path
            for path in debug_e2e_path.rglob("*")
            if path.is_file() and page_token in str(path.relative_to(debug_e2e_path)).replace("\\", "/")
        ]
        _require(
            bool(matches),
            f"Expected debug/e2e artifact for page {page_number} missing from run {manifest['run_id']}.",
        )

    for artifact in manifest["sample_artifacts"]:
        crop_path = _resolve_run_artifact_path(run_path, artifact["crop_path"])
        _require(crop_path.exists(), f"{manifest_path} missing sample artifact: {crop_path}")


@pytest.mark.parametrize("manifest_path", _manifest_paths(), ids=lambda path: path.stem)
def test_visual_regression_manifest(manifest_path: Path):
    validate_manifest(_load_manifest(manifest_path), manifest_path)


def test_visual_regression_report_is_generated(tmp_path: Path):
    tracked_report_before = REPORT_PATH.read_bytes()
    report_path = write_visual_regression_report(
        _manifest_paths(),
        output_path=tmp_path / "visual_regression_report.md",
    )

    content = report_path.read_text(encoding="utf-8")
    assert REPORT_PATH.read_bytes() == tracked_report_before
    assert "| Chapter | Pages | Flags before/current | Flags after/target | Final status | Needs review | Evidence |" in content
    expected_run_id = "ci_visual_smoke" if _running_in_ci() else "one_second_ch2"
    assert expected_run_id in content


def test_manifest_qa_report_hash_mismatch_fails(tmp_path: Path):
    run_path = tmp_path / "run"
    (run_path / "debug" / "e2e" / "09_typeset").mkdir(parents=True)
    (run_path / "debug" / "e2e" / "09_typeset" / "page_001_artifact.png").write_bytes(b"png")
    (run_path / "project.json").write_text(
        json.dumps({"schema_version": 12, "qa": {}, "text_layers": []}),
        encoding="utf-8",
    )
    (run_path / "qa_report.json").write_text("{}", encoding="utf-8")
    (run_path / "performance_timing.json").write_text(json.dumps({"total_sec": 1.0}), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "run_id": "temp_hash",
        "run_path": str(run_path),
        "ci_fixture": False,
        "schema_version": 12,
        "current_issue_classes": [],
        "target_issue_classes_after_fix": ["none_or_waived"],
        "pages_of_interest": [1],
        "sample_artifacts": [],
        "qa_report_sha256_at_record_time": "0" * 64,
        "baseline_total_sec": 1.0,
    }

    with pytest.raises(AssertionError, match="qa_report_sha256_at_record_time does not match"):
        validate_manifest(manifest, tmp_path / "temp_hash.json")


def test_expired_manifest_waiver_fails():
    manifest = {
        "manifest_version": 1,
        "run_id": "temp_waiver",
        "run_path": "N:/TraduzAI/DEBUGM/runs/temp_waiver",
        "ci_fixture": False,
        "schema_version": 12,
        "current_issue_classes": [],
        "target_issue_classes_after_fix": ["waive_known_review_flags_explicitly"],
        "pages_of_interest": [],
        "sample_artifacts": [],
        "qa_report_sha256_at_record_time": "0" * 64,
        "baseline_total_sec": 1.0,
        "waivers": [
            {
                "waiver_id": "expired",
                "flag": "TEXT_CLIPPED",
                "scope": "debug_only",
                "reason": "expired fixture waiver",
                "expires": "2026-01-01",
                "approved_by": "codex",
            }
        ],
    }

    with pytest.raises(AssertionError, match="Waiver expired expired 2026-01-01"):
        _validate_manifest_shape(manifest, Path("temp_manifest.json"))


def test_missing_expected_issue_class_fails_with_required_message(tmp_path: Path):
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "debug" / "e2e").mkdir(parents=True)
    (run_path / "debug" / "e2e" / "page_001_artifact.txt").write_text("artifact", encoding="utf-8")
    (run_path / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 12,
                "qa": {
                    "export_gate": {"issues": [{"flag": "TEXT_OVERFLOW"}]},
                    "issues": [],
                    "summary": {},
                },
                "text_layers": [],
            }
        ),
        encoding="utf-8",
    )
    qa_report_content = json.dumps({"export_gate": {"issues": [{"flag": "TEXT_OVERFLOW"}]}})
    (run_path / "qa_report.json").write_text(qa_report_content, encoding="utf-8")
    (run_path / "performance_timing.json").write_text(json.dumps({"total_sec": 1.0}), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "run_id": "temp_missing_flag",
        "run_path": str(run_path),
        "ci_fixture": False,
        "schema_version": 12,
        "current_issue_classes": ["TEXT_CLIPPED"],
        "target_issue_classes_after_fix": ["none_or_waived"],
        "pages_of_interest": [1],
        "sample_artifacts": [],
        "qa_report_sha256_at_record_time": hashlib.sha256(qa_report_content.encode("utf-8")).hexdigest(),
        "baseline_total_sec": 1.0,
    }

    with pytest.raises(AssertionError, match="Expected issue class 'TEXT_CLIPPED' missing from run temp_missing_flag"):
        validate_manifest(manifest, Path("temp_manifest.json"))


def _resolve_run_path(raw_path: str, manifest_path: Path) -> Path:
    run_path = Path(raw_path)
    _require(not str(raw_path).startswith("\\\\"), "run_path must not be UNC")
    _require(
        not (run_path.drive and not run_path.is_absolute()),
        "run_path must not be drive-relative",
    )
    if run_path.is_absolute():
        return run_path.resolve()
    return (manifest_path.parent / run_path).resolve()


def _resolve_run_artifact_path(run_root: Path, raw_path: str) -> Path:
    artifact_path = Path(str(raw_path or ""))
    _require(
        bool(str(raw_path or "").strip())
        and not artifact_path.is_absolute()
        and not artifact_path.drive
        and not artifact_path.root,
        "sample artifact path must stay inside run_path",
    )
    resolved_root = run_root.resolve()
    candidate = (resolved_root / artifact_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise AssertionError("sample artifact path must stay inside run_path") from exc
    return candidate


@pytest.mark.parametrize(
    "raw_path",
    [
        "../outside.png",
        "..\\outside.png",
        r"C:\outside.png",
        r"C:relative.png",
        r"\rooted\outside.png",
        r"\\server\share\outside.png",
    ],
)
def test_manifest_artifact_path_cannot_escape_run_root(tmp_path: Path, raw_path: str):
    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(AssertionError, match="must stay inside run_path"):
        _resolve_run_artifact_path(run_root, raw_path)


def test_manifest_artifact_path_accepts_nested_relative_path(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    resolved = _resolve_run_artifact_path(
        run_root,
        "debug/e2e/09_typeset/page_001.png",
    )

    assert resolved == (run_root / "debug" / "e2e" / "09_typeset" / "page_001.png").resolve()


def test_pages_of_interest_requires_corresponding_debug_e2e_artifact(tmp_path: Path):
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "debug" / "e2e" / "09_typeset").mkdir(parents=True)
    (run_path / "project.json").write_text(
        json.dumps({"schema_version": 12, "qa": {}, "text_layers": []}),
        encoding="utf-8",
    )
    qa_report_content = "{}"
    (run_path / "qa_report.json").write_text(qa_report_content, encoding="utf-8")
    (run_path / "performance_timing.json").write_text(json.dumps({"total_sec": 1.0}), encoding="utf-8")

    manifest = {
        "manifest_version": 1,
        "run_id": "temp_page_artifact",
        "run_path": str(run_path),
        "ci_fixture": False,
        "schema_version": 12,
        "current_issue_classes": [],
        "target_issue_classes_after_fix": ["none_or_waived"],
        "pages_of_interest": [7],
        "sample_artifacts": [],
        "qa_report_sha256_at_record_time": hashlib.sha256(qa_report_content.encode("utf-8")).hexdigest(),
        "baseline_total_sec": 1.0,
    }

    with pytest.raises(
        AssertionError,
        match="Expected debug/e2e artifact for page 7 missing from run temp_page_artifact",
    ):
        validate_manifest(manifest, Path("temp_manifest.json"))


def test_manifest_paths_in_ci_only_selects_ci_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    local_manifest = tmp_path / "local.json"
    local_manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "run_id": "local_only",
                "run_path": "N:/TraduzAI/DEBUGM/runs/local_only",
                "ci_fixture": False,
                "schema_version": 12,
                "current_issue_classes": [],
                "target_issue_classes_after_fix": ["none_or_waived"],
                "pages_of_interest": [],
                "sample_artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    ci_manifest = tmp_path / "ci.json"
    ci_manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "run_id": "ci_fixture",
                "run_path": "fixtures/visual/ci_fixture",
                "ci_fixture": True,
                "schema_version": 12,
                "current_issue_classes": [],
                "target_issue_classes_after_fix": ["none_or_waived"],
                "pages_of_interest": [],
                "sample_artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys.modules[__name__], "MANIFEST_DIR", tmp_path)
    monkeypatch.setenv("CI", "true")

    assert _manifest_paths() == [ci_manifest]
