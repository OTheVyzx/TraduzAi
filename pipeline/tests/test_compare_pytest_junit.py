from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


def _module():
    spec = importlib.util.find_spec("tools.compare_pytest_junit")
    assert spec is not None, "tools.compare_pytest_junit must exist"
    return importlib.import_module("tools.compare_pytest_junit")


def _write_junit(
    path: Path,
    *,
    passed: list[tuple[str, str]] | None = None,
    failures: list[tuple[str, str]] | None = None,
    errors: list[tuple[str, str]] | None = None,
    skipped: list[tuple[str, str]] | None = None,
) -> Path:
    root = ET.Element("testsuite", name="pytest")
    for status, rows in (
        ("passed", passed or []),
        ("failure", failures or []),
        ("error", errors or []),
        ("skipped", skipped or []),
    ):
        for classname, name in rows:
            testcase = ET.SubElement(root, "testcase", classname=classname, name=name)
            if status != "passed":
                ET.SubElement(testcase, status, message=f"{status} fixture")
    root.set("tests", str(len(root)))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_junit_comparator_reports_only_new_failure_error_nodeids(tmp_path):
    module = _module()
    baseline = _write_junit(tmp_path / "baseline.xml", failures=[("a.Test", "test_old[x]")])
    candidate = _write_junit(
        tmp_path / "candidate.xml",
        failures=[("a.Test", "test_old[x]"), ("b.Test", "test_new[y]")],
    )

    result = module.compare_junit(baseline, candidate)

    assert result["new_failures"] == [{"classname": "b.Test", "name": "test_new[y]"}]
    assert result["resolved_failures"] == []
    assert result["status"] == "BLOCK"


def test_junit_comparator_counts_errors_as_failures(tmp_path):
    module = _module()
    baseline = _write_junit(tmp_path / "baseline.xml", passed=[("a.Test", "test_old")])
    candidate = _write_junit(tmp_path / "candidate.xml", errors=[("a.Test", "test_old")])

    result = module.compare_junit(baseline, candidate)

    assert result["new_failures"] == [{"classname": "a.Test", "name": "test_old"}]
    assert result["status"] == "BLOCK"


@pytest.mark.parametrize("contents", ["<bad>", "<testsuite tests='0'/>", ""])
def test_junit_comparator_rejects_malformed_or_empty_suite(tmp_path, contents):
    module = _module()
    bad = tmp_path / "bad.xml"
    bad.write_text(contents, encoding="utf-8")
    valid = _write_junit(tmp_path / "valid.xml", passed=[("a.Test", "test_ok")])

    with pytest.raises(module.JunitComparisonError):
        module.compare_junit(bad, valid)


def test_junit_comparator_rejects_missing_xml(tmp_path):
    module = _module()
    valid = _write_junit(tmp_path / "valid.xml", passed=[("a.Test", "test_ok")])
    with pytest.raises(module.JunitComparisonError):
        module.compare_junit(tmp_path / "missing.xml", valid)


def test_junit_comparator_blocks_baseline_nodeid_missing_from_candidate(tmp_path):
    module = _module()
    baseline = _write_junit(
        tmp_path / "baseline.xml",
        passed=[("a.Test", "test_kept"), ("a.Test", "test_must_not_disappear")],
    )
    candidate = _write_junit(tmp_path / "candidate.xml", passed=[("a.Test", "test_kept")])

    result = module.compare_junit(baseline, candidate)

    assert result["missing_nodeids"] == [
        {"classname": "a.Test", "name": "test_must_not_disappear"}
    ]
    assert result["status"] == "BLOCK"


def test_junit_comparator_reports_resolved_and_allows_new_passing_nodeids(tmp_path):
    module = _module()
    baseline = _write_junit(tmp_path / "baseline.xml", failures=[("a.Test", "test_fixed")])
    candidate = _write_junit(
        tmp_path / "candidate.xml",
        passed=[("a.Test", "test_fixed"), ("b.Test", "test_added")],
    )

    result = module.compare_junit(baseline, candidate)

    assert result["resolved_failures"] == [{"classname": "a.Test", "name": "test_fixed"}]
    assert result["new_failures"] == []
    assert result["missing_nodeids"] == []
    assert result["status"] == "PASS"
