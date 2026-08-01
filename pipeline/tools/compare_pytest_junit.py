"""Compare two pytest JUnit inventories without hiding collection drift."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


class JunitComparisonError(ValueError):
    """Raised when a JUnit document cannot serve as comparison evidence."""


NodeId = tuple[str, str]


def _node_payload(nodeid: NodeId) -> dict[str, str]:
    return {"classname": nodeid[0], "name": nodeid[1]}


def _parse_junit(path: str | Path) -> dict[NodeId, str]:
    source = Path(path)
    if not source.is_file():
        raise JunitComparisonError(f"JUnit XML does not exist: {source}")
    try:
        if source.stat().st_size == 0:
            raise JunitComparisonError(f"JUnit XML is empty: {source}")
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        raise JunitComparisonError(f"invalid JUnit XML {source}: {exc}") from exc

    cases: dict[NodeId, str] = {}
    for testcase in root.iter("testcase"):
        classname = str(testcase.get("classname") or "").strip()
        name = str(testcase.get("name") or "").strip()
        if not classname or not name:
            raise JunitComparisonError(
                f"JUnit testcase lacks classname/name in {source}"
            )
        nodeid = (classname, name)
        if nodeid in cases:
            raise JunitComparisonError(
                f"duplicate JUnit testcase identity in {source}: {nodeid!r}"
            )
        if testcase.find("failure") is not None:
            status = "failure"
        elif testcase.find("error") is not None:
            status = "error"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        cases[nodeid] = status
    if not cases:
        raise JunitComparisonError(f"JUnit XML has zero testcases: {source}")
    return cases


def _status_counts(cases: dict[NodeId, str]) -> dict[str, int]:
    return {
        status: sum(value == status for value in cases.values())
        for status in ("passed", "skipped", "failure", "error")
    }


def compare_junit(baseline: str | Path, candidate: str | Path) -> dict[str, Any]:
    """Return a deterministic differential report for two JUnit documents."""

    baseline_cases = _parse_junit(baseline)
    candidate_cases = _parse_junit(candidate)
    baseline_ids = set(baseline_cases)
    candidate_ids = set(candidate_cases)
    baseline_failed = {
        nodeid for nodeid, status in baseline_cases.items() if status in {"failure", "error"}
    }
    candidate_failed = {
        nodeid for nodeid, status in candidate_cases.items() if status in {"failure", "error"}
    }
    missing = baseline_ids - candidate_ids
    new_failures = sorted(candidate_failed - baseline_failed)
    resolved = sorted((baseline_failed - candidate_failed) - missing)
    missing_nodeids = sorted(missing)
    status = "BLOCK" if new_failures or missing_nodeids else "PASS"
    return {
        "schema_version": 1,
        "status": status,
        "baseline": {
            "path": str(Path(baseline).resolve()),
            "testcase_count": len(baseline_cases),
            "status_counts": _status_counts(baseline_cases),
        },
        "candidate": {
            "path": str(Path(candidate).resolve()),
            "testcase_count": len(candidate_cases),
            "status_counts": _status_counts(candidate_cases),
        },
        "new_failures": [_node_payload(nodeid) for nodeid in new_failures],
        "resolved_failures": [_node_payload(nodeid) for nodeid in resolved],
        "missing_nodeids": [_node_payload(nodeid) for nodeid in missing_nodeids],
        "new_nodeids": [
            _node_payload(nodeid) for nodeid in sorted(candidate_ids - baseline_ids)
        ],
    }


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument("--fail-on-new", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = compare_junit(args.baseline, args.candidate)
        encoded = _canonical_json(result)
        if args.output:
            _write_atomic(Path(args.output), encoded)
        sys.stdout.write(encoded)
        return 2 if args.fail_on_new and result["status"] == "BLOCK" else 0
    except JunitComparisonError as exc:
        sys.stderr.write(f"junit comparison error: {exc}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
