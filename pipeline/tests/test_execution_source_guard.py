from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from qa.execution_source_guard import (
    ExecutionSourceError,
    collect_execution_source_ledger,
    finalize_acceptance_execution_ledgers,
    finalize_execution_source_ledger,
)


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"; source = repo / "pipeline" / "module.py"; source.parent.mkdir(parents=True); source.write_text("VALUE = 1\n")
    import hashlib
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    bundle = {"repo_root": str(repo), "execution_sources": {"pipeline/module.py": {"content_sha256": digest}}, "source_exclusions": []}
    ledger = {"schema_version": 1, "loaded_sources": {"pipeline/module.py": digest}}
    return repo, source, bundle, ledger


def test_execution_guard_accepts_hash_bound_loaded_source(tmp_path: Path):
    _repo, _source, bundle, ledger = _fixture(tmp_path)
    assert finalize_execution_source_ledger(bundle, ledger=ledger)["status"] == "PASS"


def test_execution_guard_ignores_loaded_sources_covered_by_bundle_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, _source, bundle, _ledger = _fixture(tmp_path)
    excluded = repo / "pipeline" / "venv" / "Lib" / "site-packages" / "dependency.py"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("VALUE = 1\n", encoding="utf-8")
    bundle["source_exclusions"] = ["pipeline/venv/**"]
    monkeypatch.setitem(
        sys.modules,
        "traduzai_test_excluded_dependency",
        SimpleNamespace(__file__=str(excluded)),
    )

    ledger = collect_execution_source_ledger(bundle)

    assert "pipeline/venv/Lib/site-packages/dependency.py" not in ledger["loaded_sources"]


def test_execution_guard_resolves_synthetic_relative_module_file_from_spec_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, _source, bundle, _ledger = _fixture(tmp_path)
    dependency = repo / "pipeline" / "venv" / "Lib" / "site-packages" / "torch" / "_ops.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    bundle["source_exclusions"] = ["pipeline/venv/**"]
    monkeypatch.chdir(repo / "pipeline")
    monkeypatch.setitem(
        sys.modules,
        "traduzai_test_synthetic_file",
        SimpleNamespace(
            __file__="_ops.py",
            __spec__=SimpleNamespace(origin=str(dependency)),
        ),
    )

    ledger = collect_execution_source_ledger(bundle)

    assert "pipeline/_ops.py" not in ledger["loaded_sources"]


def test_execution_guard_blocks_loaded_hash_mismatch(tmp_path: Path):
    _repo, _source, bundle, ledger = _fixture(tmp_path); ledger["loaded_sources"]["pipeline/module.py"] = "0" * 64
    with pytest.raises(ExecutionSourceError, match="loaded_hash_mismatch"):
        finalize_execution_source_ledger(bundle, ledger=ledger)


def test_execution_guard_blocks_missing_child_ledger(tmp_path: Path):
    _repo, _source, bundle, ledger = _fixture(tmp_path)
    with pytest.raises(ExecutionSourceError, match="missing_child_ledger"):
        finalize_acceptance_execution_ledgers(bundle, {"benchmark": ledger}, required_children={"benchmark", "matrix", "pipeline"})
