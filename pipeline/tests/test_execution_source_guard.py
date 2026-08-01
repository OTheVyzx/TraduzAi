from __future__ import annotations

import json
from pathlib import Path

import pytest

from qa.execution_source_guard import (
    ExecutionSourceError,
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


def test_execution_guard_blocks_loaded_hash_mismatch(tmp_path: Path):
    _repo, _source, bundle, ledger = _fixture(tmp_path); ledger["loaded_sources"]["pipeline/module.py"] = "0" * 64
    with pytest.raises(ExecutionSourceError, match="loaded_hash_mismatch"):
        finalize_execution_source_ledger(bundle, ledger=ledger)


def test_execution_guard_blocks_missing_child_ledger(tmp_path: Path):
    _repo, _source, bundle, ledger = _fixture(tmp_path)
    with pytest.raises(ExecutionSourceError, match="missing_child_ledger"):
        finalize_acceptance_execution_ledgers(bundle, {"benchmark": ledger}, required_children={"benchmark", "matrix", "pipeline"})
