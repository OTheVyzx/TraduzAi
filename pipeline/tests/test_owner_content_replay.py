from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

import main
from ownership.chapter_contract import ChapterSourceManifest
from ownership.execution import PageCandidateTransaction
from strip.page_pipeline import (
    ContentReplayIntegrityError,
    adapt_page_execution_result_to_output_page,
    load_owner_content_replay,
)
from test_chapter_publication import _generation, _verified_result


def _publication(tmp_path: Path):
    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    ref = PageCandidateTransaction(
        private_execution_root=root,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-001",
        transaction_id="transaction-001",
    ).commit_verified_generation(result)
    source = ChapterSourceManifest.from_extracted_pages(
        (root / "originals" / "page_001.png",), root,
        run_id=marker.run_id, execution_id=marker.execution_id,
    )
    inputs = main._project_inputs_from_output_pages(
        source,
        [adapt_page_execution_result_to_output_page(result, evidence_ref=ref)],
        private_execution_root=root,
    )
    return main._wrap_up_verified_owner_pages(
        inputs, source_private_execution_root=root
    )


def test_content_replay_reopens_complete_verified_page_journal(tmp_path):
    bundle = _publication(tmp_path)

    replay = load_owner_content_replay(bundle.runtime_staging_root)

    assert replay.run_id == "run-a"
    assert replay.execution_id == "execution-a"
    assert len(replay.pages) == 1
    assert replay.pages[0].status == "final_verified"
    assert replay.pages[0].result_sha256 == replay.verified_inputs.pages[0].page_result_sha256


def test_content_replay_rejects_tampered_generation_before_exposing_pages(tmp_path):
    bundle = _publication(tmp_path)
    final_path = bundle.runtime_staging_root / bundle.export_manifest.pages[0].translated_path
    Image.new("RGB", (140, 90), (0, 0, 0)).save(final_path)

    with pytest.raises(ContentReplayIntegrityError):
        load_owner_content_replay(bundle.runtime_staging_root)
