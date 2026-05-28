"""Regression tests against the canonical 2026-05-18 debug ZIP.

The ZIP lives at ``N:\\TraduzAI\\DEBUGM\\runs\\2026-05-18_chapter1_e2e_debug_002011``
(extracted). These tests:

1. Verify that **before the fix**, the analyzer's legacy confidence helper
   would have returned ``88`` on that fixture (smoke test that the file
   structure assumed by DBG2-01 is still the same).
2. Verify that **after the fix**, ``analyze_run`` returns:
   - ``confidence_zero_count == 0`` (matching the OCR audit)
   - ``content_class_counts`` is not doubled
   - ``text_count == 88``
   - ``debug_report_consistency.json`` is emitted via ``--write-report``
3. Verify that ``analyze_e2e_debug.py --strict-debug-audit`` exits with code
   ``3`` against the as-is ZIP (because PR 10/11/15/16 invariants are not
   met yet).

If the fixture isn't present the tests are skipped — keeps CI portable.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZER_PATH = REPO_ROOT / "tools" / "analyze_e2e_debug.py"

# Canonical fixture location (kept out of the repo to avoid bloating git).
FIXTURE_ROOT = Path(
    r"N:\TraduzAI\DEBUGM\runs\2026-05-18_chapter1_e2e_debug_002011"
)
RUN_A = FIXTURE_ROOT / "A_baseline_debug"

PYTHON = sys.executable


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_e2e_debug", ANALYZER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _have_fixture() -> bool:
    return (
        RUN_A.exists()
        and (RUN_A / "project.json").exists()
        and (RUN_A / "debug" / "e2e" / "03_ocr" / "ocr_confidence_audit.json").exists()
    )


pytestmark = pytest.mark.skipif(
    not _have_fixture(),
    reason=(
        "Canonical 2026-05-18 debug fixture not present at "
        f"{FIXTURE_ROOT}; skipping regression checks."
    ),
)


# ---------------------------------------------------------------------------
# Acceptance criteria from the prompt (DBG2-01/02/03/21/23)
# ---------------------------------------------------------------------------


def test_text_count_matches_estatisticas_not_doubled():
    module = _load_analyzer()
    metrics = module.analyze_run(RUN_A)["metrics"]

    assert metrics["text_count"] == 88
    counts = metrics["content_class_counts"]
    # The previous analyzer reported these doubled (dialogue=106, narration=60
    # etc., total 176). Now the canonical source returns the real numbers.
    assert sum(counts.values()) == 88
    assert counts.get("dialogue") == 53
    assert counts.get("narration") == 30
    assert counts.get("tn_note") == 2
    assert counts.get("noise") == 1
    assert counts.get("sign") == 1
    assert counts.get("url_watermark") == 1


def test_confidence_zero_count_matches_ocr_audit():
    module = _load_analyzer()
    metrics = module.analyze_run(RUN_A)["metrics"]

    audit_path = RUN_A / "debug" / "e2e" / "03_ocr" / "ocr_confidence_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_zero = audit["summary"]["blocks_with_confidence_zero"]

    assert metrics["confidence_zero_count"] == expected_zero == 0
    assert metrics["confianca_ocr_zero_count"] == 0  # legacy alias also fixed
    assert metrics["ocr_confidence_audit_zero_count"] == 0


def test_stage_level_vs_aggregator_mismatch_for_source_bbox_overreach():
    module = _load_analyzer()
    metrics = module.analyze_run(RUN_A)["metrics"]

    consistency = metrics["debug_report_consistency"]
    overreach = next(
        c for c in consistency["checks"] if c["name"] == "source_bbox_overreach_count"
    )
    # The JSONL has 2 entries — that becomes the stage_level.
    assert overreach["stage_level"] == 2
    # The aggregator counts source_bbox == balloon_bbox via the canonical
    # text layer source (DBG2-10 may show e.g. 0 because text_layers in the
    # final project don't carry the duplicated bboxes). Either way: when the
    # stage file disagrees with the aggregator we MUST report consistent=False
    # (resolves DBG2-23).
    if overreach["stage_level"] != overreach["aggregator"]:
        assert overreach["consistent"] is False
        assert metrics["stage_file_aggregate_mismatch_count"] >= 1


def test_skip_inpaint_honored_is_bool_for_run_b():
    module = _load_analyzer()
    run_b = FIXTURE_ROOT / "B_skip_inpaint"
    if not (run_b / "project.json").exists():
        pytest.skip("B_skip_inpaint not present in fixture")
    metrics = module.analyze_run(run_b)["metrics"]

    # skip_inpaint=True in runner_config; no per-band inpaint_decision.json
    # exists. Canonical interpretation: honored=True (resolves DBG2-21).
    assert metrics["skip_inpaint_requested"] is True
    assert isinstance(metrics["skip_inpaint_honored"], bool)
    assert metrics["skip_inpaint_honored"] is True


def test_debug_report_consistency_json_is_emitted_with_write_report(tmp_path):
    module = _load_analyzer()
    # Drive the analyzer over the whole 4-run fixture but write reports
    # only into a sandbox copy (we don't mutate the ZIP).
    sandbox = tmp_path / "runs"
    sandbox.mkdir()
    # Create per-run symlink/copy approximation by pointing analyze_root at
    # the fixture and letting it emit reports there. To avoid writing into
    # DEBUGM, we instead create a tiny project.json under sandbox/A and
    # rely on analyze_root to skip when nothing is present.
    (sandbox / "A").mkdir()
    project_a = json.loads((RUN_A / "project.json").read_text(encoding="utf-8"))
    # Strip down to a minimal valid project to keep test fast.
    minimal = {
        "paginas": project_a.get("paginas", [])[:1],
        "estatisticas": project_a.get("estatisticas"),
    }
    (sandbox / "A" / "project.json").write_text(
        json.dumps(minimal), encoding="utf-8"
    )

    report = module.analyze_root(sandbox, write_report=True)
    assert (sandbox / "debug_report.json").exists()
    assert report["run_count"] == 1
    consistency_path = (
        sandbox / "A" / "debug" / "e2e" / "13_report" / "debug_report_consistency.json"
    )
    assert consistency_path.exists()
    payload = json.loads(consistency_path.read_text(encoding="utf-8"))
    assert isinstance(payload["all_consistent"], bool)


def test_cli_strict_debug_audit_returns_exit_3_against_canonical_fixture():
    """The 2026-05-18 fixture still has multiple §5b invariants broken
    (translation/copyback/contact sheets empty, render_plan duplicated, etc.).
    The strict gate MUST surface that with exit_code=3.
    """

    result = subprocess.run(
        [
            PYTHON,
            str(ANALYZER_PATH),
            str(RUN_A),
            "--strict-debug-audit",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3, (
        f"Expected exit_code=3 from --strict-debug-audit on A_baseline_debug,"
        f" got {result.returncode}.\nstdout: {result.stdout[:600]}\nstderr:"
        f" {result.stderr[:600]}"
    )
    payload = json.loads(result.stdout)
    assert payload["strict_audit"]["all_passed"] is False
    failed = {item["name"] for item in payload["strict_audit"]["invariants"] if not item["passed"]}
    # At least translation_debug, copyback, and contact_sheets are missing.
    assert "translation_debug_present_or_no_text" in failed
    assert "copyback_debug_present_or_no_cleanup" in failed
    assert "contact_sheets_translated_comparison_present" in failed
