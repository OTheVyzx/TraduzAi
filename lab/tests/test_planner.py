from __future__ import annotations

import unittest

from lab.critics.base import Finding
from lab.planner import (
    ISSUE_PROFILES,
    Proposal,
    build_proposals,
    proposal_to_lab_payload,
)


def _finding(
    *,
    issue_type: str = "low_confidence",
    severity: str = "warning",
    suggested_file: str = "pipeline/vision_stack/ocr.py",
    evidence: dict | None = None,
    page_index: int = 0,
    chapter_number: int = 1,
    critic_id: str = "ocr_critic",
    suggested_anchor: str = "",
) -> Finding:
    return Finding(
        critic_id=critic_id,
        chapter_number=chapter_number,
        page_index=page_index,
        issue_type=issue_type,
        severity=severity,
        evidence=evidence or {},
        suggested_file=suggested_file,
        suggested_anchor=suggested_anchor,
        bbox=None,
    )


BENCHMARK_GREEN = {
    "green": True,
    "metrics": {
        "textual_similarity": 92.0,
        "term_consistency": 90.0,
        "layout_occupancy": 85.0,
        "readability": 80.0,
        "visual_cleanup": 95.0,
        "manual_edits_saved": 70.0,
    },
}

BENCHMARK_POOR = {
    "green": False,
    "metrics": {
        "textual_similarity": 45.0,
        "term_consistency": 50.0,
        "layout_occupancy": 40.0,
        "readability": 55.0,
        "visual_cleanup": 60.0,
        "manual_edits_saved": 20.0,
    },
}


class BuildProposalsTests(unittest.TestCase):
    def test_empty_findings_yields_no_proposals(self) -> None:
        proposals = build_proposals([], BENCHMARK_GREEN, run_id="abcdef123456")
        self.assertEqual(proposals, [])

    def test_groups_by_issue_and_file(self) -> None:
        findings = [
            _finding(issue_type="low_confidence", evidence={"confidence": 0.3}),
            _finding(issue_type="low_confidence", evidence={"confidence": 0.4}, page_index=1),
            _finding(issue_type="watermark_leaked", severity="error", suggested_file="lab/benchmarking.py"),
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="run00001")
        self.assertEqual(len(proposals), 2)
        issue_types = {p.issue_type for p in proposals}
        self.assertIn("low_confidence", issue_types)
        self.assertIn("watermark_leaked", issue_types)

    def test_prioritizes_higher_severity_and_count(self) -> None:
        findings = [
            _finding(issue_type="ocr_artifact_repeated_digits", severity="info"),
            _finding(issue_type="watermark_leaked", severity="error", suggested_file="lab/benchmarking.py"),
            _finding(issue_type="watermark_leaked", severity="error", suggested_file="lab/benchmarking.py", page_index=2),
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="runxxxx")
        self.assertGreaterEqual(len(proposals), 2)
        # Watermark deve vir primeiro (error + 2 occurrences)
        self.assertEqual(proposals[0].issue_type, "watermark_leaked")

    def test_change_kind_from_profile(self) -> None:
        findings = [
            _finding(
                issue_type="watermark_leaked",
                severity="error",
                suggested_file="lab/benchmarking.py",
                evidence={"matched_token": "asura"},
            )
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="rrrrrrr1")
        self.assertEqual(proposals[0].change_kind, "regex_add")
        self.assertFalse(proposals[0].needs_coder)
        self.assertIn("asura", proposals[0].local_patch_hint["new_watermark_tokens"])

    def test_logic_fix_requires_coder(self) -> None:
        findings = [
            _finding(
                issue_type="untranslated",
                severity="error",
                suggested_file="pipeline/translator/translate.py",
            )
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="run55555")
        self.assertEqual(proposals[0].change_kind, "logic_fix")
        self.assertTrue(proposals[0].needs_coder)

    def test_threshold_tune_low_confidence_hint(self) -> None:
        findings = [
            _finding(issue_type="low_confidence", evidence={"confidence": 0.3}),
            _finding(issue_type="low_confidence", evidence={"confidence": 0.5}, page_index=1),
            _finding(issue_type="low_confidence", evidence={"confidence": 0.45}, page_index=2),
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="runzzzzz")
        self.assertEqual(proposals[0].change_kind, "threshold_tune")
        self.assertIn("median_observed", proposals[0].local_patch_hint)

    def test_metric_gap_raises_priority(self) -> None:
        findings_same = [
            _finding(issue_type="low_confidence", evidence={"confidence": 0.4})
        ]
        prop_poor = build_proposals(findings_same, BENCHMARK_POOR, run_id="a")[0]
        prop_green = build_proposals(findings_same, BENCHMARK_GREEN, run_id="b")[0]
        self.assertGreater(prop_poor.priority_score, prop_green.priority_score)

    def test_unknown_issue_defaults_to_logic_fix(self) -> None:
        findings = [_finding(issue_type="unknown_new_bug", severity="warning")]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="x")
        self.assertEqual(proposals[0].change_kind, "logic_fix")
        self.assertTrue(proposals[0].needs_coder)

    def test_touched_domains_inferred(self) -> None:
        findings = [
            _finding(suggested_file="pipeline/vision_stack/ocr.py"),
            _finding(suggested_file="src-tauri/src/lib.rs", issue_type="watermark_leaked", severity="error"),
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="y")
        domains = {p.touched_domains[0] for p in proposals}
        self.assertIn("pipeline/**", domains)
        self.assertIn("src-tauri/**", domains)

    def test_suggested_anchor_captured(self) -> None:
        findings = [
            _finding(suggested_anchor="_is_meaningful_benchmark_text", issue_type="watermark_leaked", severity="error"),
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="z")
        self.assertEqual(proposals[0].target_anchor, "_is_meaningful_benchmark_text")


class ProposalToLabPayloadTests(unittest.TestCase):
    def test_payload_shape(self) -> None:
        findings = [
            _finding(
                issue_type="watermark_leaked",
                severity="error",
                suggested_file="lab/benchmarking.py",
                evidence={"matched_token": "discord.gg"},
            )
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="pay12345")
        payload = proposal_to_lab_payload(proposals[0], "pay12345", git_available=True)

        # Campos obrigatorios (legacy LabProposal)
        for key in (
            "proposal_id",
            "batch_id",
            "title",
            "summary",
            "author",
            "risk",
            "touched_domains",
            "review_findings",
            "proposal_status",
            "pr_status",
            "git_available",
        ):
            self.assertIn(key, payload)

        # Campos novos
        self.assertIn("change_kind", payload)
        self.assertIn("needs_coder", payload)
        self.assertIn("local_patch_hint", payload)
        self.assertIn("expected_metric_gain", payload)
        self.assertEqual(payload["risk"], "alto")  # severity=error
        self.assertIsInstance(payload["local_patch_hint"], dict)

    def test_warning_severity_maps_to_medium_risk(self) -> None:
        findings = [
            _finding(
                issue_type="occupancy_too_low",
                severity="info",
                suggested_file="pipeline/typesetter/renderer.py",
            ),
            _finding(
                issue_type="occupancy_too_low",
                severity="warning",
                suggested_file="pipeline/typesetter/renderer.py",
                page_index=1,
            ),
        ]
        proposals = build_proposals(findings, BENCHMARK_POOR, run_id="warn0000")
        payload = proposal_to_lab_payload(proposals[0], "warn0000", git_available=False)
        self.assertEqual(payload["risk"], "medio")
        self.assertFalse(payload["git_available"])


class IssueProfilesIntegrityTests(unittest.TestCase):
    def test_all_profiles_have_required_keys(self) -> None:
        for issue_type, profile in ISSUE_PROFILES.items():
            self.assertIn("change_kind", profile, f"Missing change_kind in {issue_type}")
            self.assertIn("linked_metric", profile, f"Missing linked_metric in {issue_type}")
            self.assertIn(
                profile["change_kind"],
                {"regex_add", "threshold_tune", "logic_fix"},
                f"Invalid change_kind in {issue_type}",
            )


if __name__ == "__main__":
    unittest.main()
