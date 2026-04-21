"""Testes offline para lab.agents — sem rede e sem depender do Claude CLI.

Cobre:
- get_agent_prompt: roteamento target_file → agente correto
- seleção de prompt especialista por reviewer
- _parse_review_response: parser de resposta estruturada
- ClaudeReviewerAgent: fallback rule-based quando Claude indisponível
- ClaudeSDKCoder: fallback quando Claude indisponível, patch local primeiro
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Adiciona raiz do repo ao sys.path para importações relativas
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class TestGetAgentPrompt(unittest.TestCase):
    """Testa roteamento de arquivo → agente correto."""

    def _assert_prompt_contains(
        self,
        target_file: str,
        expected_fragment: str,
        touched_domains: list[str] | None = None,
    ) -> None:
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt(target_file, touched_domains)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn(expected_fragment, result)

    def test_detect_prefix_routes_to_detect(self):
        self._assert_prompt_contains(
            "pipeline/vision_stack/detector.py",
            "TraduzAi **detect expert**",
        )

    def test_ocr_prefix_routes_to_ocr(self):
        self._assert_prompt_contains(
            "pipeline/ocr/postprocess.py",
            "TraduzAi **OCR expert**",
        )

    def test_inpaint_prefix_routes_to_inpaint(self):
        self._assert_prompt_contains(
            "pipeline/inpainter/mask_builder.py",
            "TraduzAi **inpaint expert**",
        )

    def test_runtime_routes_to_vision_stack_orchestrator(self):
        self._assert_prompt_contains(
            "pipeline/vision_stack/runtime.py",
            "TraduzAi **vision stack orchestrator**",
        )

    def test_layout_prefix_routes_to_typesetter(self):
        self._assert_prompt_contains(
            "pipeline/layout/balloon_layout.py",
            "TraduzAi **typesetter expert**",
        )

    def test_translator_prefix_routes_to_translator(self):
        self._assert_prompt_contains(
            "pipeline/translator/translate.py",
            "TraduzAi **translator expert**",
        )

    def test_frontend_prefix_routes_to_tauri_frontend(self):
        self._assert_prompt_contains(
            "src/pages/Lab.tsx",
            "TraduzAi **tauri-frontend expert**",
        )

    def test_tauri_command_prefix_routes_to_tauri_frontend(self):
        self._assert_prompt_contains(
            "src-tauri/src/commands/lab.rs",
            "TraduzAi **tauri-frontend expert**",
        )

    def test_touched_domains_fallback_routes_to_frontend_specialist(self):
        self._assert_prompt_contains(
            "",
            "TraduzAi **tauri-frontend expert**",
            ["src-tauri/src/"],
        )

    def test_unknown_file_returns_none(self):
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt("README.md")
        self.assertIsNone(result)

    def test_empty_file_returns_none(self):
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt("")
        self.assertIsNone(result)

    def test_is_claude_available_false_without_cli(self):
        """is_claude_available deve refletir indisponibilidade do CLI."""
        from lab.agents.base import is_claude_available

        with patch("lab.agents.claude_cli_runner.is_cli_available", return_value=False):
            self.assertFalse(is_claude_available())


class TestReviewerSystemSelection(unittest.TestCase):
    """Testa a priorização de prompts especialistas nos reviewers."""

    def _pick(
        self,
        reviewer_id: str,
        target_file: str,
        touched_domains: list[str] | None = None,
    ) -> str:
        from lab.agents.reviewer_agent import ClaudeReviewerAgent

        return ClaudeReviewerAgent()._pick_system(
            reviewer_id,
            target_file,
            touched_domains or [],
        )

    def test_tauri_boundary_prefers_frontend_specialist(self):
        system = self._pick("tauri_boundary_reviewer", "src/lib/tauri.ts")
        self.assertIn("TraduzAi **tauri-frontend expert**", system)

    def test_rust_reviewer_prefers_frontend_specialist_for_tauri_command(self):
        system = self._pick("rust_senior_reviewer", "src-tauri/src/commands/project.rs")
        self.assertIn("TraduzAi **tauri-frontend expert**", system)

    def test_react_reviewer_prefers_frontend_specialist_for_src(self):
        system = self._pick("react_ts_senior_reviewer", "src/pages/Home.tsx")
        self.assertIn("TraduzAi **tauri-frontend expert**", system)

    def test_python_reviewer_prefers_detect_specialist(self):
        system = self._pick("python_senior_reviewer", "pipeline/vision_stack/detector.py")
        self.assertIn("TraduzAi **detect expert**", system)

    def test_tauri_boundary_falls_back_to_generic_boundary_prompt_when_no_specialist_exists(self):
        with patch("lab.agents.reviewer_agent.get_agent_prompt", return_value=None):
            system = self._pick("tauri_boundary_reviewer", "docs/notes.md")
        self.assertIn("fronteira Tauri/IPC", system)


class TestParseReviewResponse(unittest.TestCase):
    """Testa o parser de resposta estruturada do reviewer Claude."""

    def _parse(self, raw: str, reviewer_id: str = "python_senior_reviewer"):
        from lab.agents.reviewer_agent import _parse_review_response

        return _parse_review_response(raw, {"target_file": "pipeline/typesetter/renderer.py"}, reviewer_id)

    def test_approve_verdict(self):
        raw = "VERDICT: approve\nFINDING_1: info|Tudo ok|Sem problemas detectados."
        verdict, payload = self._parse(raw)
        self.assertEqual(verdict, "approve")
        self.assertEqual(len(payload["findings"]), 1)
        self.assertEqual(payload["findings"][0]["severity"], "info")

    def test_request_changes_verdict(self):
        raw = "VERDICT: request_changes\nFINDING_1: warning|Risco de overflow|Texto longo sem wrap."
        verdict, payload = self._parse(raw)
        self.assertEqual(verdict, "request_changes")

    def test_needs_benchmark_focus_verdict(self):
        raw = "VERDICT: needs_benchmark_focus\nFINDING_1: error|Score baixo|Métrica abaixo do limiar."
        verdict, payload = self._parse(raw)
        self.assertEqual(verdict, "needs_benchmark_focus")

    def test_unknown_verdict_defaults_to_approve(self):
        raw = "VERDICT: unknown_value\n"
        verdict, _ = self._parse(raw)
        self.assertEqual(verdict, "approve")

    def test_no_verdict_defaults_to_approve(self):
        raw = "Sem verdict aqui."
        verdict, _ = self._parse(raw)
        self.assertEqual(verdict, "approve")

    def test_multiple_findings_parsed(self):
        raw = (
            "VERDICT: approve\n"
            "FINDING_1: info|Finding um|Detalhe um\n"
            "FINDING_2: warning|Finding dois|Detalhe dois\n"
            "FINDING_3: error|Finding três|Detalhe três\n"
        )
        _, payload = self._parse(raw)
        self.assertEqual(len(payload["findings"]), 3)

    def test_invalid_severity_normalized_to_info(self):
        raw = "VERDICT: approve\nFINDING_1: critical|Título|Corpo"
        _, payload = self._parse(raw)
        self.assertEqual(payload["findings"][0]["severity"], "info")

    def test_empty_response_produces_fallback_finding(self):
        verdict, payload = self._parse("")
        self.assertEqual(verdict, "approve")
        self.assertEqual(len(payload["findings"]), 1)
        self.assertIn("Revisão Claude", payload["findings"][0]["title"])

    def test_finding_title_truncated_at_120(self):
        long_title = "A" * 200
        raw = f"VERDICT: approve\nFINDING_1: info|{long_title}|Corpo"
        _, payload = self._parse(raw)
        self.assertLessEqual(len(payload["findings"][0]["title"]), 120)

    def test_finding_body_truncated_at_400(self):
        long_body = "B" * 500
        raw = f"VERDICT: approve\nFINDING_1: info|Título|{long_body}"
        _, payload = self._parse(raw)
        self.assertLessEqual(len(payload["findings"][0]["body"]), 400)


class TestClaudeReviewerAgentFallback(unittest.TestCase):
    """Testa que ClaudeReviewerAgent cai para rule-based quando Claude indisponível."""

    def setUp(self):
        self.proposal = {
            "proposal_id": "proposal-abc123",
            "batch_id": "batch-abc123",
            "title": "Test proposal",
            "target_file": "pipeline/typesetter/renderer.py",
            "touched_domains": ["pipeline/**"],
            "issue_type": "occupancy_too_high",
            "change_kind": "threshold_tune",
            "risk": "baixo",
            "motivation": "Ajustar limiar de ocupação",
        }
        self.benchmark = {
            "green": True,
            "score_after": 78.5,
            "metrics": {"layout_occupancy": 72.0, "textual_similarity": 85.0},
            "summary": "Benchmark passou.",
        }

    def test_fallback_when_claude_unavailable(self):
        from lab.agents.reviewer_agent import ClaudeReviewerAgent

        with patch("lab.agents.reviewer_agent.is_claude_available", return_value=False):
            agent = ClaudeReviewerAgent()
            verdict, payload = agent.review(self.proposal, self.benchmark, "integration_architect")

        self.assertEqual(verdict, "approve")  # green=True → approve
        self.assertIn("findings", payload)
        self.assertGreater(len(payload["findings"]), 0)

    def test_fallback_request_changes_when_not_green(self):
        from lab.agents.reviewer_agent import ClaudeReviewerAgent

        benchmark_failing = dict(self.benchmark, green=False, score_after=45.0)

        with patch("lab.agents.reviewer_agent.is_claude_available", return_value=False):
            agent = ClaudeReviewerAgent()
            verdict, _ = agent.review(self.proposal, benchmark_failing, "python_senior_reviewer")

        self.assertEqual(verdict, "request_changes")

    def test_fallback_tauri_reviewer_needs_benchmark_focus(self):
        from lab.agents.reviewer_agent import ClaudeReviewerAgent

        benchmark_failing = dict(self.benchmark, green=False)

        with patch("lab.agents.reviewer_agent.is_claude_available", return_value=False):
            agent = ClaudeReviewerAgent()
            verdict, _ = agent.review(self.proposal, benchmark_failing, "tauri_boundary_reviewer")

        self.assertEqual(verdict, "needs_benchmark_focus")

    def test_exception_in_claude_falls_back(self):
        """Se Claude lança exceção, cai para rule-based e adiciona finding de erro."""
        from lab.agents.reviewer_agent import ClaudeReviewerAgent

        with patch("lab.agents.reviewer_agent.is_claude_available", return_value=True):
            with patch.object(ClaudeReviewerAgent, "_claude_review", side_effect=RuntimeError("API timeout")):
                agent = ClaudeReviewerAgent()
                verdict, payload = agent.review(self.proposal, self.benchmark, "python_senior_reviewer")

        # Deve ter caído para rule-based + adicionado finding de erro
        self.assertIn(verdict, ("approve", "request_changes"))
        error_findings = [f for f in payload["findings"] if "falhou" in f["title"]]
        self.assertEqual(len(error_findings), 1)
        self.assertIn("API timeout", error_findings[0]["body"])


class TestClaudeSDKCoderFallback(unittest.TestCase):
    """Testa que ClaudeSDKCoder cai graciosamente quando Claude indisponível."""

    def setUp(self):
        self.proposal = {
            "proposal_id": "proposal-xyz789",
            "title": "Fix threshold",
            "target_file": "pipeline/typesetter/renderer.py",
            "touched_domains": ["pipeline/typesetter"],
            "issue_type": "occupancy_too_high",
            "change_kind": "threshold_tune",
            "motivation": "Limiar muito alto",
            "local_patch_hint": None,
            "review_findings": [],
        }
        self.repo_root = _REPO_ROOT

    def test_returns_patch_proposal_without_api_key(self):
        from lab.agents.coder_agent import ClaudeSDKCoder

        with patch("lab.agents.coder_agent.is_claude_available", return_value=False):
            coder = ClaudeSDKCoder()
            result = coder.propose_patch(self.proposal, self.repo_root)

        self.assertEqual(result.proposal_id, "proposal-xyz789")
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("Claude Code CLI", result.error)

    def test_coder_id_is_claude_sdk(self):
        from lab.agents.coder_agent import ClaudeSDKCoder

        self.assertEqual(ClaudeSDKCoder.coder_id, "claude_sdk")

    def test_exception_returns_patch_proposal_with_error(self):
        from lab.agents.coder_agent import ClaudeSDKCoder

        with patch("lab.agents.coder_agent.is_claude_available", return_value=True):
            with patch.object(ClaudeSDKCoder, "_call_claude", side_effect=RuntimeError("sem conexão")):
                coder = ClaudeSDKCoder()
                result = coder.propose_patch(self.proposal, self.repo_root)

        self.assertIn("sem conexão", result.error)
        self.assertEqual(result.confidence, 0.0)

    def test_local_patch_hint_takes_priority(self):
        """Quando local_patch_hint produz patch, não deve chamar Claude."""
        from lab.agents.coder_agent import ClaudeSDKCoder
        from lab.coders.base import PatchProposal

        proposal_with_hint = dict(
            self.proposal,
            change_kind="regex_add",
            local_patch_hint={"pattern": r"threshold\s*=\s*0\.8", "replacement": "threshold = 0.75"},
        )

        local_patch = PatchProposal(
            proposal_id="proposal-xyz789",
            patch_unified_diff="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new",
            author="local_patch_hint",
            confidence=0.9,
        )

        with patch("lab.agents.coder_agent.build_local_patch_from_hint", return_value=local_patch):
            with patch.object(ClaudeSDKCoder, "_call_claude") as mock_call:
                coder = ClaudeSDKCoder()
                result = coder.propose_patch(proposal_with_hint, self.repo_root)

        self.assertEqual(result.author, "local_patch_hint")
        mock_call.assert_not_called()


class TestEstimateConfidence(unittest.TestCase):
    """Testa a função de estimativa de confiança do coder."""

    def test_empty_diff_zero_confidence(self):
        from lab.agents.coder_agent import _estimate_confidence

        self.assertEqual(_estimate_confidence("", "qualquer resposta"), 0.0)

    def test_valid_diff_above_zero(self):
        from lab.agents.coder_agent import _estimate_confidence

        diff = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n context\n-old\n+new"
        confidence = _estimate_confidence(diff, diff)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_full_diff_markers_max_score(self):
        from lab.agents.coder_agent import _estimate_confidence

        diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n+nova linha adicionada"
        # raw longo para bônus
        raw = diff + " " * 200
        confidence = _estimate_confidence(diff, raw)
        self.assertGreaterEqual(confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
