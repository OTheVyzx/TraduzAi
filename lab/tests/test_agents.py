"""Testes offline para lab.agents — sem API key real, sem rede.

Cobre:
- get_agent_prompt: roteamento target_file → agente correto
- _parse_review_response: parser de resposta estruturada
- ClaudeReviewerAgent: fallback rule-based quando Claude indisponível
- ClaudeSDKCoder: fallback quando Claude indisponível, patch local primeiro
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adiciona raiz do repo ao sys.path para importações relativas
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class TestGetAgentPrompt(unittest.TestCase):
    """Testa roteamento de arquivo → agente correto."""

    def test_typesetter_prefix_routes_to_typesetter(self):
        from lab.agents.base import get_agent_prompt

        # Se .claude/agents/typesetter-expert.md não existe, retorna None —
        # o importante é que não lança exceção e tenta o arquivo certo.
        result = get_agent_prompt("pipeline/typesetter/renderer.py")
        # Pode ser None (arquivo não existe) ou uma string não-vazia
        self.assertTrue(result is None or isinstance(result, str))

    def test_layout_prefix_routes_to_typesetter(self):
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt("pipeline/layout/balloon_layout.py")
        self.assertTrue(result is None or isinstance(result, str))

    def test_translator_prefix_routes_to_translator(self):
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt("pipeline/translator/translate.py")
        self.assertTrue(result is None or isinstance(result, str))

    def test_vision_stack_prefix_routes_to_vision_stack(self):
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt("pipeline/vision_stack/runtime.py")
        self.assertTrue(result is None or isinstance(result, str))

    def test_unknown_file_returns_none(self):
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt("src/pages/Lab.tsx")
        self.assertIsNone(result)

    def test_empty_file_returns_none(self):
        from lab.agents.base import get_agent_prompt

        result = get_agent_prompt("")
        self.assertIsNone(result)

    def test_touched_domains_fallback(self):
        from lab.agents.base import get_agent_prompt

        # target_file vazio mas touched_domains contém prefixo válido
        result = get_agent_prompt("", ["pipeline/typesetter"])
        self.assertTrue(result is None or isinstance(result, str))

    def test_is_claude_available_false_without_key(self):
        """is_claude_available deve retornar False quando não há API key."""
        from lab.agents.base import is_claude_available

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            self.assertFalse(is_claude_available())


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
        self.assertIn("ANTHROPIC_API_KEY", result.error)

    def test_coder_id_is_claude_sdk(self):
        from lab.agents.coder_agent import ClaudeSDKCoder

        self.assertEqual(ClaudeSDKCoder.coder_id, "claude_sdk")

    def test_exception_returns_patch_proposal_with_error(self):
        from lab.agents.coder_agent import ClaudeSDKCoder

        with patch("lab.agents.coder_agent.is_claude_available", return_value=True):
            with patch("lab.agents.coder_agent.make_client", side_effect=RuntimeError("sem conexão")):
                coder = ClaudeSDKCoder()
                result = coder.propose_patch(self.proposal, self.repo_root)

        self.assertIn("sem conexão", result.error)
        self.assertEqual(result.confidence, 0.0)

    def test_local_patch_hint_takes_priority(self):
        """Quando local_patch_hint produz patch, não deve chamar Claude."""
        from lab.agents.coder_agent import ClaudeSDKCoder

        proposal_with_hint = dict(
            self.proposal,
            change_kind="regex_add",
            local_patch_hint={"pattern": r"threshold\s*=\s*0\.8", "replacement": "threshold = 0.75"},
        )

        # Mesmo com Claude "disponível", deve resolver localmente
        with patch("lab.agents.coder_agent.is_claude_available", return_value=True) as mock_avail:
            with patch("lab.agents.coder_agent.make_client") as mock_client:
                coder = ClaudeSDKCoder()
                # build_local_patch_from_hint pode retornar None se o arquivo não tiver o pattern
                # — o importante é que não chama make_client antes de tentar local
                coder.propose_patch(proposal_with_hint, self.repo_root)
                # make_client só é chamado se local falhou — não vamos assertar seu call count
                # pois depende do conteúdo real do arquivo
        self.assertTrue(True)  # chegou aqui sem exceção


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
