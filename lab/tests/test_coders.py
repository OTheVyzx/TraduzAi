"""Testes para lab/coders/ — base, local patch hints, e coder offline.

Nenhum teste aqui chama rede ou Ollama de verdade.
OllamaCoder e ClaudeCodeCoder sao testados apenas para:
- Importar sem erro
- Retornar erro gracioso quando Ollama/CLI nao esta disponivel
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from lab.coders.base import (
    PatchProposal,
    _make_unified_diff,
    _patch_low_confidence,
    _patch_min_font_size,
    _patch_watermark_tokens,
    build_local_patch_from_hint,
)
from lab.coders.claude_code_coder import ClaudeCodeCoder, _find_claude_cli, _parse_cli_output
from lab.coders.ollama_coder import (
    OllamaCoder,
    _estimate_confidence,
    _extract_diff_from_response,
    _extract_rationale,
    _list_ollama_models,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp_file(content: str, suffix: str = ".py") -> Path:
    tmp = Path(tempfile.mkdtemp())
    path = tmp / f"target{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


def _minimal_proposal(
    *,
    issue_type: str = "watermark_leaked",
    change_kind: str = "regex_add",
    target_file: str = "",
    hint: dict | None = None,
) -> dict:
    return {
        "proposal_id": "test-proposal-01",
        "title": "Teste",
        "summary": "Descricao de teste",
        "motivation": "teste",
        "target_file": target_file,
        "target_anchor": "",
        "change_kind": change_kind,
        "issue_type": issue_type,
        "needs_coder": change_kind == "logic_fix",
        "local_patch_hint": hint or {"kind": change_kind},
        "expected_metric_gain": {},
        "review_findings": [],
    }


# ---------------------------------------------------------------------------
# PatchProposal dataclass
# ---------------------------------------------------------------------------


class PatchProposalTests(unittest.TestCase):
    def test_to_dict_contains_required_keys(self) -> None:
        patch = PatchProposal(
            proposal_id="p1",
            patch_unified_diff="--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new",
            files_affected=["pipeline/ocr.py"],
            rationale="Teste",
            author="local_patch_hint",
            confidence=0.9,
        )
        d = patch.to_dict()
        for key in ("proposal_id", "patch_unified_diff", "files_affected", "rationale", "author", "confidence", "error"):
            self.assertIn(key, d)

    def test_default_dry_run_true(self) -> None:
        patch = PatchProposal(proposal_id="x", patch_unified_diff="")
        self.assertTrue(patch.dry_run)


# ---------------------------------------------------------------------------
# _make_unified_diff
# ---------------------------------------------------------------------------


class UnifiedDiffTests(unittest.TestCase):
    def test_diff_contains_change_markers(self) -> None:
        original = "a = 1\nb = 2\n"
        patched = "a = 1\nb = 99\n"
        diff = _make_unified_diff("test.py", original, patched)
        self.assertIn("---", diff)
        self.assertIn("+++", diff)
        self.assertIn("-b = 2", diff)
        self.assertIn("+b = 99", diff)

    def test_identical_files_produce_empty_diff(self) -> None:
        src = "print('hello')\n"
        self.assertEqual(_make_unified_diff("f.py", src, src), "")


# ---------------------------------------------------------------------------
# _patch_watermark_tokens
# ---------------------------------------------------------------------------


class PatchWatermarkTests(unittest.TestCase):
    def _write_ocr_critic(self, tokens: str = "scan|toon|lagoon|asura|mangaflix|discord.gg") -> tuple[Path, Path]:
        content = (
            'import re\n'
            f'WATERMARK_RE = re.compile(r"(?i)\\b({tokens})\\b")\n'
        )
        path = _write_temp_file(content)
        return path.parent, path

    def test_adds_new_token_to_regex(self) -> None:
        _dir, target = self._write_ocr_critic()
        proposal_id = "wptest01"
        result = _patch_watermark_tokens(
            proposal_id=proposal_id,
            target_file_rel=target.name,
            target_path=target,
            original_source=target.read_text(encoding="utf-8"),
            new_tokens=["newsite"],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("newsite", result.patch_unified_diff)
        self.assertIn("+", result.patch_unified_diff)
        self.assertGreater(result.confidence, 0.5)

    def test_noop_if_token_already_present(self) -> None:
        _dir, target = self._write_ocr_critic()
        result = _patch_watermark_tokens(
            proposal_id="x",
            target_file_rel=target.name,
            target_path=target,
            original_source=target.read_text(encoding="utf-8"),
            new_tokens=["asura"],  # ja presente
        )
        assert result is not None
        self.assertEqual(result.patch_unified_diff, "")
        self.assertAlmostEqual(result.confidence, 1.0)

    def test_returns_none_if_pattern_not_found(self) -> None:
        path = _write_temp_file("# sem regex de watermark aqui\n")
        result = _patch_watermark_tokens(
            proposal_id="x",
            target_file_rel=path.name,
            target_path=path,
            original_source=path.read_text(encoding="utf-8"),
            new_tokens=["newsite"],
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _patch_low_confidence
# ---------------------------------------------------------------------------


class PatchLowConfidenceTests(unittest.TestCase):
    def test_raises_threshold(self) -> None:
        content = "LOW_CONFIDENCE_THRESHOLD = 0.60\n"
        path = _write_temp_file(content)
        result = _patch_low_confidence(
            proposal_id="lc01",
            target_file_rel=path.name,
            target_path=path,
            original_source=content,
        )
        assert result is not None
        self.assertIn("+LOW_CONFIDENCE_THRESHOLD", result.patch_unified_diff)
        new_val_match = re.search(r"\+LOW_CONFIDENCE_THRESHOLD = ([\d.]+)", result.patch_unified_diff)
        self.assertIsNotNone(new_val_match)
        assert new_val_match
        self.assertGreaterEqual(float(new_val_match.group(1)), 0.65)

    def test_noop_if_already_high(self) -> None:
        content = "LOW_CONFIDENCE_THRESHOLD = 0.75\n"
        path = _write_temp_file(content)
        result = _patch_low_confidence(
            proposal_id="lc02",
            target_file_rel=path.name,
            target_path=path,
            original_source=content,
        )
        assert result is not None
        self.assertEqual(result.patch_unified_diff, "")


# ---------------------------------------------------------------------------
# _patch_min_font_size
# ---------------------------------------------------------------------------


class PatchMinFontTests(unittest.TestCase):
    def test_raises_min_font_size(self) -> None:
        content = "MIN_FONT_SIZE = 10.0\n"
        path = _write_temp_file(content)
        result = _patch_min_font_size(
            proposal_id="mf01",
            target_file_rel=path.name,
            target_path=path,
            original_source=content,
            min_font=14,
        )
        assert result is not None
        self.assertIn("+MIN_FONT_SIZE", result.patch_unified_diff)

    def test_noop_if_already_large_enough(self) -> None:
        content = "MIN_FONT_SIZE = 16\n"
        path = _write_temp_file(content)
        result = _patch_min_font_size(
            proposal_id="mf02",
            target_file_rel=path.name,
            target_path=path,
            original_source=content,
            min_font=14,
        )
        assert result is not None
        self.assertEqual(result.patch_unified_diff, "")


# ---------------------------------------------------------------------------
# build_local_patch_from_hint (integration)
# ---------------------------------------------------------------------------


class BuildLocalPatchHintTests(unittest.TestCase):
    def test_watermark_resolved_locally(self) -> None:
        content = 'WATERMARK_RE = re.compile(r"(?i)\\b(scan|toon)\\b")\n'
        target = _write_temp_file(content)
        repo_root = target.parent
        proposal = _minimal_proposal(
            issue_type="watermark_leaked",
            change_kind="regex_add",
            target_file=target.name,
            hint={"kind": "regex_add", "new_watermark_tokens": ["newsite"]},
        )
        result = build_local_patch_from_hint(proposal, repo_root)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.author, "local_patch_hint")

    def test_logic_fix_returns_none(self) -> None:
        proposal = _minimal_proposal(
            issue_type="untranslated",
            change_kind="logic_fix",
            target_file="pipeline/translator/translate.py",
            hint={"kind": "logic_fix"},
        )
        result = build_local_patch_from_hint(proposal, Path("."))
        self.assertIsNone(result)

    def test_missing_target_file_returns_none(self) -> None:
        proposal = _minimal_proposal(
            issue_type="watermark_leaked",
            change_kind="regex_add",
            target_file="does/not/exist.py",
            hint={"kind": "regex_add", "new_watermark_tokens": ["x"]},
        )
        result = build_local_patch_from_hint(proposal, Path("."))
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# OllamaCoder — testes offline
# ---------------------------------------------------------------------------


class OllamaCoderOfflineTests(unittest.TestCase):
    def test_list_models_returns_empty_on_unreachable_host(self) -> None:
        models = _list_ollama_models("http://127.0.0.1:29999")
        self.assertEqual(models, [])

    def test_propose_patch_error_on_no_ollama(self) -> None:
        content = "LOW_CONFIDENCE_THRESHOLD = 0.60\n"
        target = _write_temp_file(content)
        proposal = _minimal_proposal(
            issue_type="untranslated",
            change_kind="logic_fix",
            target_file=target.name,
            hint={"kind": "logic_fix"},
        )
        coder = OllamaCoder(host="http://127.0.0.1:29999")
        result = coder.propose_patch(proposal, target.parent)
        self.assertIsInstance(result, PatchProposal)
        self.assertTrue(result.error)
        self.assertEqual(result.confidence, 0.0)

    def test_extract_diff_from_fenced_block(self) -> None:
        response = "Aqui o diff:\n```diff\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x=1\n+x=2\n```\nExplicacao."
        diff = _extract_diff_from_response(response)
        self.assertIn("@@ -1 +1 @@", diff)

    def test_extract_rationale_strips_fences(self) -> None:
        response = "```diff\n--- a\n+++ b\n```\nMotivo: bug no OCR."
        rationale = _extract_rationale(response)
        self.assertIn("Motivo", rationale)
        self.assertNotIn("```", rationale)

    def test_estimate_confidence_full_diff(self) -> None:
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n x=1\n-y=2\n+y=99"
        # raw_response deve ser > 50 chars para nao acionar penalidade de texto curto
        long_response = "Este e o patch sugerido pelo coder Ollama: ```diff\n" + diff + "```\nFim."
        confidence = _estimate_confidence(diff, long_response)
        self.assertGreaterEqual(confidence, 0.75)

    def test_estimate_confidence_empty_diff(self) -> None:
        self.assertEqual(_estimate_confidence("", ""), 0.0)


# ---------------------------------------------------------------------------
# ClaudeCodeCoder — testes offline
# ---------------------------------------------------------------------------


class ClaudeCodeCoderOfflineTests(unittest.TestCase):
    def test_parse_cli_output_json_result(self) -> None:
        payload = json.dumps({"type": "result", "result": "```diff\n--- a/x.py\n```"})
        text = _parse_cli_output(payload)
        self.assertIn("diff", text)

    def test_parse_cli_output_ndjson(self) -> None:
        lines = [
            json.dumps({"type": "assistant", "result": "thinking..."}),
            json.dumps({"type": "result", "result": "```diff\n--- a\n```"}),
        ]
        text = _parse_cli_output("\n".join(lines))
        self.assertIn("diff", text)

    def test_parse_cli_output_plain_fallback(self) -> None:
        plain = "just plain text with diff"
        self.assertEqual(_parse_cli_output(plain), plain)

    def test_propose_patch_error_when_cli_missing(self) -> None:
        import unittest.mock as mock

        content = "x = 1\n"
        target = _write_temp_file(content)
        proposal = _minimal_proposal(
            issue_type="untranslated",
            change_kind="logic_fix",
            target_file=target.name,
            hint={"kind": "logic_fix"},
        )
        with mock.patch("lab.coders.claude_code_coder._find_claude_cli", return_value=None):
            coder = ClaudeCodeCoder()
            result = coder.propose_patch(proposal, target.parent)
        self.assertTrue(result.error)
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
