"""Testa a funcao _resolve_strip_target_pages em main.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


class StripTargetPagesConfigTests(unittest.TestCase):
    def test_default_target_pages_is_60(self):
        from main import _resolve_strip_target_pages
        self.assertEqual(_resolve_strip_target_pages({}), 60)

    def test_explicit_target_overrides_default(self):
        from main import _resolve_strip_target_pages
        self.assertEqual(_resolve_strip_target_pages({"strip_target_pages": 80}), 80)

    def test_explicit_target_as_string_is_coerced(self):
        from main import _resolve_strip_target_pages
        self.assertEqual(_resolve_strip_target_pages({"strip_target_pages": "45"}), 45)

    def test_target_below_1_falls_back_to_60(self):
        from main import _resolve_strip_target_pages
        self.assertEqual(_resolve_strip_target_pages({"strip_target_pages": 0}), 60)
        self.assertEqual(_resolve_strip_target_pages({"strip_target_pages": -5}), 60)

    def test_target_clamped_to_total_pages_when_too_high(self):
        from main import _resolve_strip_target_pages
        # Se total_pages=50 e config pede 200, retorna 50
        self.assertEqual(_resolve_strip_target_pages({"strip_target_pages": 200}, total_pages=50), 50)

    def test_target_equal_to_total_pages_is_ok(self):
        from main import _resolve_strip_target_pages
        self.assertEqual(_resolve_strip_target_pages({"strip_target_pages": 60}, total_pages=60), 60)

    def test_no_total_pages_constraint_returns_target(self):
        from main import _resolve_strip_target_pages
        self.assertEqual(_resolve_strip_target_pages({"strip_target_pages": 120}), 120)

    def test_connected_reasoner_config_disables_ollama_when_unavailable(self):
        from main import _build_connected_reasoner_config

        config = _build_connected_reasoner_config(
            {"connected_balloon_reasoner": "ollama", "connected_balloon_reasoner_enabled": True},
            ollama_status={"running": False, "models": [], "has_translator": False},
        )

        self.assertEqual(config["provider"], "disabled")
        self.assertFalse(config["enabled"])

    def test_connected_reasoner_config_keeps_running_ollama_and_short_timeout(self):
        from main import _build_connected_reasoner_config

        config = _build_connected_reasoner_config(
            {"connected_balloon_reasoner": "ollama", "connected_balloon_reasoner_enabled": True},
            ollama_status={"running": True, "models": ["qwen2.5:3b"], "has_translator": False},
        )

        self.assertEqual(config["provider"], "ollama")
        self.assertTrue(config["enabled"])
        self.assertEqual(config["timeout_sec"], 12)
        self.assertEqual(config["_ollama_status"]["models"], ["qwen2.5:3b"])

    def test_runtime_profile_config_records_eco_decision_and_applies_env_defaults(self):
        from main import _apply_runtime_profile_config

        config = {"runtime_profile": "eco"}
        with patch.dict(os.environ, {}, clear=True):
            decision = _apply_runtime_profile_config(config)

            self.assertEqual(decision.profile, "eco")
            self.assertFalse(config["runtime_profile_decision"]["visual_stack_warmup"])
            self.assertEqual(os.environ["TRADUZAI_STRIP_INPAINTER_PREWARM"], "0")
            self.assertEqual(config["runtime_profile_env"]["TRADUZAI_STRIP_INPAINTER_PREWARM"], "applied")
