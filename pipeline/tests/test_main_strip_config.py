"""Testa a funcao _resolve_strip_target_pages em main.py."""

import unittest


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
