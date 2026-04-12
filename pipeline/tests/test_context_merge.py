import unittest

from translator.context import merge_context


class ContextMergeTests(unittest.TestCase):
    def test_merge_context_preserves_enriched_fields(self):
        existing = {
            "sinopse": "",
            "genero": [],
            "personagens": ["Ghislain"],
            "aliases": ["Mercenario Regressado"],
            "fontes_usadas": [{"fonte": "webnovel"}],
        }
        fallback = {
            "sinopse": "Canon synopsis",
            "genero": ["Action"],
            "personagens": ["Ghislain", "Vanessa"],
            "aliases": [],
            "fontes_usadas": [],
        }

        merged = merge_context(existing, fallback)

        self.assertEqual(merged["sinopse"], "Canon synopsis")
        self.assertEqual(merged["genero"], ["Action"])
        self.assertEqual(merged["personagens"], ["Ghislain"])
        self.assertEqual(merged["aliases"], ["Mercenario Regressado"])
        self.assertEqual(merged["fontes_usadas"], [{"fonte": "webnovel"}])


if __name__ == "__main__":
    unittest.main()
