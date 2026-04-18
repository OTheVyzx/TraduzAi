import unittest
from unittest.mock import patch

from translator.translate import (
    _build_context_hints,
    _build_text_payload,
    _lookup_memory_translation,
    _postprocess,
    _preprocess_text,
    _prepare_source_text_for_translation,
    _resolve_translation_backend,
    _review_translation_grammar_semantics,
    list_supported_google_languages,
    normalize_google_language_code,
    translate_pages,
)


class TranslateContextTests(unittest.TestCase):
    def test_list_supported_google_languages_returns_sorted_metadata(self):
        class _FakeGoogleTranslator:
            @staticmethod
            def get_supported_languages(as_dict=False):  # noqa: FBT002
                self.assertTrue(as_dict)
                return {
                    "spanish": "es",
                    "english": "en",
                    "portuguese": "pt",
                    "russian": "ru",
                }

        with patch("deep_translator.GoogleTranslator", _FakeGoogleTranslator):
            languages = list_supported_google_languages()

        self.assertEqual([item["code"] for item in languages], ["en", "pt", "ru", "es"])
        self.assertEqual(languages[0]["label"], "English")
        self.assertEqual(languages[2]["ocr_strategy"], "dedicated")

    def test_normalize_google_language_code_handles_aliases_and_regions(self):
        self.assertEqual(normalize_google_language_code("pt-BR"), "pt")
        self.assertEqual(normalize_google_language_code("en-GB"), "en")
        self.assertEqual(normalize_google_language_code("zh"), "zh-CN")
        self.assertEqual(normalize_google_language_code("zh-Hant"), "zh-TW")
        self.assertEqual(normalize_google_language_code("ES"), "es")

    def test_translate_pages_normalizes_source_and_target_language_codes(self):
        created = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                created.append((source, target))
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "test" else f"{self.target}:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return [f"{self.target}:{text}" for text in texts]

        ocr_results = [{"texts": [{"text": "Hello there", "tipo": "fala"}]}]

        with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
            with patch(
                "translator.translate._check_ollama",
                return_value={"running": False, "models": [], "has_translator": False},
            ):
                translated = translate_pages(
                    ocr_results=ocr_results,
                    obra="obra-teste",
                    context={},
                    glossario={},
                    idioma_origem="en-GB",
                    idioma_destino="pt-BR",
                )

        self.assertEqual(created[0], ("en", "pt"))
        self.assertTrue(translated[0]["texts"][0]["translated"].lower().startswith("pt:"))

    def test_translate_pages_repairs_empty_or_unchanged_ollama_outputs_with_google_when_available(self):
        class _FakeGoogleTranslator:
            def translate(self, text: str):
                return "teste" if text == "test" else f"pt:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return [f"pt:{text}" for text in texts]

        ocr_results = [
            {
                "texts": [
                    {"text": "HELLO", "tipo": "fala"},
                    {"text": "I'LL WIN.", "tipo": "fala"},
                ]
            }
        ]

        with patch("translator.translate._GoogleTranslator", return_value=_FakeGoogleTranslator()):
            with patch(
                "translator.translate._check_ollama",
                return_value={"running": True, "models": ["mangatl-translator:latest"], "has_translator": True},
            ):
                with patch(
                    "translator.translate._call_ollama",
                    return_value=[
                        {"id": "t1", "translated": ""},
                        {"id": "t2", "translated": "I'LL WIN."},
                    ],
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={},
                        glossario={},
                    )

        texts = translated[0]["texts"]
        self.assertEqual(texts[0]["translated"], "PT:HELLO")
        self.assertEqual(texts[1]["translated"], "PT:I'LL WIN.")

    def test_sfx_preprocess_preserves_uppercase(self):
        processed = _preprocess_text("BANG!!", tipo="sfx")
        self.assertEqual(processed, "BANG!!")

    def test_sfx_postprocess_keeps_uppercase(self):
        processed = _postprocess("estrondo!!", was_upper=True, tipo="sfx")
        self.assertEqual(processed, "ESTRONDO!!")

    def test_build_text_payload_includes_local_context(self):
        texts = [
            {"text": "Who are you?", "tipo": "fala"},
            {"text": "Martha...", "tipo": "fala"},
            {"text": "Boom", "tipo": "sfx"},
        ]
        history = [{"source": "Who are you?", "translated": "Quem e voce?"}]

        payload = _build_text_payload(texts, 1, history)

        self.assertEqual(payload["tipo"], "fala")
        self.assertEqual(payload["context_before"], "Who are you?")
        self.assertEqual(payload["context_after"], "Boom")
        self.assertEqual(payload["history_tail"][0]["translated"], "Quem e voce?")

    def test_context_hints_include_new_structured_fields(self):
        hints = _build_context_hints(
            {
                "aliases": ["Mercenario"],
                "termos": ["Mana Core"],
                "faccoes": ["Legiao Cinzenta"],
                "relacoes": ["Ghislain -> Vanessa"],
                "resumo_por_arco": ["Arco 1"],
                "memoria_lexical": {"Mana Core": "Nucleo de Mana"},
            },
            {"Ghislain": "Ghislain"},
        )
        self.assertIn("ALIASES: Mercenario", hints)
        self.assertIn("TERMOS: Mana Core", hints)
        self.assertIn('"Mana Core": "Nucleo de Mana"', hints)

    def test_lookup_memory_translation_prefers_structured_context(self):
        translated = _lookup_memory_translation(
            "Mana Core",
            "fala",
            {"memoria_lexical": {"Mana Core": "Nucleo de Mana"}},
            {},
        )
        self.assertEqual(translated, "Nucleo de Mana")

    def test_lookup_memory_translation_uses_corpus_memory_map(self):
        translated = _lookup_memory_translation(
            "Do you think",
            "fala",
            {"corpus_memoria_lexical": {"Do you think": "Voce acha que"}},
            {},
        )
        self.assertEqual(translated, "Voce acha que")

    def test_context_hints_include_corpus_candidates(self):
        hints = _build_context_hints(
            {
                "corpus_memory_candidates": [
                    {"source_text": "Do you think", "target_text": "Voce acha que"},
                    {"source_text": "Knight", "target_text": "Cavaleiro"},
                ]
            },
            {},
        )
        self.assertIn("MEMORIA_CORPUS:", hints)
        self.assertIn("Do you think => Voce acha que", hints)

    def test_prepare_source_text_for_translation_repairs_ocr_artifacts(self):
        prepared = _prepare_source_text_for_translation("COYLD THAT LIGHTBES", tipo="fala")
        self.assertEqual(prepared, "Could that light be...?!")

    def test_review_translation_grammar_semantics_fixes_literal_combat_phrase(self):
        reviewed = _review_translation_grammar_semantics(
            source_text="YOU SAID YOU COULD SEE THROUGH ALL MY ATTACKS, RIGHT?",
            translated_text="Vocę disse que podia ver através de todos os meus ataques, certo?",
            tipo="fala",
        )
        self.assertEqual(reviewed, "Você disse que podia enxergar todos os meus golpes, certo?")

    def test_postprocess_applies_source_aware_light_question_fix(self):
        processed = _postprocess(
            "PODE SER ESSA LUZ",
            was_upper=True,
            tipo="fala",
            source_text="COYLD THAT LIGHTBES",
        )
        self.assertEqual(processed, "PODERIA SER AQUELA LUZ...?!")

    def test_resolve_translation_backend_prefers_local_ollama_when_available(self):
        with patch.dict("os.environ", {}, clear=False):
            backend = _resolve_translation_backend(
                google_ok=True,
                ollama_status={
                    "running": True,
                    "models": ["mangatl-translator:latest"],
                    "has_translator": True,
                },
            )

        self.assertEqual(backend, "ollama")

    def test_resolve_translation_backend_can_fall_back_to_google_when_local_preference_is_disabled(self):
        with patch.dict("os.environ", {"TRADUZAI_PREFER_LOCAL_TRANSLATION": "0"}, clear=False):
            backend = _resolve_translation_backend(
                google_ok=True,
                ollama_status={
                    "running": True,
                    "models": ["mangatl-translator:latest"],
                    "has_translator": True,
                },
            )

        self.assertEqual(backend, "google")


if __name__ == "__main__":
    unittest.main()
