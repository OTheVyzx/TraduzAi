import unittest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import translator.translate as translate_module

from translator.term_protection import PLACEHOLDER_TEMPLATE
from translator.translate import (
    build_translation_context_header,
    _build_context_hints,
    _build_text_payload,
    _fix_infinitive_to_imperative,
    _is_likely_proper_noun,
    _lookup_special_literal_translation,
    _lookup_memory_translation,
    _postprocess,
    _pick_ollama_model_for_language_pair,
    _preprocess_text,
    _probe_google_backend,
    _prepare_source_text_for_translation,
    _resolve_translation_backend,
    _review_translation_grammar_semantics,
    _repair_translation_after_source_prefix_cleanup,
    _should_skip_translation_item,
    _source_text_before_normalization,
    _source_text_for_translation,
    _translate_google_parallel_chunks,
    _translation_quality_flags,
    list_supported_google_languages,
    normalize_google_language_code,
    translate_pages,
)


class TranslateContextTests(unittest.TestCase):
    def setUp(self):
        translate_module._google = None
        translate_module._google_health_key = None
        translate_module._google_health_ok = False
        translate_module._google_health_failed_at = {}

    def test_should_skip_translation_item_ignores_legacy_skip_fields(self):
        assert _should_skip_translation_item({"content_class": "tn_note"}) is False
        assert _should_skip_translation_item({"content_class": "scanlator_credit"}) is False
        assert _should_skip_translation_item({"content_class": "noise"}) is False
        assert _should_skip_translation_item({"content_class": "dialogue"}) is False
        assert _should_skip_translation_item({"content_class": "sign"}) is False
        assert _should_skip_translation_item({"skip_processing": True}) is False
        assert _should_skip_translation_item({"preserve_original": True}) is False
        assert _should_skip_translation_item({"translate_policy": "skip_translation"}) is False
        assert _should_skip_translation_item({"route_action": "inpaint_only", "skip_processing": False}) is True
        assert _should_skip_translation_item({"route_action": "preserve", "skip_processing": False}) is False
        assert _should_skip_translation_item({"route_action": "review_required", "skip_processing": False}) is True
        assert _should_skip_translation_item({"route_action": "translate_render_only", "skip_processing": False}) is False

    def test_repair_translation_after_source_prefix_cleanup_removes_stale_dark_lobe_prefix(self):
        text = {
            "text": "If you exceed that time, you will return to your original world!",
            "qa_flags": ["leading_dark_lobe_duplicate_fragment_removed"],
            "qa_metrics": {
                "leading_dark_lobe_duplicate_fragment_removed": {
                    "from": "space is only utes. If you exceed that time, you will return to your original world!",
                    "to": "If you exceed that time, you will return to your original world!",
                }
            },
        }

        repaired, flags = _repair_translation_after_source_prefix_cleanup(
            text,
            "Space is only utes. se voce ultrapassar esse tempo, voce retornara ao seu mundo original!",
        )

        self.assertEqual(
            repaired,
            "se voce ultrapassar esse tempo, voce retornara ao seu mundo original!",
        )
        self.assertEqual(flags, ["translation_leading_duplicate_fragment_removed"])

    def test_source_text_before_normalization_prefers_dark_lobe_prefix_cleanup(self):
        text = {
            "raw_ocr": "space is only utes. If you exceed that time, you will return to your original world!",
            "text": "If you exceed that time, you will return to your original world!",
            "qa_flags": ["leading_dark_lobe_duplicate_fragment_removed"],
            "qa_metrics": {
                "leading_dark_lobe_duplicate_fragment_removed": {
                    "from": "space is only utes. If you exceed that time, you will return to your original world!",
                    "to": "If you exceed that time, you will return to your original world!",
                }
            },
        }

        self.assertEqual(
            _source_text_before_normalization(text),
            "If you exceed that time, you will return to your original world!",
        )
        self.assertEqual(
            _source_text_for_translation(text),
            "If you exceed that time, you will return to your original world!",
        )

    def test_google_parallel_chunks_default_off_uses_single_batch_call(self):
        calls = []

        def translate_batch(texts: list[str]) -> list[str]:
            calls.append(list(texts))
            return [f"pt:{text}" for text in texts]

        with patch.dict("os.environ", {}, clear=True):
            translated = _translate_google_parallel_chunks(
                ["one", "two", "one"],
                translate_batch,
                min_unique_texts=2,
                max_texts_per_chunk=1,
            )

        self.assertEqual(translated, ["pt:one", "pt:two", "pt:one"])
        self.assertEqual(calls, [["one", "two", "one"]])

    def test_google_parallel_chunks_deduplicates_chunks_and_preserves_order(self):
        calls = []

        def translate_batch(texts: list[str]) -> list[str]:
            calls.append(list(texts))
            return [f"pt:{text}" for text in texts]

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_GOOGLE_PARALLEL_CHUNKS": "1",
                "TRADUZAI_GOOGLE_TRANSLATE_WORKERS": "2",
            },
            clear=True,
        ):
            translated = _translate_google_parallel_chunks(
                ["one", "two", "one", "three", "four", "two"],
                translate_batch,
                min_unique_texts=3,
                max_texts_per_chunk=2,
            )

        self.assertEqual(translated, ["pt:one", "pt:two", "pt:one", "pt:three", "pt:four", "pt:two"])
        self.assertEqual(calls, [["one", "two"], ["three", "four"]])

    def test_list_supported_google_languages_returns_sorted_metadata(self):
        class _FakeGoogleTranslator:
            def __init__(self, source="auto", target="en"):
                self.source = source
                self.target = target

            def get_supported_languages(self, as_dict=False):  # noqa: FBT002
                assert as_dict is True
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
                self._translator = self

            def translate(self, text: str):
                return f"{self.target}:{text}"

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

    def test_translate_pages_repairs_mojibake_and_flags_translation(self):
        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self._translator = self
                self.target = target

            def translate(self, text: str):
                if text == "__traduzai_probe__":
                    return "ok"
                return "VOCÃƒÅ  SABE"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return ["VOCÃƒÅ  SABE" for _text in texts]

        ocr_results = [{"texts": [{"id": "ocr_001", "text": "YOU KNOW", "tipo": "fala"}]}]

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
                    idioma_origem="en",
                    idioma_destino="pt-BR",
                )

        text = translated[0]["texts"][0]
        self.assertEqual(text["translated"], "VOCÊ SABE")
        self.assertIn("mojibake_in_translation", text["qa_flags"])
        self.assertEqual(text["mojibake_audit"]["suggested_fix"], "VOCÊ SABE")

    def test_translate_pages_repairs_cp1250_portuguese_mojibake(self):
        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self._translator = self
                self.target = target

            def translate(self, text: str):
                if text == "__traduzai_probe__":
                    return "ok"
                return "VOC\u0118 N\u0102O TEM TR\u0118S DIAS"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return ["VOC\u0118 N\u0102O TEM TR\u0118S DIAS" for _text in texts]

        ocr_results = [{"texts": [{"id": "ocr_001", "text": "YOU DO NOT HAVE THREE DAYS", "tipo": "fala"}]}]

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
                    idioma_origem="en",
                    idioma_destino="pt-BR",
                )

        text = translated[0]["texts"][0]
        self.assertEqual(text["translated"], "VOCÊ NÃO TEM TRÊS DIAS")
        self.assertIn("mojibake_in_translation", text["qa_flags"])
        self.assertEqual(text["mojibake_audit"]["suggested_fix"], "VOCÊ NÃO TEM TRÊS DIAS")

    def test_translate_pages_skips_ollama_probe_when_google_is_available_by_default(self):
        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                return f"pt:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return [f"pt:{text}" for text in texts]

        ocr_results = [{"texts": [{"text": "Hello", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=_FakeGoogleTranslator()):
                        with patch("translator.translate._check_ollama") as check_ollama:
                            translated = translate_pages(
                                ocr_results=ocr_results,
                                obra="obra-teste",
                                context={},
                                glossario={},
                            )

        check_ollama.assert_not_called()
        self.assertEqual(translated[0]["texts"][0]["translated"], "pt:Hello")

    def test_translate_pages_uses_context_for_split_sake_phrase_without_grouping_layout(self):
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                return "saude" if text == "health" else f"pt:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.append(list(texts))
                if [text.lower() for text in texts] == ["please, for the child's sake."]:
                    return ["POR FAVOR, PELO BEM DA CRIANCA."]
                return ["POR FAVOR, PARA A CRIANCA", "SAQUE."]

        ocr_results = [
            {
                "texts": [
                    {
                        "text": "PLEASE, FOR THE CHILD'S",
                        "tipo": "fala",
                        "bbox": [112, 3728, 330, 3794],
                        "source_bbox": [90, 3690, 360, 3860],
                        "text_pixel_bbox": [116, 3738, 326, 3792],
                        "balloon_bbox": [90, 3690, 360, 3860],
                        "balloon_type": "white",
                    },
                    {
                        "text": "SAKE.",
                        "tipo": "fala",
                        "bbox": [170, 3800, 260, 3830],
                        "source_bbox": [90, 3690, 360, 3860],
                        "text_pixel_bbox": [174, 3802, 258, 3828],
                        "balloon_bbox": [90, 3690, 360, 3860],
                        "balloon_type": "white",
                    },
                ]
            }
        ]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=_FakeGoogleTranslator()):
                        translated = translate_pages(
                            ocr_results=ocr_results,
                            obra="obra-teste",
                            context={},
                            glossario={},
                            idioma_origem="en",
                            idioma_destino="pt",
                        )

        texts = translated[0]["texts"]
        self.assertEqual([text.lower() for text in captured_batches[0]], ["please, for the child's sake."])
        joined = " ".join(text["translated"] for text in texts)
        self.assertIn("PELO BEM", joined)
        self.assertIn("CRIANCA", joined)
        self.assertNotIn("SAQUE", joined)
        self.assertEqual(texts[0]["translation_context_group_id"], texts[1]["translation_context_group_id"])
        self.assertEqual(texts[0].get("layout_group_size", 1), 1)
        self.assertEqual(texts[1].get("layout_group_size", 1), 1)

    def test_translate_pages_reuses_successful_google_health_check_for_same_language_pair(self):
        class _FakeGoogleTranslator:
            def __init__(self):
                self.health_checks = 0
                self._translator = self

            def translate(self, text: str):
                if text == "hello":
                    self.health_checks += 1
                return f"pt:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return [f"pt:{text}" for text in texts]

        fake = _FakeGoogleTranslator()
        ocr_results = [{"texts": [{"text": "Hello", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=fake):
                        with patch("translator.translate._check_ollama"):
                            translate_pages(ocr_results=ocr_results, obra="obra-teste", context={}, glossario={})
                            translate_pages(ocr_results=ocr_results, obra="obra-teste", context={}, glossario={})

        self.assertEqual(fake.health_checks, 1)

    def test_translate_pages_caches_google_failure_and_skips_reprobe_temporarily(self):
        class _BrokenGoogleTranslator:
            def __init__(self):
                self._translator = self
                self.health_checks = 0

            def translate(self, text: str):
                self.health_checks += 1
                raise RuntimeError("proxy down")

            def translate_batch(self, texts: list[str]) -> list[str]:
                raise RuntimeError("proxy down")

        fake = _BrokenGoogleTranslator()
        ocr_results = [{"texts": [{"text": "Ã¬â€¢Ë†Ã«â€¦â€¢Ã­â€¢ËœÃ¬â€žÂ¸Ã¬Å¡â€", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._google_health_failed_at", {}):
                        with patch("translator.translate._GoogleTranslator", return_value=fake):
                            with patch(
                                "translator.translate._check_ollama",
                                return_value={"running": False, "models": [], "has_translator": False},
                            ):
                                translate_pages(
                                    ocr_results=ocr_results,
                                    obra="obra-teste",
                                    context={},
                                    glossario={},
                                    idioma_origem="ko",
                                    idioma_destino="pt-BR",
                                )
                                translate_pages(
                                    ocr_results=ocr_results,
                                    obra="obra-teste",
                                    context={},
                                    glossario={},
                                    idioma_origem="ko",
                                    idioma_destino="pt-BR",
                                )

        self.assertEqual(fake.health_checks, 1)

    def test_translate_pages_repairs_empty_or_unchanged_ollama_outputs_with_google_when_available(self):
        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                return f"pt:{text}"

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

    def test_probe_google_backend_rejects_unchanged_korean_probe(self):
        class _EchoTranslator:
            def translate(self, text: str):
                return "teste" if text == "hello" else text

        fake = type("FakeGoogle", (), {"_translator": _EchoTranslator()})()

        with self.assertRaisesRegex(RuntimeError, "source sem traduzir"):
            _probe_google_backend(fake, "ko", "pt")

    def test_translate_pages_falls_back_when_google_health_probe_fails(self):
        class _BrokenGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                raise RuntimeError("proxy down")

            def translate_batch(self, texts: list[str]) -> list[str]:
                return [f"pt:{text}" for text in texts]

        ocr_results = [{"texts": [{"text": "Ã¬â€¢Ë†Ã«â€¦â€¢Ã­â€¢ËœÃ¬â€žÂ¸Ã¬Å¡â€", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=_BrokenGoogleTranslator()):
                        with patch(
                            "translator.translate._check_ollama",
                            return_value={"running": False, "models": [], "has_translator": False},
                        ):
                            translated = translate_pages(
                                ocr_results=ocr_results,
                                obra="obra-teste",
                                context={},
                                glossario={},
                                idioma_origem="ko",
                                idioma_destino="pt-BR",
                            )

        self.assertEqual(translated[0]["texts"][0]["translated"], "Ã¬â€¢Ë†Ã«â€¦â€¢Ã­â€¢ËœÃ¬â€žÂ¸Ã¬Å¡â€")

    def test_translate_pages_does_not_call_ollama_when_google_health_fails(self):
        class _BrokenGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                raise RuntimeError("proxy down")

            def translate_batch(self, texts: list[str]) -> list[str]:
                raise RuntimeError("proxy down")


        ocr_results = [{"texts": [{"text": "Ã¬â€¢Ë†Ã«â€¦â€¢Ã­â€¢ËœÃ¬â€žÂ¸Ã¬Å¡â€", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=_BrokenGoogleTranslator()):
                        with patch("translator.translate._check_ollama") as check_ollama:
                            with patch("translator.translate._call_ollama") as call_ollama:
                                translated = translate_pages(
                                    ocr_results=ocr_results,
                                    obra="obra-teste",
                                    context={},
                                    glossario={},
                                    idioma_origem="ko",
                                    idioma_destino="pt-BR",
                                )

        check_ollama.assert_not_called()
        call_ollama.assert_not_called()
        self.assertEqual(translated[0]["texts"][0]["translated"], ocr_results[0]["texts"][0]["text"])

    def test_none_is_not_treated_as_proper_noun(self):
        self.assertFalse(_is_likely_proper_noun("NONE."))

    def test_common_caps_dialogue_words_are_not_treated_as_proper_nouns(self):
        for token in [
            "UNCONTROLLABLE",
            "ALREADY",
            "COMMANDER",
            "RAID",
            "SQUAD",
            "HOUSEHOLD",
            "LOST",
            "EVERYTHING",
            "SOLDIER",
        ]:
            with self.subTest(token=token):
                self.assertFalse(_is_likely_proper_noun(token))

    def test_none_gets_context_safe_literal_translation(self):
        self.assertEqual(_lookup_special_literal_translation("NONE.", "fala"), "Nenhuma.")

    def test_short_quoted_dialogue_gets_literal_translation(self):
        self.assertEqual(_lookup_special_literal_translation('"WE"?!', "fala"), '"N\u00f3s"?!')

    def test_no_is_a_no_gets_context_safe_literal_translation(self):
        self.assertEqual(
            _lookup_special_literal_translation("A NO IS A NO!", "fala"),
            "Um n\u00e3o \u00e9 um n\u00e3o!",
        )

    def test_sfx_preprocess_preserves_uppercase(self):
        processed = _preprocess_text("BANG!!", tipo="sfx")
        self.assertEqual(processed, "BANG!!")

    def test_sfx_postprocess_keeps_uppercase(self):
        processed = _postprocess("estrondo!!", was_upper=True, tipo="sfx")
        self.assertEqual(processed, "ESTRONDO!!")

    def test_postprocess_removes_zero_width_format_chars(self):
        processed = _postprocess("Hosu \u200b\u200b24 anos desempregado", was_upper=False, tipo="text")
        self.assertEqual(processed, "Hosu 24 anos desempregado")

    def test_postprocess_repairs_stutter_prefix_from_translated_word(self):
        processed = _postprocess(
            "S-jovem mestre...?",
            was_upper=True,
            tipo="fala",
            source_text="y-YOUNG MASTER...?",
        )

        self.assertEqual(processed, "J-JOVEM MESTRE...?")

    def test_postprocess_does_not_rewrite_non_stutter_rank_prefix(self):
        processed = _postprocess(
            "S-rank",
            was_upper=False,
            tipo="fala",
            source_text="S-RANK",
        )

        self.assertEqual(processed, "S-rank")

    def test_postprocess_repairs_ranker_and_national_selection_terms(self):
        processed = _postprocess(
            "Um rankeador entrou na seleção da seleção.",
            was_upper=False,
            tipo="fala",
            source_text="A ranker entered the national team selection.",
        )

        self.assertEqual(processed, "Um Ranker entrou na seletiva nacional.")

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

    def test_translate_pages_repairs_proper_name_using_context_entities(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "test" else f"{self.target}:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.append(list(texts))
                return [f"pt:{text}" for text in texts]

        ocr_results = [{"texts": [{"text": "GHHISLAN PERDIUM", "tipo": "fala"}]}]
        decisions = []

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    with patch(
                        "translator.translate.record_decision",
                        side_effect=lambda **kwargs: decisions.append(kwargs),
                    ):
                        translated = translate_pages(
                            ocr_results=ocr_results,
                            obra="obra-teste",
                            context={"personagens": ["Ghislain Perdium"]},
                            glossario={},
                            idioma_origem="en",
                            idioma_destino="pt-BR",
                        )

        self.assertEqual(captured_batches[0], [placeholder])
        entity_meta = translated[0]["texts"][0]
        self.assertIn("Ghislain Perdium", entity_meta["translated"])
        self.assertEqual(entity_meta["entity_flags"], ["source_entity_repaired"])
        self.assertEqual(
            entity_meta["entity_repairs"][0],
            {
                "phase": "source",
                "kind": "character",
                "from": "GHHISLAN PERDIUM",
                "to": "Ghislain Perdium",
            },
        )
        self.assertTrue(
            any(
                item.get("action") == "repair_entity"
                and item.get("reason") == "source_entity_match"
                for item in decisions
            )
        )

    def test_translate_pages_does_not_preserve_multi_word_caps_name_by_shape_only(self):
        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "test" else f"{self.target}:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return ["P\u00c9RDIO GHISLAIN."]

        ocr_results = [{"texts": [{"text": "GHISLAIN PERDIUM.", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
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
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        result = translated[0]["texts"][0]["translated"]
        self.assertEqual(result, "P\u00c9RDIO GHISLAIN.")
        self.assertNotIn("target_proper_name_repaired", translated[0]["texts"][0].get("entity_flags", []))

    def test_translate_pages_translates_common_caps_dialogue_instead_of_preserving_it(self):
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "test" else f"{self.target}:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.append(list(texts))
                return [
                    {
                        "Uncontrollable": "Incontrolavel",
                        "Already?": "Ja?",
                        "Commander!": "Comandante!",
                        "Raid squad commander skovan": "Comandante Skovan do esquadrao de ataque",
                        "Killing somehow..": "Matando de alguma forma..",
                    }.get(text, f"pt:{text}")
                    for text in texts
                ]

        ocr_results = [
            {
                "texts": [
                    {"text": "UNCONTROLLABLE", "tipo": "fala"},
                    {"text": "ALREADY?", "tipo": "fala"},
                    {"text": "COMMANDER!", "tipo": "fala"},
                    {"text": "RAID SQUAD COMMANDER SKOVAN", "tipo": "fala"},
                    {"text": "KILLING SOMEHOW..", "tipo": "narracao"},
                ]
            }
        ]

        with patch("translator.translate._google", None):
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
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertEqual(
            captured_batches[0],
            ["Uncontrollable", "Already?", "Commander!", "Raid squad commander skovan", "Killing somehow.."],
        )
        outputs = [item["translated"] for item in translated[0]["texts"]]
        self.assertEqual(outputs[0], "INCONTROLAVEL")
        self.assertEqual(outputs[1], "JA?")
        self.assertEqual(outputs[2], "COMANDANTE!")
        self.assertEqual(outputs[3], "COMANDANTE SKOVAN DO ESQUADRAO DE ATAQUE")
        self.assertEqual(outputs[4], "MATANDO DE ALGUMA FORMA..")
        self.assertNotIn("target_proper_name_repaired", translated[0]["texts"][4].get("entity_flags", []))
        self.assertFalse(translated[0]["texts"][0].get("proper_noun_preserved", False))

    def test_translate_pages_does_not_preserve_arbitrary_single_caps_dialogue(self):
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "test" else f"{self.target}:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.append(list(texts))
                return ["Seu bastardo!" if text == "Bastard!" else f"pt:{text}" for text in texts]

        ocr_results = [{"texts": [{"text": "BASTARD!", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
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
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertEqual(captured_batches[0], ["Bastard!"])
        self.assertEqual(translated[0]["texts"][0]["translated"], "SEU BASTARDO!")
        self.assertFalse(translated[0]["texts"][0].get("proper_noun_preserved", False))

    def test_translate_pages_repairs_embedded_proper_name_inside_short_phrase(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return f"{self.target}:{text}"

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.append(list(texts))
                return [f"pt:{text}" for text in texts]

        ocr_results = [{"texts": [{"text": "OVER, GHHISLAN PERDIUM!", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={"personagens": ["Ghislain Perdium"]},
                        glossario={},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertIn(placeholder, captured_batches[0][0])
        entity_meta = translated[0]["texts"][0]
        self.assertIn("Ghislain Perdium", entity_meta["translated"])
        self.assertIn("source_entity_repaired", entity_meta["entity_flags"])
        self.assertTrue(
            any(item.get("to") == "Ghislain Perdium" for item in entity_meta["entity_repairs"])
        )

    def test_translate_pages_locks_glossary_terms_inside_sentence(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                if text == "test":
                    return "teste"
                if text == "hello":
                    return "ol\u00e1"
                return text

            def translate_batch(self, texts: list[str]) -> list[str]:
                return [f"o {placeholder} esta instavel."]

        ocr_results = [{"texts": [{"text": "The Mana Core is unstable.", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={},
                        glossario={"Mana Core": "N\u00facleo de Mana"},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        entity_meta = translated[0]["texts"][0]
        self.assertIn("N\u00facleo de Mana", entity_meta["translated"])
        self.assertIn(
            {"phase": "placeholder", "source": "Mana Core", "target": "N\u00facleo de Mana"},
            entity_meta["glossary_hits"],
        )
        self.assertNotIn("unrestored_placeholder", entity_meta.get("qa_flags", []))

    def test_translate_pages_restores_glossary_placeholder_when_preserved(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "hello" else text

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.extend(texts)
                return [f"{placeholder} despertou."]

        ocr_results = [{"texts": [{"text": "Ghislain woke up.", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={},
                        glossario={"Ghislain": "Ghislain"},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertIn(placeholder, captured_batches[0])
        entity_meta = translated[0]["texts"][0]
        self.assertEqual(entity_meta["translated"], "Ghislain despertou.")
        self.assertIn(
            {"phase": "placeholder", "source": "Ghislain", "target": "Ghislain"},
            entity_meta["glossary_hits"],
        )
        self.assertNotIn("unrestored_placeholder", entity_meta.get("qa_flags", []))

    def test_translate_pages_name_locks_context_character_before_google(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "hello" else text

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.extend(texts)
                if placeholder in texts[0]:
                    return [f'{placeholder} disse "estou indo"']
                return ['maravilhoso disse "estou indo"']

        ocr_results = [{"texts": [{"text": 'Wonho said "I am going"', "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={"personagens": ["Wonho"]},
                        glossario={},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertIn(placeholder, captured_batches[0])
        entity_meta = translated[0]["texts"][0]
        self.assertEqual(entity_meta["translated"], 'Wonho disse "estou indo"')
        self.assertNotIn("maravilhoso", entity_meta["translated"])
        self.assertNotIn("unrestored_placeholder", entity_meta.get("qa_flags", []))

    def test_translate_pages_name_lock_preserves_alias_punctuation(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "hello" else text

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.extend(texts)
                return [f"{placeholder}...?"]

        ocr_results = [{"texts": [{"text": "Hosu...?", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={"personagens": ["Wonho"], "aliases": ["Hosu"]},
                        glossario={},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertIn(placeholder, captured_batches[0])
        entity_meta = translated[0]["texts"][0]
        self.assertEqual(entity_meta["translated"], "Hosu...?")
        self.assertNotIn("unrestored_placeholder", entity_meta.get("qa_flags", []))

    def test_translate_pages_flags_dropped_placeholder_as_unrestored(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "hello" else text

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.extend(texts)
                return ['maravilhoso disse "estou indo"']

        ocr_results = [{"texts": [{"text": 'Wonho said "I am going"', "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={"personagens": ["Wonho"]},
                        glossario={},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertIn(placeholder, captured_batches[0])
        entity_meta = translated[0]["texts"][0]
        self.assertIn("unrestored_placeholder", entity_meta.get("qa_flags", []))
        self.assertIn("translation_render_blocked", entity_meta.get("qa_flags", []))
        self.assertEqual(entity_meta.get("translation_blocked_text"), 'maravilhoso disse "estou indo"')

    def test_ollama_repair_google_batch_keeps_name_lock_placeholders(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self._translator = self
                self.target = target
                self.seen_batches = []

            def translate(self, text: str):
                return "teste" if text == "hello" else text

            def translate_batch(self, texts: list[str]) -> list[str]:
                self.seen_batches.extend(texts)
                if texts and placeholder in texts[0]:
                    return [f"{placeholder} venceu."]
                return ["maravilhoso venceu."]

        fake_google = _FakeGoogleTranslator()

        ocr_results = [{"texts": [{"text": "Wonho wins", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=fake_google):
                        with patch(
                            "translator.translate._check_ollama",
                            return_value={"running": True, "models": ["mangatl-translator:latest"], "has_translator": True},
                        ):
                            with patch("translator.translate._resolve_translation_backend", return_value="ollama"):
                                with patch(
                                    "translator.translate._call_ollama",
                                    return_value=[{"id": "t1", "translated": "Wonho wins"}],
                                ):
                                    translated = translate_pages(
                                        ocr_results=ocr_results,
                                        obra="obra-teste",
                                        context={"personagens": ["Wonho"]},
                                        glossario={},
                                        idioma_origem="en",
                                        idioma_destino="pt-BR",
                                    )

        self.assertEqual(fake_google.seen_batches, [f"{placeholder} wins"])
        entity_meta = translated[0]["texts"][0]
        self.assertEqual(entity_meta["translated"], "Wonho venceu.")
        self.assertNotIn("maravilhoso", entity_meta["translated"])
        self.assertNotIn("unrestored_placeholder", entity_meta.get("qa_flags", []))

    def test_translate_pages_does_not_name_lock_common_uppercase_one(self):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=0)
        captured_batches = []

        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "hello" else text

            def translate_batch(self, texts: list[str]) -> list[str]:
                captured_batches.extend(texts)
                return ["aquele poder permite que uma pessoa leia."]

        ocr_results = [{"texts": [{"text": "THAT POWER LETS ONE READ.", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._GoogleTranslator", _FakeGoogleTranslator):
                with patch(
                    "translator.translate._check_ollama",
                    return_value={"running": False, "models": [], "has_translator": False},
                ):
                    translated = translate_pages(
                        ocr_results=ocr_results,
                        obra="obra-teste",
                        context={"personagens": ["ONE", "READ"]},
                        glossario={},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        self.assertNotIn(placeholder, captured_batches[0])
        entity_meta = translated[0]["texts"][0]
        self.assertEqual(entity_meta["translated"], "AQUELE PODER PERMITE QUE UMA PESSOA LEIA.")
        self.assertNotIn("unrestored_placeholder", entity_meta.get("qa_flags", []))

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
            translated_text="Voc\u0119 disse que podia ver atrav\u00e9s de todos os meus ataques, certo?",
            tipo="fala",
        )
        self.assertEqual(reviewed, "Voc\u00ea disse que podia enxergar todos os meus golpes, certo?")

    def test_postprocess_applies_source_aware_light_question_fix(self):
        processed = _postprocess(
            "PODE SER ESSA LUZ",
            was_upper=True,
            tipo="fala",
            source_text="COYLD THAT LIGHTBES",
        )
        self.assertEqual(processed, "PODERIA SER AQUELA LUZ...?!")

    def test_review_translation_grammar_semantics_rewrites_trading_with_them_phrase(self):
        reviewed = _review_translation_grammar_semantics(
            source_text='YOU MEAN THE POWER HE GOT BY TRADING WITH "THEM"?',
            translated_text='Voc\u00ea quer dizer o poder que ele obteve negociando com "eles"?',
            tipo="fala",
        )
        self.assertEqual(reviewed, "Voc\u00ea quer dizer o poder que ele conseguiu em um acordo com eles?")

    def test_review_translation_grammar_semantics_compacts_half_mana_technique_line(self):
        reviewed = _review_translation_grammar_semantics(
            source_text=(
                "EVEN THOUGH IT'S ONLY HALF OF A MANA TECHNIQUE, "
                "ITS EFFECTS WILL BE MORE THAN ENOUGH. "
                "THAT POWER LETS ONE INSTANTLY SURPASS THEIR OWN LIMITS."
            ),
            translated_text=(
                "Mesmo sendo apenas metade de uma t\u00e9cnica de mana, "
                "seus efeitos ser\u00e3o mais que suficientes. "
                "Esse poder permite superar instantaneamente seus pr\u00f3prios limites."
            ),
            tipo="fala",
        )
        self.assertEqual(
            reviewed,
            "Pode at\u00e9 ser s\u00f3 metade de uma t\u00e9cnica de mana, mas seus efeitos j\u00e1 s\u00e3o mais do que suficientes. "
            "Esse poder permite ultrapassar instantaneamente os pr\u00f3prios limites.",
        )

    def test_review_translation_grammar_semantics_shortens_desmond_power_line(self):
        reviewed = _review_translation_grammar_semantics(
            source_text="IF DESMOND ENDS UP USING THAT POWER...",
            translated_text="Se Desmond acabar usando esse poder...",
            tipo="fala",
        )
        self.assertEqual(reviewed, "Se Desmond usar esse poder...")

    def test_review_translation_grammar_semantics_rewrites_vanessa_mana_line(self):
        reviewed = _review_translation_grammar_semantics(
            source_text="THE RAMPAGING VANESSA ALSO USED A SIMILAR MANA TECHNIQUE.",
            translated_text="A violenta Vanessa tamb\u00e9m usou uma t\u00e9cnica de mana semelhante.",
            tipo="fala",
        )
        self.assertEqual(
            reviewed,
            "A Vanessa em f\u00faria tamb\u00e9m usava um m\u00e9todo semelhante de circula\u00e7\u00e3o de mana.",
        )

    def test_review_translation_grammar_semantics_preserves_split_sentence_continuation(self):
        reviewed = _review_translation_grammar_semantics(
            source_text="YOU COULD SEE ALL OF MY ATTACKS?",
            translated_text="VocÃƒÂª pode ver todos os meus ataques?",
            tipo="fala",
        )
        self.assertEqual(reviewed, "Que conseguia ver todos os meus ataques?")

    def test_resolve_translation_backend_prefers_google_by_default_when_available(self):
        with patch.dict("os.environ", {}, clear=False):
            backend = _resolve_translation_backend(
                google_ok=True,
                ollama_status={
                    "running": True,
                    "models": ["mangatl-translator:latest"],
                    "has_translator": True,
                },
            )

        self.assertEqual(backend, "google")

    def test_resolve_translation_backend_ignores_local_flag(self):
        with patch.dict("os.environ", {"TRADUZAI_PREFER_LOCAL_TRANSLATION": "1"}, clear=False):
            backend = _resolve_translation_backend(
                google_ok=True,
                ollama_status={
                    "running": True,
                    "models": ["mangatl-translator:latest"],
                    "has_translator": True,
                },
            )

        self.assertEqual(backend, "google")

    def test_resolve_translation_backend_does_not_fallback_to_ollama(self):
        backend = _resolve_translation_backend(
            google_ok=False,
            ollama_status={
                "running": True,
                "models": ["mangatl-translator:latest"],
                "has_translator": True,
            },
        )

        self.assertEqual(backend, "passthrough")

    def test_pick_ollama_model_for_korean_portuguese_prefers_gemma(self):
        picked = _pick_ollama_model_for_language_pair(
            ["qwen2.5:3b", "gemma4:e4b", "mangatl-translator:latest"],
            "traduzai-translator",
            "ko",
            "pt-BR",
        )

        self.assertEqual(picked, "gemma4:e4b")

    def test_pick_ollama_model_for_korean_portuguese_honors_explicit_hf_gemma(self):
        hf_model = "hf.co/stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small:Q4_0_S"
        picked = _pick_ollama_model_for_language_pair(
            ["gemma4:e4b", hf_model, "qwen2.5:3b"],
            hf_model,
            "ko",
            "pt-BR",
        )

        self.assertEqual(picked, hf_model)

    def test_translate_pages_uses_semantic_review_with_google_for_korean(self):
        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                return "ok"

            def translate_batch(self, texts: list[str]) -> list[str]:
                mapping = {
                    "Ã«Ââ€žÃ¬Â â‚¬Ã­Å¾Ë† Ã¬Æ’ÂÃ«Â¬Â¸Ã¬Ââ€ž Ã¬Â°Â¾Ã¬Ââ€ž Ã¬Ë†ËœÃªÂ°â‚¬ Ã¬â€”â€ Ã«â€¹Â¤": "NÃƒÂ£o consigo encontrar o texto original",
                    "Ã¬Â§â€žÃ«Â²â€¢Ã¬Ââ‚¬ Ã¬ÂÂ´ ÃªÂ°Ë†Ã¬â€šÂ¬Ã«Å¸â€°Ã¬ÂËœ Ã¬Â§â€œÃ¬ÂÂ´ Ã«Â¶â€žÃ«Âªâ€¦Ã­â€¢Â©Ã«â€¹Ë†Ã«â€¹Â¤": "Ãƒâ€° claro que este ÃƒÂ© o trabalho desta garota.",
                    "Ã­ÂÂ¬Ã¬â€¢â€žÃ¬â€¢â€žÃ¬â€¢â€¦!!": "UAU!!",
                }
                return [mapping.get(text, f"pt:{text}") for text in texts]

        captured = {}

        def _fake_call_ollama(model, system, user_msg, host):
            captured["model"] = model
            captured["system"] = system
            captured["user_msg"] = user_msg
            return [
                {"id": "p1_t1", "translated": "NÃƒÂ£o consigo encontrar uma saÃƒÂ­da."},
            ]

        ocr_results = [
            {
                "texts": [
                    {"text": "Ã«Ââ€žÃ¬Â â‚¬Ã­Å¾Ë† Ã¬Æ’ÂÃ«Â¬Â¸Ã¬Ââ€ž Ã¬Â°Â¾Ã¬Ââ€ž Ã¬Ë†ËœÃªÂ°â‚¬ Ã¬â€”â€ Ã«â€¹Â¤", "tipo": "fala"},
                    {"text": "Ã¬Â§â€žÃ«Â²â€¢Ã¬Ââ‚¬ Ã¬ÂÂ´ ÃªÂ°Ë†Ã¬â€šÂ¬Ã«Å¸â€°Ã¬ÂËœ Ã¬Â§â€œÃ¬ÂÂ´ Ã«Â¶â€žÃ«Âªâ€¦Ã­â€¢Â©Ã«â€¹Ë†Ã«â€¹Â¤", "tipo": "fala"},
                    {"text": "Ã­ÂÂ¬Ã¬â€¢â€žÃ¬â€¢â€žÃ¬â€¢â€¦!!", "tipo": "sfx"},
                ]
            }
        ]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch.dict("os.environ", {"TRADUZAI_SEMANTIC_REVIEW": "1"}):
                        with patch("translator.translate._GoogleTranslator", return_value=_FakeGoogleTranslator()):
                            with patch(
                                "translator.translate._check_ollama",
                                return_value={"running": True, "models": ["gemma4:e4b"], "has_translator": False},
                            ) as check_ollama:
                                with patch("translator.translate._call_ollama", side_effect=_fake_call_ollama) as call_ollama:
                                    translated = translate_pages(
                                        ocr_results=ocr_results,
                                        obra="obra-teste",
                                        context={},
                                        glossario={},
                                        idioma_origem="ko",
                                        idioma_destino="pt-BR",
                                    )

        check_ollama.assert_not_called()
        call_ollama.assert_not_called()
        self.assertEqual(captured, {})
        texts = translated[0]["texts"]
        self.assertEqual(len(texts), 3)

    def test_translate_pages_rejects_invalid_semantic_review_output(self):
        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                return "ok"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return ["NÃƒÂ£o consigo encontrar o texto original" for _ in texts]

        ocr_results = [{"texts": [{"text": "Ã«Ââ€žÃ¬Â â‚¬Ã­Å¾Ë† Ã¬Æ’ÂÃ«Â¬Â¸Ã¬Ââ€ž Ã¬Â°Â¾Ã¬Ââ€ž Ã¬Ë†ËœÃªÂ°â‚¬ Ã¬â€”â€ Ã«â€¹Â¤", "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch.dict("os.environ", {"TRADUZAI_SEMANTIC_REVIEW": "1"}):
                        with patch("translator.translate._GoogleTranslator", return_value=_FakeGoogleTranslator()):
                            with patch(
                                "translator.translate._check_ollama",
                                return_value={"running": True, "models": ["gemma4:e4b"], "has_translator": False},
                            ):
                                with patch(
                                    "translator.translate._call_ollama",
                                    return_value=[{"id": "p1_t1", "translated": "..."}],
                                ):
                                    translated = translate_pages(
                                        ocr_results=ocr_results,
                                        obra="obra-teste",
                                        context={},
                                        glossario={},
                                        idioma_origem="ko",
                                        idioma_destino="pt-BR",
                                    )

        self.assertEqual(
            translated[0]["texts"][0]["translated"],
            "",
        )
        self.assertEqual(
            translated[0]["texts"][0]["translation_blocked_text"],
            "NÃƒÂ£o consigo encontrar o texto original",
        )
        self.assertIn("translation_render_blocked", translated[0]["texts"][0]["qa_flags"])

    def test_korean_google_translation_flags_ocr_artifacts_and_script_leaks(self):
        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                return "ok"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return ["Bukmyeong TCH \u00e9 invenc\u00edvel", "Mu-\u8449!"]

        ocr_results = [
            {
                "texts": [
                    {"text": "\ubd81\uba85\ub300\ub294 \ubb34\uc801\uc785\ub2c8\ub2e4", "tipo": "fala"},
                    {"text": "\ubb34\uc5fd!", "tipo": "fala"},
                ]
            }
        ]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=_FakeGoogleTranslator()):
                        translated = translate_pages(
                            ocr_results=ocr_results,
                            obra="obra-teste",
                            context={},
                            glossario={},
                            idioma_origem="ko",
                            idioma_destino="pt-BR",
                        )

        flags = [item["qa_flags"] for item in translated[0]["texts"]]
        self.assertIn("suspected_ocr_error", flags[0])
        self.assertIn("source_script_leak", flags[1])

    def test_google_translation_legacy_skip_fields_do_not_preserve_source_text(self):
        ocr_results = [
            {
                "texts": [
                    {
                        "id": "noise_001",
                        "text": "\u5514\u2026\u2026",
                        "tipo": "fala",
                        "content_class": "noise",
                        "skip_processing": True,
                    }
                ]
            }
        ]

        translated = translate_pages(
            ocr_results=ocr_results,
            obra="obra-teste",
            context={},
            glossario={},
            idioma_origem="zh",
            idioma_destino="pt-BR",
        )

        item = translated[0]["texts"][0]
        self.assertNotEqual(item["translated"], "\u5514\u2026\u2026")
        self.assertNotIn("source_script_leak", item.get("qa_flags") or [])
        self.assertNotIn("translation_blocked_text", item)

    def test_chinese_mojibake_kana_passthrough_is_preserved_as_sfx(self):
        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self
                self.seen = []

            def translate(self, text: str):
                return "traduzido"

            def translate_batch(self, texts: list[str]) -> list[str]:
                self.seen.extend(texts)
                return list(texts)

        fake = _FakeGoogleTranslator()
        mojibake_kana = (
            "\u00e3\u201a\u00a2\u00e3\u0192\u00aa\u00e3\u0192\u2022"
            "\u00e3\u0192\u00bc\u00e3\u0192\u2022\u00e3\u0192\u00bc"
        )
        ocr_results = [{"texts": [{"id": "ocr_001", "text": mojibake_kana, "tipo": "fala"}]}]

        with patch("translator.translate._google", None):
            with patch("translator.translate._google_health_key", None):
                with patch("translator.translate._google_health_ok", False):
                    with patch("translator.translate._GoogleTranslator", return_value=fake):
                        translated = translate_pages(
                            ocr_results=ocr_results,
                            obra="obra-teste",
                            context={},
                            glossario={},
                            idioma_origem="zh",
                            idioma_destino="pt-BR",
                        )

        item = translated[0]["texts"][0]
        self.assertEqual(fake.seen, ["\u30a2\u30ea\u30d5\u30fc\u30d5\u30fc"])
        self.assertEqual(item["tipo"], "sfx")
        self.assertEqual(item["content_class"], "sfx")
        self.assertEqual(item["route_action"], "preserve")
        self.assertEqual(item["route_reason"], "untranslated_kana_sfx_preserved")
        self.assertFalse(item["skip_processing"])
        self.assertTrue(item["preserve_original"])
        self.assertIn("untranslated_kana_sfx_preserved", item.get("qa_flags") or [])
        self.assertNotIn("source_script_leak", item.get("qa_flags") or [])
        self.assertNotIn("translation_blocked_text", item)

    def test_chinese_kana_sfx_preservation_keeps_translation_debug_pair(self):
        from debug_tools import DebugRecorder, bind_recorder

        class _FakeGoogleTranslator:
            def __init__(self):
                self._translator = self

            def translate(self, text: str):
                return "traduzido"

            def translate_batch(self, texts: list[str]) -> list[str]:
                return list(texts)

        with tempfile.TemporaryDirectory() as tmp:
            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="run-translation-test")
            bind_recorder(recorder)
            try:
                with patch("translator.translate._google", None):
                    with patch("translator.translate._google_health_key", None):
                        with patch("translator.translate._google_health_ok", False):
                            with patch("translator.translate._GoogleTranslator", return_value=_FakeGoogleTranslator()):
                                translate_pages(
                                    ocr_results=[
                                        {
                                            "texts": [
                                                {
                                                    "id": "ocr_001",
                                                    "text": "アリフーフー",
                                                    "tipo": "fala",
                                                    "band_id": "page_008_band_012",
                                                }
                                            ]
                                        }
                                    ],
                                    obra="obra-teste",
                                    context={},
                                    glossario={},
                                    idioma_origem="zh",
                                    idioma_destino="pt-BR",
                                )
            finally:
                bind_recorder(None)

            root = Path(tmp) / "debug" / "e2e" / "07_translation"
            inputs = [
                json.loads(line)
                for line in (root / "translation_inputs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            outputs = [
                json.loads(line)
                for line in (root / "translation_outputs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(inputs[0]["trace_id"], outputs[0]["trace_id"])
        self.assertEqual(outputs[0]["final_translation_after_postprocess"], "アリフーフー")
        self.assertIn("untranslated_kana_sfx_preserved", outputs[0].get("qa_flags") or [])

    def test_korean_translation_quality_flags_fallback_and_literal_ocr_phrases(self):
        self.assertIn(
            "translation_fallback_phrase",
            _translation_quality_flags(
                "\uB3C4\uC800\uD788 \uC0DD\uBB38\uC744 \uCC3E\uC744 \uC218\uAC00 \uC5C6\uB2E4.",
                "Nao consigo encontrar o texto original.",
                "ko",
            ),
        )
        self.assertIn(
            "literal_ocr_translation",
            _translation_quality_flags(
                "\uC800\uB807\uAC8C \uC80A\uC740 \uC790\uAC1C \uBC30\uD6C4\uC5D0",
                "Quem poderia imaginar que uma madreperola tao jovem estaria por tras disso?",
                "ko",
            ),
        )

    def test_korean_life_gate_mistranslation_is_repaired_before_render_block(self):
        translated = _postprocess(
            "Nao consigo... nao consigo encontrar o texto original.",
            was_upper=False,
            tipo="fala",
            source_text="\uB3C4\uC800\uD788... \uC0DD\uBB38\uC744 \uCC3E\uC744 \uC218\uAC00 \uC5C6\uB2E4.",
            lang="ko",
        )

        self.assertEqual(translated, "Nao consigo... nao consigo encontrar a porta da vida.")
        self.assertNotIn("translation_fallback_phrase", _translation_quality_flags("", translated, "ko"))

    def test_translation_context_header_includes_title_synopsis_and_genre(self):
        header = build_translation_context_header(
            {
                "title": "Reincarnated Murim Lord",
                "synopsis": "Um guerreiro retorna e tenta corrigir o passado.",
                "genre": ["murim", "fantasia", "regressao"],
            }
        )

        self.assertIn("TITULO_OBRA: Reincarnated Murim Lord", header)
        self.assertIn("SINOPSE: Um guerreiro retorna", header)
        self.assertIn("GENERO: murim, fantasia, regressao", header)


class ProperNounProtectionTests(unittest.TestCase):
    """Verifica que nomes prÃƒÂ³prios em CAPS isolados sÃƒÂ£o preservados."""

    def test_single_word_caps_treated_as_proper_noun(self):
        # Casos reais do Cap 78: nomes que o Google estava traduzindo errado
        self.assertTrue(_is_likely_proper_noun("GILLION"))
        self.assertTrue(_is_likely_proper_noun("WILLOW"))
        self.assertTrue(_is_likely_proper_noun("VANESSA"))
        self.assertTrue(_is_likely_proper_noun("FENRIS"))
        self.assertTrue(_is_likely_proper_noun("DESMOND"))
        self.assertTrue(_is_likely_proper_noun("RACHEL"))

    def test_proper_noun_with_trailing_punctuation(self):
        # GILLION. e WILLOW, do project.json
        self.assertTrue(_is_likely_proper_noun("GILLION."))
        self.assertTrue(_is_likely_proper_noun("WILLOW,"))
        self.assertTrue(_is_likely_proper_noun("VANESSA!"))

    def test_common_caps_words_are_not_proper_nouns(self):
        # Palavras comuns em CAPS que nÃƒÂ£o devem ser preservadas
        self.assertFalse(_is_likely_proper_noun("WAKE"))
        self.assertFalse(_is_likely_proper_noun("STOP"))
        self.assertFalse(_is_likely_proper_noun("HELP"))
        self.assertFalse(_is_likely_proper_noun("MAGIC"))
        self.assertFalse(_is_likely_proper_noun("PLEASE"))
        self.assertFalse(_is_likely_proper_noun("LORD"))
        self.assertFalse(_is_likely_proper_noun("MASTER"))
        self.assertFalse(_is_likely_proper_noun("BECAUSE."))
        self.assertFalse(_is_likely_proper_noun("AMAZING."))
        self.assertFalse(_is_likely_proper_noun("REALLY..."))

    def test_multi_word_text_is_not_proper_noun(self):
        self.assertFalse(_is_likely_proper_noun("WAKE UP"))
        self.assertFalse(_is_likely_proper_noun("STOP THEM"))
        self.assertFalse(_is_likely_proper_noun("MY LORD"))

    def test_short_words_are_not_proper_nouns(self):
        # Curtas demais para serem nomes prÃƒÂ³prios confiÃƒÂ¡veis
        self.assertFalse(_is_likely_proper_noun("GO"))
        self.assertFalse(_is_likely_proper_noun("HI"))
        self.assertFalse(_is_likely_proper_noun("NGH"))

    def test_text_with_digits_is_not_proper_noun(self):
        # OCR errors com dÃƒÂ­gitos (M9, i999) nÃƒÂ£o sÃƒÂ£o tratados como nomes
        self.assertFalse(_is_likely_proper_noun("M9"))
        self.assertFalse(_is_likely_proper_noun("i999"))
        self.assertFalse(_is_likely_proper_noun("R2D2"))

    def test_lowercase_words_are_not_proper_nouns(self):
        self.assertFalse(_is_likely_proper_noun("gillion"))
        self.assertFalse(_is_likely_proper_noun("willow"))


class ImperativeFixTests(unittest.TestCase):
    """Verifica que infinitivos em comandos curtos viram imperativos."""

    def test_wake_up_becomes_imperative(self):
        # Caso real do Cap 78 pÃƒÂ¡g. 58: "WAKE UP." -> "ACORDAR." (errado)
        result = _fix_infinitive_to_imperative("Acordar.", "WAKE UP.", "fala")
        self.assertEqual(result, "Acorde.")

    def test_imperative_preserves_caps(self):
        result = _fix_infinitive_to_imperative("ACORDAR.", "WAKE UP.", "fala")
        self.assertEqual(result, "ACORDE.")

    def test_imperative_preserves_exclamation(self):
        result = _fix_infinitive_to_imperative("Parar!", "STOP!", "fala")
        self.assertEqual(result, "Pare!")

    def test_imperative_preserves_ellipsis(self):
        result = _fix_infinitive_to_imperative("Esperar...", "WAIT...", "fala")
        self.assertEqual(result, "Espere...")

    def test_long_source_does_not_trigger_imperative_fix(self):
        # Source com mais de 4 palavras nÃƒÂ£o dispara o fix
        long_source = "I really need you to wake up right now."
        result = _fix_infinitive_to_imperative("Acordar.", long_source, "fala")
        # MantÃƒÂ©m infinitivo porque a frase ÃƒÂ© longa o suficiente para o Google
        # ter contexto e traduzir corretamente como verbo principal.
        self.assertEqual(result, "Acordar.")

    def test_sfx_does_not_trigger_imperative_fix(self):
        # SFX nunca recebe o fix de imperativo
        result = _fix_infinitive_to_imperative("Acordar!", "WAKE UP!", "sfx")
        self.assertEqual(result, "Acordar!")

    def test_multi_word_translation_does_not_trigger_fix(self):
        # TraduÃƒÂ§ÃƒÂ£o multi-palavra nÃƒÂ£o bate na tabela; passa direto.
        result = _fix_infinitive_to_imperative("Por favor, acordar.", "WAKE UP.", "fala")
        self.assertEqual(result, "Por favor, acordar.")

    def test_unknown_infinitive_passes_through(self):
        # Verbo fora da tabela nÃƒÂ£o ÃƒÂ© alterado
        result = _fix_infinitive_to_imperative("Discutir.", "DISCUSS.", "fala")
        self.assertEqual(result, "Discutir.")


class OcrRepairExpansionTests(unittest.TestCase):
    """Verifica que os novos padrÃƒÂµes de OCR sÃƒÂ£o aplicados em _preprocess_text."""

    def test_m9_repaired_to_my(self):
        result = _prepare_source_text_for_translation("M9 LORD", "fala", lang="en")
        self.assertIn("MY", result.upper())

    def test_nd_squad_repaired_to_2nd(self):
        result = _prepare_source_text_for_translation("ND SQUAD ATTACK", "fala", lang="en")
        self.assertIn("2ND", result.upper())

    def test_ghk_normalized_to_ngh(self):
        # SFX comum: GHK era traduzido como "OBRIGADO" pelo Google
        result = _prepare_source_text_for_translation("GHK", "fala", lang="en")
        self.assertNotIn("GHK", result.upper())


class PostprocessImperativeIntegrationTests(unittest.TestCase):
    """Garante que o fix de imperativo ÃƒÂ© executado durante _postprocess."""

    def test_postprocess_converts_short_infinitive(self):
        # Pipeline completo: source curta + traduÃƒÂ§ÃƒÂ£o em infinitivo -> imperativo
        result = _postprocess(
            text="Acordar.",
            was_upper=True,
            tipo="fala",
            source_text="WAKE UP.",
            lang="en",
        )
        self.assertEqual(result, "ACORDE.")

    def test_postprocess_keeps_long_infinitive(self):
        # Source longa: nÃƒÂ£o aciona o fix de imperativo
        result = _postprocess(
            text="Eu preciso acordar agora.",
            was_upper=False,
            tipo="fala",
            source_text="I really need to wake up right now.",
            lang="en",
        )
        self.assertEqual(result, "Eu preciso acordar agora.")


if __name__ == "__main__":
    unittest.main()
