import unittest
from unittest.mock import patch

from translator.translate import (
    _build_context_hints,
    _build_text_payload,
    _fix_infinitive_to_imperative,
    _is_likely_proper_noun,
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

    def test_translate_pages_repairs_proper_name_using_context_entities(self):
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

        self.assertEqual(captured_batches[0], ["Ghislain Perdium"])
        entity_meta = translated[0]["texts"][0]
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

    def test_translate_pages_repairs_embedded_proper_name_inside_short_phrase(self):
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

        self.assertIn("Ghislain Perdium", captured_batches[0][0])
        entity_meta = translated[0]["texts"][0]
        self.assertIn("source_entity_repaired", entity_meta["entity_flags"])
        self.assertTrue(
            any(item.get("to") == "Ghislain Perdium" for item in entity_meta["entity_repairs"])
        )

    def test_translate_pages_locks_glossary_terms_inside_sentence(self):
        class _FakeGoogleTranslator:
            def __init__(self, source="en", target="pt"):
                self.target = target

            def translate(self, text: str):
                return "teste" if text == "test" else text

            def translate_batch(self, texts: list[str]) -> list[str]:
                return ["o nucleo de mana esta instavel."]

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
                        glossario={"Mana Core": "Núcleo de Mana"},
                        idioma_origem="en",
                        idioma_destino="pt-BR",
                    )

        entity_meta = translated[0]["texts"][0]
        self.assertIn("Núcleo de Mana", entity_meta["translated"])
        self.assertEqual(
            entity_meta["glossary_hits"],
            [{"phase": "target", "source": "Mana Core", "target": "Núcleo de Mana"}],
        )
        self.assertIn("glossary_locked", entity_meta["entity_flags"])

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

    def test_review_translation_grammar_semantics_rewrites_trading_with_them_phrase(self):
        reviewed = _review_translation_grammar_semantics(
            source_text='YOU MEAN THE POWER HE GOT BY TRADING WITH "THEM"?',
            translated_text='Você quer dizer o poder que ele obteve negociando com "eles"?',
            tipo="fala",
        )
        self.assertEqual(reviewed, "Você quer dizer o poder que ele conseguiu em um acordo com eles?")

    def test_review_translation_grammar_semantics_compacts_half_mana_technique_line(self):
        reviewed = _review_translation_grammar_semantics(
            source_text=(
                "EVEN THOUGH IT'S ONLY HALF OF A MANA TECHNIQUE, "
                "ITS EFFECTS WILL BE MORE THAN ENOUGH. "
                "THAT POWER LETS ONE INSTANTLY SURPASS THEIR OWN LIMITS."
            ),
            translated_text=(
                "Mesmo sendo apenas metade de uma técnica de mana, "
                "seus efeitos serão mais que suficientes. "
                "Esse poder permite superar instantaneamente seus próprios limites."
            ),
            tipo="fala",
        )
        self.assertEqual(
            reviewed,
            "Pode até ser só metade de uma técnica de mana, mas seus efeitos já são mais do que suficientes. "
            "Esse poder permite ultrapassar instantaneamente os próprios limites.",
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
            translated_text="A violenta Vanessa também usou uma técnica de mana semelhante.",
            tipo="fala",
        )
        self.assertEqual(
            reviewed,
            "A Vanessa em fúria também usava um método semelhante de circulação de mana.",
        )

    def test_review_translation_grammar_semantics_preserves_split_sentence_continuation(self):
        reviewed = _review_translation_grammar_semantics(
            source_text="YOU COULD SEE ALL OF MY ATTACKS?",
            translated_text="Você pode ver todos os meus ataques?",
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

    def test_resolve_translation_backend_prefers_local_when_flag_enabled(self):
        with patch.dict("os.environ", {"TRADUZAI_PREFER_LOCAL_TRANSLATION": "1"}, clear=False):
            backend = _resolve_translation_backend(
                google_ok=True,
                ollama_status={
                    "running": True,
                    "models": ["mangatl-translator:latest"],
                    "has_translator": True,
                },
            )

        self.assertEqual(backend, "ollama")


class ProperNounProtectionTests(unittest.TestCase):
    """Verifica que nomes próprios em CAPS isolados são preservados."""

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
        # Palavras comuns em CAPS que não devem ser preservadas
        self.assertFalse(_is_likely_proper_noun("WAKE"))
        self.assertFalse(_is_likely_proper_noun("STOP"))
        self.assertFalse(_is_likely_proper_noun("HELP"))
        self.assertFalse(_is_likely_proper_noun("MAGIC"))
        self.assertFalse(_is_likely_proper_noun("PLEASE"))
        self.assertFalse(_is_likely_proper_noun("LORD"))
        self.assertFalse(_is_likely_proper_noun("MASTER"))

    def test_multi_word_text_is_not_proper_noun(self):
        self.assertFalse(_is_likely_proper_noun("WAKE UP"))
        self.assertFalse(_is_likely_proper_noun("STOP THEM"))
        self.assertFalse(_is_likely_proper_noun("MY LORD"))

    def test_short_words_are_not_proper_nouns(self):
        # Curtas demais para serem nomes próprios confiáveis
        self.assertFalse(_is_likely_proper_noun("GO"))
        self.assertFalse(_is_likely_proper_noun("HI"))
        self.assertFalse(_is_likely_proper_noun("NGH"))

    def test_text_with_digits_is_not_proper_noun(self):
        # OCR errors com dígitos (M9, i999) não são tratados como nomes
        self.assertFalse(_is_likely_proper_noun("M9"))
        self.assertFalse(_is_likely_proper_noun("i999"))
        self.assertFalse(_is_likely_proper_noun("R2D2"))

    def test_lowercase_words_are_not_proper_nouns(self):
        self.assertFalse(_is_likely_proper_noun("gillion"))
        self.assertFalse(_is_likely_proper_noun("willow"))


class ImperativeFixTests(unittest.TestCase):
    """Verifica que infinitivos em comandos curtos viram imperativos."""

    def test_wake_up_becomes_imperative(self):
        # Caso real do Cap 78 pág. 58: "WAKE UP." -> "ACORDAR." (errado)
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
        # Source com mais de 4 palavras não dispara o fix
        long_source = "I really need you to wake up right now."
        result = _fix_infinitive_to_imperative("Acordar.", long_source, "fala")
        # Mantém infinitivo porque a frase é longa o suficiente para o Google
        # ter contexto e traduzir corretamente como verbo principal.
        self.assertEqual(result, "Acordar.")

    def test_sfx_does_not_trigger_imperative_fix(self):
        # SFX nunca recebe o fix de imperativo
        result = _fix_infinitive_to_imperative("Acordar!", "WAKE UP!", "sfx")
        self.assertEqual(result, "Acordar!")

    def test_multi_word_translation_does_not_trigger_fix(self):
        # Tradução multi-palavra não bate na tabela; passa direto.
        result = _fix_infinitive_to_imperative("Por favor, acordar.", "WAKE UP.", "fala")
        self.assertEqual(result, "Por favor, acordar.")

    def test_unknown_infinitive_passes_through(self):
        # Verbo fora da tabela não é alterado
        result = _fix_infinitive_to_imperative("Discutir.", "DISCUSS.", "fala")
        self.assertEqual(result, "Discutir.")


class OcrRepairExpansionTests(unittest.TestCase):
    """Verifica que os novos padrões de OCR são aplicados em _preprocess_text."""

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
    """Garante que o fix de imperativo é executado durante _postprocess."""

    def test_postprocess_converts_short_infinitive(self):
        # Pipeline completo: source curta + tradução em infinitivo -> imperativo
        result = _postprocess(
            text="Acordar.",
            was_upper=True,
            tipo="fala",
            source_text="WAKE UP.",
            lang="en",
        )
        self.assertEqual(result, "ACORDE.")

    def test_postprocess_keeps_long_infinitive(self):
        # Source longa: não aciona o fix de imperativo
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
