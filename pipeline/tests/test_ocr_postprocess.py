import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.postprocess import (
    classify_text_type,
    fix_ocr_errors,
    has_run_on_tokens,
    is_cover_title_logo,
    is_editorial_credit,
    is_ghost_ocr_noise,
    is_hallucination,
    is_korean_sfx,
    is_low_confidence_visual_noise,
    is_short_textured_sfx_or_noise,
    is_vlm_failure_phrase,
    is_watermark,
    looks_suspicious,
    should_preserve_cjk_sfx_candidate,
)
from ocr.text_router import route_text_record


class OcrPostprocessTests(unittest.TestCase):
    def test_watermark_detects_asura_discord_variants(self):
        self.assertTrue(is_watermark("Asura.gg/discord Asuracomic.net"))
        self.assertTrue(is_watermark("Read only at ASURACOMIC.NET"))
        self.assertTrue(is_watermark("ASURASOANS. COM"))
        self.assertTrue(is_watermark("FOR THE FASTEST RELEASES"))
        self.assertTrue(is_watermark("Reader that does not support our website has been detected!"))
        self.assertTrue(is_watermark("READERTHATDOESNOTSUPPORT OURWEBSITEHASBEENDETECTED!"))

    def test_editorial_credit_keeps_ambiguous_single_words(self):
        self.assertFalse(is_editorial_credit("STAFF"))
        self.assertFalse(is_editorial_credit("RAW"))

    def test_editorial_credit_still_detects_role_credit_lines(self):
        self.assertTrue(is_editorial_credit("TRANSLATOR AKIRA"))
        self.assertTrue(is_editorial_credit("CLEANER REDRAWER"))
        self.assertTrue(is_editorial_credit("STAFF EDITOR"))
        self.assertTrue(is_editorial_credit("RAW PROVIDER"))

    def test_editorial_credit_detects_scan_team_role_line_and_tl_note(self):
        self.assertTrue(is_editorial_credit("TL Kiki Pr Mars Shadow CI Ts Erian Qc Shadow Rp Shadow"))
        self.assertTrue(is_editorial_credit("TL/N: AISH IS A FORM OF IRRITATED EXPRESSION IN KOREA."))
        self.assertTrue(is_editorial_credit("TL/NAISHISAFORMOF IRRITATED OR ANNOYED EXPRESSION IN KOREA."))
        self.assertFalse(is_editorial_credit("PLEASE, FOR THE CHILD'S SAKE."))

    def test_editorial_credit_detects_cover_view_metadata_line(self):
        self.assertTrue(
            is_editorial_credit(
                "'COVER ~\"ABSOLUTE\" = WOOJIN'S KANG L CARA VER:] MILEY [KOREAN I EGO ALTER / VIEWS IS. EEM"
            )
        )

    def test_cover_title_logo_requires_user_provided_title(self):
        self.assertFalse(
            is_cover_title_logo(
                "SIE ooi",
                [53, 1226, 800, 2283],
                0.74,
                (2400, 800, 3),
                "fala",
                False,
                page_profile="cover_opening",
                work_title_user_provided=False,
            )
        )
        self.assertTrue(
            is_cover_title_logo(
                "The Regressed Mercenary",
                [53, 1226, 800, 2283],
                0.74,
                (2400, 800, 3),
                "fala",
                False,
                page_profile="cover_opening",
                work_title="The Regressed Mercenary Has a Plan",
                work_title_aliases=["The Regressed Mercenary"],
                work_title_user_provided=True,
            )
        )

    def test_cover_logo_rules_are_disabled_without_user_title(self):
        routed = route_text_record(
            {
                "text": "REGIONAL FIREMAN RECRUITMENT TEST",
                "bbox": [10, 10, 320, 80],
                "confidence": 0.91,
                "page_number": 1,
                "page_height": 900,
            },
            work_title="",
            work_title_aliases=[],
            work_title_user_provided=False,
        )

        self.assertIn(routed.get("route_action"), (None, "", "translate_inpaint_render"))
        self.assertIn(routed.get("content_class"), (None, "", "text", "dialogue"))
        self.assertIsNot(routed.get("skip_processing"), True)
        self.assertIsNot(routed.get("preserve_original"), True)

    def test_router_treats_legacy_noise_credit_and_note_as_text(self):
        for text in (
            "",
            "T/N: translator note",
            "Read at scanlator.com",
            "Kgm sini_@naver.com llshinheell",
            "TEXT: hospital sign",
        ):
            routed = route_text_record(
                {
                    "text": text,
                    "bbox": [10, 10, 320, 80],
                    "confidence": 0.91,
                },
                work_title="",
                work_title_aliases=[],
                work_title_user_provided=False,
            )

            self.assertEqual(routed.get("route_action"), "translate_inpaint_render")
            self.assertEqual(routed.get("content_class"), "text")
            self.assertFalse(routed.get("skip_processing"))
            self.assertIsNot(routed.get("preserve_original"), True)

    def test_korean_dialogue_is_not_forced_to_sfx(self):
        self.assertFalse(is_korean_sfx("도저히 생문을 찾을 수가 없다"))
        self.assertFalse(is_korean_sfx("진법은 이 갈사량의 짓이 분명합니다"))
        self.assertNotEqual(classify_text_type("과연 그럴까?", [130, 16, 394, 75], 720), "sfx")

    def test_korean_onomatopoeia_stays_sfx(self):
        self.assertTrue(is_korean_sfx("크아아악!!"))
        self.assertTrue(is_korean_sfx("하하하"))
        self.assertTrue(is_korean_sfx("즈으으"))

    def test_known_short_korean_sfx_stays_sfx(self):
        self.assertTrue(is_korean_sfx("\uD5C8\uC5EC"))
        self.assertTrue(is_korean_sfx("\uBE44\uD2C0"))

    def test_korean_dialogue_question_is_not_sfx(self):
        self.assertFalse(is_korean_sfx("뭐?!"))
        self.assertNotEqual(classify_text_type("뭐?!", [109, 1135, 277, 1207], 690), "sfx")

    def test_korean_source_cleanup_preserves_korean_characters(self):
        self.assertEqual(
            fix_ocr_errors("북명대는 무적입니다", idioma_origem="ko"),
            "북명대는 무적입니다",
        )

    def test_low_confidence_korean_dialogue_is_not_suspicious_noise(self):
        self.assertFalse(looks_suspicious("죽여라!", 0.49))

    def test_run_on_token_detector_catches_joined_ocr_word(self):
        self.assertTrue(has_run_on_tokens("JLSTASMAROUESS BRANEORD SAID."))

    def test_run_on_token_detector_ignores_normal_long_word(self):
        self.assertFalse(has_run_on_tokens("CONGRATULATIONS, SENIOR BROTHER."))

    def test_quoted_short_dialogue_is_not_dropped_as_textured_sfx(self):
        self.assertFalse(is_short_textured_sfx_or_noise('"WE"?!', [502, 16, 718, 108], 0.828, False))
        self.assertTrue(is_short_textured_sfx_or_noise("SH", [502, 16, 560, 108], 0.828, False))

    def test_unquoted_short_caps_dialogue_with_punctuation_is_not_dropped_as_sfx(self):
        self.assertFalse(is_short_textured_sfx_or_noise("WE?!", [502, 16, 718, 108], 0.828, False))
        self.assertFalse(is_short_textured_sfx_or_noise("NO!", [502, 16, 718, 108], 0.828, False))
        self.assertFalse(is_short_textured_sfx_or_noise("OK?", [502, 16, 718, 108], 0.828, False))
        self.assertTrue(is_short_textured_sfx_or_noise("SH", [502, 16, 560, 108], 0.828, False))

    def test_cjk_sfx_preserve_gate_catches_short_artifacts_before_translation(self):
        self.assertTrue(
            should_preserve_cjk_sfx_candidate(
                "gioi",
                [392, 1469, 626, 1629],
                0.91,
                is_white_balloon=True,
                source_lang="ko",
                image_shape=(2400, 690, 3),
                block_profile="white_balloon",
            )
        )
        self.assertTrue(
            should_preserve_cjk_sfx_candidate(
                "\uB204\uC774\uC57C",
                [94, 617, 289, 729],
                0.88,
                is_white_balloon=False,
                source_lang="ko",
                image_shape=(1800, 690, 3),
                block_profile="standard",
            )
        )
        self.assertTrue(
            should_preserve_cjk_sfx_candidate(
                "bloto",
                [195, 1177, 318, 1269],
                0.90,
                is_white_balloon=False,
                source_lang="ko",
                image_shape=(2400, 690, 3),
                block_profile="standard",
            )
        )
        self.assertFalse(
            should_preserve_cjk_sfx_candidate(
                "bloto",
                [195, 1177, 318, 1269],
                0.90,
                is_white_balloon=True,
                source_lang="ko",
                image_shape=(2400, 690, 3),
                block_profile="white_balloon",
            )
        )
        self.assertFalse(
            should_preserve_cjk_sfx_candidate(
                "\uBB50?!",
                [109, 1135, 277, 1207],
                0.91,
                is_white_balloon=True,
                source_lang="ko",
                image_shape=(1800, 690, 3),
                block_profile="white_balloon",
            )
        )
        self.assertFalse(
            should_preserve_cjk_sfx_candidate(
                "\uC8FD\uC5EC\uB77C!",
                [245, 44, 472, 132],
                0.90,
                is_white_balloon=True,
                source_lang="ko",
                image_shape=(1800, 690, 3),
                block_profile="white_balloon",
            )
        )

    def test_vlm_failure_phrase_is_dropped_even_with_high_confidence(self):
        text = "The image is too blurry to recognize any text content."
        self.assertTrue(is_vlm_failure_phrase(text))
        self.assertTrue(is_hallucination(text, [12, 18, 420, 58], 0.99))

    def test_low_confidence_visual_noise_is_metric_only_not_a_drop_gate(self):
        self.assertFalse(
            is_low_confidence_visual_noise(
                "Oq ^ I s9aat",
                [396, 864, 621, 1008],
                0.38,
                is_white_balloon=True,
                image_shape=(5000, 1600, 3),
            )
        )
        self.assertFalse(
            is_low_confidence_visual_noise(
                "W I KO",
                [217, 147, 1322, 388],
                0.38,
                is_white_balloon=True,
                image_shape=(5000, 1600, 3),
            )
        )
        self.assertFalse(
            is_low_confidence_visual_noise(
                "W I KO",
                [217, 147, 1322, 388],
                0.40,
                is_white_balloon=True,
                image_shape=(5000, 1600, 3),
            )
        )
        self.assertFalse(
            is_low_confidence_visual_noise(
                "LET'S GO!",
                [92, 16, 233, 38],
                0.457,
                is_white_balloon=True,
                image_shape=(307, 371, 3),
            )
        )
        self.assertFalse(
            is_low_confidence_visual_noise(
                "WHAT?",
                [120, 180, 280, 260],
                0.38,
                is_white_balloon=True,
                image_shape=(5000, 1600, 3),
            )
        )

    def test_ghost_ocr_noise_drops_tiny_outside_balloon_tokens(self):
        self.assertTrue(
            is_ghost_ocr_noise(
                "1",
                [10, 10, 20, 28],
                0.99,
                is_white_balloon=False,
                image_shape=(1600, 720, 3),
            )
        )
        self.assertFalse(
            is_ghost_ocr_noise(
                "1",
                [10, 10, 20, 28],
                0.99,
                is_white_balloon=True,
                image_shape=(1600, 720, 3),
            )
        )

    def test_korean_sfx_inside_white_balloon_is_not_preserved(self):
        self.assertFalse(
            should_preserve_cjk_sfx_candidate(
                "\uD558\uD558\uD558.",
                [140, 210, 360, 300],
                0.96,
                is_white_balloon=True,
                source_lang="ko",
                image_shape=(1800, 690, 3),
                block_profile="white_balloon",
            )
        )
        self.assertTrue(
            should_preserve_cjk_sfx_candidate(
                "\uD558\uD558\uD558.",
                [140, 210, 360, 300],
                0.96,
                is_white_balloon=False,
                source_lang="ko",
                image_shape=(1800, 690, 3),
                block_profile="standard",
            )
        )


if __name__ == "__main__":
    unittest.main()
