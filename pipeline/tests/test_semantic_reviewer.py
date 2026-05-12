import unittest

from ocr.semantic_reviewer import semantic_refine_text


class SemanticReviewerTests(unittest.TestCase):
    def test_repairs_common_dialogue_confusions(self):
        reviewed = semantic_refine_text("D0NT Y0U M0VE!", tipo="fala", confidence=0.48)
        self.assertEqual(reviewed, "DON'T YOU MOVE!")

    def test_repairs_contraction_like_im(self):
        reviewed = semantic_refine_text("1M N0T G0ING", tipo="fala", confidence=0.43)
        self.assertEqual(reviewed, "I'M NOT GOING")

    def test_preserves_clean_high_confidence_text(self):
        reviewed = semantic_refine_text("GET OUT OF HERE!", tipo="fala", confidence=0.91)
        self.assertEqual(reviewed, "GET OUT OF HERE!")

    def test_preserves_high_confidence_abbreviations_and_dotted_titles(self):
        reviewed = semantic_refine_text("Read Dr.Stone at the U.S.A. HQ", tipo="fala", confidence=0.96)
        self.assertEqual(reviewed, "Read Dr.Stone at the U.S.A. HQ")

    def test_splits_common_merged_words_before_translation(self):
        reviewed = semantic_refine_text("BYANY MEANS", tipo="narracao", confidence=0.80)
        self.assertEqual(reviewed, "BY ANY MEANS")

    def test_splits_common_merged_words_even_when_confident(self):
        reviewed = semantic_refine_text(
            "WHAT'SWITH THE THINGSBETWEEN US?",
            tipo="fala",
            confidence=0.95,
        )
        self.assertEqual(reviewed, "WHAT'S WITH THE THINGS BETWEEN US?")

    def test_repairs_common_phrase_boundaries(self):
        reviewed = semantic_refine_text(
            "T CANNOT BE STOPPED. PREPARE YOURSELF,SENIOR BROTHER.",
            tipo="fala",
            confidence=0.93,
        )
        self.assertEqual(reviewed, "IT CANNOT BE STOPPED. PREPARE YOURSELF, SENIOR BROTHER.")

    def test_splits_so_one_must_compound(self):
        reviewed = semantic_refine_text(
            "SOONEMUST UNDERSTAND THE QUIETEST PRINCIPLE",
            tipo="fala",
            confidence=0.58,
        )
        self.assertEqual(reviewed, "SO ONE MUST UNDERSTAND THE QUIETEST PRINCIPLE")

    def test_repairs_ocr_period_inserted_inside_its_about_time(self):
        reviewed = semantic_refine_text(
            "...It's. ABOUT TIME TO FINISH THIS.",
            tipo="fala",
            confidence=0.89,
        )
        self.assertEqual(reviewed, "...IT'S ABOUT TIME TO FINISH THIS.")

    def test_splits_known_title_words_that_ocr_joined(self):
        reviewed = semantic_refine_text(
            "martialwildwest",
            tipo="narracao",
            confidence=0.83,
        )
        self.assertEqual(reviewed, "martial wild west")

    def test_repairs_observed_merged_narration_tokens(self):
        reviewed = semantic_refine_text(
            "THISHUNWONTHUNDER REACHESA CERTAIN POINT",
            tipo="narracao",
            confidence=0.74,
        )
        self.assertEqual(reviewed, "THIS HUNWON THUNDER REACHES A CERTAIN POINT")

    def test_repairs_observed_merged_action_tokens(self):
        reviewed = semantic_refine_text(
            "STOPSALLMOVEMENT ANDE RANSFORMSINTO A SINGLEBOLT OFLIGHTNING",
            tipo="narracao",
            confidence=0.72,
        )
        self.assertEqual(
            reviewed,
            "STOPS ALL MOVEMENT AND TRANSFORMS INTO A SINGLE BOLT OF LIGHTNING",
        )

    def test_repairs_observed_split_action_phrase_without_global_ani_rewrite(self):
        reviewed = semantic_refine_text(
            "STOPSALL MOVEMENT ANI TRANSFORMS INTO A SINGLE BOLT",
            tipo="narracao",
            confidence=0.95,
        )
        self.assertEqual(reviewed, "STOPS ALL MOVEMENT AND TRANSFORMS INTO A SINGLE BOLT")

        unrelated = semantic_refine_text("ANI WAITS OUTSIDE.", tipo="fala", confidence=0.95)
        self.assertEqual(unrelated, "ANI WAITS OUTSIDE.")

    def test_repairs_observed_merged_dialogue_tokens(self):
        reviewed = semantic_refine_text(
            "I'M THE ONE WHOPRACTICEDIT FOR TWENTY YEARS, YET YOU CLAIM TO KNOWITBETTER",
            tipo="fala",
            confidence=0.82,
        )
        self.assertEqual(
            reviewed,
            "I'M THE ONE WHO PRACTICED IT FOR TWENTY YEARS, YET YOU CLAIM TO KNOW IT BETTER",
        )

    def test_repairs_observed_merged_punctuation_tokens(self):
        reviewed = semantic_refine_text(
            "OFCOURSE.THAT OUTDATED MARTIAL ARTIKNEW COULDN'TPOSSIBLYHAVE SUCH POWER",
            tipo="fala",
            confidence=0.84,
        )
        self.assertEqual(
            reviewed,
            "OF COURSE. THAT OUTDATED MARTIAL ART I KNEW COULDN'T POSSIBLY HAVE SUCH POWER",
        )

    def test_repairs_remaining_observed_joined_tokens(self):
        reviewed = semantic_refine_text(
            "SAMADH RUEFIREOFDRAGON-SUBDUING PALM. IREMOVED IT. SOICAN'T. WASTHAT ARMYMARTIAL ART OE THE FOUNDER?",
            tipo="fala",
            confidence=0.88,
        )
        self.assertEqual(
            reviewed,
            "SAMADHI TRUE FIRE OF DRAGON-SUBDUING PALM. I REMOVED IT. SO I CAN'T. WAS THAT ARMY MARTIAL ART OF THE FOUNDER?",
        )

    def test_repairs_observed_out_of_order_arrogant_tone_merge(self):
        reviewed = semantic_refine_text(
            "WHAT'S WITHE TONE? ARE YOU ACCUSING ME OF AND DESTROYING OUR THAT ARROGANT BETRAYING OUR MASTER LINEAGE?",
            tipo="fala",
            confidence=0.64,
        )
        self.assertEqual(
            reviewed,
            "WHAT'S WITH THAT ARROGANT TONE? ARE YOU ACCUSING ME OF BETRAYING OUR MASTER AND DESTROYING OUR LINEAGE?",
        )

    def test_repairs_remaining_high_noise_exact_tokens(self):
        reviewed = semantic_refine_text(
            "KOMWTWYHATYOO WANTED. OOAMSILN FINAL MESSAGE. DON'T SAY THAT ARE YOURINJURIES ALRIGHT?",
            tipo="fala",
            confidence=0.82,
        )
        self.assertEqual(
            reviewed,
            "KNOW WHAT YOU WANTED. MASTER'S FINAL MESSAGE. DON'T SAY THAT. ARE YOUR INJURIES ALRIGHT?",
        )


if __name__ == "__main__":
    unittest.main()
