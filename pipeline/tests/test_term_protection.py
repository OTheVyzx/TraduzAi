import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.term_protection import PLACEHOLDER_TEMPLATE, protect_terms, restore_terms, validate_placeholders


GLOSSARY = [
    {
        "id": "char_ghislain_perdium",
        "source": "GHISLAIN PERDIUM",
        "target": "Ghislain Perdium",
        "type": "character",
        "protect": True,
        "aliases": ["Ghislain Perdium"],
        "forbidden": ["Perdium Ghislain"],
    },
    {
        "id": "rank_knight",
        "source": "KNIGHT",
        "target": "Cavaleiro",
        "type": "rank",
        "protect": False,
        "aliases": [],
        "forbidden": ["Night"],
    },
]


def test_preserves_name_with_safe_placeholder_and_restores_exact_target():
    protected = protect_terms("GHISLAIN PERDIUM arrived.", GLOSSARY)

    assert protected["protected_source"] == "__TZN_NAME_0__ arrived."

    restored = restore_terms("__TZN_NAME_0__ chegou.", protected["terms"])
    assert restored["text"] == "Ghislain Perdium chegou."
    assert restored["flags"] == []


def test_fixed_translation_term_is_restored():
    protected = protect_terms("THE KNIGHT OF THE NORTH", GLOSSARY)
    restored = restore_terms("O __TZN_NAME_0__ do norte", protected["terms"])

    assert restored["text"] == "O Cavaleiro do norte"


def test_corrupted_placeholder_blocks_region():
    flags = validate_placeholders("TZN_NAME_0 chegou", [{"placeholder": "__TZN_NAME_0__"}])

    assert any(flag["reason"] == "unrestored_placeholder" for flag in flags)
    assert any(flag.get("issue") == "placeholder_corrupted" for flag in flags)


def test_forbidden_translation_generates_critical_flag():
    protected = protect_terms("GHISLAIN PERDIUM arrived.", GLOSSARY)
    restored = restore_terms("Perdium Ghislain chegou __TZN_NAME_0__", protected["terms"])

    assert any(flag["severity"] == "critical" for flag in restored["flags"])


def test_multiple_terms_in_same_sentence_are_protected():
    protected = protect_terms("GHISLAIN PERDIUM, THE KNIGHT OF THE NORTH", GLOSSARY)

    assert len(protected["terms"]) == 2
    assert protected["protected_source"] == "__TZN_NAME_0__, THE __TZN_NAME_1__ OF THE NORTH"


def test_placeholder_template_is_ascii_and_zero_based():
    assert PLACEHOLDER_TEMPLATE.format(index=0) == "__TZN_NAME_0__"


def test_common_uppercase_words_are_not_name_locked():
    glossary = [
        {
            "source": word,
            "target": word.title(),
            "type": "character",
            "protect": True,
            "aliases": [],
        }
        for word in ["ONE", "HOSPITAL", "READ", "THE", "I"]
    ]
    glossary.append(
        {
            "source": "Hosu",
            "target": "Hosu",
            "type": "character",
            "protect": True,
            "aliases": [],
        }
    )

    protected = protect_terms("ONE! HOSPITAL READ THE I Hosu...?", glossary)

    assert protected["terms"] == [
        {
            "placeholder": "__TZN_NAME_0__",
            "source": "Hosu",
            "target": "Hosu",
            "mode": "preserve",
            "protect": True,
            "forbidden": [],
        }
    ]
    assert protected["protected_source"] == "ONE! HOSPITAL READ THE I __TZN_NAME_0__...?"


def test_repeated_name_occurrences_get_distinct_placeholders():
    glossary = [
        {
            "source": "Wonho",
            "target": "Wonho",
            "type": "character",
            "protect": True,
            "aliases": [],
        }
    ]

    protected = protect_terms("Wonho met Wonho.", glossary)

    assert protected["protected_source"] == "__TZN_NAME_0__ met __TZN_NAME_1__."
    assert [term["source"] for term in protected["terms"]] == ["Wonho", "Wonho"]


def test_alias_occurrence_is_protected_when_canonical_name_also_matches():
    glossary = [
        {
            "source": "Wonho",
            "target": "Wonho",
            "type": "character",
            "protect": True,
            "aliases": ["Hosu"],
        }
    ]

    protected = protect_terms("Wonho met Hosu.", glossary)

    assert protected["protected_source"] == "__TZN_NAME_0__ met __TZN_NAME_1__."
    assert [term["source"] for term in protected["terms"]] == ["Wonho", "Hosu"]


def test_duplicated_placeholder_is_flagged_before_restore_can_pass_silently():
    terms = [
        {
            "placeholder": "__TZN_NAME_0__",
            "source": "Wonho",
            "target": "Wonho",
            "mode": "preserve",
            "protect": True,
            "forbidden": [],
        }
    ]

    restored = restore_terms("__TZN_NAME_0__ e __TZN_NAME_0__ chegaram.", terms)

    assert restored["blocked"] is True
    assert any(
        flag["reason"] == "unrestored_placeholder" and flag.get("issue") == "placeholder_count_mismatch"
        for flag in restored["flags"]
    )


def test_validate_placeholders_flags_extra_placeholder_count():
    flags = validate_placeholders(
        "__TZN_NAME_0__ e __TZN_NAME_0__ chegaram.",
        [{"placeholder": "__TZN_NAME_0__"}],
    )

    assert any(
        flag["reason"] == "unrestored_placeholder" and flag.get("issue") == "placeholder_count_mismatch"
        for flag in flags
    )
