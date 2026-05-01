from translator.term_protection import protect_terms, restore_terms, validate_placeholders


GLOSSARY = [
    {
        "id": "char_ghislain_perdium",
        "source": "GHISLAIN PERDIUM",
        "target": "Ghislain Perdium",
        "type": "character",
        "protect": True,
        "aliases": ["Ghislain Perdium"],
        "forbidden": ["Pérdium Ghislain"],
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

    assert "⟦TA_TERM_" in protected["protected_source"]

    restored = restore_terms("⟦TA_TERM_001⟧ chegou.", protected["terms"])
    assert restored["text"] == "Ghislain Perdium chegou."
    assert restored["flags"] == []


def test_fixed_translation_term_is_restored():
    protected = protect_terms("THE KNIGHT OF THE NORTH", GLOSSARY)
    restored = restore_terms("O ⟦TA_TERM_001⟧ do norte", protected["terms"])

    assert restored["text"] == "O Cavaleiro do norte"


def test_corrupted_placeholder_blocks_region():
    flags = validate_placeholders("TA_TERM_001 chegou", [{"placeholder": "⟦TA_TERM_001⟧"}])

    assert any(flag["reason"] == "placeholder_missing" for flag in flags)
    assert any(flag["reason"] == "placeholder_corrupted" for flag in flags)


def test_forbidden_translation_generates_critical_flag():
    protected = protect_terms("GHISLAIN PERDIUM arrived.", GLOSSARY)
    restored = restore_terms("Pérdium Ghislain chegou ⟦TA_TERM_001⟧", protected["terms"])

    assert any(flag["severity"] == "critical" for flag in restored["flags"])


def test_multiple_terms_in_same_sentence_are_protected():
    protected = protect_terms("GHISLAIN PERDIUM, THE KNIGHT OF THE NORTH", GLOSSARY)

    assert len(protected["terms"]) == 2
    assert protected["protected_source"].count("⟦TA_TERM_") == 2
