from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ownership.delivery import (
    GlyphRunObservation,
    build_owner_text_delivery_contract,
    seal_owner_text_execution_authority,
    validate_owner_text_delivery_contract,
)


def _mask(x1: int, x2: int) -> np.ndarray:
    mask = np.zeros((24, 80), dtype=np.uint8)
    mask[6:18, x1:x2] = 255
    return mask


def _contract(rendered_lines, *, translated="O QUE ELES ESTÃO DIZENDO?"):
    authority = seal_owner_text_execution_authority(
        owner_id="owner_a",
        page_id="page_001",
        source_payload="WHAT ARE THEY SAYING?",
        translated_payload=translated,
        normalized_chunks=[translated],
    )
    span_masks = tuple(_mask(index * 20 + 2, index * 20 + 18) for index in range(len(rendered_lines)))
    core = np.maximum.reduce(span_masks) if span_masks else np.zeros((24, 80), dtype=np.uint8)
    runs = tuple(
        GlyphRunObservation.build(text=line, font_identity="Komika", span_index=index)
        for index, line in enumerate(rendered_lines)
    )
    return authority, build_owner_text_delivery_contract(
        execution_authority=authority,
        layout_payload=translated,
        rendered_lines=rendered_lines,
        rendered_glyph_runs=runs,
        glyph_core_mask=core,
        glyph_span_core_masks=span_masks,
        rendered_patch_sha256="a" * 64,
    )


def test_delivery_contract_binds_one_complete_owner_payload_to_render_input():
    authority, contract = _contract(["O QUE ELES", "ESTÃO DIZENDO?"])

    assert contract.status == "delivered"
    assert contract.translated_payload_sha256 == contract.layout_payload_sha256
    assert contract.translated_payload_sha256 == contract.rendered_payload_sha256
    assert contract.execution_authority_sha256 == authority.authority_sha256
    assert validate_owner_text_delivery_contract(contract, execution_authority=authority)


@pytest.mark.parametrize(
    ("rendered", "reason"),
    [
        (["O QUE ELES"], "render_payload_incomplete"),
        (["O QUE ELES", "ESTÃO"], "render_payload_incomplete"),
        (["O QUE ELES", "ESTÃO DIZENDO?", "ESTÃO DIZENDO?"], "render_payload_duplicated"),
    ],
)
def test_partial_or_duplicated_render_body_is_not_delivered(rendered, reason):
    _authority, contract = _contract(rendered)

    assert contract.status == "review_required"
    assert contract.reason == reason


def test_delivery_preserves_unicode_name_and_cross_tile_body_exactly_once():
    translated = "NÃO, JOÃO! O CORPO CONECTADO CONTINUA AQUI."
    authority, contract = _contract(
        ["NÃO, JOÃO!", "O CORPO CONECTADO", "CONTINUA AQUI."],
        translated=translated,
    )

    assert contract.status == "delivered"
    assert validate_owner_text_delivery_contract(contract, execution_authority=authority)


def test_delivery_rejects_tampered_self_hash_against_sealed_authority():
    authority, contract = _contract(["O QUE ELES", "ESTÃO DIZENDO?"])
    tampered = replace(contract, rendered_payload_sha256="f" * 64)

    with pytest.raises(ValueError, match="text delivery contract hash mismatch"):
        validate_owner_text_delivery_contract(tampered, execution_authority=authority)


def test_delivery_requires_exact_once_glyph_span_masks():
    authority = seal_owner_text_execution_authority(
        owner_id="owner_a", page_id="page_001", source_payload="BODY",
        translated_payload="CORPO COMPLETO", normalized_chunks=["CORPO COMPLETO"],
    )
    duplicated = _mask(2, 30)
    contract = build_owner_text_delivery_contract(
        execution_authority=authority,
        layout_payload="CORPO COMPLETO",
        rendered_lines=["CORPO", "COMPLETO"],
        rendered_glyph_runs=[
            GlyphRunObservation.build(text="CORPO", font_identity="Komika", span_index=0),
            GlyphRunObservation.build(text="COMPLETO", font_identity="Komika", span_index=1),
        ],
        glyph_core_mask=duplicated,
        glyph_span_core_masks=[duplicated, duplicated],
        rendered_patch_sha256="a" * 64,
    )

    assert contract.status == "review_required"
    assert contract.reason == "glyph_span_duplicated"
