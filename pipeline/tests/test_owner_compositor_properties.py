"""Metamorphic properties for deterministic page owner composition."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from compositor.owner_compositor import OwnerCompositionError, compose_page
from test_owner_compositor import (
    _array_sha256,
    _empty_protected,
    _glyph_patch,
    _mutation,
    _original,
)


def test_composition_is_invariant_to_tile_partition_and_input_order() -> None:
    original = _original()
    mutations = [
        _mutation(original, owner_id="owner_a", box=(1, 1, 5, 5)),
        _mutation(
            original,
            owner_id="owner_b",
            box=(9, 2, 14, 7),
            color=(70, 80, 90),
        ),
        _mutation(
            original,
            owner_id="owner_c",
            box=(16, 10, 22, 16),
            color=(100, 110, 120),
        ),
    ]
    glyphs = [
        _glyph_patch(
            original,
            owner_id=mutation.owner_id,
            mutation=mutation,
            box=box,
        )
        for mutation, box in zip(
            mutations,
            ((2, 2, 4, 4), (10, 3, 13, 6), (18, 12, 21, 15)),
            strict=True,
        )
    ]
    baseline = compose_page(
        original,
        mutations,
        glyphs,
        _empty_protected(original),
    )

    for mutation_order in permutations(mutations):
        for glyph_order in (glyphs, list(reversed(glyphs))):
            result = compose_page(
                original,
                [*mutation_order, mutation_order[0]],
                [*glyph_order, glyph_order[0]],
                _empty_protected(original),
            )
            assert result.committed is True
            assert result.sha256 == baseline.sha256
            assert result.write_counts == baseline.write_counts
            np.testing.assert_array_equal(result.final_rgb, baseline.final_rgb)
            np.testing.assert_array_equal(
                result.cleanup_owner_map,
                baseline.cleanup_owner_map,
            )
            np.testing.assert_array_equal(
                result.glyph_owner_map,
                baseline.glyph_owner_map,
            )


def test_cross_phase_claims_between_different_owners_fail_closed() -> None:
    original = _original()
    cleanup = _mutation(original, owner_id="owner_a", box=(2, 2, 9, 9))
    glyph = _glyph_patch(
        original,
        owner_id="owner_b",
        mutation=None,
        box=(7, 7, 12, 12),
    )

    result = compose_page(
        original,
        [cleanup],
        [glyph],
        _empty_protected(original),
    )

    assert result.committed is False
    assert any(conflict.phase == "cross_phase" for conflict in result.conflicts)
    np.testing.assert_array_equal(result.final_rgb, original)


def test_compositor_is_pure_and_freezes_result_arrays() -> None:
    original = _original()
    original_before = original.copy()
    mutation = _mutation(original)
    mutation_before = np.asarray(mutation.result_rgb).copy()

    result = compose_page(
        original,
        [mutation],
        [],
        _empty_protected(original),
    )

    np.testing.assert_array_equal(original, original_before)
    np.testing.assert_array_equal(mutation.result_rgb, mutation_before)
    assert result.final_rgb.flags.writeable is False
    assert result.cleanup_owner_map.flags.writeable is False
    assert result.glyph_owner_map.flags.writeable is False
    for frozen_array in (
        result.final_rgb,
        result.cleanup_owner_map,
        result.glyph_owner_map,
    ):
        with pytest.raises(ValueError):
            frozen_array.setflags(write=True)


@pytest.mark.parametrize(
    "invalid_original",
    [
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_noncanonical_page_arrays_are_rejected(
    invalid_original: np.ndarray,
) -> None:
    protected = np.zeros(invalid_original.shape[:2], dtype=np.uint8)

    with pytest.raises(OwnerCompositionError, match="RGB uint8 page"):
        compose_page(invalid_original, [], [], protected)


def test_final_hash_covers_exact_composed_array() -> None:
    original = _original()
    mutation = _mutation(original)

    result = compose_page(
        original,
        [mutation],
        [],
        _empty_protected(original),
    )

    assert result.sha256 == _array_sha256(result.final_rgb)
