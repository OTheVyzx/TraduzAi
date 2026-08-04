from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.hash_contract import (  # noqa: E402
    canonical_json_sha256,
    canonical_page_sha256,
    canonical_source_tree_sha256,
    sha256_bytes,
    sha256_file,
    sha256_text,
)


def _write_image(path: Path, rgb: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (5, 3), rgb).save(path)
    return path


def test_raw_hash_primitives_match_sha256(tmp_path):
    payload = "olá source replacement"
    encoded = payload.encode("utf-8")
    path = tmp_path / "payload.bin"
    path.write_bytes(encoded)

    expected = hashlib.sha256(encoded).hexdigest()
    assert sha256_bytes(encoded) == expected
    assert sha256_text(payload) == expected
    assert sha256_file(path) == expected


def test_canonical_json_hash_is_key_order_independent_but_list_order_sensitive():
    left = {"z": [1, 2], "a": {"enabled": True, "value": None}}
    reordered = {"a": {"value": None, "enabled": True}, "z": [1, 2]}
    reversed_list = {"a": {"value": None, "enabled": True}, "z": [2, 1]}

    assert canonical_json_sha256(left) == canonical_json_sha256(reordered)
    assert canonical_json_sha256(left) != canonical_json_sha256(reversed_list)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), ("tuple",)])
def test_canonical_json_rejects_values_outside_json_contract(invalid):
    with pytest.raises((TypeError, ValueError)):
        canonical_json_sha256({"invalid": invalid})


def test_canonical_page_hash_normalizes_equivalent_pixels_to_rgb():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 37
    rgba = np.dstack((rgb, np.full((4, 6), 255, dtype=np.uint8)))

    assert canonical_page_sha256(rgb) == canonical_page_sha256(Image.fromarray(rgba, "RGBA"))
    mutated = rgb.copy()
    mutated[0, 0, 1] = 1
    assert canonical_page_sha256(rgb) != canonical_page_sha256(mutated)


def test_canonical_source_tree_hash_binds_order_paths_files_and_pixels(tmp_path):
    root = tmp_path / "source"
    first_path = _write_image(root / "nested" / "001.png", (20, 40, 60))
    second_path = _write_image(root / "002.png", (80, 100, 120))
    ordered = (first_path, second_path)

    first = canonical_source_tree_sha256(ordered, root)
    assert first == canonical_source_tree_sha256(ordered, root)
    assert first != canonical_source_tree_sha256(tuple(reversed(ordered)), root)

    Image.new("RGB", (5, 3), (21, 40, 60)).save(first_path)
    assert first != canonical_source_tree_sha256(ordered, root)


def test_canonical_source_tree_rejects_outside_duplicate_and_casefold_collision(tmp_path):
    root = tmp_path / "source"
    inside = _write_image(root / "001.png", (20, 40, 60))
    outside = _write_image(tmp_path / "outside.png", (80, 100, 120))

    with pytest.raises(ValueError, match="outside"):
        canonical_source_tree_sha256((inside, outside), root)
    with pytest.raises(ValueError, match="duplicate"):
        canonical_source_tree_sha256((inside, inside), root)

    upper = _write_image(root / "Page.png", (1, 2, 3))
    lower = _write_image(root / "page.png", (4, 5, 6))
    with pytest.raises(ValueError, match="casefold"):
        canonical_source_tree_sha256((upper, lower), root)
