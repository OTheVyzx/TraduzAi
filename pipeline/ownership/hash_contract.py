"""Canonical hashing primitives shared by ownership pipeline contracts.

The functions in this module are the single authority for persisted identities.
They intentionally hash decoded RGB pixels separately from source-file bytes so
that both visual identity and container/encoding identity remain auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, TypeAlias


JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for an exact byte sequence."""

    if not isinstance(value, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without loading it all into memory."""

    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"hash input is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text without platform-dependent newline conversion."""

    if not isinstance(value, str):
        raise TypeError("sha256_text requires str")
    return sha256_bytes(value.encode("utf-8"))


def _validate_json_value(value: object, *, location: str = "$") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string JSON object key at {location}")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise TypeError(f"unsupported canonical JSON value at {location}: {type(value).__name__}")


def _canonical_json_bytes(value: JSONValue) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: JSONValue) -> str:
    """Hash JSON using sorted keys, UTF-8 and no insignificant whitespace."""

    return sha256_bytes(_canonical_json_bytes(value))


def canonical_json_bytes(value: JSONValue) -> bytes:
    """Serialize a JSON value using the same canonical form used for hashing."""

    return _canonical_json_bytes(value)


def _canonical_rgb(image: object):
    """Return a C-contiguous uint8 RGB ndarray and its canonical metadata."""

    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise TypeError("canonical page pixels must use uint8 samples")
        if array.ndim == 2:
            rgb = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 1:
            rgb = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 3:
            rgb = array
        elif array.ndim == 3 and array.shape[2] == 4:
            rgb = np.asarray(Image.fromarray(array, "RGBA").convert("RGB"), dtype=np.uint8)
        else:
            raise ValueError("canonical page pixels must be HxW, HxWx1, HxWx3 or HxWx4")
    else:
        raise TypeError("canonical_page_sha256 requires a PIL image or numpy array")

    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ValueError("canonical page pixels cannot be empty")
    contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)
    metadata: JSONValue = {
        "height": int(contiguous.shape[0]),
        "mode": "RGB",
        "width": int(contiguous.shape[1]),
    }
    return contiguous, metadata


def canonical_page_sha256(image: object) -> str:
    """Hash canonical dimensions, RGB mode and exact decoded pixel bytes."""

    rgb, metadata = _canonical_rgb(image)
    digest = hashlib.sha256()
    digest.update(b"traduzai-canonical-page-v1\0")
    digest.update(_canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(rgb.tobytes(order="C"))
    return digest.hexdigest()


def canonical_source_tree_sha256(
    ordered_paths: Iterable[str | Path], root: str | Path
) -> str:
    """Hash an explicitly ordered image tree using portable relative paths.

    Each entry binds the source file bytes and decoded RGB pixels. Paths that
    escape the root, repeat, or collide after case-folding are rejected before
    any digest is returned.
    """

    from PIL import Image

    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError(f"source tree root is not a directory: {root_path}")

    entries: list[JSONValue] = []
    lexical_paths: set[str] = set()
    casefold_paths: set[str] = set()
    for supplied in ordered_paths:
        candidate = Path(supplied)
        candidate = candidate if candidate.is_absolute() else root_path / candidate
        candidate_absolute = candidate.absolute()
        try:
            lexical_relative = candidate_absolute.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"source path is outside root: {candidate_absolute}") from exc
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"source path is outside root: {resolved}") from exc
        if not resolved.is_file():
            raise ValueError(f"source path is not a regular file: {relative}")
        if lexical_relative in lexical_paths:
            raise ValueError(f"duplicate source path: {lexical_relative}")
        folded = lexical_relative.casefold()
        if folded in casefold_paths:
            raise ValueError(f"casefold source path collision: {lexical_relative}")
        lexical_paths.add(lexical_relative)
        casefold_paths.add(folded)

        with Image.open(resolved) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            pixel_sha256 = canonical_page_sha256(rgb)
        entries.append(
            {
                "height": height,
                "mode": "RGB",
                "page_source_sha256": pixel_sha256,
                "relative_path": relative,
                "source_file_sha256": sha256_file(resolved),
                "width": width,
            }
        )

    if not entries:
        raise ValueError("canonical source tree requires at least one image")
    tree: JSONValue = {"entries": entries, "schema_version": 1}
    return canonical_json_sha256(tree)
