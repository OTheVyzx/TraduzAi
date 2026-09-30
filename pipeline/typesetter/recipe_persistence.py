"""Immutable persistence and deterministic replay for RendererRecipe artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Callable

from typesetter.recipe_contract import RendererRecipe


def _canonical_recipe_bytes(recipe: RendererRecipe) -> bytes:
    return json.dumps(
        recipe.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def persist_renderer_recipe(root: str | Path, recipe: RendererRecipe) -> Path:
    """Persist one immutable recipe under its content identity; identical writes are idempotent."""

    validated = RendererRecipe.from_dict(recipe.to_dict())
    directory = Path(root).resolve() / "renderer-recipes"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{validated.recipe_sha256}.json"
    expected = _canonical_recipe_bytes(validated)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != expected:
            raise ValueError("persisted renderer recipe conflicts with its immutable identity")
        return destination

    descriptor, temporary_name = tempfile.mkstemp(prefix=".renderer-recipe-", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != expected:
                raise ValueError("persisted renderer recipe conflicts with its immutable identity")
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def load_renderer_recipe(path: str | Path) -> RendererRecipe:
    """Load a persisted recipe and revalidate its complete canonical hash."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("renderer recipe artifact must be a regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("persisted renderer recipe must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("persisted renderer recipe must be an object")
    return RendererRecipe.from_dict(payload)


def rerender_typography_edit(
    previous: RendererRecipe,
    revised: RendererRecipe,
    *,
    rasterize: Callable[[RendererRecipe], bytes],
    persist_root: str | Path | None = None,
) -> tuple[bytes, Path | None]:
    """Replay an edited recipe and require same-analysis lineage and exact output bytes."""

    old = RendererRecipe.from_dict(previous.to_dict())
    new = RendererRecipe.from_dict(revised.to_dict())
    if new.target_text == old.target_text:
        raise ValueError("typography edit must change target text")
    if new.owner_id != old.owner_id or new.source_sha256 != old.source_sha256:
        raise ValueError("typography edit must retain owner and source identity")
    old_analysis = old.dependency_hashes.get("analysis_record")
    if not old_analysis or new.dependency_hashes.get("analysis_record") != old_analysis:
        raise ValueError("typography edit must remain bound to the same AnalysisRecord")
    if new.font != old.font or new.rasterizer != old.rasterizer:
        raise ValueError("typography edit cannot silently change renderer or font identity")

    rendered = rasterize(new)
    if not isinstance(rendered, bytes):
        raise ValueError("renderer replay must return canonical output bytes")
    new.verify_output(rendered)
    persisted = persist_renderer_recipe(persist_root, new) if persist_root is not None else None
    return rendered, persisted

