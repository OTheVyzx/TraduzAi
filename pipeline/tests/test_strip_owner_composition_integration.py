from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _action_ref(owner_id: str) -> str:
    safe = owner_id.replace("/", "_")
    identity = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    return f"owner_masks/{safe}--{identity}/mask/action_mask.png"


def _mutation(
    original: np.ndarray,
    *,
    owner_id: str,
    page_id: str = "page_001",
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
):
    from ownership.model import OwnerMutation

    x1, y1, x2, y2 = bbox
    action = np.zeros(original.shape[:2], dtype=np.uint8)
    action[y1 + 2 : y2 - 2 : 3, x1 + 2 : x2 - 2] = 255
    result = original.copy()
    result[action > 0] = color
    changed = np.where(np.any(result != original, axis=2), 255, 0).astype(np.uint8)
    protected = np.zeros(original.shape[:2], dtype=np.uint8)
    return OwnerMutation(
        owner_id=owner_id,
        page_id=page_id,
        coordinate_space="page",
        action_mask_ref=_action_ref(owner_id),
        result_rgb=result,
        action_mask=action,
        protected_art_mask=protected,
        changed_mask=changed,
        engine="test-engine",
        mask_pixels=int(np.count_nonzero(action)),
        changed_pixels=int(np.count_nonzero(changed)),
        changed_outside_owner_pixels=0,
        protected_art_changed_pixels=0,
        before_sha256=_array_sha256(original),
        after_sha256=_array_sha256(result),
        action_mask_sha256=_array_sha256(action),
        changed_mask_sha256=_array_sha256(changed),
        engine_crop_bbox_page=bbox,
        owner_bbox_page=bbox,
        component_geometry_sha256=sha256(owner_id.encode("utf-8")).hexdigest(),
        execution_tile_id=f"tile-{owner_id}",
    )


def _chapter_fixture():
    from strip.process_bands import apply_atomic_owner_execution
    from strip.types import Band, VerticalStrip

    original = np.full((80, 100, 3), 240, dtype=np.uint8)
    strip = VerticalStrip(
        image=original.copy(),
        width=100,
        height=80,
        source_page_breaks=[0, 80],
        page_x_offsets=[0],
        source_page_widths=[100],
    )
    left = Band(y_top=0, y_bottom=55, original_slice=original[:55].copy())
    right = Band(y_top=25, y_bottom=80, original_slice=original[25:].copy())
    left_mutation = _mutation(
        original,
        owner_id="owner-left",
        bbox=(12, 18, 34, 38),
        color=(20, 80, 160),
    )
    right_mutation = _mutation(
        original,
        owner_id="owner-right",
        bbox=(62, 44, 88, 68),
        color=(180, 40, 30),
    )
    left.owner_execution_commits = [
        apply_atomic_owner_execution(
            original,
            left_mutation,
            _glyph_patch(left_mutation, bbox=(16, 21, 28, 27)),
        )
    ]
    right.owner_execution_commits = [
        apply_atomic_owner_execution(
            original,
            right_mutation,
            _glyph_patch(right_mutation, bbox=(68, 50, 80, 56)),
        )
    ]
    return original, strip, [left, right]


def _glyph_patch(mutation, *, bbox=(16, 21, 28, 27)):
    import json
    from ownership.model import OwnerGlyphPatch

    before = np.asarray(mutation.result_rgb)
    x1, y1, x2, y2 = bbox
    result = before.copy()
    result[y1:y2, x1:x2] = (3, 3, 3)
    mask = np.zeros(before.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    polygon = (
        (max(0, x1 - 4), max(0, y1 - 4)),
        (min(before.shape[1] - 1, x2 + 4), max(0, y1 - 4)),
        (min(before.shape[1] - 1, x2 + 4), min(before.shape[0] - 1, y2 + 4)),
        (max(0, x1 - 4), min(before.shape[0] - 1, y2 + 4)),
    )
    return OwnerGlyphPatch(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        coordinate_space="page",
        result_rgb=result,
        glyph_mask=mask,
        glyph_bbox_page=bbox,
        render_completed=True,
        fit_status="ok",
        before_sha256=mutation.after_sha256,
        after_sha256=_array_sha256(result),
        glyph_mask_sha256=_array_sha256(mask),
        changed_outside_glyph_mask_pixels=0,
        render_safe_polygon_page=polygon,
        render_safe_polygon_sha256=sha256(
            json.dumps(polygon, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        component_geometry_sha256=mutation.component_geometry_sha256,
        execution_tile_id=mutation.execution_tile_id,
    )


def test_owner_stage_outputs_preserve_executor_mutation_and_glyph_patch():
    from strip.process_bands import _run_inpaint_stage, _run_typeset_stage
    from strip.types import Band

    original = np.full((80, 100, 3), 240, dtype=np.uint8)
    mutation = _mutation(
        original,
        owner_id="owner-left",
        bbox=(12, 18, 34, 38),
        color=(20, 80, 160),
    )
    glyph = _glyph_patch(mutation)
    band = Band(y_top=0, y_bottom=80, strip_slice=original, original_slice=original)

    class Inpainter:
        @staticmethod
        def inpaint_band_image(_image, _page):
            return mutation

    class Typesetter:
        @staticmethod
        def render_band_image(_image, _page):
            return glyph

    translated_page = {"texts": []}
    inpaint_stage = _run_inpaint_stage(
        band,
        inpainter=Inpainter(),
        translated_page=translated_page,
    )
    typeset_stage = _run_typeset_stage(
        inpaint_stage.to_image(),
        typesetter=Typesetter(),
        translated_page=translated_page,
        owner_mutation=mutation,
    )

    assert inpaint_stage.owner_mutation is mutation
    assert typeset_stage.owner_glyph_patch is glyph
    np.testing.assert_array_equal(inpaint_stage.to_image(), mutation.result_rgb)
    np.testing.assert_array_equal(typeset_stage.to_image(), glyph.result_rgb)


def test_owner_composition_rejects_raw_artifacts_that_bypass_atomic_commit():
    from compositor.owner_compositor import OwnerCompositionError
    from strip.run import _compose_owner_output_pages
    from strip.types import Band, VerticalStrip

    original = np.full((40, 50, 3), 240, dtype=np.uint8)
    strip = VerticalStrip(
        image=original.copy(),
        width=50,
        height=40,
        source_page_breaks=[0, 40],
        page_x_offsets=[0],
        source_page_widths=[50],
    )
    band = Band(y_top=0, y_bottom=40, original_slice=original.copy())
    band.owner_mutations = [
        _mutation(
            original,
            owner_id="owner-raw",
            bbox=(8, 8, 20, 20),
            color=(10, 20, 30),
        )
    ]

    try:
        _compose_owner_output_pages(
            original_strip_image=original,
            strip=strip,
            bands=[band],
            balloons=[],
            target_count=1,
        )
    except OwnerCompositionError as exc:
        assert "atomic" in str(exc)
    else:
        raise AssertionError("raw owner mutation bypassed atomic execution")


def test_owner_typeset_patch_without_cleanup_mutation_fails_closed():
    from strip.process_bands import _run_typeset_stage

    original = np.full((80, 100, 3), 240, dtype=np.uint8)
    mutation = _mutation(
        original,
        owner_id="owner-left",
        bbox=(12, 18, 34, 38),
        color=(20, 80, 160),
    )
    glyph = _glyph_patch(mutation)

    class Typesetter:
        @staticmethod
        def render_band_image(_image, _page):
            return glyph

    try:
        _run_typeset_stage(
            original,
            typesetter=Typesetter(),
            translated_page={"texts": []},
        )
    except ValueError as exc:
        assert "cleanup mutation" in str(exc)
    else:
        raise AssertionError("owner glyph patch was accepted without cleanup mutation")


def test_owner_final_page_binding_preserves_composed_bytes_without_late_clamp():
    from strip.run import _bind_owner_final_page_images
    from strip.types import OutputPage

    final = np.full((30, 40, 3), [17, 91, 203], dtype=np.uint8)
    clean = np.full((30, 40, 3), [210, 220, 230], dtype=np.uint8)
    original = np.full((30, 40, 3), 240, dtype=np.uint8)
    page = OutputPage(y_top=0, y_bottom=30, image=final.copy())
    clean_page = OutputPage(y_top=0, y_bottom=30, image=clean.copy())
    original_page = OutputPage(y_top=0, y_bottom=30, image=original.copy())

    _bind_owner_final_page_images(page, original_page, clean_page)

    np.testing.assert_array_equal(page.image, final)
    np.testing.assert_array_equal(page.inpainted_image, clean)
    np.testing.assert_array_equal(page.original_image, original)


def test_run_chapter_uses_owner_compositor_as_only_final_pixel_authority(monkeypatch):
    import strip.run as run

    original, strip, bands = _chapter_fixture()
    monkeypatch.setattr(
        run,
        "_paste_band_attr_into_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy paste called")),
    )

    result = run._compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    assert result.compositions["page_001"].committed is True
    assert np.all(result.output_pages[0].image[20, 20] == (20, 80, 160))
    assert np.all(result.output_pages[0].image[61, 70] == (180, 40, 30))


def test_run_chapter_output_is_identical_after_band_permutation():
    from strip.run import _compose_owner_output_pages

    original, strip, bands = _chapter_fixture()
    forward = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )
    reverse = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=list(reversed(bands)),
        balloons=[],
        target_count=1,
    )

    np.testing.assert_array_equal(forward.output_pages[0].image, reverse.output_pages[0].image)
    assert forward.compositions["page_001"].sha256 == reverse.compositions["page_001"].sha256


def test_changed_context_from_overlapping_tile_never_restores_source():
    from strip.run import _compose_owner_output_pages

    original, strip, bands = _chapter_fixture()
    bands[1].rendered_slice = original[25:].copy()
    bands[1].rendered_slice[0:20, 12:34] = original[25:45, 12:34]

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    assert np.all(result.output_pages[0].image[29, 20] == (20, 80, 160))


def test_production_composition_succeeds_without_debug_directory(tmp_path, monkeypatch):
    from strip.run import _compose_owner_output_pages

    original, strip, bands = _chapter_fixture()
    monkeypatch.setenv("TRADUZAI_DEBUG_DIR", str(tmp_path / "does-not-exist"))

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    assert result.output_pages[0].image.shape == original.shape
    assert not (tmp_path / "does-not-exist").exists()
