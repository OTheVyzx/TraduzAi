"""Regression gates for append-only operational recovery publication."""
from __future__ import annotations

import json

import pytest

from consumer_operational_publish import _next_ordinal, publish
from consumer_operational_recovery import _reviewed_translation, prepare_seam_member


def test_ordinal_skips_inactive_historical_recipe(tmp_path):
    path = tmp_path / "text_layers" / "page_any"
    path.mkdir(parents=True)
    (path / "0007.recipe.pickle").write_bytes(b"historical")
    page = dict(page_id="page_any")
    meta = dict(raster_cache=[dict(path="overlay/page_any/0006.png")])
    refs = dict(recipes=[dict(path="text_layers/page_any/0006.recipe.pickle")])
    assert _next_ordinal(tmp_path, page, meta, refs) == 8


def test_editorial_review_requires_exact_source_binding(tmp_path):
    path = tmp_path / "review.json"
    binding = dict(target="OLD", provenance="accepted_existing_translation_reuse")
    review = dict(schema="consumer_model_editorial_translation_v1",
                  reviewer_type="model", human_review=False,
                  units=[dict(source_member="member.webp", source_sha256="hash",
                              source="VISIBLE SOURCE", previous_target="OLD",
                              target="NEW", reason="model review")])
    path.write_text(json.dumps(review), encoding="utf-8")
    assert _reviewed_translation(binding, path, "member.webp", "hash",
                                 "VISIBLE SOURCE")["target"] == "NEW"
    with pytest.raises(ValueError, match="lineage"):
        _reviewed_translation(binding, path, "member.webp", "changed", "VISIBLE SOURCE")
    assert _reviewed_translation(binding, path, "other.webp", "hash",
                                 "VISIBLE SOURCE") == binding


def test_publish_failure_keeps_prior_and_does_not_create_destination(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    original = b'{"pages": []}'
    (control / "project.json").write_bytes(original)
    destination = tmp_path / "new-revision"
    with pytest.raises((ValueError, KeyError)):
        publish(control_root=control, destination=destination,
                outcomes=[dict(status="rendered_candidate", proposal=dict(
                    kind="unsupported"), page=dict(source_member="anything"))])
    assert not destination.exists()
    assert (control / "project.json").read_bytes() == original
    stages = list(tmp_path.glob("new-revision.building-*"))
    assert len(stages) == 1 and (stages[0] / "failure.txt").is_file()


def test_seam_rejects_nonadjacent_member_before_layout(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "project.json").write_text(json.dumps(dict(pages=[
        dict(source_member="a.webp", source_index=1),
        dict(source_member="b.webp", source_index=3)])), encoding="utf-8")
    with pytest.raises(ValueError, match="not adjacent"):
        prepare_seam_member(control_root=root, member="b.webp",
            seam_evidence=tmp_path / "missing.json",
            translation_binding=tmp_path / "missing-binding.json",
            editorial_candidate=tmp_path / "missing-candidate.json",
            font_path=tmp_path / "missing.ttf")


def test_seam_rejects_changed_source_member_before_layout(tmp_path):
    root = tmp_path / "project"
    (root / "source_members").mkdir(parents=True)
    (root / "source_members" / "a.webp").write_bytes(b"changed")
    (root / "source_members" / "b.webp").write_bytes(b"original")
    (root / "project.json").write_text(json.dumps(dict(pages=[
        dict(source_member="a.webp", source_index=1, source_sha256="expected",
             original="original/a.png"),
        dict(source_member="b.webp", source_index=2, source_sha256="other",
             original="original/b.png")])), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(dict(source_members=[
        dict(source_member="a.webp", source_sha256="expected"),
        dict(source_member="b.webp", source_sha256="other")])), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes changed"):
        prepare_seam_member(control_root=root, member="b.webp",
            seam_evidence=evidence, translation_binding=tmp_path / "unused-binding.json",
            editorial_candidate=tmp_path / "unused-candidate.json",
            font_path=tmp_path / "unused.ttf")
