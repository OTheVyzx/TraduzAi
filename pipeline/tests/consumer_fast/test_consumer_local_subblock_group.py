"""A logical utterance cannot publish only one of its physical text bodies."""

import hashlib
import json

import pytest

from consumer_fast_project import validate_local_subblock_groups


def _fixture(tmp_path):
    (tmp_path / "source_members").mkdir()
    source = b"source member"
    (tmp_path / "source_members" / "014.webp").write_bytes(source)
    source_sha = hashlib.sha256(source).hexdigest()
    (tmp_path / "group").mkdir()
    (tmp_path / "meta").mkdir()
    members = [dict(order=i, owner_id=f"part:{i}",
                    recipe_path=f"recipe/{i}.pickle", recipe_sha256=f"hash:{i}")
               for i in range(2)]
    group = dict(schema="consumer_local_subblock_group_v1", group_id="logical:014",
                 page_id="page_014", source_member="014.webp", source_sha256=source_sha,
                 members=members)
    (tmp_path / "group" / "014.json").write_text(json.dumps(group))
    refs = dict(recipes=[dict(owner_id=x["owner_id"], path=x["recipe_path"],
                              sha256=x["recipe_sha256"]) for x in members])
    cache = dict(raster_cache=[dict(owner_id=x["owner_id"]) for x in members])
    (tmp_path / "meta" / "refs.json").write_text(json.dumps(refs))
    (tmp_path / "meta" / "texts.json").write_text(json.dumps(cache))
    project = dict(local_subblock_groups=["group/014.json"],
                   pages=[dict(page_id="page_014", source_member="014.webp",
                               source_sha256=source_sha,
                               rerender_metadata="meta/refs.json", text_layers="meta/texts.json")])
    return project, group, refs


def test_both_physical_members_are_required(tmp_path):
    project, _group, refs = _fixture(tmp_path)
    assert validate_local_subblock_groups(tmp_path, project) == ["logical:014"]
    refs["recipes"].pop()
    (tmp_path / "meta" / "refs.json").write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="partially published"):
        validate_local_subblock_groups(tmp_path, project)


def test_wrong_source_or_member_order_is_rejected(tmp_path):
    project, group, _refs = _fixture(tmp_path)
    group["members"][0]["order"] = 1
    (tmp_path / "group" / "014.json").write_text(json.dumps(group))
    with pytest.raises(ValueError, match="ordered physical members"):
        validate_local_subblock_groups(tmp_path, project)
    group["members"][0]["order"] = 0
    group["source_sha256"] = "0" * 64
    (tmp_path / "group" / "014.json").write_text(json.dumps(group))
    with pytest.raises(ValueError, match="source/page binding"):
        validate_local_subblock_groups(tmp_path, project)
