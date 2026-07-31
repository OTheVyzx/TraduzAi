import json
from pathlib import Path

from qa.style_fidelity import audit_style_fidelity, delta_e_2000, merge_style_and_functional_gates, resolve_original_path


def _owner(owner_id="owner_a", *, status="applied", expected="#FFFFFF", observed="#FFFFFF", confidence=0.95):
    return {
        "owner_id": owner_id,
        "visual_profile_v2": {
            "owner_id": owner_id,
            "source_sha256": "a" * 64,
            "glyph_mask_sha256": "b" * 64,
            "status": status,
            "applied_style": {"fill": expected},
            "style_evidence_v2": {"attributes": {"fill": {"confidence": confidence}}},
        },
        "style_v2_raster_contract": {"applied_attributes": {"fill": observed}},
    }


def test_png_original_is_resolved_from_project_record(tmp_path):
    image = tmp_path / "source" / "page-one.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    page = {"original_path": "source/page-one.png"}

    assert resolve_original_path(tmp_path, page, 1) == image.resolve()


def test_audit_pairs_source_and_final_by_owner_id(tmp_path):
    report = audit_style_fidelity({"paginas": [{"text_layers": [_owner()]}]}, tmp_path, mode="render")

    assert report["owners"][0]["owner_id"] == "owner_a"
    assert report["owners"][0]["fields"]["fill"]["status"] == "applied"


def test_unrelated_record_on_same_page_cannot_satisfy_expectation(tmp_path):
    expected = _owner("owner_a", observed="")
    unrelated = _owner("owner_b", expected="#000000", observed="#FFFFFF")
    report = audit_style_fidelity({"paginas": [{"text_layers": [expected, unrelated]}]}, tmp_path, mode="render")

    owner_a = next(item for item in report["owners"] if item["owner_id"] == "owner_a")
    assert owner_a["fields"]["fill"]["status"] == "mismatch"


def test_fallback_owner_is_not_high_confidence_failure(tmp_path):
    report = audit_style_fidelity({"paginas": [{"text_layers": [_owner(status="fallback", observed="")]}]}, tmp_path, mode="enforce")

    assert report["gate"]["status"] == "PASS"
    assert report["owners"][0]["fields"]["fill"]["status"] == "fallback"


def test_high_confidence_style_mismatch_blocks_enforce_mode(tmp_path):
    report = audit_style_fidelity({"paginas": [{"text_layers": [_owner(observed="#000000")]}]}, tmp_path, mode="enforce")

    assert report["gate"]["status"] == "BLOCK"
    assert report["gate"]["blocking_owner_ids"] == ["owner_a"]


def test_style_gate_cannot_override_blocked_functional_gate():
    merged = merge_style_and_functional_gates({"status": "BLOCK", "issues": [{"code": "english"}]}, {"status": "PASS"})

    assert merged["status"] == "BLOCK"
    assert merged["functional_gate"]["issues"][0]["code"] == "english"


def test_color_fidelity_reports_delta_e_2000():
    assert delta_e_2000("#FFFFFF", "#FFFFFF") == 0.0
    assert delta_e_2000("#FFFFFF", "#000000") > 90.0
