from tools.build_style_owner_target_manifest import build_effective_style_config


def test_functional_matrix_builder_preserves_explicit_style_copy_off():
    effective = build_effective_style_config(
        {"owner_graph_mode": "enforce", "style_copy_mode": "off"},
        required_categories=[],
    )
    assert effective["owner_graph_mode"] == "enforce"
    assert effective["style_copy_mode"] == "off"
