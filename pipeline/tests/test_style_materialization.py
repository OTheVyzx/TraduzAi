from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_contract import STYLE_V2_ATTRIBUTE_NAME_SET


def _materialization_module():
    spec = importlib.util.find_spec("typesetter.style_materialization")
    assert spec is not None, "typesetter.style_materialization must exist"
    return importlib.import_module("typesetter.style_materialization")


def test_materialization_module_and_domain_registry_are_complete():
    module = _materialization_module()

    assert set(module.ATTRIBUTE_DOMAIN) == STYLE_V2_ATTRIBUTE_NAME_SET
    assert set(module.attributes_for_domain("layout")) == {
        "font_size_px",
        "alignment",
        "container",
        "tracking_xh",
        "curve",
    }
    assert set(module.attributes_for_domain("font")) == {
        "font_name",
        "font_weight",
        "font_width",
    }
    assert set(module.attributes_for_domain("raster")) == (
        STYLE_V2_ATTRIBUTE_NAME_SET
        - set(module.attributes_for_domain("layout"))
        - set(module.attributes_for_domain("font"))
    )


def test_canonicalization_normalizes_equivalent_values():
    module = _materialization_module()

    assert module.canonicalize_style_attribute("fill", "fff") == "#FFFFFF"
    assert module.canonicalize_style_attribute("rotation_deg", -0.0) == 0.0
    assert module.canonicalize_style_attribute(
        "stroke", {"width_xh": 0.1, "color": "#fff"}
    ) == {"color": "#FFFFFF", "width_xh": 0.1}


def test_material_difference_is_not_normalized_away():
    module = _materialization_module()

    result = module.compare_style_attribute("fill", "#FFFFFF", "#F0A000")

    assert result.matches is False
    assert result.delta_e_2000 is not None and result.delta_e_2000 > 12


def test_unknown_or_incomplete_domain_registry_is_rejected():
    module = _materialization_module()

    with pytest.raises(ValueError, match="domain registry"):
        module.validate_attribute_domain_registry({"fill": "raster"})
    with pytest.raises(ValueError, match="domain registry"):
        module.validate_attribute_domain_registry(
            dict(module.ATTRIBUTE_DOMAIN) | {"unknown": "raster"}
        )
