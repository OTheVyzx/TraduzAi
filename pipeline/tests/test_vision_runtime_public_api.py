from __future__ import annotations


def test_public_api_declares_versioned_capabilities_and_all_mask_channels() -> None:
    import vision_runtime

    manifest = vision_runtime.capability_manifest()

    assert manifest["schema"] == "traduzai.vision-runtime.v1"
    assert manifest["version"] == vision_runtime.VISION_RUNTIME_VERSION
    assert set(manifest["capabilities"]) == {
        "analysis_payload",
        "dependency_cache",
        "ocr_consensus",
        "ocr_refinement",
        "observation_evidence",
        "structure",
        "authenticated_context",
        "raster_authority",
    }
    assert manifest["mask_channels"] == [
        "glyph", "outline", "shadow", "glow", "ignore_or_uncertain"
    ]


def test_public_api_exports_runtime_entry_points() -> None:
    import vision_runtime

    assert callable(vision_runtime.build_analysis_payload)
    assert callable(vision_runtime.select_ocr_invocation)
    assert callable(vision_runtime.plan_vertical_refinement)
    assert callable(vision_runtime.reconcile_observations)
    assert callable(vision_runtime.assess_observation_evidence)
    assert callable(vision_runtime.build_structural_analysis)
    assert callable(vision_runtime.discover_white_containers)
    assert vision_runtime.DeterministicFailureCache is not None
