from __future__ import annotations

from PIL import Image

import main


def test_page_text_coordinate_audit_flags_detects_local_safe_box():
    texts = [
        {
            "id": "ocr_002",
            "band_id": "page_002_band_005",
            "band_y_top": 5420,
            "band_height": 895,
            "bbox": [25, 5436, 667, 5745],
            "text_pixel_bbox": [498, 5655, 656, 5740],
            "balloon_bbox": [466, 5606, 696, 5777],
            "bubble_inner_bbox": [513, 230, 649, 313],
            "safe_text_box": [525, 242, 637, 301],
            "render_bbox": [542, 246, 620, 296],
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
        }
    ]

    flags = main._page_text_coordinate_audit_flags(texts, height=13832, width=800)

    assert "layout_bbox_coordinate_mismatch" in flags
    assert "bubble_inner_bbox_coordinate_mismatch" in flags
    assert "page_space_rerender_mixed_coordinates" in flags


def test_final_page_space_renderer_layers_drop_stale_band_metadata():
    normalized = main._final_page_space_text_layers_for_renderer(
        {
            "texts": [
                {
                    "id": "ocr_001",
                    "band_id": "page_007_band_012",
                    "band_y_top": 112,
                    "_band_y_top": 112,
                    "strip_band_y_top": 112,
                    "_strip_band_y_top": 112,
                    "source_coordinate_space": "band",
                    "coordinate_space": "band",
                    "bbox": [39, 107, 199, 152],
                    "source_bbox": [39, 107, 199, 152],
                    "text_pixel_bbox": [63, 119, 175, 140],
                    "balloon_bbox": [39, 107, 199, 152],
                    "bubble_inner_bbox": [43, 111, 195, 148],
                    "safe_text_box": [53, 111, 185, 148],
                    "render_bbox": [70, 122, 167, 136],
                    "translated": "E-ESPERE!!",
                }
            ]
        },
        page_number=7,
    )

    assert normalized[0]["coordinate_space"] == "page"
    assert normalized[0]["source_coordinate_space"] == "page"
    assert normalized[0].get("band_y_top") is None
    flags = main._page_text_coordinate_audit_flags(normalized, height=1400, width=760)
    assert "page_space_rerender_mixed_coordinates" not in flags


def test_final_page_space_renderer_drops_cross_page_local_source_render_conflict():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "direct_paddle_reocr_001",
                "trace_id": "direct_paddle_reocr_001@page_003_band_035",
                "page_id": "page_002",
                "band_id": "page_003_band_035",
                "route_action": "translate_inpaint_render",
                "translated": "Isso mesmo!",
                "bbox": [573, 389, 704, 427],
                "source_bbox": [573, 389, 704, 427],
                "text_pixel_bbox": [573, 389, 704, 427],
                "target_bbox": [413, 11572, 800, 12183],
                "safe_text_box": [518, 11728, 695, 12027],
                "render_bbox": [523, 11859, 690, 11895],
                "balloon_bbox": [413, 11572, 800, 12183],
                "bubble_mask_bbox": [413, 11572, 800, 12183],
                "qa_flags": ["candidate_crop_direct_paddle_reocr"],
            }
        ],
        page_number=2,
    )

    assert normalized == []


def test_final_page_space_renderer_drops_stale_render_layout_contract():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_005_band_078",
                "band_id": "page_005_band_078",
                "coordinate_space": "band",
                "source_coordinate_space": "band",
                "translated": "SE VOCE ULTRAPASSAR ESSE TEMPO, VOCE RETORNARA AO SEU MUNDO ORIGINAL!",
                "bbox": [393, 64, 714, 110],
                "source_bbox": [393, 64, 714, 110],
                "text_pixel_bbox": [393, 64, 714, 110],
                "target_bbox": [385, 32, 744, 691],
                "safe_text_box": [410, 78, 719, 645],
                "render_bbox": [345, 140, 571, 308],
                "render_layout_contract": {
                    "coordinate_space": "band",
                    "band_y_top": 8378,
                    "block_bbox": [345, 140, 571, 308],
                    "line_bboxes": [[345, 140, 571, 180]],
                    "line_positions": [[345, 170]],
                    "font_size": 34,
                },
                "qa_flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon"],
            }
        ],
        page_number=5,
    )[0]

    assert "render_layout_contract" not in normalized
    assert "_render_layout_contract_hydrated_from_debug" not in normalized
    assert "render_bbox" not in normalized
    assert "stale_final_render_contract_dropped" in normalized["qa_flags"]


def test_final_page_space_renderer_offsets_source_text_mask_bbox():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "direct_paddle_reocr_001",
                "band_id": "page_005_band_078",
                "coordinate_space": "page",
                "source_coordinate_space": "page",
                "translated": "A RETENCAO DO SUBESPACO E DE APENAS CINCO MINUTOS.",
                "bbox": [129, 60633, 312, 60756],
                "source_bbox": [129, 60633, 312, 60756],
                "text_pixel_bbox": [129, 60633, 312, 60756],
                "target_bbox": [83, 60554, 385, 61213],
                "safe_text_box": [104, 60600, 364, 61167],
                "render_bbox": [216, 60668, 364, 60823],
                "source_text_mask_bbox": [122, 8431, 319, 8572],
                "_source_text_mask_bbox": [122, 8431, 319, 8572],
            }
        ],
        page_number=5,
    )[0]

    assert normalized["source_text_mask_bbox"] == [122, 60633, 319, 60774]
    assert normalized["_source_text_mask_bbox"] == [122, 60633, 319, 60774]


def test_final_page_space_renderer_drops_contract_misaligned_from_source_text_mask():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "direct_paddle_reocr_001",
                "band_id": "page_005_band_078",
                "coordinate_space": "page",
                "source_coordinate_space": "page",
                "translated": "A RETENCAO DO SUBESPACO E DE APENAS CINCO MINUTOS.",
                "bbox": [129, 60633, 312, 60756],
                "source_bbox": [129, 60633, 312, 60756],
                "text_pixel_bbox": [129, 60633, 312, 60756],
                "target_bbox": [83, 60554, 385, 61213],
                "safe_text_box": [104, 60600, 364, 61167],
                "render_bbox": [216, 60668, 364, 60823],
                "source_text_mask_bbox": [122, 8431, 319, 8572],
                "render_layout_contract": {
                    "coordinate_space": "page",
                    "band_y_top": 0,
                    "block_bbox": [204, 60648, 375, 60844],
                    "font_size": 33,
                },
            }
        ],
        page_number=5,
    )[0]

    assert "render_layout_contract" not in normalized
    assert "render_bbox" not in normalized
    assert "source_text_mask_render_contract_dropped" in normalized["qa_flags"]


def test_sync_page_legacy_aliases_drops_contract_misaligned_from_source_text_mask():
    page = {
        "numero": 5,
        "image_layers": {"base": {"path": "005.jpg"}, "rendered": {"path": "005.jpg"}},
        "text_layers": [
            {
                "id": "direct_paddle_reocr_001",
                "band_id": "page_005_band_078",
                "coordinate_space": "page",
                "source_coordinate_space": "page",
                "translated": "A RETENCAO DO SUBESPACO E DE APENAS CINCO MINUTOS.",
                "bbox": [129, 60633, 312, 60756],
                "source_bbox": [129, 60633, 312, 60756],
                "text_pixel_bbox": [129, 60633, 312, 60756],
                "target_bbox": [83, 60554, 385, 61213],
                "safe_text_box": [104, 60600, 364, 61167],
                "render_bbox": [216, 60668, 364, 60823],
                "source_text_mask_bbox": [122, 60633, 319, 60774],
                "render_layout_contract": {
                    "coordinate_space": "page",
                    "band_y_top": 0,
                    "block_bbox": [204, 60648, 375, 60844],
                    "font_size": 33,
                },
            }
        ],
    }

    main._sync_page_legacy_aliases(page)
    layer = page["text_layers"][0]

    assert "render_layout_contract" not in layer
    assert "source_text_mask_render_contract_dropped" in layer["qa_flags"]
    assert layer["source_text_mask_bbox"] == [122, 60633, 319, 60774]


def test_final_page_space_preserves_dark_component_partition_contract():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "ocr_001",
                "band_id": "page_002_band_023",
                "coordinate_space": "page",
                "source_coordinate_space": "page",
                "translated": "VOCE CRESCEU EM UM ORFANATO SEM PAIS.",
                "bbox": [68, 16575, 491, 16780],
                "source_bbox": [68, 16575, 491, 16780],
                "text_pixel_bbox": [68, 16575, 491, 16780],
                "source_text_mask_bbox": [63, 16575, 391, 16772],
                "_source_text_mask_bbox": [63, 16575, 391, 16772],
                "target_bbox": [55, 16509, 407, 16856],
                "safe_text_box": [55, 16509, 407, 16856],
                "_debug_safe_text_box": [55, 16509, 407, 16856],
                "render_bbox": [55, 16509, 407, 16856],
                "_debug_render_bbox": [55, 16509, 407, 16856],
                "bubble_mask_source": "image_dark_bubble_mask",
                "_render_bbox_from_repaired_safe_text_box": True,
                "qa_flags": [
                    "source_text_mask_bbox_from_inpaint_component",
                    "dark_connected_component_safe_partition",
                ],
                "render_layout_contract": {
                    "coordinate_space": "page",
                    "block_bbox": [450, 16680, 730, 16780],
                    "font_size": 40,
                },
            }
        ],
        page_number=2,
    )[0]

    assert normalized["safe_text_box"] == [55, 16509, 407, 16856]
    assert normalized["render_bbox"] == [55, 16509, 407, 16856]
    assert normalized["target_bbox"] == [55, 16509, 407, 16856]
    assert "render_layout_contract" not in normalized
    assert "source_text_mask_render_contract_dropped" not in normalized.get("qa_flags", [])
    assert normalized["_final_band_render_contract_preserved"] is True


def test_dark_component_partition_updates_all_render_geometry_fields():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "left",
                        "band_id": "page_002_band_023",
                        "bubble_mask_source": "image_dark_bubble_mask",
                        "source_text_mask_bbox": [63, 16575, 391, 16772],
                        "safe_text_box": [55, 16509, 595, 16856],
                        "target_bbox": [11, 16481, 639, 16884],
                        "render_bbox": [55, 16509, 595, 16856],
                        "render_layout_contract": {"block_bbox": [450, 16680, 730, 16780]},
                        "qa_flags": ["source_text_mask_bbox_from_inpaint_component"],
                    },
                    {
                        "id": "right",
                        "band_id": "page_002_band_023",
                        "bubble_mask_source": "image_dark_bubble_mask",
                        "source_text_mask_bbox": [423, 16672, 751, 16794],
                        "safe_text_box": [393, 16546, 756, 16859],
                        "target_bbox": [364, 16521, 785, 16884],
                        "render_bbox": [447, 16668, 726, 16788],
                        "qa_flags": ["source_text_mask_bbox_from_inpaint_component"],
                    },
                ]
            }
        ]
    }

    updated = main._partition_dark_connected_lobe_safe_boxes_from_components(project)
    left, right = project["paginas"][0]["text_layers"]

    assert updated == 1
    assert left["safe_text_box"][2] == 407
    assert right["safe_text_box"][0] == 407
    for layer in (left, right):
        for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "position_bbox", "capacity_bbox", "layout_safe_bbox", "render_bbox", "_debug_render_bbox"):
            assert layer[key] == layer["safe_text_box"]
        assert "render_layout_contract" not in layer
        assert "dark_connected_component_safe_partition" in layer["qa_flags"]


def test_strip_crop_localization_uses_band_y_top_for_source_geometry():
    layer = {
        "bbox": [60, 16575, 390, 16772],
        "source_bbox": [60, 16575, 390, 16772],
        "text_pixel_bbox": [60, 16575, 390, 16772],
        "source_text_mask_bbox": [63, 16575, 391, 16772],
        "target_bbox": [55, 16509, 407, 16856],
        "safe_text_box": [55, 16509, 407, 16856],
        "render_bbox": [55, 16509, 407, 16856],
        "balloon_subregions": [[11, 3413, 352, 3798], [352, 3413, 492, 3798]],
        "connected_lobe_bboxes": [[11, 3413, 352, 3798], [352, 3413, 492, 3798]],
        "connected_position_bboxes": [[37, 3482, 312, 3717], [440, 3619, 492, 3699]],
    }

    localized = main._localize_layer_to_crop(
        layer,
        [0, 3336, 800, 3887],
        source_y_top=16404,
    )

    assert localized["safe_text_box"] == [55, 105, 407, 452]
    assert localized["source_text_mask_bbox"] == [63, 171, 391, 368]
    assert localized["connected_lobe_bboxes"] == [[11, 77, 352, 462], [352, 77, 492, 462]]
    assert localized["connected_position_bboxes"] == [[37, 146, 312, 381], [440, 283, 492, 363]]
    assert localized["band_y_top"] == 0
    assert localized["coordinate_space"] == "band"


def test_final_render_cleanup_uses_source_text_mask_only(tmp_path):
    base = tmp_path / "base.jpg"
    out = tmp_path / "clean.jpg"
    img = Image.new("RGB", (240, 120), (0, 0, 0))
    for x in range(80, 160):
        for y in range(46, 70):
            img.putpixel((x, y), (245, 245, 245))
    img.save(base)

    cleaned = main._apply_final_text_mask_cleanup_for_render(
        inpainted_path=base,
        texts=[
            {
                "route_action": "translate_inpaint_render",
                "translated": "TEXTO",
                "source_text_mask_bbox": [82, 48, 158, 68],
                "text_pixel_bbox": [82, 48, 158, 68],
                "background_rgb": [0, 0, 0],
            }
        ],
        temp_output_path=out,
        update_inpaint=False,
    )

    assert cleaned == out
    with Image.open(out) as cleaned_img:
        assert cleaned_img.getpixel((120, 58))[0] < 20


def test_final_render_cleanup_skips_sfx(tmp_path):
    base = tmp_path / "base.jpg"
    out = tmp_path / "clean.jpg"
    img = Image.new("RGB", (160, 90), (0, 0, 0))
    for x in range(40, 120):
        for y in range(30, 56):
            img.putpixel((x, y), (245, 245, 245))
    img.save(base)

    cleaned = main._apply_final_text_mask_cleanup_for_render(
        inpainted_path=base,
        texts=[
            {
                "route_action": "translate_sfx_inpaint_render",
                "content_class": "sfx",
                "translated": "SFX",
                "source_text_mask_bbox": [42, 32, 118, 54],
                "background_rgb": [0, 0, 0],
            }
        ],
        temp_output_path=out,
        update_inpaint=False,
    )

    assert cleaned == base
    assert not out.exists()


def test_page_text_coordinate_audit_allows_page_space_band_id_without_band_offset():
    texts = [
        {
            "id": "ocr_002",
            "band_id": "page_002_band_005",
            "coordinate_space": "page",
            "source_coordinate_space": "page",
            "bbox": [499, 4576, 656, 4661],
            "source_bbox": [499, 4576, 656, 4661],
            "text_pixel_bbox": [499, 4576, 656, 4661],
            "layout_bbox": [461, 4549, 675, 4670],
            "target_bbox": [461, 4549, 675, 4670],
            "safe_text_box": [461, 4549, 675, 4670],
            "render_bbox": [488, 4582, 648, 4638],
            "balloon_bbox": [461, 4549, 675, 4670],
            "bubble_mask_bbox": [461, 4549, 675, 4670],
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
        }
    ]

    flags = main._page_text_coordinate_audit_flags(texts, height=16383, width=800)

    assert "layout_bbox_coordinate_mismatch" not in flags
    assert "page_space_rerender_mixed_coordinates" not in flags


def test_page_space_typeset_repair_promotes_real_bubble_mask_to_render_target():
    layers = [
        {
            "id": "ocr_001",
            "band_id": "page_002_band_005",
            "coordinate_space": "page",
            "source_coordinate_space": "page",
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
            "bbox": [499, 4576, 656, 4661],
            "source_bbox": [499, 4576, 656, 4661],
            "text_pixel_bbox": [499, 4576, 656, 4661],
            "layout_bbox": [499, 4576, 656, 4661],
            "target_bbox": [25, 4357, 667, 4629],
            "balloon_bbox": [25, 4473, 667, 4745],
            "bubble_mask_bbox": [435, 4516, 701, 4703],
            "bubble_mask_source": "image_white_bubble_mask",
            "safe_text_box": [102, 4390, 590, 4596],
            "render_bbox": [152, 4482, 540, 4504],
            "qa_flags": [
                "same_balloon_fragment_merged",
                "same_band_dependent_fragment_merged",
                "safe_text_box_recomputed",
            ],
        }
    ]

    repaired, audit = main._repair_page_space_text_layers_for_typeset(layers, page_number=2)

    layer = repaired[0]
    assert audit["safe_area_repaired_count"] == 1
    assert layer["balloon_bbox"] == [435, 4516, 701, 4703]
    assert layer["target_bbox"] == [435, 4516, 701, 4703]
    assert layer["layout_safe_bbox"] == layer["safe_text_box"]
    assert layer["render_bbox"] == layer["safe_text_box"]


def test_final_page_space_clamps_edge_clipped_safe_box_to_visible_bubble():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "ocr_002",
                "text_id": "ocr_002",
                "trace_id": "ocr_002@page_002_band_019",
                "page_id": "page_003",
                "band_id": "page_002_band_019",
                "route_action": "translate_inpaint_render",
                "translated": "A QUANTIA ESTA CERTA. ESSA VADIA E REAL ATRIZ...",
                "bbox": [298, 107, 525, 229],
                "source_bbox": [298, 107, 525, 229],
                "text_pixel_bbox": [298, 107, 525, 229],
                "target_bbox": [298, 107, 525, 241],
                "balloon_bbox": [324, 0, 508, 38],
                "bubble_mask_bbox": [234, 0, 575, 109],
                "safe_text_box": [318, 124, 504, 224],
                "render_bbox": [321, 140, 503, 207],
                "qa_flags": ["same_balloon_fragment_merged", "safe_text_box_recomputed"],
            }
        ],
        page_number=3,
    )[0]

    safe = normalized["safe_text_box"]
    render = normalized["render_bbox"]
    assert safe[1] >= 0
    assert safe[3] <= 109
    assert render[1] >= safe[1]
    assert render[3] <= safe[3]
    assert normalized["target_bbox"] == safe
    assert normalized["_final_edge_clipped_bubble_safe_box"] is True


def test_final_page_space_shifts_local_render_box_to_page_bubble_mask():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "ocr_005",
                "text_id": "ocr_005",
                "trace_id": "ocr_005@page_002_band_014",
                "band_id": "page_002_band_014",
                "route_action": "translate_inpaint_render",
                "translated": "AFINAL, E CANCER, POR QUE SE PREOCUPAR?",
                "bbox": [301, 761, 690, 930],
                "source_bbox": [301, 761, 690, 930],
                "text_pixel_bbox": [301, 761, 690, 930],
                "target_bbox": [209, 690, 792, 996],
                "balloon_bbox": [209, 690, 792, 996],
                "bubble_mask_bbox": [274, 12380, 727, 12620],
                "safe_text_box": [335, 729, 657, 957],
                "render_bbox": [345, 776, 646, 909],
                "qa_flags": [
                    "ocr_joined_repaired",
                    "same_balloon_fragment_merged",
                    "debug_derived_bubble_mask_rejected",
                ],
            }
        ],
        page_number=2,
    )[0]

    assert normalized["render_bbox"][1] >= 12000
    assert normalized["safe_text_box"][1] >= 12000
    assert normalized["target_bbox"][1] >= 12000
    assert normalized["text_pixel_bbox"][1] >= 12000
    assert normalized["bubble_mask_bbox"] == [274, 12380, 727, 12620]


def test_final_page_space_realigns_stale_page_render_box_to_page_bubble_mask():
    normalized = main._final_page_space_text_layers_for_renderer(
        [
            {
                "id": "ocr_005",
                "text_id": "ocr_005",
                "trace_id": "ocr_005@page_002_band_014",
                "band_id": "page_002_band_014",
                "coordinate_space": "page",
                "source_coordinate_space": "page",
                "route_action": "translate_inpaint_render",
                "translated": "AFINAL, E CANCER, POR QUE SE PREOCUPAR?",
                "bbox": [301, 11794, 690, 11963],
                "source_bbox": [301, 11794, 690, 11963],
                "text_pixel_bbox": [301, 11794, 690, 11963],
                "target_bbox": [209, 11794, 792, 12100],
                "balloon_bbox": [209, 11794, 792, 12100],
                "bubble_mask_bbox": [274, 12380, 727, 12620],
                "safe_text_box": [335, 11833, 657, 12061],
                "render_bbox": [345, 11880, 646, 12013],
                "qa_flags": [
                    "ocr_joined_repaired",
                    "same_balloon_fragment_merged",
                    "debug_derived_bubble_mask_rejected",
                ],
            }
        ],
        page_number=2,
    )[0]

    safe = normalized["safe_text_box"]
    render = normalized["render_bbox"]
    bubble = normalized["bubble_mask_bbox"]
    assert safe[1] >= bubble[1] - 8
    assert safe[3] <= bubble[3] + 8
    assert render[1] >= bubble[1]
    assert render[3] <= bubble[3]
    assert normalized.get("_final_page_space_reanchored_to_bubble_mask") is True


def test_append_page_text_flags_marks_all_processable_texts():
    texts = [{"id": "ocr_1", "qa_flags": ["TEXT_OVERFLOW"]}, {"id": "ocr_2", "skip_processing": True}]

    main._append_page_text_flags(texts, ["page_space_rerender_mixed_coordinates"])

    assert texts[0]["qa_flags"] == ["TEXT_OVERFLOW", "page_space_rerender_mixed_coordinates"]
    assert texts[1]["qa_flags"] == ["page_space_rerender_mixed_coordinates"]


def test_strip_crop_rerender_allows_center_anchor_partition_layer_without_qa_metrics():
    layer = {
        "id": "ocr_001",
        "qa_flags": ["dark_connected_component_safe_partition"],
        "_anchor_center_only_layout": True,
        "source_text_anchor_bbox": [68, 169, 383, 354],
        "qa_metrics": {"original_text_scale_min_underflow": {"source_bbox": [11, 16481, 492, 16866]}},
    }

    assert main._layer_requires_strip_crop_rerender(layer) is True
    localized = main._localize_layer_to_crop(layer, [0, 3336, 800, 3887], source_y_top=16404)
    assert "qa_metrics" not in localized
