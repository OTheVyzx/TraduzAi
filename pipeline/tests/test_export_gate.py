from qa.export_gate import evaluate_export_gate


def test_export_gate_blocks_renderable_p0_flags():
    project = {
        "idioma_origem": "ko",
        "paginas": [
            {
                "numero": 5,
                "text_layers": [
                    {
                        "id": "t1",
                        "translated": "Nao consigo encontrar o texto original.",
                        "qa_flags": ["translation_fallback_phrase"],
                        "bbox": [10, 20, 80, 50],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["issues"][0]["page"] == 5
    assert "translation_fallback_phrase" in gate["issues"][0]["flags"]


def test_export_gate_marks_cjk_script_left_inside_translated_balloon():
    project = {
        "idioma_origem": "ko",
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {"id": "t1", "translated": "\ud558\ud558\ud558.", "qa_flags": []}
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert "speech_cjk_preserved_inside_balloon" in gate["issues"][0]["flags"]


def test_export_gate_can_record_override():
    project = {
        "paginas": [
            {"text_layers": [{"translated": "x", "qa_flags": ["placeholder_lost"]}]}
        ]
    }

    gate = evaluate_export_gate(project, override=True)

    assert gate["status"] == "OVERRIDDEN"
    assert gate["allowed"] is True
    assert gate["override"] is True


def test_export_gate_blocks_unrestored_placeholder():
    project = {
        "paginas": [
            {"text_layers": [{"translated": "", "qa_flags": ["unrestored_placeholder"]}]}
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert "unrestored_placeholder" in gate["issues"][0]["flags"]


def test_export_gate_blocks_text_clipped_review_flag():
    project = {
        "paginas": [
            {"text_layers": [{"id": "t1", "translated": "texto", "qa_flags": ["TEXT_CLIPPED"]}]}
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["needs_review"] is True
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["type"] == "needs_review"
    assert gate["issues"][0]["blocks_export"] is True


def test_export_gate_keeps_rotated_text_policy_unmet_as_review_flag():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "t1",
                        "translated": "texto",
                        "qa_flags": ["rotated_text_policy_unmet"],
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["allowed"] is True
    assert gate["needs_review"] is True
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["blocks_export"] is False


def test_export_gate_blocks_low_confidence_lobe_assignment_on_renderable_route():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "t1",
                        "translated": "texto traduzido",
                        "route_action": "translate_inpaint_render",
                        "skip_processing": True,
                        "lobe_assignment_confidence": 0.42,
                        "qa_flags": ["lobe_assignment_low_confidence"],
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["needs_review"] is True
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["flags"] == ["lobe_assignment_low_confidence"]
    assert gate["issues"][0]["blocks_export"] is True


def test_export_gate_does_not_block_synthetic_lobe_parent_without_lobe_evidence():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_parent",
                        "translated": "O que e...? por que ele esta ficando",
                        "route_action": "review_required",
                        "qa_flags": [
                            "lobe_assignment_low_confidence",
                            "weak_text_residual_after_inpaint",
                        ],
                        "bbox": [148, 655, 667, 998],
                        "balloon_bbox": [148, 655, 667, 998],
                        "safe_text_box": None,
                        "render_bbox": None,
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["blocks_export"] is False


def test_export_gate_ignores_ocr_truncated_or_joined_before_repair_fails():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "What!Then,why",
                        "route_action": "review_required",
                        "qa_flags": ["ocr_truncated_or_joined"],
                        "bbox": [10, 10, 120, 40],
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is False
    assert gate["issues"] == []


def test_export_gate_keeps_ocr_truncated_or_joined_when_repair_fails():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "What!Then,why",
                        "route_action": "review_required",
                        "qa_flags": ["ocr_truncated_or_joined"],
                        "ocr_repair_status": "repair_failed",
                        "bbox": [10, 10, 120, 40],
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["blocks_export"] is False
    assert gate["issues"][0]["flags"] == ["ocr_truncated_or_joined"]


def test_export_gate_ignores_removed_legacy_filter_flags():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_legacy",
                        "translated": "Texto",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": [
                            "low_confidence_visual_noise",
                            "cover_title_logo",
                            "mask_density_high",
                        ],
                        "bbox": [10, 10, 120, 40],
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is False
    assert gate["issues"] == []


def test_export_gate_keeps_partial_low_confidence_fragment_as_review_signal():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "fragmento",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["ocr_partial_low_confidence_fragment"],
                        "bbox": [10, 10, 120, 40],
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["blocks_export"] is False


def test_export_gate_blocks_critical_visual_flags():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "t1",
                        "translated": "texto",
                        "qa_flags": ["bbox_overreach_critical", "TEXT_OVERFLOW"],
                    }
                ]
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["needs_review"] is True
    assert gate["critical_issue_count"] == 1
    assert gate["review_issue_count"] == 1


def test_export_gate_render_outside_balloon_issue_carries_debug_bboxes():
    project = {
        "paginas": [
            {
                "numero": 7,
                "page_id": "page_007",
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_007_band_003",
                        "trace_id": "ocr_001@page_007_band_003",
                        "translated": "texto",
                        "qa_flags": ["render_outside_balloon"],
                        "bbox": [40, 40, 120, 90],
                        "balloon_bbox": [30, 30, 130, 100],
                        "safe_text_box": [42, 42, 118, 88],
                        "render_bbox": [20, 20, 150, 120],
                        "qa_metrics": {"render_balloon_containment": 0.42},
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    issue = gate["issues"][0]
    assert gate["status"] == "BLOCK"
    assert issue["severity"] == "critical"
    assert issue["flags"] == ["render_outside_balloon"]
    assert issue["page_id"] == "page_007"
    assert issue["band_id"] == "page_007_band_003"
    assert issue["trace_id"] == "ocr_001@page_007_band_003"
    assert issue["balloon_bbox"] == [30, 30, 130, 100]
    assert issue["safe_text_box"] == [42, 42, 118, 88]
    assert issue["render_bbox"] == [20, 20, 150, 120]
    assert issue["qa_metrics"]["render_balloon_containment"] == 0.42


def test_export_gate_text_overflow_review_issue_carries_debug_bboxes():
    project = {
        "paginas": [
            {
                "numero": 8,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "translated": "texto",
                        "qa_flags": ["TEXT_OVERFLOW"],
                        "bbox": [10, 10, 80, 40],
                        "balloon_bbox": [8, 8, 82, 42],
                        "safe_text_box": [12, 12, 78, 38],
                        "render_bbox": [4, 4, 90, 48],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    issue = gate["issues"][0]
    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["needs_review"] is True
    assert issue["severity"] == "warning"
    assert issue["flags"] == ["TEXT_OVERFLOW"]
    assert issue["blocks_export"] is True
    assert issue["balloon_bbox"] == [8, 8, 82, 42]
    assert issue["safe_text_box"] == [12, 12, 78, 38]
    assert issue["render_bbox"] == [4, 4, 90, 48]


def test_export_gate_blocks_top_level_text_flags_even_when_render_fit_metric_is_stale():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "Obrigado pela ajuda! a partir de agora, deixe com a gente!",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW", "weak_text_residual_after_inpaint"],
                        "balloon_bbox": [0, 3402, 800, 4039],
                        "safe_text_box": [200, 3455, 661, 3907],
                        "render_bbox": [200, 3420, 661, 3902],
                        "qa_metrics": {
                            "render_fit": {
                                "flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW"],
                                "render_bbox": [200, 3420, 318, 3458],
                                "safe_text_box": [200, 3455, 370, 3464],
                                "target_bbox": [134, 3415, 304, 3464],
                                "balloon_bbox": [134, 3415, 304, 3464],
                            }
                        },
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    issue_flags = [flag for issue in gate["issues"] for flag in issue["flags"]]
    assert "TEXT_CLIPPED" in issue_flags
    assert "TEXT_OVERFLOW" in issue_flags


def test_export_gate_ignores_stale_top_level_text_clipped_when_final_overhang_is_small():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "Voce disse que era maravilhoso, certo? me ajude",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["TEXT_CLIPPED"],
                        "balloon_bbox": [0, 12404, 800, 13281],
                        "safe_text_box": [156, 12425, 639, 13064],
                        "render_bbox": [149, 12425, 627, 13059],
                        "qa_metrics": {
                            "render_fit": {
                                "flags": ["TEXT_CLIPPED"],
                                "render_bbox": [149, 12425, 314, 12483],
                                "safe_text_box": [156, 12425, 307, 12483],
                                "target_bbox": [146, 12418, 317, 12490],
                                "balloon_bbox": [146, 12418, 317, 12490],
                            }
                        },
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["allowed"] is True
    assert gate["issue_count"] == 0


def test_export_gate_ignores_top_level_text_clipped_when_final_geometry_is_contained():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "Voce disse que era maravilhoso, certo? me ajude",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["TEXT_CLIPPED"],
                        "balloon_bbox": [0, 12404, 800, 13281],
                        "safe_text_box": [156, 12425, 639, 13064],
                        "render_bbox": [149, 12425, 627, 13059],
                        "qa_metrics": {"render_balloon_containment": 1.0},
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["allowed"] is True
    assert gate["issue_count"] == 0


def test_export_gate_ignores_metric_only_stale_tiny_render_fit_against_broad_final_balloon():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "Obrigado pela ajuda! a partir de agora, deixe com a gente!",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["weak_text_residual_after_inpaint"],
                        "balloon_bbox": [0, 3402, 800, 4039],
                        "safe_text_box": [200, 3455, 661, 3907],
                        "render_bbox": [200, 3420, 661, 3902],
                        "qa_metrics": {
                            "render_fit": {
                                "flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW"],
                                "render_bbox": [200, 3420, 318, 3458],
                                "safe_text_box": [200, 3455, 370, 3464],
                                "target_bbox": [134, 3415, 304, 3464],
                                "balloon_bbox": [134, 3415, 304, 3464],
                            }
                        },
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["flags"] == ["weak_text_residual_after_inpaint"]


def test_export_gate_blocks_mask_outside_critical_when_source_and_render_are_displaced():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_credits",
                        "translated": "BRONZE",
                        "qa_flags": ["mask_density_high", "mask_outside_balloon_critical"],
                        "bbox": [527, 13666, 653, 13691],
                        "source_bbox": [524, 13662, 653, 13692],
                        "balloon_bbox": [427, 13719, 651, 13862],
                        "safe_text_box": [444, 13736, 634, 13845],
                        "render_bbox": [488, 13781, 590, 13800],
                        "qa_metrics": {"render_balloon_containment": 1.0},
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["critical_issue_count"] == 1
    assert "mask_outside_balloon_critical" in gate["issues"][0]["flags"]


def test_export_gate_demotes_aligned_contained_mask_outside_balloon_critical_without_visual_damage():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_credits",
                        "translated": "BRONZE",
                        "qa_flags": ["mask_outside_balloon_critical"],
                        "bbox": [527, 13666, 653, 13691],
                        "source_bbox": [524, 13662, 653, 13692],
                        "balloon_bbox": [510, 13650, 670, 13705],
                        "safe_text_box": [524, 13658, 656, 13698],
                        "render_bbox": [530, 13666, 650, 13690],
                        "qa_metrics": {"render_balloon_containment": 1.0},
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["blocks_export"] is False


def test_export_gate_demotes_mask_outside_critical_when_render_overlaps_text_bbox_despite_tiny_source():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_remember",
                        "translated": "Voce se lembra",
                        "qa_flags": ["mask_outside_balloon_critical"],
                        "bbox": [132, 6996, 326, 7014],
                        "source_bbox": [125, 6992, 152, 7021],
                        "text_pixel_bbox": [132, 6996, 326, 7014],
                        "balloon_bbox": [113, 6980, 326, 7033],
                        "safe_text_box": [136, 6985, 302, 7028],
                        "render_bbox": [144, 6998, 294, 7015],
                        "qa_metrics": {"render_balloon_containment": 0.1333},
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["blocks_export"] is False


def test_export_gate_demotes_mask_outside_when_render_is_inside_real_bubble_mask():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "Ajussi quanto tempo levara para chegar ao hospital mais proximo?",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": [
                            "safe_text_box_recomputed",
                            "mask_outside_balloon",
                            "mask_outside_balloon_critical",
                        ],
                        "bbox": [214, 4325, 269, 4378],
                        "source_bbox": [214, 4325, 269, 4378],
                        "balloon_bbox": [214, 4325, 269, 4378],
                        "bubble_mask_bbox": [125, 4294, 656, 4853],
                        "bubble_inner_bbox": [162, 4331, 619, 4816],
                        "safe_text_box": [162, 4331, 619, 4816],
                        "render_bbox": [162, 4413, 547, 4456],
                        "qa_metrics": {"render_balloon_containment": 0.0},
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["blocks_export"] is False
    assert "mask_outside_balloon_critical" in gate["issues"][0]["flags"]


def test_export_gate_keeps_aligned_credit_mask_density_review_only():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_credit",
                        "translated": "PRATA",
                        "qa_flags": [
                            "mask_density_high",
                            "mask_outside_balloon",
                            "mask_outside_balloon_critical",
                            "safe_text_box_recomputed",
                        ],
                        "bbox": [161, 13454, 253, 13470],
                        "source_bbox": [151, 13450, 263, 13474],
                        "text_pixel_bbox": [151, 13450, 263, 13474],
                        "balloon_bbox": [140, 13442, 278, 13482],
                        "safe_text_box": [150, 13448, 268, 13478],
                        "render_bbox": [161, 13454, 253, 13470],
                        "qa_metrics": {
                            "render_balloon_containment": 0.0,
                            "render_validated_containment": 1.0,
                            "mask_density_in_band": 0.194,
                            "outside_balloon_ratio": 0.217,
                        },
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["blocks_export"] is False
    assert "mask_outside_balloon_critical" in gate["issues"][0]["flags"]


def test_export_gate_keeps_stale_containment_mask_warning_review_only_when_geometry_is_contained():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_platina",
                        "translated": "PLATINA",
                        "qa_flags": ["mask_outside_balloon", "safe_text_box_recomputed"],
                        "bbox": [161, 13454, 253, 13470],
                        "source_bbox": [151, 13450, 263, 13474],
                        "balloon_bbox": [140, 13442, 278, 13482],
                        "safe_text_box": [150, 13448, 268, 13478],
                        "render_bbox": [161, 13454, 253, 13470],
                        "qa_metrics": {
                            "render_balloon_containment": 0.0,
                            "render_validated_containment": 1.0,
                        },
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["blocks_export"] is False


def test_export_gate_keeps_low_containment_mask_warning_review_only_when_bbox_is_contained():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_regional",
                        "translated": "2o ** teste regional de recrutamento de bombeiros",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["mask_outside_balloon", "safe_text_box_recomputed"],
                        "bbox": [73, 304, 118, 314],
                        "source_bbox": [48, 296, 356, 324],
                        "text_pixel_bbox": [73, 304, 118, 314],
                        "balloon_bbox": [45, 286, 356, 332],
                        "safe_text_box": [68, 291, 332, 327],
                        "render_bbox": [112, 296, 288, 322],
                        "qa_metrics": {"render_balloon_containment": 0.1932},
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    issue = gate["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["blocks_export"] is False
    assert "mask_outside_balloon" in issue["flags"]


def test_export_gate_blocks_low_containment_mask_warning_on_renderable_route():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_regional",
                        "translated": "2o ** teste regional de recrutamento de bombeiros",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["mask_outside_balloon", "safe_text_box_recomputed"],
                        "bbox": [112, 296, 288, 322],
                        "source_bbox": [48, 296, 356, 324],
                        "text_pixel_bbox": [73, 304, 118, 314],
                        "balloon_bbox": [45, 286, 146, 332],
                        "safe_text_box": [68, 291, 332, 327],
                        "render_bbox": [112, 296, 288, 322],
                        "qa_metrics": {"render_balloon_containment": 0.1932},
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    issue = gate["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["blocks_export"] is True
    assert "mask_outside_balloon" in issue["flags"]


def test_export_gate_keeps_mask_outside_balloon_critical_blocking_with_confirmed_residual():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_scene",
                        "translated": "APENAS VIRAR A DIREITA",
                        "qa_flags": ["mask_outside_balloon_critical", "text_residual_after_inpaint"],
                        "balloon_bbox": [370, 7567, 435, 7640],
                        "safe_text_box": [378, 7575, 428, 7632],
                        "render_bbox": [379, 7579, 426, 7627],
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    issue_flags = [flag for issue in gate["issues"] for flag in issue["flags"]]
    assert "text_residual_after_inpaint" in issue_flags


def test_export_gate_blocks_microtext_bbox_overreach_when_render_is_upscaled():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_micro",
                        "translated": "Isso",
                        "qa_flags": ["bbox_overreach_critical", "low_ocr_confidence"],
                        "bbox": [123, 7873, 179, 7877],
                        "source_bbox": [118, 7863, 198, 7892],
                        "balloon_bbox": [101, 7851, 215, 7904],
                        "safe_text_box": [107, 7857, 209, 7898],
                        "render_bbox": [110, 7873, 177, 7895],
                        "qa_metrics": {"render_balloon_containment": 1.0},
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["critical_issue_count"] == 1
    assert "bbox_overreach_critical" in gate["issues"][0]["flags"]


def test_export_gate_demotes_microtext_bbox_overreach_when_render_stays_micro():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_micro",
                        "translated": "Isso",
                        "qa_flags": ["bbox_overreach_critical", "low_ocr_confidence"],
                        "bbox": [123, 7873, 179, 7877],
                        "source_bbox": [118, 7863, 198, 7892],
                        "balloon_bbox": [101, 7851, 215, 7904],
                        "safe_text_box": [107, 7857, 209, 7898],
                        "render_bbox": [130, 7872, 170, 7884],
                        "qa_metrics": {"render_balloon_containment": 1.0},
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["issues"][0]["severity"] == "warning"
    assert "bbox_overreach_critical" in gate["issues"][0]["flags"]


def test_export_gate_demotes_nonrendered_microtext_fit_below_minimum():
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_ad_title",
                        "translated": "ARTES MARCIAIS GLOBAIS",
                        "route_action": "preserve_original",
                        "qa_flags": ["fit_below_minimum_legible"],
                        "bbox": [449, 15238, 525, 15261],
                        "source_bbox": [447, 15233, 572, 15263],
                        "balloon_bbox": [447, 15233, 572, 15263],
                        "safe_text_box": [456, 15236, 562, 15260],
                        "render_bbox": [461, 15245, 557, 15250],
                        "qa_metrics": {"render_balloon_containment": 1.0},
                    }
                ]
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is False
    assert gate["issues"] == []


def test_export_gate_blocks_render_fit_text_overflow_without_top_level_flag():
    project = {
        "paginas": [
            {
                "numero": 5,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "texto",
                        "qa_flags": ["safe_text_box_recomputed"],
                        "balloon_bbox": [3, 10860, 651, 10986],
                        "safe_text_box": [144, 10888, 535, 10919],
                        "render_bbox": [173, 10900, 506, 10919],
                        "qa_metrics": {
                            "render_fit": {
                                "flags": ["TEXT_OVERFLOW"],
                                "target_bbox": [3, 10860, 285, 10986],
                                "render_bbox": [173, 10900, 506, 10919],
                            }
                        },
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    issue_flags = [flag for issue in gate["issues"] for flag in issue["flags"]]
    assert "TEXT_OVERFLOW" in issue_flags


def test_export_gate_blocks_unpropagated_debug_qa_flags():
    project = {
        "qa": {
            "flag_propagation_audit": {
                "summary": {"qa_flag_not_propagated_count": 1},
                "missing_in_project": [
                    {
                        "identity": "ocr_999@page_003_band_042",
                        "text_id": "ocr_999@page_003_band_042",
                        "flag": "mask_outside_balloon_critical",
                        "source": "mask_decision",
                    }
                ],
            }
        },
        "paginas": [{"numero": 3, "text_layers": []}],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["critical_issue_count"] == 1
    issue = gate["issues"][0]
    assert issue["type"] == "p0_traceability_blocker"
    assert issue["issue_scope"] == "run"
    assert issue["page_id"] == "page_003"
    assert issue["band_id"] == "page_003_band_042"
    assert issue["missing_flag"] == "mask_outside_balloon_critical"
    assert issue["flags"] == ["qa_flag_not_propagated"]


def test_export_gate_blocks_page_space_rerender_mixed_coordinates():
    project = {
        "paginas": [
            {
                "numero": 1,
                "page_id": "page_001",
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "band_id": "page_002_band_005",
                        "trace_id": "ocr_002@page_002_band_005",
                        "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
                        "qa_flags": ["page_space_rerender_mixed_coordinates"],
                        "bbox": [25, 5436, 667, 5745],
                        "balloon_bbox": [466, 5606, 696, 5777],
                        "safe_text_box": [525, 242, 637, 301],
                        "render_bbox": [542, 246, 620, 296],
                    }
                ],
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["critical_issue_count"] == 1
    issue = gate["issues"][0]
    assert issue["severity"] == "critical"
    assert issue["flags"] == ["page_space_rerender_mixed_coordinates"]
    assert issue["band_id"] == "page_002_band_005"
    assert issue["trace_id"] == "ocr_002@page_002_band_005"
