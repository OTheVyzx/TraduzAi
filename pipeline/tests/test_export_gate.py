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


def test_export_gate_blocks_source_glyph_area_ratio_critical():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_023",
                        "band_id": "page_003_band_023",
                        "translated": "Não aperte minha mãe!",
                        "qa_flags": ["source_glyph_area_ratio_critical"],
                        "source_bbox": [0, 5320, 555, 5979],
                        "text_pixel_bbox": [38, 5829, 445, 5901],
                        "render_bbox": [82, 5433, 474, 5866],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["critical_issue_count"] == 1
    assert "source_glyph_area_ratio_critical" in gate["issues"][0]["flags"]


def test_export_gate_ignores_suppressed_scanlation_credit_layer():
    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_004",
                        "text": "The God of Death SUPPORTUS ON ko-fi.com/Secretscans patreon.com/Secretscans JOIN US AT DISCORD Discordggxzeknv",
                        "translated": "The God of Death SUPPORTUS ON ko-fi.com/Secretscans patreon.com/Secretscans JOIN US AT DISCORD Discordggxzeknv",
                        "route_action": "review_required",
                        "route_reason": "scanlation_credit_suppressed",
                        "skip_reason": "scanlation_credit_suppressed",
                        "skip_processing": True,
                        "qa_flags": [
                            "scanlation_credit_suppressed",
                            "mask_outside_balloon_critical",
                            "text_residual_after_inpaint",
                        ],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["issue_count"] == 0


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


def test_export_gate_keeps_low_confidence_lobe_assignment_review_only_on_renderable_route():
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

    assert gate["status"] == "PASS"
    assert gate["allowed"] is True
    assert gate["needs_review"] is True
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["flags"] == ["lobe_assignment_low_confidence"]
    assert gate["issues"][0]["blocks_export"] is False


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


def test_export_gate_demotes_strong_scanlation_credit_visual_flags_to_review_only():
    project = {
        "paginas": [
            {
                "numero": 15,
                "text_layers": [
                    {
                        "id": "ocr_credit",
                        "text": "RESET SCANSHOMEMANGA CONTACT SPECIAL THANKS TO OUR PATREONS",
                        "raw_ocr": "RESET SCANSHOMEMANGA CONTACT SPECIAL THANKS TO OUR PATREONS",
                        "translated": "REDEFINIR CONTATO SCANSHOMEMANGA AGRADECIMENTO ESPECIAL AOS NOSSOS PATREONS",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": [
                            "render_on_art_suspected",
                            "TEXT_CLIPPED",
                            "TEXT_OVERFLOW",
                            "render_outside_balloon",
                            "text_residual_after_inpaint",
                            "missing_render_bbox",
                            "fit_below_minimum_legible",
                            "layout_bbox_coordinate_mismatch",
                            "mask_outside_balloon_critical",
                            "mask_outside_balloon",
                            "ocr_partial_low_confidence_fragment",
                            "compact_small_text_capacity",
                            "connected_lobe_boxes_missing_source_anchor_fallback",
                            "ocr_art_fragment_suspected",
                            "rotated_text_recovery",
                            "safe_text_box_recomputed",
                            "tiny_bubble_inner_bbox_rejected",
                        ],
                        "bbox": [10, 200, 680, 260],
                        "source_bbox": [10, 200, 680, 260],
                        "text_pixel_bbox": [10, 200, 680, 260],
                        "balloon_bbox": [10, 200, 680, 260],
                        "safe_text_box": [10, 200, 680, 260],
                        "render_bbox": [12, 204, 650, 250],
                        "qa_metrics": {"render_balloon_containment": 1.0},
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["allowed"] is True
    assert gate["needs_review"] is True
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["severity"] == "warning"
    assert gate["issues"][0]["blocks_export"] is False


def test_export_gate_keeps_content_missing_render_blocking():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "dialogue",
                        "text": "Virtual Image Resembles Langit.",
                        "translated": "A imagem virtual se parece com Langit.",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["missing_render_bbox"],
                        "bbox": [100, 100, 320, 180],
                        "source_bbox": [100, 100, 320, 180],
                        "balloon_bbox": [80, 80, 360, 220],
                        "safe_text_box": None,
                        "render_bbox": None,
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["allowed"] is False
    assert gate["critical_issue_count"] == 1


def test_export_gate_demotes_credit_tier_only_when_band_has_strong_credit_context():
    project = {
        "paginas": [
            {
                "numero": 15,
                "text_layers": [
                    {
                        "id": "credit_header",
                        "text": "RESET SCANSHOMEMANGA SPECIAL THANKS TO OUR PATREONS",
                        "translated": "REDEFINIR SCANSHOMEMANGA AGRADECIMENTO AOS PATREONS",
                        "band_id": "page_015_band_165",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["render_on_art_suspected"],
                    },
                    {
                        "id": "tier",
                        "text": "SILVER",
                        "translated": "PRATA",
                        "band_id": "page_015_band_165",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["missing_render_bbox", "mask_outside_balloon"],
                    },
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 2


def test_export_gate_keeps_isolated_tier_missing_render_blocking():
    project = {
        "paginas": [
            {
                "numero": 4,
                "text_layers": [
                    {
                        "id": "dialogue",
                        "text": "SILVER",
                        "translated": "PRATA",
                        "band_id": "page_004_band_010",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["missing_render_bbox"],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["blocking_issue_count"] == 1
    assert gate["critical_issue_count"] == 1


def test_export_gate_demotes_contextual_warning_on_scanlation_credit_page():
    project = {
        "paginas": [
            {
                "numero": 26,
                "text_layers": [
                    {
                        "id": "warning",
                        "text": "WARNING!!",
                        "translated": "AVISO!!",
                        "band_id": "page_026_band_265",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["text_residual_after_inpaint"],
                    },
                    {
                        "id": "site",
                        "text": "READ THIS FROM THE OFFICIAL SITE TO HELP US RELEASE FASTER!",
                        "translated": "LEIA ISTO NO SITE OFICIAL PARA NOS AJUDAR A LANÇAR MAIS RÁPIDO!",
                        "band_id": "page_026_band_266",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": [],
                    },
                    {
                        "id": "hive",
                        "text": "HIVESCANS. COM",
                        "translated": "COLMEIAS. COM",
                        "band_id": "page_026_band_267",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": [],
                    },
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["flags"] == ["text_residual_after_inpaint"]


def test_export_gate_demotes_newtoki_watermark_residuals():
    project = {
        "paginas": [
            {
                "numero": 4,
                "text_layers": [
                    {
                        "id": "watermark",
                        "text": "83469 NEWTORITGO. COM",
                        "translated": "83469 NEWTORITGO. COM",
                        "band_id": "page_003_band_050",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["fit_below_minimum_legible", "text_residual_after_inpaint"],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1


def test_export_gate_demotes_corrupted_tok_watermark_residuals():
    project = {
        "paginas": [
            {
                "numero": 6,
                "text_layers": [
                    {
                        "id": "tok",
                        "text": "3469 SWTOKLEGO. COM",
                        "translated": "3469 SWTOKLEGO. COM",
                        "band_id": "page_005_band_082",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["text_residual_after_inpaint"],
                    },
                    {
                        "id": "digit_com",
                        "text": "BEETAG9 COM",
                        "translated": "BEETAG9 COM",
                        "band_id": "page_005_band_083",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["text_residual_after_inpaint"],
                    },
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 2


def test_export_gate_keeps_isolated_warning_residual_blocking():
    project = {
        "paginas": [
            {
                "numero": 7,
                "text_layers": [
                    {
                        "id": "warning",
                        "text": "WARNING!!",
                        "translated": "AVISO!!",
                        "band_id": "page_007_band_010",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["text_residual_after_inpaint"],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["blocking_issue_count"] == 1
    assert gate["critical_issue_count"] == 1


def test_export_gate_demotes_dark_textured_sfx_residual_with_contained_render():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "sfx",
                        "text": "Kouaak",
                        "translated": "Kouaak",
                        "band_id": "page_001_band_011",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["text_residual_after_inpaint"],
                        "bbox": [48, 405, 571, 539],
                        "balloon_bbox": [47, 393, 622, 546],
                        "safe_text_box": [106, 424, 563, 515],
                        "render_bbox": [259, 454, 410, 484],
                        "qa_metrics": {
                            "render_background_luma": 27.63,
                            "render_background_luma_std": 6.07,
                            "render_balloon_containment": 1.0,
                        },
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["flags"] == ["text_residual_after_inpaint"]


def test_export_gate_demotes_render_on_art_when_render_replaces_source_text_area():
    project = {
        "paginas": [
            {
                "numero": 11,
                "text_layers": [
                    {
                        "id": "caption",
                        "text": "AWARENESS",
                        "translated": "CONHECIMENTO",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["render_on_art_suspected"],
                        "bbox": [913, 3141, 1293, 3196],
                        "source_bbox": [894, 3126, 1305, 3217],
                        "text_pixel_bbox": [913, 3141, 1293, 3196],
                        "safe_text_box": [922, 3138, 1278, 3205],
                        "render_bbox": [938, 3156, 1261, 3186],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["flags"] == ["render_on_art_suspected"]


def test_export_gate_keeps_displaced_render_on_art_blocking():
    project = {
        "paginas": [
            {
                "numero": 11,
                "text_layers": [
                    {
                        "id": "caption",
                        "text": "AWARENESS",
                        "translated": "CONHECIMENTO",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["render_on_art_suspected"],
                        "bbox": [913, 3141, 1293, 3196],
                        "source_bbox": [894, 3126, 1305, 3217],
                        "text_pixel_bbox": [913, 3141, 1293, 3196],
                        "safe_text_box": [100, 100, 500, 180],
                        "render_bbox": [120, 120, 420, 160],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["blocking_issue_count"] == 1
    assert gate["critical_issue_count"] == 1


def test_export_gate_demotes_fast_fill_no_glyph_when_render_replaces_source_text_area():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "label",
                        "text": "STAR PHOTO",
                        "translated": "FOTO ESTRELA",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["fast_fill_no_glyph_evidence"],
                        "bbox": [341, 2247, 621, 2323],
                        "text_pixel_bbox": [348, 2251, 625, 2318],
                        "safe_text_box": [348, 2251, 625, 2318],
                        "render_bbox": [354, 2271, 624, 2298],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1


def test_export_gate_drops_render_fit_overflow_when_final_render_is_contained():
    project = {
        "paginas": [
            {
                "numero": 5,
                "text_layers": [
                    {
                        "id": "dialogue",
                        "translated": "Eles são chineses?",
                        "qa_flags": ["TEXT_OVERFLOW", "safe_text_box_recomputed"],
                        "balloon_bbox": [907, 6369, 1248, 6532],
                        "safe_text_box": [965, 6408, 1192, 6498],
                        "render_bbox": [978, 6418, 1185, 6488],
                        "qa_metrics": {
                            "render_fit": {
                                "flags": ["TEXT_OVERFLOW"],
                                "target_bbox": [841, 6445, 1309, 6645],
                                "balloon_bbox": [841, 6445, 1309, 6645],
                            }
                        },
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["flags"] == ["safe_text_box_recomputed"]


def test_export_gate_drops_overflow_when_final_fit_is_ok_inside_safe_box():
    project = {
        "paginas": [
            {
                "numero": 17,
                "text_layers": [
                    {
                        "id": "dialogue",
                        "translated": "Ano... mas...",
                        "qa_flags": ["TEXT_OVERFLOW"],
                        "fit_status": "ok",
                        "fit_attempts": [{"font_px": 36, "lines": 2, "status": "ok"}],
                        "balloon_bbox": [922, 3588, 1294, 3688],
                        "safe_text_box": [368, 3539, 1294, 3737],
                        "render_bbox": [806, 3602, 1293, 3674],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0


def test_export_gate_demotes_dark_bubble_render_on_art_when_render_is_contained():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "dark_dialogue",
                        "translated": "Você pensou amizade.",
                        "route_action": "translate_inpaint_render",
                        "layout_profile": "dark_bubble",
                        "block_profile": "dark_bubble",
                        "bubble_mask_source": "image_dark_bubble_mask",
                        "qa_flags": ["render_on_art_suspected"],
                        "balloon_bbox": [48, 3400, 800, 3823],
                        "safe_text_box": [236, 3505, 611, 3718],
                        "render_bbox": [237, 3543, 611, 3718],
                        "qa_metrics": {
                            "image_dark_bubble_mask": {"mask_bbox": [48, 3400, 800, 3823]},
                            "render_balloon_containment": 1.0,
                            "render_background_luma": 0.0,
                        },
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1


def test_export_gate_drops_dark_bubble_overflow_when_render_fit_safe_box_contains_render():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "dark_lobe",
                        "translated": "No entanto, nenhum deles visitou você uma única vez.",
                        "route_action": "translate_inpaint_render",
                        "layout_profile": "dark_bubble",
                        "block_profile": "dark_bubble",
                        "bubble_mask_source": "image_dark_bubble_mask",
                        "qa_flags": ["TEXT_OVERFLOW", "fast_fill_no_glyph_evidence", "render_on_art_suspected"],
                        "balloon_bbox": [384, 4312, 761, 4606],
                        "bubble_mask_bbox": [474, 4488, 552, 4519],
                        "qa_metrics": {
                            "render_fit": {
                                "flags": ["TEXT_OVERFLOW"],
                                "render_bbox": [474, 4458, 618, 4508],
                                "safe_text_box": [474, 4451, 671, 4508],
                                "target_bbox": [474, 4488, 552, 4519],
                                "balloon_bbox": [474, 4488, 552, 4519],
                            },
                            "image_dark_bubble_mask": {"mask_bbox": [348, 4225, 800, 4606]},
                            "render_balloon_containment": 1.0,
                        },
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0
    assert all("TEXT_OVERFLOW" not in issue["flags"] for issue in gate["issues"])


def test_export_gate_demotes_compact_small_text_fit_when_render_is_contained():
    project = {
        "paginas": [
            {
                "numero": 4,
                "text_layers": [
                    {
                        "id": "uied_label",
                        "translated": "MAS EU VIM ATÉ AQUI! EU NÃO POSSO PARECER UM AMADOR!!",
                        "qa_flags": ["uied_form_label_split", "compact_small_text_capacity", "fit_below_minimum_legible"],
                        "fit_status": "below_minimum_legible",
                        "safe_text_box": [822, 5961, 1125, 5996],
                        "render_bbox": [831, 5974, 1115, 5983],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0


def test_export_gate_demotes_fast_fill_for_group_sibling_render_geometry():
    project = {
        "paginas": [
            {
                "numero": 20,
                "text_layers": [
                    {
                        "id": "group_child",
                        "translated": "POSSO CHEGAR ATÉ AQUI COM BASTANTE FACILIDADE.",
                        "qa_flags": ["fast_fill_no_glyph_evidence"],
                        "_render_metadata_group_sibling_geometry": True,
                        "safe_text_box": [286, 1749, 1403, 2349],
                        "render_bbox": [410, 2008, 1280, 2089],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["blocking_issue_count"] == 0
    assert gate["critical_issue_count"] == 0


def test_export_gate_keeps_displaced_fast_fill_no_glyph_blocking():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "label",
                        "text": "STAR PHOTO",
                        "translated": "FOTO ESTRELA",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["fast_fill_no_glyph_evidence"],
                        "bbox": [341, 2247, 621, 2323],
                        "text_pixel_bbox": [348, 2251, 625, 2318],
                        "safe_text_box": [452, 1776, 1162, 2353],
                        "render_bbox": [599, 2049, 1015, 2079],
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["blocking_issue_count"] == 1
    assert gate["critical_issue_count"] == 1


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


def test_export_gate_does_not_block_white_balloon_fit_below_minimum_legible():
    project = {
        "idioma_origem": "en",
        "paginas": [
            {
                "numero": 19,
                "text_layers": [
                    {
                        "id": "ocr_003",
                        "translated": "Sim, eles ainda não sabem!",
                        "qa_flags": ["fit_below_minimum_legible"],
                        "bbox": [482, 822, 610, 894],
                        "source_bbox": [482, 817, 613, 890],
                        "balloon_bbox": [454, 796, 641, 911],
                        "safe_text_box": [502, 827, 593, 880],
                        "render_bbox": [506, 840, 591, 867],
                        "qa_metrics": {
                            "render_balloon_containment": 1.0,
                            "render_background_luma": 255.0,
                            "render_balloon_background_luma": 255.0,
                            "render_balloon_background_luma_std": 0.0,
                            "render_flat_balloon_background": True,
                        },
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    issue = gate["issues"][0]
    assert issue["type"] == "needs_review"
    assert issue["severity"] == "warning"
    assert issue["blocks_export"] is False
    assert issue["flags"] == ["fit_below_minimum_legible"]


def test_export_gate_demotes_translator_note_fit_below_minimum_on_flat_white_background():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "translated": "T/N: HYUNGNIM É UM TERMO USADO PARA CHAMAR O CHEFE DA MÁFIA.",
                        "qa_flags": ["fit_below_minimum_legible", "safe_text_box_recomputed"],
                        "bbox": [595, 14319, 643, 14365],
                        "source_bbox": [595, 14319, 643, 14365],
                        "balloon_bbox": [535, 14278, 797, 14413],
                        "safe_text_box": None,
                        "render_bbox": None,
                        "qa_metrics": {
                            "render_balloon_containment": 1.0,
                            "render_background_luma": 255.0,
                            "render_background_luma_std": 2.53,
                            "render_balloon_background_luma": 255.0,
                            "render_balloon_background_luma_std": 1.72,
                            "render_flat_balloon_background": True,
                        },
                    }
                ],
            }
        ],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["critical_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    issue = gate["issues"][0]
    assert issue["type"] == "needs_review"
    assert issue["blocks_export"] is False
    assert "fit_below_minimum_legible" in issue["flags"]


def test_export_gate_does_not_block_unpropagated_fast_fill_no_glyph_evidence():
    project = {
        "qa": {
            "flag_propagation_audit": {
                "summary": {"qa_flag_not_propagated_count": 1},
                "missing_in_project": [
                    {
                        "identity": "ocr_003@page_051_band_127",
                        "text_id": "ocr_003@page_051_band_127",
                        "flag": "fast_fill_no_glyph_evidence",
                        "source": "render_plan",
                        "is_review_only": True,
                    }
                ],
            }
        },
        "paginas": [{"numero": 3, "text_layers": []}],
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "PASS"
    assert gate["needs_review"] is True
    assert gate["critical_issue_count"] == 0
    issue = gate["issues"][0]
    assert issue["type"] == "needs_review"
    assert issue["severity"] == "warning"
    assert issue["flags"] == ["qa_flag_not_propagated"]
    assert issue["missing_flag"] == "fast_fill_no_glyph_evidence"
    assert issue["blocks_export"] is False


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


def test_export_gate_marks_blocked_sfx_inpaint_as_review_status():
    project = {
        "paginas": [
            {
                "numero": 2,
                "page_id": "page_002",
                "text_layers": [
                    {
                        "id": "sfx_001",
                        "text": "\ucff5",
                        "translated": "TUM",
                        "route_action": "translate_sfx_inpaint_render",
                        "content_class": "sfx",
                        "bbox": [40, 60, 140, 120],
                        "sfx": {
                            "source_text": "\ucff5",
                            "adapted_text": "TUM",
                            "inpaint_allowed": False,
                            "qa_flags": ["complex_background"],
                        },
                    }
                ],
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "REVIEW"
    assert gate["allowed"] is True
    assert gate["needs_review"] is True
    assert gate["blocking_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    issue = gate["issues"][0]
    assert issue["type"] == "sfx_inpaint_review"
    assert issue["severity"] == "warning"
    assert issue["blocks_export"] is False
    assert issue["flags"] == ["sfx_inpaint_review_required", "complex_background"]


def test_export_gate_reviews_only_unsafe_sfx_in_mixed_page():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "dialogue",
                        "translated": "Ola!",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": [],
                    },
                    {
                        "id": "safe_sfx",
                        "text": "\ucff5",
                        "translated": "TUM",
                        "route_action": "translate_sfx_inpaint_render",
                        "content_class": "sfx",
                        "bbox": [10, 10, 80, 80],
                        "sfx": {
                            "source_text": "\ucff5",
                            "adapted_text": "TUM",
                            "inpaint_allowed": True,
                            "qa_flags": [],
                        },
                    },
                    {
                        "id": "unsafe_sfx",
                        "text": "\ucff5",
                        "translated": "TUM",
                        "route_action": "translate_sfx_inpaint_render",
                        "content_class": "sfx",
                        "bbox": [110, 10, 180, 80],
                        "sfx": {
                            "source_text": "\ucff5",
                            "adapted_text": "TUM",
                            "inpaint_allowed": False,
                            "qa_flags": ["complex_background"],
                        },
                    },
                ],
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "REVIEW"
    assert gate["allowed"] is True
    assert gate["needs_review"] is True
    assert gate["blocking_issue_count"] == 0
    assert gate["review_issue_count"] == 1
    assert gate["issues"][0]["layer"] == "unsafe_sfx"
    assert gate["issues"][0]["type"] == "sfx_inpaint_review"
