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
