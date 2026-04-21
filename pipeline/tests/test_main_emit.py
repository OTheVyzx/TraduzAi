import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from typesetter.renderer import build_render_blocks  # noqa: E402


class MainEmitTests(unittest.TestCase):
    def setUp(self) -> None:
        main._EMIT_STDOUT_FAILED = False

    def test_emit_swallow_oserror_from_stdout_once(self) -> None:
        stderr = io.StringIO()

        with patch("builtins.print", side_effect=OSError(22, "Invalid argument")):
            with patch.object(main.sys, "stderr", stderr):
                main.emit("progress", message="primeira")
                main.emit("progress", message="segunda")

        log_output = stderr.getvalue()
        self.assertIn("Falha ao emitir evento JSON no stdout", log_output)
        self.assertEqual(log_output.count("Falha ao emitir evento JSON no stdout"), 1)

    def test_main_lists_supported_languages_in_cli_mode(self) -> None:
        stdout = io.StringIO()

        with patch.object(main.sys, "argv", ["main.py", "--list-supported-languages"]):
            with patch("main.list_supported_google_languages", return_value=[{"code": "en", "label": "English"}]):
                with patch.object(main.sys, "stdout", stdout):
                    main.main()

        self.assertEqual(stdout.getvalue().strip(), '[{"code": "en", "label": "English"}]')

    def test_build_text_layer_preserves_connected_balloon_metadata_for_renderer(self) -> None:
        ocr_text = {
            "id": "tl_001_003",
            "text": "IT MAY BE NOTHING MORE THAN A HALF-FINISHED CULTIVATION METHOD...",
            "bbox": [113, 1513, 705, 1767],
            "balloon_bbox": [113, 1513, 705, 1767],
            "balloon_subregions": [[113, 1513, 395, 1767], [409, 1513, 705, 1767]],
            "tipo": "fala",
            "confidence": 0.97,
            "layout_group_size": 1,
            "connected_balloon_orientation": "left-right",
            "connected_detection_confidence": 1.0,
            "connected_group_confidence": 0.898,
            "connected_position_confidence": 0.952,
            "subregion_confidence": 1.0,
            "connected_text_groups": [[113, 1513, 373, 1722], [432, 1558, 705, 1767]],
            "connected_lobe_bboxes": [[113, 1513, 395, 1767], [409, 1513, 705, 1767]],
            "connected_position_bboxes": [[113, 1513, 345, 1712], [462, 1568, 705, 1767]],
            "connected_focus_bboxes": [[113, 1513, 345, 1712], [462, 1568, 705, 1767]],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
            },
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=2,
            ocr_text=ocr_text,
            translated=(
                "PODE SER NADA MAIS DO QUE UM METODO DE CULTIVO INACABADO, "
                "MAS SEUS EFEITOS SAO MAIS DO QUE SUFICIENTES. UM PODER QUE "
                "PERMITE SUPERAR SEUS PROPRIOS LIMITES EM UM INSTANTE."
            ),
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        blocks = build_render_blocks([layer])

        self.assertEqual(layer.get("connected_position_bboxes"), ocr_text["connected_position_bboxes"])
        self.assertEqual(layer.get("connected_text_groups"), ocr_text["connected_text_groups"])
        self.assertEqual(layer.get("connected_detection_confidence"), 1.0)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].get("balloon_subregions"), ocr_text["balloon_subregions"])
        self.assertEqual(blocks[0].get("connected_position_bboxes"), ocr_text["connected_position_bboxes"])


if __name__ == "__main__":
    unittest.main()
