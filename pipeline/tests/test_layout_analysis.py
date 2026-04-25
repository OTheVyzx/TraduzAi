import unittest
import tempfile
import json
import importlib
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from layout.balloon_layout import (
    _analyze_connected_subregions,
    _apply_geometric_fallback_subregions,
    _build_balloon_subregions_from_groups,
    _detect_connected_balloon_subregions_from_fill,
    _detect_connected_lobes_from_outline,
    _detect_lobes_via_distance_transform,
    _enforce_min_lobe_size,
    _extract_balloon_outline_polygon,
    _geometric_fallback_subregions,
    _refine_connected_position_bboxes_with_ollama,
    _score_subregion_quality,
    enrich_page_layout,
)


def _fixture_image_path(name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "testes" / name,
        root / "testes" / "debug_pipeline" / "originals" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


class LayoutAnalysisTests(unittest.TestCase):
    def test_connected_reasoner_accepts_valid_ollama_position_boxes(self):
        image = np.full((240, 420, 3), 245, dtype=np.uint8)
        settings = {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "gemma4",
            "enabled": True,
        }
        text_groups = [[52, 44, 182, 136], [222, 118, 360, 214]]
        lobes = [[30, 24, 210, 182], [198, 86, 392, 232]]
        heuristic = [[40, 34, 178, 146], [230, 102, 372, 222]]

        with patch(
            "layout.balloon_layout._check_ollama",
            return_value={"running": True, "models": ["gemma4:e4b"], "has_translator": False},
        ):
            with patch(
                "layout.balloon_layout._call_ollama_json",
                return_value={
                    "position_bboxes": [[46, 36, 184, 144], [236, 112, 378, 228]],
                    "confidence": 0.91,
                    "notes": "left lobe top-left, right lobe lower-right",
                },
            ):
                refined = _refine_connected_position_bboxes_with_ollama(
                    image,
                    [30, 24, 392, 232],
                    [30, 24, 392, 232],
                    text_groups,
                    lobes,
                    heuristic,
                    "left-right",
                    settings,
                )

        self.assertIsNotNone(refined)
        self.assertEqual(refined["position_bboxes"], [[46, 36, 184, 144], [236, 112, 378, 228]])
        self.assertEqual(refined["source"], "ollama")
        self.assertEqual(refined["model"], "gemma4:e4b")
        self.assertGreaterEqual(float(refined["confidence"]), 0.9)

    def test_connected_reasoner_rejects_invalid_ollama_boxes_and_falls_back(self):
        image = np.full((240, 420, 3), 245, dtype=np.uint8)
        settings = {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "gemma4",
            "enabled": True,
        }
        text_groups = [[52, 44, 182, 136], [222, 118, 360, 214]]
        lobes = [[30, 24, 210, 182], [198, 86, 392, 232]]
        heuristic = [[40, 34, 178, 146], [230, 102, 372, 222]]

        with patch(
            "layout.balloon_layout._check_ollama",
            return_value={"running": True, "models": ["gemma4:e4b"], "has_translator": False},
        ):
            with patch(
                "layout.balloon_layout._call_ollama_json",
                return_value={
                    "position_bboxes": [[8, 8, 404, 210], [16, 16, 408, 220]],
                    "confidence": 0.97,
                    "notes": "invalid because boxes escape the lobes",
                },
            ):
                refined = _refine_connected_position_bboxes_with_ollama(
                    image,
                    [30, 24, 392, 232],
                    [30, 24, 392, 232],
                    text_groups,
                    lobes,
                    heuristic,
                    "left-right",
                    settings,
                )

        self.assertIsNone(refined)

    def test_connected_reasoner_accepts_anchor_label_selection(self):
        image = np.full((240, 420, 3), 245, dtype=np.uint8)
        settings = {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "qwen2.5",
            "enabled": True,
        }
        text_groups = [[52, 44, 182, 136], [222, 118, 360, 214]]
        lobes = [[30, 24, 210, 182], [198, 86, 392, 232]]
        heuristic = [[40, 34, 178, 146], [230, 102, 372, 222]]

        with patch(
            "layout.balloon_layout._check_ollama",
            return_value={"running": True, "models": ["qwen2.5:3b"], "has_translator": False},
        ):
            with patch(
                "layout.balloon_layout._call_ollama_json",
                return_value={
                    "selected_anchor_labels": ["outer-upper", "outer-lower"],
                    "confidence": 0.87,
                    "notes": "picked diagonal-biased anchors",
                },
            ):
                refined = _refine_connected_position_bboxes_with_ollama(
                    image,
                    [30, 24, 392, 232],
                    [30, 24, 392, 232],
                    text_groups,
                    lobes,
                    heuristic,
                    "left-right",
                    settings,
                )

        self.assertIsNotNone(refined)
        left_box, right_box = refined["position_bboxes"]
        left_heuristic, right_heuristic = heuristic
        self.assertLess((left_box[0] + left_box[2]) / 2.0, (left_heuristic[0] + left_heuristic[2]) / 2.0)
        self.assertGreater((right_box[1] + right_box[3]) / 2.0, (right_heuristic[1] + right_heuristic[3]) / 2.0)
        self.assertEqual(refined["source"], "ollama")

    def test_assigns_shared_balloon_bbox_for_clustered_dialogue(self):
        page = {
            "width": 800,
            "height": 1200,
            "texts": [
                {"text": "Who are you?", "bbox": [120, 200, 260, 250], "tipo": "fala", "confidence": 0.9},
                {"text": "Answer me!", "bbox": [130, 270, 250, 320], "tipo": "fala", "confidence": 0.88},
            ],
        }

        enriched = enrich_page_layout(page)
        texts = enriched["texts"]

        self.assertEqual(texts[0]["balloon_bbox"], texts[1]["balloon_bbox"])
        self.assertEqual(texts[0]["layout_shape"], "tall")
        self.assertEqual(texts[0]["layout_group_size"], 2)

    def test_prefers_detected_bubble_region_for_single_dialogue(self):
        page = {
            "width": 800,
            "height": 1200,
            "_bubble_regions": [
                {"bbox": [150, 150, 450, 390], "confidence": 0.97},
                {"bbox": [520, 180, 720, 340], "confidence": 0.91},
            ],
            "texts": [
                {"text": "Who are you?", "bbox": [220, 218, 382, 286], "tipo": "fala", "confidence": 0.93},
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(text["balloon_bbox"], [150, 150, 450, 390])
        self.assertEqual(text["ocr_text_bbox"], [220, 218, 382, 286])

    def test_geometric_fallback_uses_numbered_mask_layer_path_from_project(self):
        texts = [
            {
                "id": "txt-1",
                "tipo": "fala",
                "bbox": [60, 70, 240, 170],
                "balloon_bbox": [20, 20, 340, 240],
                "layout_group_size": 1,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            mask_dir = work_dir / "layers" / "mask"
            mask_dir.mkdir(parents=True)
            mask = np.zeros((260, 360), dtype=np.uint8)
            mask[70:140, 80:170] = 255
            cv2.imwrite(str(mask_dir / "001.png"), mask)

            page_result = {
                "_work_dir": str(work_dir),
                "numero": 1,
                "arquivo_original": "originals/page-1.jpg",
                "image_layers": {
                    "mask": {"path": "layers/mask/001.png"},
                },
            }

            with patch(
                "layout.balloon_layout._geometric_fallback_subregions",
                return_value=[[20, 20, 180, 240], [180, 20, 340, 240]],
            ) as geometric_fallback:
                _apply_geometric_fallback_subregions(
                    texts,
                    page_result,
                    np.zeros((260, 360, 3), dtype=np.uint8),
                )

        geometric_fallback.assert_called_once()

    def test_narration_prefers_wide_layout(self):
        page = {
            "width": 1200,
            "height": 1800,
            "texts": [
                {"text": "Three days later...", "bbox": [220, 80, 860, 180], "tipo": "narracao", "confidence": 0.95},
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(text["layout_shape"], "wide")
        self.assertEqual(text["layout_align"], "top")

    def test_enrich_page_layout_clamps_top_narration_bbox_and_records_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            page = {
                "image": "page-041.jpg",
                "width": 800,
                "height": 2600,
                "texts": [
                    {
                        "text": "MY BIGGEST FEAR IS DESMOND UNLEASHING THAT POWER AT THE VERY END.",
                        "bbox": [300, 80, 500, 130],
                        "tipo": "narracao",
                        "confidence": 0.95,
                    }
                ],
            }

            with patch(
                "layout.balloon_layout._load_page_image",
                return_value=np.zeros((2600, 800, 3), dtype=np.uint8),
            ), patch(
                "layout.balloon_layout.refine_balloon_bbox_from_image",
                return_value=[20, 30, 780, 220],
            ):
                enriched = enrich_page_layout(page)

            decision_log.finalize_decision_trace()

            self.assertEqual(enriched["texts"][0]["balloon_bbox"], [190, 30, 610, 220])
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertEqual(payloads[0]["reason"], "top_caption_overexpand")

    def test_enrich_page_layout_adaptive_top_narration_clamps_earlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            page = {
                "image": "page-007.jpg",
                "width": 800,
                "height": 2600,
                "texts": [
                    {
                        "text": "Three days later, the northern wall had already fallen.",
                        "bbox": [300, 80, 500, 130],
                        "tipo": "narracao",
                        "confidence": 0.95,
                        "block_profile": "top_narration",
                    }
                ],
            }

            with patch(
                "layout.balloon_layout._load_page_image",
                return_value=np.zeros((2600, 800, 3), dtype=np.uint8),
            ), patch(
                "layout.balloon_layout.refine_balloon_bbox_from_image",
                return_value=[185, 30, 615, 220],
            ):
                enriched = enrich_page_layout(page)

            decision_log.finalize_decision_trace()

            self.assertEqual(enriched["texts"][0]["balloon_bbox"], [216, 30, 584, 220])
            self.assertEqual(enriched["texts"][0]["layout_profile"], "top_narration")
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertEqual(payloads[0]["reason"], "top_caption_overexpand")
            self.assertEqual(payloads[0]["details"]["layout_profile"], "top_narration")

    def test_analyze_connected_subregions_white_balloon_profile_preserves_close_split(self):
        subregions = [[20, 20, 120, 120], [102, 20, 202, 120]]
        balloon_bbox = [0, 0, 220, 140]

        standard = _analyze_connected_subregions(subregions, balloon_bbox)
        adaptive = _analyze_connected_subregions(subregions, balloon_bbox, profile="white_balloon")

        self.assertEqual(standard["ordered_subregions"], [balloon_bbox])
        self.assertEqual(adaptive["orientation"], "left-right")
        self.assertEqual(len(adaptive["ordered_subregions"]), 2)

    def test_real_009_connected_balloon_creates_two_subregions(self):
        page = {
            "image": str(_fixture_image_path("009__001.jpg")),
            "width": 800,
            "height": 2600,
            "texts": [
                {
                    "text": "JTMAY BE NOTHING MORE THAN A HALF-FINISHED CULTINATIONMETHOD, BUT IT'S EFFECTS ARE MORE THAN ENOUGH. A POWER THAT LET'S YOU SURPASS YOUR OWN LIMITS IN AN INSTANT",
                    "bbox": [113, 1514, 705, 1767],
                    "tipo": "fala",
                    "confidence": 0.874,
                }
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(len(text.get("balloon_subregions", [])), 2)
        self.assertEqual(text.get("ocr_text_bbox"), [113, 1514, 705, 1767])
        self.assertEqual(text.get("connected_lobe_bboxes"), text.get("balloon_subregions"))
        self.assertEqual(text.get("connected_position_bboxes"), text.get("connected_focus_bboxes"))
        self.assertEqual(len(text.get("connected_text_groups", [])), 2)
        self.assertEqual(len(text.get("connected_lobe_bboxes", [])), 2)
        self.assertEqual(len(text.get("connected_position_bboxes", [])), 2)
        self.assertNotEqual(text["balloon_subregions"][0], text["balloon_subregions"][1])
        self.assertLess(_intersection_area(text["balloon_subregions"][0], text["balloon_subregions"][1]), 12)
        balloon_area = max(
            1,
            (text["balloon_bbox"][2] - text["balloon_bbox"][0]) * (text["balloon_bbox"][3] - text["balloon_bbox"][1]),
        )
        covered_area = sum(
            max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            for bbox in text.get("balloon_subregions", [])
        )
        self.assertGreater(covered_area / float(balloon_area), 0.55)
        self.assertEqual(len(text.get("connected_focus_bboxes", [])), 2)
        self.assertGreater(float(text.get("connected_detection_confidence", 0.0) or 0.0), 0.5)
        self.assertGreater(float(text.get("connected_group_confidence", 0.0) or 0.0), 0.5)
        self.assertGreater(float(text.get("connected_position_confidence", 0.0) or 0.0), 0.5)
        left_group, right_group = text["connected_text_groups"]
        left_focus, right_focus = text["connected_focus_bboxes"]
        left_sub, right_sub = text["balloon_subregions"]
        left_group_cx = (left_group[0] + left_group[2]) / 2.0
        left_group_cy = (left_group[1] + left_group[3]) / 2.0
        right_group_cx = (right_group[0] + right_group[2]) / 2.0
        right_group_cy = (right_group[1] + right_group[3]) / 2.0
        left_focus_cx = (left_focus[0] + left_focus[2]) / 2.0
        left_focus_cy = (left_focus[1] + left_focus[3]) / 2.0
        right_focus_cx = (right_focus[0] + right_focus[2]) / 2.0
        right_focus_cy = (right_focus[1] + right_focus[3]) / 2.0
        left_sub_cx = (left_sub[0] + left_sub[2]) / 2.0
        left_sub_cy = (left_sub[1] + left_sub[3]) / 2.0
        right_sub_cx = (right_sub[0] + right_sub[2]) / 2.0
        right_sub_cy = (right_sub[1] + right_sub[3]) / 2.0
        self.assertLess(left_group_cx, left_sub_cx)
        self.assertLess(left_group_cy, left_sub_cy)
        self.assertGreater(right_group_cx, right_sub_cx)
        self.assertGreater(right_group_cy, right_sub_cy)
        self.assertLess(left_focus_cx, left_sub_cx)
        self.assertLess(left_focus_cy, left_sub_cy)
        self.assertGreater(right_focus_cx, right_sub_cx)
        self.assertGreater(right_focus_cy, right_sub_cy)

    def test_real_009_single_balloon_keeps_single_region(self):
        page = {
            "image": str(_fixture_image_path("009__001.jpg")),
            "width": 800,
            "height": 2600,
            "texts": [
                {
                    "text": "YOU MEAN THE POWER HE FORCED OUT THROUGHA DEAL WITH 'THEM'",
                    "bbox": [241, 1095, 578, 1235],
                    "tipo": "fala",
                    "confidence": 0.94,
                }
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(text.get("balloon_subregions", []), [])
        self.assertLess(text["balloon_bbox"][2] - text["balloon_bbox"][0], 520)
        self.assertLess(text["balloon_bbox"][3] - text["balloon_bbox"][1], 260)

    def test_textured_standard_region_never_promotes_to_connected_balloon(self):
        page = {
            "width": 800,
            "height": 2560,
            "texts": [
                {
                    "id": "tl_008_001",
                    "text": "THERE'S NO TURNING BACK NOW.",
                    "bbox": [111, 238, 707, 610],
                    "tipo": "fala",
                    "block_profile": "standard",
                    "estilo": {"fonte": "Newrotic.ttf"},
                    "confidence": 0.924,
                }
            ],
        }

        rich_subregions = [
            {"bbox": [111, 238, 362, 610], "polygon": None, "area": 1000},
            {"bbox": [376, 238, 707, 610], "polygon": None, "area": 1000},
        ]
        with patch(
            "layout.balloon_layout._load_page_image",
            return_value=np.zeros((2560, 800, 3), dtype=np.uint8),
        ), patch(
            "layout.balloon_layout._detect_connected_balloon_subregions_rich",
            return_value=rich_subregions,
        ) as detector:
            enriched = enrich_page_layout(page)

        text = enriched["texts"][0]
        self.assertFalse(detector.called, "Texturizado nao deve nem consultar detector conectado")
        self.assertNotEqual(text.get("layout_profile"), "connected_balloon")
        self.assertEqual(text.get("layout_group_size"), 1)
        self.assertEqual(text.get("balloon_subregions"), [])

    def test_narration_region_never_promotes_to_connected_balloon(self):
        page = {
            "width": 800,
            "height": 2560,
            "texts": [
                {
                    "id": "tl_040_001",
                    "text": "I'VE MET A FEW WHO USE THAT POWER.",
                    "bbox": [118, 780, 652, 1104],
                    "tipo": "narracao",
                    "block_profile": "white_balloon",
                    "confidence": 0.892,
                }
            ],
        }

        rich_subregions = [
            {"bbox": [118, 780, 456, 1104], "polygon": None, "area": 1000},
            {"bbox": [468, 780, 652, 1104], "polygon": None, "area": 1000},
        ]
        with patch(
            "layout.balloon_layout._load_page_image",
            return_value=np.zeros((2560, 800, 3), dtype=np.uint8),
        ), patch(
            "layout.balloon_layout._detect_connected_balloon_subregions_rich",
            return_value=rich_subregions,
        ) as detector:
            enriched = enrich_page_layout(page)

        text = enriched["texts"][0]
        self.assertFalse(detector.called, "Narracao nao deve entrar no fluxo de balao conectado")
        self.assertNotEqual(text.get("layout_profile"), "connected_balloon")
        self.assertEqual(text.get("layout_group_size"), 1)
        self.assertEqual(text.get("balloon_subregions"), [])

    def test_single_white_balloon_connected_outline_splits_into_two_subregions(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "connected-white-balloon.jpg"
            image = np.full((420, 520, 3), 235, dtype=np.uint8)
            cv2.ellipse(image, (170, 150), (120, 80), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(image, (320, 250), (140, 90), 0, 0, 360, (255, 255, 255), -1)
            cv2.rectangle(image, (180, 165), (305, 235), (255, 255, 255), -1)
            cv2.ellipse(image, (170, 150), (120, 80), 0, 0, 360, (0, 0, 0), 3)
            cv2.ellipse(image, (320, 250), (140, 90), 0, 0, 360, (0, 0, 0), 3)
            cv2.line(image, (180, 165), (305, 165), (0, 0, 0), 3)
            cv2.line(image, (180, 235), (305, 235), (0, 0, 0), 3)
            cv2.imwrite(str(image_path), image)

            page = {
                "image": str(image_path),
                "width": 520,
                "height": 420,
                "texts": [
                    {
                        "id": "tl_036_002",
                        "text": "EVEN THOUGH IT'S ONLY HALF OF A MANA TECHNIQUE, ITS EFFECTS WILL BE MORE THAN ENOUGH. THAT POWER LETS ONE INSTANTLY SURPASS THEIR OWN LIMITS.",
                        "bbox": [95, 110, 410, 295],
                        "tipo": "fala",
                        "block_profile": "white_balloon",
                        "confidence": 0.909,
                    }
                ],
            }

            enriched = enrich_page_layout(page)

        text = enriched["texts"][0]
        self.assertEqual(text.get("layout_profile"), "connected_balloon")
        self.assertGreaterEqual(text.get("layout_group_size", 0), 2)
        self.assertEqual(len(text.get("balloon_subregions", [])), 2)
        self.assertNotEqual(text["balloon_subregions"][0], text["balloon_subregions"][1])

    def test_real_002_textured_balloon_does_not_merge_with_distant_white_gap_text(self):
        page = {
            "image": str(_fixture_image_path("002__002.jpg")),
            "width": 800,
            "height": 2560,
            "texts": [
                {
                    "text": "Etetob",
                    "bbox": [349, 1733, 759, 2020],
                    "tipo": "fala",
                    "confidence": 0.81,
                },
                {
                    "text": "THERE'S NO TURNING BACK MON",
                    "bbox": [190, 2273, 625, 2476],
                    "tipo": "fala",
                    "confidence": 0.92,
                },
            ],
        }

        enriched = enrich_page_layout(page)
        bottom = next(text for text in enriched["texts"] if "TURNING BACK" in text["text"])

        self.assertEqual(bottom["layout_group_size"], 1)
        self.assertGreater(bottom["balloon_bbox"][1], 2200)
        self.assertLess(bottom["balloon_bbox"][3], 2520)

    def test_partial_balloon_touching_page_edge_still_expands_layout_bbox(self):
        import cv2
        import numpy as np

        page = {
            "width": 260,
            "height": 140,
            "texts": [
                {
                    "text": "USELESS",
                    "bbox": [92, 12, 168, 42],
                    "tipo": "fala",
                    "confidence": 0.96,
                }
            ],
        }

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        image[:] = 18
        cv2.ellipse(image, (130, 10), (94, 52), 0, 0, 360, (246, 246, 246), -1)
        page["_cached_image_bgr"] = image

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertLessEqual(text["balloon_bbox"][0], 54)
        self.assertEqual(text["balloon_bbox"][1], 0)
        self.assertGreaterEqual(text["balloon_bbox"][2], 206)
        self.assertGreaterEqual(text["balloon_bbox"][3], 54)

    def test_connected_vertical_balloons_split_into_top_and_bottom_subregions(self):
        import cv2
        import numpy as np

        page = {
            "width": 380,
            "height": 320,
            "texts": [
                {
                    "text": "A. B.",
                    "bbox": [70, 45, 305, 245],
                    "tipo": "fala",
                    "confidence": 0.95,
                }
            ],
        }

        image = np.full((320, 380, 3), 230, dtype=np.uint8)
        cv2.ellipse(image, (190, 95), (120, 70), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (200, 210), (125, 72), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (195, 152), (26, 22), 0, 0, 360, (248, 248, 248), -1)
        cv2.rectangle(image, (120, 80), (250, 112), (20, 20, 20), -1)
        cv2.rectangle(image, (116, 194), (258, 226), (20, 20, 20), -1)
        page["_cached_image_bgr"] = image

        enriched = enrich_page_layout(page)
        subregions = enriched["texts"][0]["balloon_subregions"]

        self.assertEqual(len(subregions), 2)
        self.assertLess(subregions[0][1], subregions[1][1])
        self.assertEqual(subregions[0][0], 70)
        self.assertEqual(subregions[1][2], 305)


def _intersection_area(a: list[int], b: list[int]) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int((ix2 - ix1) * (iy2 - iy1))


def _make_connected_balloon_image(w: int, h: int, lobe1_center, lobe1_r, lobe2_center, lobe2_r, neck_rect=None):
    """Cria imagem sintética com dois lobos brancos conectados por um pescoço."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, lobe1_center, lobe1_r, (255, 255, 255), -1)
    cv2.circle(img, lobe2_center, lobe2_r, (255, 255, 255), -1)
    if neck_rect:
        nx1, ny1, nx2, ny2 = neck_rect
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
    return img


class ConnectedBalloonDetectionTests(unittest.TestCase):
    def test_distance_transform_detects_wide_neck_balloon(self):
        """Dois lobos conectados por pescoço largo — erosão falha, distance transform detecta."""
        img = _make_connected_balloon_image(
            400, 200,
            lobe1_center=(100, 100), lobe1_r=70,
            lobe2_center=(300, 100), lobe2_r=70,
            neck_rect=(80, 70, 320, 130),
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, component = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        lobes = _detect_lobes_via_distance_transform(
            component, [0, 0, 400, 200], fill_area=int(np.count_nonzero(component)),
        )

        self.assertGreaterEqual(len(lobes), 2)
        centers = [(l["bbox"][0] + l["bbox"][2]) / 2 for l in lobes]
        self.assertTrue(any(c < 200 for c in centers))
        self.assertTrue(any(c > 200 for c in centers))

    def test_distance_transform_returns_empty_for_single_lobe(self):
        """Balão único sem pescoço — distance transform NÃO deve retornar lobos."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 80, (255, 255, 255), -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, component = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        lobes = _detect_lobes_via_distance_transform(
            component, [0, 0, 200, 200], fill_area=int(np.count_nonzero(component)),
        )

        self.assertEqual(len(lobes), 0)

    def test_fill_based_detection_uses_distance_transform_fallback(self):
        """_detect_connected_balloon_subregions_from_fill retorna 2 subregions via DT fallback."""
        img = _make_connected_balloon_image(
            500, 220,
            lobe1_center=(120, 110), lobe1_r=80,
            lobe2_center=(380, 110), lobe2_r=80,
            neck_rect=(100, 75, 400, 145),
        )

        subregions = _detect_connected_balloon_subregions_from_fill(
            img, [80, 40, 420, 180], [0, 0, 500, 220],
        )

        self.assertGreaterEqual(len(subregions), 2)

    def test_geometric_fallback_splits_wide_balloon_with_two_texts(self):
        """Fallback geométrico: balão largo sem imagem + 2 textos próximos → subregions inferidas."""
        # Textos com gap de 20px (< 25.5px threshold) → serão mesclados numa região
        page = {
            "width": 600,
            "height": 200,
            "image": None,
            "texts": [
                {"text": "Hello", "bbox": [20, 50, 240, 150], "tipo": "fala", "confidence": 0.9},
                {"text": "World", "bbox": [260, 50, 480, 150], "tipo": "fala", "confidence": 0.9},
            ],
        }

        enriched = enrich_page_layout(page)
        texts = enriched["texts"]

        has_subregions = any(len(t.get("balloon_subregions", [])) >= 2 for t in texts)
        self.assertTrue(has_subregions, "Fallback geométrico deveria ter inferido subregions")

    def test_geometric_fallback_skips_single_text_wide_balloon(self):
        """1 texto em balão largo → sem subregions pelo fallback."""
        page = {
            "width": 500,
            "height": 200,
            "image": None,
            "texts": [
                {"text": "Hello World", "bbox": [30, 50, 470, 150], "tipo": "fala", "confidence": 0.9},
            ],
        }

        enriched = enrich_page_layout(page)
        texts = enriched["texts"]

        for t in texts:
            self.assertFalse(t.get("balloon_subregions"), "Texto único não deve ter subregions")

    def test_geometric_fallback_handles_koharu_loose_bbox_aspect_below_two(self):
        """Regressão: bubble_bbox solto do koharu (aspect ~1.97) deve ativar Modo B.

        Na app real, o koharu devolve uma bbox maior do que o balão real; se o
        pipeline confia nela e a bbox tem aspect < 2.0, o Mode B antigo
        (aspect >= 2.0) desligava o split e o balão conectado saía como um
        bloco único sobrescrevendo os dois lobos.
        """
        # balloon [29,1455,789,1841]: bw=760, bh=386, aspect≈1.97
        # min(bw,bh)=386 >= 180, max(bw,bh)=760 >= 450
        balloon = [29, 1455, 789, 1841]
        # single-text scenario (group_size==1), so Mode B is the path that must fire
        group_text = {
            "balloon_bbox": balloon,
            "balloon_subregions": [],
            "layout_group_size": 1,
            "tipo": "fala",
            "bbox": [113, 1514, 705, 1765],
        }
        from layout.balloon_layout import _apply_geometric_fallback_subregions
        _apply_geometric_fallback_subregions([group_text])
        self.assertEqual(
            len(group_text.get("balloon_subregions", [])),
            2,
            "Mode B deveria emitir 2 subregions para balão com aspect ~1.97",
        )


class SubregionConfidenceTests(unittest.TestCase):
    def test_good_split_has_high_confidence(self):
        """Two equal halves covering the balloon → high confidence."""
        balloon = [0, 0, 400, 200]
        subregions = [[0, 0, 200, 200], [200, 0, 400, 200]]
        score = _score_subregion_quality(subregions, balloon)
        self.assertGreater(score, 0.7)

    def test_overlapping_subregions_have_lower_confidence(self):
        """Overlapping subregions should score lower."""
        balloon = [0, 0, 400, 200]
        good = [[0, 0, 200, 200], [200, 0, 400, 200]]
        bad = [[0, 0, 300, 200], [100, 0, 400, 200]]  # heavy overlap
        good_score = _score_subregion_quality(good, balloon)
        bad_score = _score_subregion_quality(bad, balloon)
        self.assertGreater(good_score, bad_score)

    def test_empty_subregions_return_zero(self):
        self.assertEqual(_score_subregion_quality([], [0, 0, 400, 200]), 0.0)
        self.assertEqual(_score_subregion_quality([[0, 0, 100, 100]], [0, 0, 400, 200]), 0.0)


class GeometricFallbackSubregionsTests(unittest.TestCase):
    """Testes para _geometric_fallback_subregions com splits horizontal, vertical e diagonal."""

    def test_horizontal_texts_produce_vertical_cut(self):
        """Textos lado-a-lado → corte vertical (L/R split)."""
        balloon = [0, 0, 800, 400]
        texts = [[50, 100, 250, 300], [550, 100, 750, 300]]  # left, right
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # Left sub should be to the left of right sub
        self.assertLess(subs[0][2], subs[1][0] + 50)

    def test_vertical_texts_produce_horizontal_cut(self):
        """Textos empilhados → corte horizontal (T/B split)."""
        balloon = [0, 0, 400, 800]
        texts = [[100, 50, 300, 250], [100, 550, 300, 750]]  # top, bottom
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # Top sub should be above bottom sub
        self.assertLess(subs[0][3], subs[1][1] + 50)

    def test_diagonal_backslash_texts_produce_quadrant_split(self):
        r"""Texto top-left + bottom-right → quadrantes diagonal \ ."""
        balloon = [0, 0, 800, 800]
        texts = [[50, 50, 250, 250], [550, 550, 750, 750]]  # TL, BR
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # One sub covers top-left quadrant, other bottom-right
        tl = min(subs, key=lambda s: s[0] + s[1])
        br = max(subs, key=lambda s: s[0] + s[1])
        self.assertLess(tl[2], balloon[2])  # TL doesn't span full width
        self.assertGreater(br[0], balloon[0])  # BR doesn't start at left edge

    def test_diagonal_slash_texts_produce_quadrant_split(self):
        """Texto top-right + bottom-left → quadrantes diagonal / ."""
        balloon = [0, 0, 800, 800]
        texts = [[550, 50, 750, 250], [50, 550, 250, 750]]  # TR, BL
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # One sub covers top-right, other bottom-left
        tr = min(subs, key=lambda s: s[1])  # lower y = top
        bl = max(subs, key=lambda s: s[1])  # higher y = bottom
        self.assertGreater(tr[0], balloon[0])  # TR starts past left
        self.assertLess(bl[2], balloon[2])  # BL doesn't reach right

    def test_close_centers_rejected(self):
        """Textos com centros muito próximos → retorna [] (balão único)."""
        balloon = [0, 0, 800, 400]
        texts = [[300, 150, 400, 200], [350, 160, 450, 210]]  # very close
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(subs, [])

    def test_single_text_wide_balloon_splits_vertically(self):
        """1 texto + balão largo → split L/R."""
        balloon = [0, 0, 800, 300]
        texts = [[200, 80, 600, 220]]
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)

    def test_single_text_tall_balloon_splits_horizontally(self):
        """1 texto + balão alto → split T/B."""
        balloon = [0, 0, 300, 800]
        texts = [[50, 200, 250, 600]]
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)

    def test_geometric_fallback_accepts_aspect_1_75_with_min_dims_160_420(self):
        """Balão 420×240 (aspect=1.75, min=160, max=420) + texto central → 2 subregions."""
        # bbox [0, 0, 420, 240]: w=420, h=240 → aspect=1.75, min(w,h)=240≥160, max(w,h)=420≥420
        text_entry = {
            "text": "Texto de teste",
            "bbox": [105, 60, 315, 180],  # centrado no balão
            "tipo": "fala",
            "confidence": 0.9,
            "balloon_bbox": [0, 0, 420, 240],
            "layout_group_size": 1,
        }
        texts = [text_entry]
        _apply_geometric_fallback_subregions(texts)
        subs = texts[0].get("balloon_subregions", [])
        self.assertEqual(len(subs), 2, f"Esperado 2 subregions, obteve {len(subs)}: {subs}")

    def test_text_pixel_gap_fallback_splits_tall_connected_balloon(self):
        """Quando há um grande gap vertical entre text_pixel_bbox, devemos dividir top/bottom."""
        texts = [
            {
                "text": "Parte superior",
                "bbox": [80, 60, 220, 120],
                "text_pixel_bbox": [92, 70, 208, 116],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [0, 0, 300, 600],
                "layout_group_size": 2,
            },
            {
                "text": "Parte inferior",
                "bbox": [76, 360, 224, 424],
                "text_pixel_bbox": [88, 372, 212, 418],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [0, 0, 300, 600],
                "layout_group_size": 2,
            },
        ]

        _apply_geometric_fallback_subregions(texts)

        subs = texts[0].get("balloon_subregions", [])
        self.assertEqual(len(subs), 2, f"Esperado 2 subregions, obteve {len(subs)}: {subs}")
        ordered = sorted(subs, key=lambda item: (item[1], item[0]))
        self.assertLess(ordered[0][3], ordered[1][1] + 40)

    def test_geometric_fallback_skips_non_connected_group_without_crashing(self):
        """Grupo sem shared layout nem heurística válida não deve disparar UnboundLocalError."""
        texts = [
            {
                "text": "Linha 1",
                "bbox": [60, 70, 240, 170],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [20, 20, 340, 240],
                "layout_group_size": 1,
            },
            {
                "text": "Linha 2",
                "bbox": [80, 260, 260, 330],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [20, 20, 340, 240],
                "layout_group_size": 1,
            },
        ]

        _apply_geometric_fallback_subregions(texts, {}, None)

        for text in texts:
            self.assertEqual(text.get("balloon_subregions", []), [])

    def test_geometric_fallback_skips_stacked_lines_in_same_balloon(self):
        """Linhas empilhadas do mesmo balÃ£o nÃ£o devem virar lobos conectados."""
        texts = [
            {
                "text": "HE BROKE",
                "bbox": [233, 734, 673, 770],
                "text_pixel_bbox": [256, 742, 648, 768],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_type": "white",
                "balloon_bbox": [233, 734, 673, 906],
                "layout_group_size": 4,
            },
            {
                "text": "THE MANA-INFUSED",
                "bbox": [233, 772, 673, 808],
                "text_pixel_bbox": [248, 780, 652, 806],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_type": "white",
                "balloon_bbox": [233, 734, 673, 906],
                "layout_group_size": 4,
            },
            {
                "text": "BLADE WITH SHEER",
                "bbox": [233, 810, 673, 846],
                "text_pixel_bbox": [246, 818, 656, 844],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_type": "white",
                "balloon_bbox": [233, 734, 673, 906],
                "layout_group_size": 4,
            },
            {
                "text": "GRIP STRENGTH.",
                "bbox": [233, 848, 673, 884],
                "text_pixel_bbox": [266, 856, 638, 882],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_type": "white",
                "balloon_bbox": [233, 734, 673, 906],
                "layout_group_size": 4,
            },
        ]

        _apply_geometric_fallback_subregions(texts, {}, None)

        for text in texts:
            self.assertEqual(text.get("balloon_subregions", []), [])

    def test_geometric_fallback_skips_dense_single_balloon_when_image_is_available(self):
        """Balão simples, largo e denso não deve ser splitado só pela geometria."""
        image = np.full((220, 420, 3), 170, dtype=np.uint8)
        cv2.ellipse(image, (210, 110), (180, 88), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (210, 110), (180, 88), 0, 0, 360, (18, 18, 18), 3)

        texts = [
            {
                "text": "Texto central",
                "bbox": [92, 62, 328, 160],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [0, 0, 420, 220],
                "layout_group_size": 1,
            }
        ]

        _apply_geometric_fallback_subregions(texts, {}, image)

        self.assertEqual(texts[0].get("balloon_subregions", []), [])

    def test_geometric_fallback_splits_diagonal_shared_group_even_with_aspect_below_one_point_seven(self):
        """Grupo diagonal real pode dividir mesmo quando o balão compartilhado não é tão largo."""
        texts = [
            {
                "text": "Parte superior",
                "bbox": [70, 7513, 442, 7761],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [33, 7484, 667, 7995],
                "layout_group_size": 2,
            },
            {
                "text": "Parte inferior",
                "bbox": [331, 7800, 631, 7969],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [33, 7484, 667, 7995],
                "layout_group_size": 2,
            },
        ]

        _apply_geometric_fallback_subregions(texts, {}, None)

        for text in texts:
            self.assertEqual(len(text.get("balloon_subregions", [])), 2)

    def test_fill_area_1500_still_accepted(self):
        """Verifica no código-fonte que fill_area_min é 1500 (não 2500)."""
        import inspect
        import layout.balloon_layout as _mod
        src = inspect.getsource(_mod._detect_connected_balloon_subregions_from_fill)
        self.assertIn("fill_area < 1500", src, "Threshold deveria ser 1500, não 2500")
        self.assertNotIn("fill_area < 2500", src, "Threshold antigo 2500 ainda presente")

    def test_erosion_ratio_relaxed_to_20_percent(self):
        """Verifica no código-fonte que o corte de erosão foi relaxado para 20%."""
        import inspect
        import layout.balloon_layout as _mod

        src = inspect.getsource(_mod._detect_connected_balloon_subregions_from_fill)
        self.assertIn("fill_area * 0.20", src, "Threshold de erosão deveria ser 20%")
        self.assertNotIn("fill_area * 0.25", src, "Threshold antigo de 25% ainda presente")

    def test_gap_threshold_relaxed_to_1_point_5_percent(self):
        """Verifica no código-fonte que o gap mínimo foi relaxado para 1.5%."""
        import inspect
        import layout.balloon_layout as _mod

        src = inspect.getsource(_mod._detect_connected_balloon_subregions_from_fill)
        self.assertIn("0.015", src, "Threshold de gap deveria ser 1.5%")
        self.assertNotIn("0.02", src, "Threshold antigo de gap 2% ainda presente")

    def test_enriched_text_has_subregion_confidence(self):
        """enrich_page_layout should attach subregion_confidence to texts."""
        image = np.full((320, 380, 3), 230, dtype=np.uint8)
        cv2.ellipse(image, (190, 95), (120, 70), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (200, 210), (125, 72), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (195, 152), (26, 22), 0, 0, 360, (248, 248, 248), -1)
        page = {
            "width": 380,
            "height": 320,
            "_cached_image_bgr": image,
            "texts": [
                {
                    "text": "A. B.",
                    "bbox": [70, 45, 305, 245],
                    "tipo": "fala",
                    "confidence": 0.95,
                }
            ],
        }
        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]
        self.assertIn("subregion_confidence", text)
        if text.get("balloon_subregions"):
            self.assertGreater(text["subregion_confidence"], 0.0)


class EnforceMinLobeSizeTests(unittest.TestCase):
    """Testes para _enforce_min_lobe_size e _build_balloon_subregions_from_groups."""

    def test_enforce_min_lobe_size_expands_narrow_right_lobe(self):
        """Lobo direito muito estreito → expandido para 30% mínimo."""
        balloon = [0, 0, 800, 400]
        subs = [[0, 0, 700, 400], [700, 0, 800, 400]]
        fixed = _enforce_min_lobe_size(subs, balloon)
        right_w = fixed[1][2] - fixed[1][0]
        self.assertGreaterEqual(right_w / 800.0, 0.29)

    def test_enforce_min_lobe_size_expands_narrow_left_lobe(self):
        """Lobo esquerdo muito estreito → expandido para 30% mínimo."""
        balloon = [0, 0, 800, 400]
        subs = [[0, 0, 100, 400], [100, 0, 800, 400]]
        fixed = _enforce_min_lobe_size(subs, balloon)
        left_w = fixed[0][2] - fixed[0][0]
        self.assertGreaterEqual(left_w / 800.0, 0.29)

    def test_enforce_min_lobe_size_keeps_balanced_split(self):
        """Split já balanceado → sem mudança."""
        balloon = [0, 0, 800, 400]
        subs = [[0, 0, 400, 400], [400, 0, 800, 400]]
        fixed = _enforce_min_lobe_size(subs, balloon)
        self.assertEqual(fixed[0], [0, 0, 400, 400])
        self.assertEqual(fixed[1], [400, 0, 800, 400])

    def test_build_subregions_horizontal_creates_gap(self):
        """Split horizontal deve ter gap entre os lobos."""
        left_bbox = [50, 100, 350, 300]
        right_bbox = [450, 100, 750, 300]
        balloon = [0, 0, 800, 400]
        subs = _build_balloon_subregions_from_groups([left_bbox, right_bbox], balloon)
        self.assertEqual(len(subs), 2)
        gap = subs[1][0] - subs[0][2]
        self.assertGreater(gap, 0, "Deve haver gap entre os lobos")

    def test_build_subregions_vertical_creates_gap(self):
        """Split vertical deve ter gap entre os lobos."""
        top_bbox = [100, 50, 300, 180]
        bottom_bbox = [100, 250, 300, 380]
        balloon = [0, 0, 400, 400]
        subs = _build_balloon_subregions_from_groups([top_bbox, bottom_bbox], balloon)
        self.assertEqual(len(subs), 2)
        gap = subs[1][1] - subs[0][3]
        self.assertGreater(gap, 0, "Deve haver gap vertical entre os lobos")

    def test_build_subregions_neither_lobe_too_narrow(self):
        """Nenhum lobo deve ter menos que 30% da dimensão principal."""
        left_bbox = [50, 100, 150, 300]
        right_bbox = [600, 100, 750, 300]
        balloon = [0, 0, 800, 400]
        subs = _build_balloon_subregions_from_groups([left_bbox, right_bbox], balloon)
        self.assertEqual(len(subs), 2)
        bw = 800
        for s in subs:
            sw = s[2] - s[0]
            self.assertGreaterEqual(sw / float(bw), 0.25,
                f"Lobo {s} muito estreito: {sw}px de {bw}px")

    def test_analyze_connected_subregions_detects_horizontal_reading_order(self):
        """Subregions horizontais devem ser ordenadas da esquerda para a direita."""
        plan = _analyze_connected_subregions(
            [[220, 0, 420, 240], [0, 0, 200, 240]],
            [0, 0, 420, 240],
        )

        self.assertEqual(plan["orientation"], "left-right")
        self.assertEqual(plan["ordered_subregions"][0], [0, 0, 200, 240])

    def test_analyze_connected_subregions_detects_vertical_reading_order(self):
        """Subregions verticais devem ser ordenadas de cima para baixo."""
        plan = _analyze_connected_subregions(
            [[0, 220, 240, 420], [0, 0, 240, 200]],
            [0, 0, 240, 420],
        )

        self.assertEqual(plan["orientation"], "top-bottom")
        self.assertEqual(plan["ordered_subregions"][0], [0, 0, 240, 200])

    # ------------------------------------------------------------------
    # Testes de detecção topológica (_extract_balloon_outline_polygon,
    # _detect_connected_lobes_from_outline)
    # ------------------------------------------------------------------

    def test_extract_balloon_outline_polygon_returns_polygon_for_white_balloon(self):
        """_extract_balloon_outline_polygon deve retornar um polígono com >= 4 pontos
        para um balão branco sintético com borda preta."""
        # Criar imagem sintética: fundo branco com borda preta de 4px
        h, w = 120, 180
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Borda preta
        img[:4, :] = 0
        img[-4:, :] = 0
        img[:, :4] = 0
        img[:, -4:] = 0
        balloon_bbox = [0, 0, w, h]

        poly = _extract_balloon_outline_polygon(img, balloon_bbox)

        self.assertIsNotNone(poly, "Deve retornar um polígono para balão com borda preta")
        self.assertGreaterEqual(len(poly), 4, "Polígono deve ter pelo menos 4 pontos")
        # Verificar que os pontos estão em coordenadas globais (dentro da imagem)
        self.assertTrue(np.all(poly[:, 0] >= 0) and np.all(poly[:, 0] <= w))
        self.assertTrue(np.all(poly[:, 1] >= 0) and np.all(poly[:, 1] <= h))

    def test_extract_balloon_outline_polygon_returns_none_for_empty_region(self):
        """_extract_balloon_outline_polygon deve retornar None para bbox vazio."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = _extract_balloon_outline_polygon(img, [0, 0, 0, 0])
        self.assertIsNone(result)

    def test_detect_connected_lobes_from_outline_separates_two_connected_circles(self):
        """Dois círculos brancos conectados por pescoço fino devem ser separados em 2 lobos.

        A imagem simula um balão manga realista: fundo cinza claro (página),
        interior branco (área do balão), borda preta (contorno do balão).
        """
        # Criar imagem realista de balão conectado:
        # fundo cinza (simulando papel), interior branco, borda preta
        h, w = 220, 440
        # Fundo cinza claro (como papel de mangá)
        img = np.full((h, w, 3), 200, dtype=np.uint8)

        # Preencher interior dos dois círculos de branco (área do balão)
        cv2.circle(img, (90, 110), 80, (255, 255, 255), -1)
        cv2.circle(img, (350, 110), 80, (255, 255, 255), -1)
        # Pescoço fino branco conectando os dois lobos
        img[100:120, 150:290] = 255

        # Desenhar borda preta ao redor dos círculos
        cv2.circle(img, (90, 110), 80, (0, 0, 0), 4)
        cv2.circle(img, (350, 110), 80, (0, 0, 0), 4)
        # Apagar a borda preta na região do pescoço para que seja contínuo
        img[100:120, 145:295] = 255

        balloon_bbox = [5, 20, 435, 200]
        seed_bbox = [10, 25, 430, 195]

        lobes = _detect_connected_lobes_from_outline(img, balloon_bbox, seed_bbox)

        self.assertGreaterEqual(
            len(lobes), 2,
            "Dois círculos conectados devem ser separados em 2+ lobos"
        )
        # Cada lobo deve ter bbox e polygon
        for lobe in lobes[:2]:
            self.assertIn("bbox", lobe)
            self.assertIn("polygon", lobe)
            self.assertIsNotNone(lobe["polygon"])
            self.assertGreater(lobe["area"], 0)

    def test_detect_connected_lobes_from_outline_returns_empty_for_single_circle(self):
        """Um único círculo não deve ser separado em lobos."""
        h, w = 200, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 80, (255, 255, 255), -1)
        # Borda preta
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 3)

        balloon_bbox = [0, 0, w, h]
        seed_bbox = [10, 10, 190, 190]

        lobes = _detect_connected_lobes_from_outline(img, balloon_bbox, seed_bbox)

        # Um único círculo não deve ser separado (pode retornar 0 ou 1 lobe)
        self.assertLess(
            len(lobes), 2,
            "Um único círculo não deve ser separado em 2 lobos"
        )

    def test_detect_connected_lobes_from_outline_returns_empty_for_none_image(self):
        """Deve retornar [] quando image é None."""
        lobes = _detect_connected_lobes_from_outline(None, [0, 0, 100, 100], [0, 0, 100, 100])
        self.assertEqual(lobes, [])


if __name__ == "__main__":
    unittest.main()
