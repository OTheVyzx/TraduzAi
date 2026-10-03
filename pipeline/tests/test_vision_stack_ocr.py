import builtins
from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
import ast
import os
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import cv2

import vision_stack.ocr as ocr_mod
from vision_stack.ocr import (
    OCREngine,
    execute_hash_bound_provider_attempt,
    normalize_easyocr_languages,
    normalize_paddleocr_language,
)
from ownership.hash_contract import canonical_page_sha256, sha256_bytes, sha256_text
from ownership.ocr_contract import (
    OCRAttempt,
    OCRBlock,
    OCRDiagnostics,
    OCRInputPixelIdentityError,
    OCRInvocationResult,
    OCRObservationRecord,
    OCRRequest,
    OCRRequestIdentityError,
    OCRTransformOperation,
    OCRTransformSpec,
)


def _contract_page(marker: int = 25) -> np.ndarray:
    return np.full((32, 48, 3), marker, dtype=np.uint8)


def _contract_request(
    page_id: str = "page_001",
    *,
    run_id: str = "run-a",
    origin_execution_id: str = "exec-a",
    root: np.ndarray | None = None,
    invocation_id: str | None = None,
) -> OCRRequest:
    pixels = _contract_page() if root is None else root
    return OCRRequest(
        run_id=run_id,
        origin_execution_id=origin_execution_id,
        page_id=page_id,
        page_source_sha256=canonical_page_sha256(pixels),
        root_input_pixel_sha256=canonical_page_sha256(pixels),
        invocation_id=invocation_id or f"ocr-{page_id}",
        provider_family="paddleocr",
    )


def _identity_spec() -> OCRTransformSpec:
    return OCRTransformSpec.build((OCRTransformOperation(kind="identity"),))


def _scale_2x_spec(root: np.ndarray) -> tuple[OCRTransformSpec, np.ndarray]:
    transformed = cv2.resize(root, (root.shape[1] * 2, root.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    return (
        OCRTransformSpec.build((
            OCRTransformOperation(kind="identity"),
            OCRTransformOperation(kind="resize", output_size=(transformed.shape[1], transformed.shape[0]), interpolation="nearest"),
        )),
        transformed,
    )


def _affine_spec(root: np.ndarray, kind: str, angle: float = 7.125) -> tuple[OCRTransformSpec, np.ndarray]:
    center = (root.shape[1] / 2.0, root.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    fixed = tuple(int(round(float(value) * 1_000_000)) for value in matrix.reshape(-1))
    operation = OCRTransformOperation(
        kind=kind,
        output_size=(root.shape[1], root.shape[0]),
        interpolation="linear",
        border_mode="constant",
        border_value_rgb=(255, 255, 255),
        affine_matrix_fixed_1e6=fixed,
        algorithm_id="opencv-warp-affine-v1",
    )
    spec = OCRTransformSpec.build((operation,))
    return spec, spec.replay(root)


def _raw_ocr(text: str):
    return [[([[2, 3], [40, 3], [40, 18], [2, 18]], (text, 0.96))]]


def _attempt_and_record(
    request: OCRRequest,
    *,
    text: str = "TEXT",
    origin_execution_id: str | None = None,
) -> tuple[OCRAttempt, OCRObservationRecord]:
    spec = _identity_spec()
    input_hash = request.root_input_pixel_sha256
    attempt = OCRAttempt(
        attempt_id="attempt-1",
        run_id=request.run_id,
        origin_execution_id=origin_execution_id or request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=request.root_input_pixel_sha256,
        invocation_id=request.invocation_id,
        provider_family=request.provider_family,
        variant_id="full_page",
        input_pixel_sha256=input_hash,
        parent_input_pixel_sha256=input_hash,
        input_bbox_page=None,
        input_kind="full_page",
        transform_spec=spec,
        input_width=48,
        input_height=32,
        input_mode="RGB",
        provider_called=True,
        cache_hit=False,
    )
    record = OCRObservationRecord(
        observation_id="observation-1",
        attempt_id=attempt.attempt_id,
        run_id=attempt.run_id,
        origin_execution_id=attempt.origin_execution_id,
        page_id=attempt.page_id,
        page_source_sha256=attempt.page_source_sha256,
        root_input_pixel_sha256=attempt.root_input_pixel_sha256,
        input_pixel_sha256=attempt.input_pixel_sha256,
        invocation_id=attempt.invocation_id,
        provider_family=attempt.provider_family,
        variant_id=attempt.variant_id,
        payload_sha256=sha256_text(text),
        text=text,
        confidence=0.96,
        bbox_page=(2, 3, 40, 18),
        polygon_page=((2, 3), (40, 3), (40, 18), (2, 18)),
        source="paddle_full_page",
    )
    return attempt, record


def _atomic_result(request: OCRRequest, text: str = "TEXT") -> OCRInvocationResult:
    attempt, record = _attempt_and_record(request, text=text)
    return OCRInvocationResult.build(
        request=request,
        blocks=(OCRBlock("block-1", text, 0.96, record.bbox_page, record.polygon_page, {"nested": {"value": 1}}),),
        observations=(record,),
        full_page_lines=(record,),
        attempts=(attempt,),
        diagnostics=OCRDiagnostics("paddleocr", {"nested": {"value": 1}}),
    )


def _production_ocr_modules() -> tuple[Path, ...]:
    pipeline_root = Path(__file__).resolve().parents[1]
    return (pipeline_root / "vision_stack" / "ocr.py", pipeline_root / "vision_stack" / "runtime.py")


def find_direct_ocr_provider_call_sites(paths: tuple[Path, ...]) -> set[str]:
    sites: set[str] = set()
    pipeline_root = Path(__file__).resolve().parents[1]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function_stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                function_stack.append(node.name)
                self.generic_visit(node)
                function_stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                function_name = function_stack[-1] if function_stack else "<module>"
                relative = path.relative_to(pipeline_root.parent).as_posix()
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"ocr", "generate", "_processor"}:
                    sites.add(f"{relative}:{function_name}")
                if function_name == "execute_hash_bound_provider_attempt" and isinstance(node.func, ast.Name) and node.func.id == "provider":
                    sites.add(f"{relative}:{function_name}")
                self.generic_visit(node)

        Visitor().visit(tree)
    return sites


class VisionStackOCRTests(unittest.TestCase):
    def _load_paddle_with_fake_modules(self, paddle_log_env: str = ""):
        engine = OCREngine.__new__(OCREngine)
        engine.model_name = "paddleocr"
        engine.lang = "en"
        engine.device = type("Device", (), {"type": "cuda"})()
        engine.half = False
        engine.batch_size = 8
        captured_kwargs = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        paddleocr_module = types.ModuleType("paddleocr")
        paddleocr_module.PaddleOCR = FakePaddleOCR
        paddle_module = types.ModuleType("paddle")
        paddle_base_module = types.ModuleType("paddle.base")
        libpaddle_module = types.ModuleType("paddle.base.libpaddle")
        paddle_module.device = SimpleNamespace(is_compiled_with_cuda=lambda: True)
        paddle_module.__path__ = []
        paddle_base_module.__path__ = []
        paddle_module.base = paddle_base_module
        paddle_base_module.libpaddle = libpaddle_module

        with patch.dict(
            sys.modules,
            {
                "paddleocr": paddleocr_module,
                "paddle": paddle_module,
                "paddle.base": paddle_base_module,
                "paddle.base.libpaddle": libpaddle_module,
            },
        ), patch.dict("os.environ", {"TRADUZAI_PADDLE_SHOW_LOG": paddle_log_env}, clear=False):
            OCREngine._load_paddle_ocr(engine)

        return engine, captured_kwargs

    def test_normalize_paddleocr_language_handles_regions_and_common_languages(self):
        self.assertEqual(normalize_paddleocr_language("en-GB"), "en")
        self.assertEqual(normalize_paddleocr_language("pt-BR"), "pt")
        self.assertEqual(normalize_paddleocr_language("zh-TW"), "chinese_cht")
        self.assertEqual(normalize_paddleocr_language("ja"), "japan")
        self.assertEqual(normalize_paddleocr_language("ko"), "korean")
        self.assertEqual(normalize_paddleocr_language("ru"), "ru")

    def test_normalize_easyocr_languages_handles_regions_and_fallbacks(self):
        self.assertEqual(normalize_easyocr_languages("en-GB"), ["en"])
        self.assertEqual(normalize_easyocr_languages("pt-BR"), ["pt", "en"])
        self.assertEqual(normalize_easyocr_languages("zh-TW"), ["ch_tra", "en"])
        self.assertEqual(normalize_easyocr_languages("ru"), ["ru", "en"])

    def test_manga_ocr_falls_back_to_paddle_when_model_load_breaks(self):
        engine = OCREngine.__new__(OCREngine)
        engine.model_name = "manga-ocr"
        engine.device = type("Device", (), {"type": "cpu"})()
        engine.half = False
        engine.batch_size = 8
        engine._model = None
        engine._processor = None
        original_import = builtins.__import__

        transformers_stub = types.ModuleType("transformers")

        class _BrokenAutoFeatureExtractor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args, kwargs
                raise ValueError("broken hf metadata")

        class _UnusedModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args, kwargs
                raise AssertionError("nao deveria chegar aqui")

        transformers_stub.AutoFeatureExtractor = _BrokenAutoFeatureExtractor
        transformers_stub.VisionEncoderDecoderModel = _UnusedModel
        transformers_stub.AutoTokenizer = _UnusedModel

        with patch("vision_stack.ocr.OCREngine._load_paddle_ocr") as load_paddle, patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: transformers_stub
            if name == "transformers"
            else original_import(name, *args, **kwargs),
        ):
            OCREngine._load_manga_ocr(engine)

        self.assertEqual(engine.model_name, "paddleocr")
        load_paddle.assert_called_once()

    def test_paddle_ocr_retries_empty_result_with_upscaled_variants(self):
        engine = OCREngine.__new__(OCREngine)

        class FakeModel:
            def ocr(self, crop, det=True, rec=True, cls=True):
                h, w = crop.shape[:2]
                if h < 60:
                    return [[]]
                return [[[[0, 0], ("A SINGLE STRIKE, SO I NEVER", 0.99)]]]

        engine._model = FakeModel()
        crop = np.zeros((36, 452, 3), dtype=np.uint8)

        texts = OCREngine._paddle_ocr_batch(engine, [crop])

        self.assertEqual(len(texts), 1)
        self.assertIn("A SINGLE STRIKE", texts[0])

    def test_paddle_ocr_detects_dot_run_when_ocr_returns_empty(self):
        engine = OCREngine.__new__(OCREngine)

        class FakeModel:
            def ocr(self, crop, det=True, rec=True, cls=True):
                return [[]]

        engine._model = FakeModel()
        crop = np.full((32, 84, 3), 255, dtype=np.uint8)
        for x in (13, 25, 37, 49, 61, 73):
            cv2.circle(crop, (x, 20), 3, (0, 0, 0), thickness=-1)

        texts = OCREngine._paddle_ocr_batch(engine, [crop])

        self.assertEqual(texts, ["......"])

    def test_paddle_retry_does_not_request_disabled_angle_classifier(self):
        engine = OCREngine.__new__(OCREngine)
        seen_cls_flags = []

        class FakeModel:
            def ocr(self, crop, det=True, rec=True, cls=True):
                del crop, det, rec
                seen_cls_flags.append(cls)
                return [[]]

        engine._model = FakeModel()
        crop = np.full((36, 84, 3), 255, dtype=np.uint8)

        with patch.object(
            engine,
            "_build_paddle_retry_variants",
            return_value=[crop.copy(), crop.copy()],
        ):
            OCREngine._recognize_single_paddle_with_retry(engine, crop)

        self.assertGreaterEqual(len(seen_cls_flags), 2)
        self.assertEqual(set(seen_cls_flags), {False})

    def test_recognize_batch_uses_crop_cache_when_enabled(self):
        engine = OCREngine.__new__(OCREngine)
        engine.batch_size = 2
        crop = np.full((32, 64, 3), 255, dtype=np.uint8)
        calls = []

        def fake_impl(crops):
            calls.append(len(crops))
            return ["HELLO" for _ in crops]

        engine._recognize_batch_impl = fake_impl

        with patch.dict(os.environ, {"TRADUZAI_OCR_CACHE": "1"}, clear=False):
            self.assertEqual(engine.recognize_batch([crop]), ["HELLO"])
            self.assertEqual(engine.recognize_batch([crop.copy()]), ["HELLO"])

        self.assertEqual(calls, [1])
        self.assertEqual(engine._observation_state().stats["ocr_cache_hits"], 1)

    def test_dedupe_ocr_records_clears_duplicate_lower_confidence_text(self):
        engine = OCREngine.__new__(OCREngine)
        records = [{"text": "HELLO THERE"}, {"text": "HELLO THERE"}]
        blocks = [
            SimpleNamespace(xyxy=(10, 10, 80, 40), confidence=0.91),
            SimpleNamespace(xyxy=(12, 11, 82, 41), confidence=0.80),
        ]

        removed = engine._dedupe_ocr_records_in_place(records, blocks)

        self.assertEqual(removed, 1)
        self.assertEqual(records[0]["text"], "HELLO THERE")
        self.assertEqual(records[1]["text"], "")

    def test_derive_text_pixel_bbox_tracks_tight_text_extent_on_synthetic_text(self):
        page = np.full((180, 320, 3), 255, dtype=np.uint8)
        cv2.putText(
            page,
            "ABC",
            (56, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            2,
            cv2.LINE_8,
        )

        bbox = ocr_mod._derive_text_pixel_bbox(page, [24, 48, 230, 140])

        gray = cv2.cvtColor(page[48:140, 24:230], cv2.COLOR_RGB2GRAY)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(binary_inv > 0)
        expected = [24 + int(xs.min()), 48 + int(ys.min()), 24 + int(xs.max()) + 1, 48 + int(ys.max()) + 1]

        self.assertIsNotNone(bbox)
        self.assertLessEqual(abs(int(bbox[0]) - expected[0]), 2)
        self.assertLessEqual(abs(int(bbox[1]) - expected[1]), 2)
        self.assertLessEqual(abs(int(bbox[2]) - expected[2]), 2)
        self.assertLessEqual(abs(int(bbox[3]) - expected[3]), 2)

    def test_paddle_full_page_blocks_preserve_line_polygons_in_rich_records(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"

        class FakeModel:
            def ocr(self, page_bgr, det=True, rec=True, cls=False):
                del det, rec, cls
                return [[
                    [
                        [[10, 10], [70, 10], [70, 28], [10, 28]],
                        ("HELLO", 0.99),
                    ],
                    [
                        [[12, 34], [88, 34], [88, 50], [12, 50]],
                        ("WORLD", 0.97),
                    ],
                ]]

        engine._model = FakeModel()
        page = np.full((80, 120, 3), 255, dtype=np.uint8)

        records = OCREngine._paddle_ocr_full_page_to_blocks(engine, cv2.cvtColor(page, cv2.COLOR_RGB2BGR), [SimpleNamespace(x1=0, y1=0, x2=120, y2=80)])

        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "HELLO WORLD")
        self.assertIn("line_polygons", records[0])
        self.assertGreaterEqual(len(records[0]["line_polygons"]), 2)
        self.assertIn("text_pixel_bbox", records[0])
        self.assertGreater(records[0]["text_pixel_bbox"][2], records[0]["text_pixel_bbox"][0])

    def test_paddle_full_page_records_rotation_from_slanted_line_polygons(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"

        class FakeModel:
            def ocr(self, page_bgr, det=True, rec=True, cls=False):
                del page_bgr, det, rec, cls
                return [[
                    [
                        [[40, 20], [56, 16], [96, 136], [80, 140]],
                        ("TILTED", 0.96),
                    ],
                ]]

        engine._model = FakeModel()
        page = np.full((180, 160, 3), 210, dtype=np.uint8)

        records = OCREngine._paddle_ocr_full_page_to_blocks(
            engine,
            cv2.cvtColor(page, cv2.COLOR_RGB2BGR),
            [SimpleNamespace(x1=20, y1=0, x2=130, y2=160)],
            allow_sparse_mapping=True,
        )

        self.assertIsInstance(records, list)
        assert records is not None
        self.assertEqual(records[0]["text"], "TILTED")
        self.assertEqual(records[0]["rotation_source"], "line_polygons")
        self.assertGreater(records[0]["rotation_deg"], 65.0)
        self.assertLess(records[0]["rotation_deg"], 80.0)

    def test_paddle_full_page_deskews_diagonal_text_when_initial_pass_misses_line(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def ocr(self, page_bgr, det=True, rec=True, cls=False):
                del page_bgr, det, rec, cls
                self.calls += 1
                if self.calls == 1:
                    return [[
                        [
                            [[44, 133], [140, 78], [154, 104], [58, 159]],
                            ("AJUMMA,", 0.98),
                        ],
                        [
                            [[91, 194], [289, 64], [305, 89], [107, 219]],
                            ("USE YOUR MONEY.", 0.94),
                        ],
                        [
                            [[116, 222], [319, 86], [334, 111], [131, 246]],
                            ("WHAT'S UP WITH THE", 0.92),
                        ],
                        [
                            [[142, 249], [327, 129], [341, 152], [156, 273]],
                            ("KID'S EDUCATION?", 0.95),
                        ],
                        [
                            [[164, 278], [257, 219], [271, 243], [178, 303]],
                            ("BE WELL.", 0.98),
                        ],
                    ]]
                return [[
                    [[[40, 40], [120, 40], [120, 60], [40, 60]], ("AJUMMA,", 0.98)],
                    [[[40, 66], [220, 66], [220, 86], [40, 86]], ("FROM NOW ON, DON'T", 0.93)],
                    [[[40, 92], [210, 92], [210, 112], [40, 112]], ("USE YOUR MONEY.", 0.94)],
                    [[[40, 118], [220, 118], [220, 138], [40, 138]], ("WHAT'S UP WITH THE", 0.92)],
                    [[[40, 144], [205, 144], [205, 164], [40, 164]], ("KID'S EDUCATION?", 0.95)],
                    [[[40, 170], [130, 170], [130, 190], [40, 190]], ("BE WELL.", 0.98)],
                ]]

        engine._model = FakeModel()
        page = np.full((360, 360, 3), 255, dtype=np.uint8)

        records = OCREngine._paddle_ocr_full_page_to_blocks(
            engine,
            cv2.cvtColor(page, cv2.COLOR_RGB2BGR),
            [SimpleNamespace(x1=0, y1=0, x2=360, y2=360)],
            allow_sparse_mapping=True,
        )

        self.assertIsInstance(records, list)
        assert records is not None
        self.assertIn("FROM NOW ON", records[0]["text"])
        self.assertIn("DON'T", records[0]["text"])
        self.assertGreaterEqual(len(records[0]["line_polygons"]), 6)
        self.assertEqual(records[0]["rotation_source"], "line_polygons")
        self.assertLess(records[0]["rotation_deg"], -20.0)
        self.assertIn("skewed_text_deskew_recovery", records[0].get("qa_flags", []))

    def test_paddle_full_page_resize_can_use_experimental_gpu_preprocess_hook(self):
        from vision_stack import gpu_image_ops

        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        seen_shapes = []

        class FakeModel:
            def ocr(self, page_bgr, det=True, rec=True, cls=False):
                del det, rec, cls
                seen_shapes.append(page_bgr.shape[:2])
                return [[
                    [
                        [[4, 4], [24, 4], [24, 12], [4, 12]],
                        ("HELLO", 0.99),
                    ],
                ]]

        engine._model = FakeModel()
        page = np.full((160, 320, 3), 255, dtype=np.uint8)

        with patch.dict(
            os.environ,
            {
                "TRADUZAI_PADDLE_FULL_PAGE_MAX_SIDE": "80",
                "TRADUZAI_EXPERIMENTAL_GPU_OCR_PREPROCESS": "1",
                "TRADUZAI_GPU_IMAGE_OPS_BACKEND": "cpu",
            },
            clear=False,
        ), patch.object(gpu_image_ops, "resize_crops_batch", wraps=gpu_image_ops.resize_crops_batch) as resize_spy:
            records = OCREngine._paddle_ocr_full_page_to_blocks(
                engine,
                cv2.cvtColor(page, cv2.COLOR_RGB2BGR),
                [SimpleNamespace(x1=0, y1=0, x2=320, y2=160)],
                allow_sparse_mapping=True,
            )

        self.assertEqual(seen_shapes, [(40, 80)])
        self.assertIsInstance(records, list)
        self.assertGreaterEqual(resize_spy.call_count, 1)
        self.assertEqual(records[0]["text"], "HELLO")

    def test_recognize_blocks_from_page_crop_fallback_updates_rich_record_text(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        block = type("Block", (), {"xyxy": (10, 10, 70, 34), "confidence": 0.91})()

        with patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=[{"text": "", "source_bbox": [10, 10, 70, 34], "line_polygons": []}],
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            return_value="HELLO",
        ):
            records = engine.recognize_blocks_from_page(np.full((80, 120, 3), 255, dtype=np.uint8), [block])

        self.assertEqual(records[0]["text"], "HELLO")
        self.assertEqual(records[0]["source_bbox"], [10, 10, 70, 34])

    def test_load_paddle_ocr_fails_closed_when_paddle_is_unavailable(self):
        engine = OCREngine.__new__(OCREngine)
        engine.model_name = "paddleocr"
        engine.lang = "en"
        engine.device = type("Device", (), {"type": "cpu"})()
        engine.half = False
        engine.batch_size = 8
        engine._model = None
        engine._processor = None
        original_import = builtins.__import__

        with patch("vision_stack.ocr.OCREngine._load_easyocr") as load_easyocr, patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError(name))
            if name == "paddleocr"
            else original_import(name, *args, **kwargs),
        ):
            with self.assertRaisesRegex(RuntimeError, "PaddleOCR"):
                OCREngine._load_paddle_ocr(engine)

        self.assertEqual(engine.model_name, "paddleocr")
        load_easyocr.assert_not_called()

    def test_load_paddle_ocr_forces_gpu_when_gpu_is_required(self):
        engine = OCREngine.__new__(OCREngine)
        engine.model_name = "paddleocr"
        engine.lang = "en"
        engine.device = type("Device", (), {"type": "cpu"})()
        engine.half = False
        engine.batch_size = 8
        captured_kwargs = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        paddleocr_module = types.ModuleType("paddleocr")
        paddleocr_module.PaddleOCR = FakePaddleOCR
        paddle_module = types.ModuleType("paddle")
        paddle_base_module = types.ModuleType("paddle.base")
        libpaddle_module = types.ModuleType("paddle.base.libpaddle")
        paddle_module.device = SimpleNamespace(is_compiled_with_cuda=lambda: True)
        paddle_module.__path__ = []
        paddle_base_module.__path__ = []
        paddle_module.base = paddle_base_module
        paddle_base_module.libpaddle = libpaddle_module

        with patch.dict(
            sys.modules,
            {
                "paddleocr": paddleocr_module,
                "paddle": paddle_module,
                "paddle.base": paddle_base_module,
                "paddle.base.libpaddle": libpaddle_module,
            },
        ), patch.dict("os.environ", {"TRADUZAI_REQUIRE_GPU": "1"}, clear=False):
            OCREngine._load_paddle_ocr(engine)

        self.assertEqual(engine.device.type, "cuda")
        self.assertEqual(engine._backend, "paddleocr")
        self.assertTrue(captured_kwargs["use_gpu"])

    def test_load_paddle_ocr_does_not_fall_back_to_easyocr_when_gpu_is_required(self):
        engine = OCREngine.__new__(OCREngine)
        engine.model_name = "paddleocr"
        engine.lang = "en"
        engine.device = type("Device", (), {"type": "cpu"})()
        engine.half = False
        engine.batch_size = 8
        original_import = builtins.__import__

        with patch("vision_stack.ocr.OCREngine._load_easyocr") as load_easyocr, patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError(name))
            if name == "paddle"
            else original_import(name, *args, **kwargs),
        ), patch.dict("os.environ", {"TRADUZAI_REQUIRE_GPU": "1"}, clear=False):
            with self.assertRaises(RuntimeError):
                OCREngine._load_paddle_ocr(engine)

        load_easyocr.assert_not_called()

    def test_load_paddle_ocr_disables_paddle_console_logs_by_default(self):
        engine, captured_kwargs = self._load_paddle_with_fake_modules()

        self.assertEqual(engine._backend, "paddleocr")
        self.assertIs(captured_kwargs["show_log"], False)

    def test_load_paddle_ocr_can_enable_paddle_console_logs_for_debugging(self):
        _engine, captured_kwargs = self._load_paddle_with_fake_modules("1")

        self.assertIs(captured_kwargs["show_log"], True)


class PaddleBlockMappingTests(unittest.TestCase):
    def _engine_with_lines(self, lines):
        class FakePaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                return [lines]

        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = FakePaddleModel()
        engine.batch_size = 8
        return engine

    def test_full_page_ocr_runs_with_empty_detector_blocks(self):
        engine = self._engine_with_lines(
            [([[10, 10], [90, 10], [90, 30], [10, 30]], ("VISIBLE ENGLISH", 0.97))]
        )
        result = engine.recognize_page_with_evidence(
            _contract_page(),
            [],
            request=_contract_request(),
            force_full_page=True,
        )

        self.assertEqual([line.text for line in result.full_page_lines], ["VISIBLE ENGLISH"])

    def test_crop_first_reads_detector_and_unassigned_pixels_once(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                received.append(image.copy())
                if image.shape[:2] == (280, 300):
                    return _raw_ocr("COMPLETE TEXT INSIDE BALLOON")
                return [[
                    ([[350, 350], [394, 350], [394, 375], [350, 375]], ("OUTSIDE", 0.96)),
                    ([[12, 22], [35, 22], [35, 36], [12, 36]], ("MASK ARTIFACT", 0.40)),
                ]]

        page = np.full((400, 400, 3), 27, dtype=np.uint8)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        block = SimpleNamespace(xyxy=(10, 20, 310, 300))
        small_block = SimpleNamespace(xyxy=(340, 340, 399, 390))
        with patch.dict(os.environ, {"TRADUZAI_OCR_CROP_FIRST": "1"}):
            result = engine.recognize_page_with_evidence(
                page, [block, small_block], request=_contract_request(root=page),
            )

        self.assertEqual(len(received), 2)
        self.assertTrue(np.array_equal(received[0], page[20:300, 10:310]))
        self.assertTrue(np.all(received[1][20:300, 10:310] == 255))
        self.assertTrue(np.all(received[1][350:375, 350:394] == 27))
        self.assertEqual(result.blocks[0].text, "COMPLETE TEXT INSIDE BALLOON")
        self.assertEqual(result.blocks[1].text, "OUTSIDE")
        self.assertEqual([line.text for line in result.observations], ["COMPLETE TEXT INSIDE BALLOON", "OUTSIDE"])
        self.assertEqual(result.observations[0].bbox_page, (12, 23, 50, 38))
        for attempt, physical in zip(result.attempts, received):
            self.assertEqual(attempt.input_pixel_sha256, canonical_page_sha256(physical))
            self.assertTrue(np.array_equal(attempt.transform_spec.replay(page), physical))

    def test_crop_first_deskews_supported_slanted_lines_with_replayable_pixels(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                received.append(image.copy())
                return _raw_ocr("SLANTED LETTER CONTENT IS READ") if len(received) == 1 else []

        page = np.full((420, 420, 3), 255, dtype=np.uint8)
        for index in range(6):
            cv2.line(page, (40, 180 + index * 20), (260, 70 + index * 20), (10, 10, 10), 2)
        native = page[20:320, 10:310]
        angle = OCREngine._crop_text_skew_angle(native)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, -26.5, delta=2.0)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        with patch.dict(os.environ, {"TRADUZAI_OCR_CROP_FIRST": "1"}):
            result = engine.recognize_page_with_evidence(
                page, [SimpleNamespace(xyxy=(10, 20, 310, 320))],
                request=_contract_request(root=page),
            )
        self.assertEqual(len(received), 2)
        self.assertEqual(result.attempts[0].transform_spec.operations[1].kind, "deskew_affine")
        self.assertTrue(np.array_equal(result.attempts[0].transform_spec.replay(page), received[0]))
        self.assertEqual(result.attempts[0].input_pixel_sha256, canonical_page_sha256(received[0]))
        self.assertEqual(result.blocks[0].text, "SLANTED LETTER CONTENT IS READ")
        self.assertTrue(all(10 <= x <= 310 and 20 <= y <= 320 for x, y in result.observations[0].polygon_page))

    def test_terminal_ink_gate_rejects_distant_and_non_punctuation_artifacts(self):
        polygon = ((20, 30), (120, 30), (120, 50), (20, 50))
        image = np.full((90, 170, 3), 255, dtype=np.uint8)
        image[46:48, 123:125] = 0
        self.assertTrue(OCREngine._terminal_ink_outside_line(image, polygon))
        image[46:48, 123:125] = 255
        image[8:10, 123:125] = 0
        self.assertFalse(OCREngine._terminal_ink_outside_line(image, polygon))
        image[8:10, 123:125] = 255
        image[33:45, 123:125] = 0
        self.assertFalse(OCREngine._terminal_ink_outside_line(image, polygon))
        image[33:45, 123:125] = 255
        image[46:48, 145:147] = 0
        self.assertFalse(OCREngine._terminal_ink_outside_line(image, polygon))

    def test_crop_first_preserves_full_page_path_for_small_regions(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                received.append(image.copy())
                return _raw_ocr("SMALL")

        page = np.full((100, 100, 3), 27, dtype=np.uint8)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        with patch.dict(os.environ, {"TRADUZAI_OCR_CROP_FIRST": "1"}):
            result = engine.recognize_page_with_evidence(
                page, [SimpleNamespace(xyxy=(0, 0, 60, 60))], request=_contract_request(root=page),
            )
        self.assertEqual(len(received), 1)
        self.assertTrue(np.array_equal(received[0], page))
        self.assertEqual(result.blocks[0].text, "SMALL")

    def test_empty_detected_crop_remains_visible_to_page_ocr(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                received.append(image.copy())
                return [] if image.shape[:2] == (280, 300) else _raw_ocr("RECOVERED")

        page = np.full((400, 400, 3), 27, dtype=np.uint8)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        with patch.dict(os.environ, {"TRADUZAI_OCR_CROP_FIRST": "1"}):
            result = engine.recognize_page_with_evidence(
                page, [SimpleNamespace(xyxy=(10, 20, 310, 300))],
                request=_contract_request(root=page),
            )
        self.assertEqual(len(received), 2)
        self.assertTrue(np.array_equal(received[1], page))
        self.assertEqual([line.text for line in result.observations], ["RECOVERED"])

    def test_short_fragment_does_not_mask_large_sfx_region(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                received.append(image.copy())
                return _raw_ocr("oF") if image.shape[:2] == (280, 300) else []

        page = np.full((400, 400, 3), 27, dtype=np.uint8)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        with patch.dict(os.environ, {"TRADUZAI_OCR_CROP_FIRST": "1"}):
            result = engine.recognize_page_with_evidence(
                page, [SimpleNamespace(xyxy=(10, 20, 310, 300))], request=_contract_request(root=page),
            )
        self.assertTrue(np.array_equal(received[1], page))
        self.assertEqual(result.observations, ())
        self.assertEqual(result.blocks[0].text, "")

    def test_overlapping_boxes_keep_single_full_page_read(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                received.append(image.copy())
                return _raw_ocr("SHARED")

        page = np.full((400, 400, 3), 27, dtype=np.uint8)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        blocks = [SimpleNamespace(xyxy=(10, 20, 310, 300)), SimpleNamespace(xyxy=(20, 30, 70, 80))]
        with patch.dict(os.environ, {"TRADUZAI_OCR_CROP_FIRST": "1"}):
            result = engine.recognize_page_with_evidence(page, blocks, request=_contract_request(root=page))
        self.assertEqual(len(received), 1)
        self.assertTrue(np.array_equal(received[0], page))
        self.assertEqual(len(result.observations), 1)

    def test_mask_rectangles_rejects_out_of_bounds_at_replay(self):
        spec = OCRTransformSpec.build((
            OCRTransformOperation(kind="mask_rectangles", mask_bboxes=((0, 0, 200, 10),)),
        ))
        with self.assertRaises(ValueError):
            spec.replay(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_full_page_attempt_hashes_exact_array_passed_to_provider(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del det, rec, cls
                received.append(image.copy())
                return _raw_ocr("DOWNSCALED")

        page = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        result = engine.recognize_page_with_evidence(
            page,
            [],
            request=_contract_request(root=page),
            force_downscale=True,
        )
        attempt = result.attempts[0]

        self.assertEqual(attempt.root_input_pixel_sha256, canonical_page_sha256(page))
        self.assertEqual(attempt.input_pixel_sha256, canonical_page_sha256(received[0]))
        self.assertNotEqual(attempt.input_pixel_sha256, attempt.root_input_pixel_sha256)
        self.assertEqual((attempt.input_width, attempt.input_height), (48, 32))

    def test_anchored_crop_attempt_hashes_crop_pixels_not_page_pixels(self):
        page = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
        engine = self._engine_with_lines(
            [([[1, 1], [30, 1], [30, 12], [1, 12]], ("CROP", 0.95))]
        )
        result = engine.recognize_region_with_evidence(
            page,
            bbox_page=(7, 11, 83, 41),
            request=_contract_request(root=page, invocation_id="anchored-crop"),
            variants=("anchored_crop",),
        )
        attempt = result.attempts[0]
        expected_crop = page[11:41, 7:83]

        self.assertEqual(attempt.root_input_pixel_sha256, canonical_page_sha256(page))
        self.assertEqual(attempt.input_pixel_sha256, canonical_page_sha256(expected_crop))
        self.assertNotEqual(attempt.input_pixel_sha256, attempt.root_input_pixel_sha256)
        self.assertEqual(attempt.input_bbox_page, (7, 11, 83, 41))

    def test_each_retry_variant_hashes_exact_provider_pixels(self):
        received = []

        class CapturingPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del det, rec, cls
                received.append(image.copy())
                return _raw_ocr("VARIANT")

        page = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = CapturingPaddleModel()
        result = engine.recognize_region_with_evidence(
            page,
            bbox_page=(7, 11, 83, 41),
            request=_contract_request(root=page, invocation_id="retry-variants"),
            variants=("native", "gray", "inverted", "scale_2x"),
        )

        self.assertEqual(len(result.attempts), 4)
        for attempt, provider_rgb in zip(result.attempts, received):
            with self.subTest(variant=attempt.variant_id):
                self.assertEqual(attempt.input_pixel_sha256, canonical_page_sha256(provider_rgb))
                self.assertEqual(
                    (attempt.input_width, attempt.input_height),
                    (provider_rgb.shape[1], provider_rgb.shape[0]),
                )
                self.assertEqual(
                    attempt.transform_spec.sha256,
                    sha256_bytes(attempt.transform_spec.canonical_json_bytes),
                )
                self.assertTrue(np.array_equal(attempt.transform_spec.replay(page), provider_rgb))

    def test_full_page_line_records_are_request_scoped_across_parallel_pages(self):
        barrier = __import__("threading").Barrier(2)

        class FakePaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del det, rec, cls
                marker = int(image[0, 0, 0])
                barrier.wait(timeout=5)
                return _raw_ocr("PAGE A" if marker == 25 else "PAGE B")

        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = FakePaddleModel()
        engine.batch_size = 8
        page_a = _contract_page(25)
        page_b = _contract_page(50)

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(engine.recognize_page_with_evidence, page_a, [], request=_contract_request("page_001", root=page_a))
            future_b = pool.submit(engine.recognize_page_with_evidence, page_b, [], request=_contract_request("page_002", root=page_b))
        result_a = future_a.result()
        result_b = future_b.result()
        self.assertEqual({line.text for line in result_a.full_page_lines}, {"PAGE A"})
        self.assertEqual({line.text for line in result_b.full_page_lines}, {"PAGE B"})
        self.assertEqual(result_a.page_id, "page_001")
        self.assertEqual(result_b.page_id, "page_002")

    def test_ocr_invocation_result_is_deeply_immutable(self):
        result = _atomic_result(_contract_request(), "TEXT")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.blocks[0].text = "MUTATED"
        with self.assertRaises(TypeError):
            result.blocks[0].extras["nested"] = "MUTATED"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.diagnostics.provider = "other"
        with self.assertRaises(TypeError):
            result.diagnostics.extras["nested"] = "MUTATED"

    def test_every_ocr_record_links_exact_logical_request_and_physical_attempt(self):
        request = _contract_request("page_010", run_id="run-a", origin_execution_id="exec-a")
        result = _atomic_result(request, "CURRENT PAGE")
        attempts = {attempt.attempt_id: attempt for attempt in result.attempts}
        for record in (*result.observations, *result.full_page_lines):
            self.assertEqual(record.request_identity, request.identity)
            attempt = attempts[record.attempt_id]
            self.assertEqual(record.attempt_identity, attempt.identity)
            self.assertEqual(record.root_input_pixel_sha256, request.root_input_pixel_sha256)
            self.assertEqual(record.input_pixel_sha256, attempt.input_pixel_sha256)

    def test_runtime_record_preserves_complete_atomic_identity(self):
        from vision_stack.runtime import _atomic_ocr_record_to_runtime_dict

        request = _contract_request("page_010", run_id="run-a", origin_execution_id="exec-a")
        _attempt, record = _attempt_and_record(request, text="CURRENT PAGE")

        payload = _atomic_ocr_record_to_runtime_dict(record)

        for field_name in (
            "run_id",
            "origin_execution_id",
            "page_id",
            "page_source_sha256",
            "root_input_pixel_sha256",
            "input_pixel_sha256",
            "invocation_id",
            "attempt_id",
            "provider_family",
            "variant_id",
            "payload_sha256",
        ):
            self.assertEqual(payload[field_name], getattr(record, field_name))

    def test_ocr_result_rejects_record_with_mismatched_request_identity(self):
        request = _contract_request("page_010", run_id="run-a")
        stale_request = _contract_request("page_010", run_id="run-previous")
        attempt, stale = _attempt_and_record(stale_request)
        with self.assertRaises(OCRRequestIdentityError):
            OCRInvocationResult.build(request=request, observations=(stale,), attempts=(attempt,))

    def test_ocr_result_rejects_record_from_other_execution_under_same_content_run(self):
        request = _contract_request("page_010", run_id="run-a", origin_execution_id="exec-a")
        stale_request = _contract_request("page_010", run_id="run-a", origin_execution_id="exec-b")
        attempt, stale = _attempt_and_record(stale_request)
        with self.assertRaises(OCRRequestIdentityError):
            OCRInvocationResult.build(request=request, observations=(stale,), attempts=(attempt,))

    def test_provider_boundary_rejects_declared_hash_b_when_pixels_a_are_passed(self):
        pixels_a = _contract_page(25)
        pixels_b = _contract_page(50)
        for physical_kind in ("full_page", "anchored_crop", "native", "gray", "inverted", "scale_2x", "final_observer", "external_auditor"):
            with self.subTest(physical_kind=physical_kind):
                provider = MagicMock()
                with self.assertRaises(OCRInputPixelIdentityError):
                    execute_hash_bound_provider_attempt(
                        request=_contract_request(root=pixels_a),
                        root_input_rgb=pixels_a,
                        actual_input_rgb=pixels_a,
                        expected_input_pixel_sha256=canonical_page_sha256(pixels_b),
                        variant_id=physical_kind,
                        transform_spec=_identity_spec(),
                        provider=provider,
                    )
                provider.assert_not_called()

    def test_provider_boundary_records_hash_of_actual_array_on_success(self):
        root = _contract_page(25)
        spec, transformed = _scale_2x_spec(root)
        provider = MagicMock(return_value=_raw_ocr("TEXT"))
        attempt, records = execute_hash_bound_provider_attempt(
            request=_contract_request(root=root),
            root_input_rgb=root,
            actual_input_rgb=transformed,
            expected_input_pixel_sha256=canonical_page_sha256(transformed),
            variant_id="scale_2x",
            transform_spec=spec,
            provider=provider,
        )
        self.assertEqual(attempt.input_pixel_sha256, canonical_page_sha256(transformed))
        self.assertTrue(all(record.attempt_id == attempt.attempt_id for record in records))
        self.assertTrue(all(record.input_pixel_sha256 == attempt.input_pixel_sha256 for record in records))
        self.assertIs(attempt.provider_called, True)
        self.assertIs(attempt.cache_hit, False)
        provider.assert_called_once()

    def test_parameterized_rotation_and_deskew_specs_replay_exact_provider_pixels(self):
        root = _contract_page(25)
        for kind in ("rotate_affine", "deskew_affine"):
            with self.subTest(kind=kind):
                spec, expected_rgb = _affine_spec(root, kind)
                provider = MagicMock(return_value=_raw_ocr("TEXT"))
                attempt, _ = execute_hash_bound_provider_attempt(
                    request=_contract_request(root=root),
                    root_input_rgb=root,
                    actual_input_rgb=expected_rgb,
                    expected_input_pixel_sha256=canonical_page_sha256(expected_rgb),
                    variant_id=kind,
                    transform_spec=spec,
                    provider=provider,
                )
                self.assertTrue(np.array_equal(spec.replay(root), provider.call_args.args[0]))
                self.assertEqual(attempt.transform_spec.canonical_json_bytes, spec.canonical_json_bytes)

    def test_cache_or_dot_heuristic_attempt_is_not_fresh_physical_ocr(self):
        request = _contract_request()
        attempt, _ = _attempt_and_record(request)
        cached = dataclasses.replace(attempt, provider_called=False, cache_hit=True)
        self.assertIs(cached.qualifies_as_fresh_physical_inference, False)

    def test_production_provider_calls_exist_only_inside_hash_bound_boundary(self):
        self.assertEqual(
            find_direct_ocr_provider_call_sites(_production_ocr_modules()),
            {"pipeline/vision_stack/ocr.py:execute_hash_bound_provider_attempt"},
        )

    def test_sparse_block_mapping_is_rejected_by_default(self):
        engine = self._engine_with_lines(
            [
                (
                    [[10, 10], [50, 10], [50, 30], [10, 30]],
                    ("HELLO", 0.91),
                )
            ]
        )
        image = np.full((100, 220, 3), 255, dtype=np.uint8)
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34)),
            SimpleNamespace(xyxy=(70, 8, 118, 34)),
            SimpleNamespace(xyxy=(140, 8, 190, 34)),
        ]

        mapped = engine._paddle_ocr_full_page_to_blocks(image, blocks)

        self.assertIsNone(mapped)

    def test_sparse_block_mapping_can_be_accepted_for_strip_bands(self):
        engine = self._engine_with_lines(
            [
                (
                    [[10, 10], [50, 10], [50, 30], [10, 30]],
                    ("HELLO", 0.91),
                )
            ]
        )
        image = np.full((100, 220, 3), 255, dtype=np.uint8)
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34)),
            SimpleNamespace(xyxy=(70, 8, 118, 34)),
            SimpleNamespace(xyxy=(140, 8, 190, 34)),
        ]

        mapped = engine._paddle_ocr_full_page_to_blocks(
            image,
            blocks,
            allow_sparse_mapping=True,
        )

        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertEqual(mapped[0]["text"], "HELLO")
        self.assertEqual(mapped[1]["text"], "")
        self.assertEqual(mapped[2]["text"], "")

    def test_full_page_mapping_preserves_line_texts_for_multiline_block(self):
        engine = self._engine_with_lines(
            [
                (
                    [[42, 44], [86, 44], [86, 58], [42, 58]],
                    ("Name", 0.91),
                ),
                (
                    [[34, 72], [112, 72], [112, 86], [34, 86]],
                    ("Resident", 0.90),
                ),
                (
                    [[18, 88], [154, 88], [154, 102], [18, 102]],
                    ("registration number", 0.89),
                ),
            ]
        )
        image = np.full((130, 360, 3), 255, dtype=np.uint8)
        blocks = [SimpleNamespace(xyxy=(18, 40, 154, 104))]

        mapped = engine._paddle_ocr_full_page_to_blocks(image, blocks)

        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertEqual(mapped[0]["text"], "Name Resident registration number")
        self.assertEqual(mapped[0]["line_texts"], ["Name", "Resident", "registration number"])

    def test_full_page_mapping_audits_spatially_disconnected_lines_without_changing_mapping(self):
        engine = self._engine_with_lines(
            [
                (
                    [[20, 20], [100, 20], [100, 42], [20, 42]],
                    ("UPPER", 0.95),
                ),
                (
                    [[20, 210], [120, 210], [120, 232], [20, 232]],
                    ("LOWER", 0.94),
                ),
            ]
        )
        image = np.full((280, 220, 3), 255, dtype=np.uint8)
        blocks = [SimpleNamespace(xyxy=(10, 10, 160, 250))]

        with patch.dict(os.environ, {"TRADUZAI_FLAG_OCR_ASSIGNMENT_AUDIT_V2": "1"}, clear=False):
            mapped = engine._paddle_ocr_full_page_to_blocks(image, blocks)

        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertEqual(mapped[0]["text"], "UPPER LOWER")
        self.assertEqual(mapped[0]["source_bbox"], [20, 20, 120, 232])
        audit = mapped[0]["_ocr_assignment_audit"]
        self.assertTrue(audit["suspicious"])
        self.assertEqual(audit["reason"], "large_vertical_gap")
        self.assertEqual(audit["assigned_line_count"], 2)
        self.assertGreater(audit["max_vertical_gap_over_median_height"], 4.0)

    def test_full_page_mapping_audit_accepts_tight_multiline_text(self):
        engine = self._engine_with_lines(
            [
                (
                    [[20, 20], [100, 20], [100, 42], [20, 42]],
                    ("UPPER", 0.95),
                ),
                (
                    [[20, 50], [120, 50], [120, 72], [20, 72]],
                    ("LOWER", 0.94),
                ),
            ]
        )
        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        blocks = [SimpleNamespace(xyxy=(10, 10, 160, 90))]

        with patch.dict(os.environ, {"TRADUZAI_FLAG_OCR_ASSIGNMENT_AUDIT_V2": "1"}, clear=False):
            mapped = engine._paddle_ocr_full_page_to_blocks(image, blocks)

        self.assertIsNotNone(mapped)
        assert mapped is not None
        audit = mapped[0]["_ocr_assignment_audit"]
        self.assertFalse(audit["suspicious"])
        self.assertEqual(audit["reason"], "spatially_coherent")

    def test_full_page_mapping_omits_assignment_audit_when_flag_is_disabled(self):
        engine = self._engine_with_lines(
            [
                (
                    [[20, 20], [100, 20], [100, 42], [20, 42]],
                    ("HELLO", 0.95),
                ),
            ]
        )
        image = np.full((100, 220, 3), 255, dtype=np.uint8)
        blocks = [SimpleNamespace(xyxy=(10, 10, 160, 90))]

        with patch.dict(os.environ, {"TRADUZAI_FLAG_OCR_ASSIGNMENT_AUDIT_V2": "0"}, clear=False):
            mapped = engine._paddle_ocr_full_page_to_blocks(image, blocks)

        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertNotIn("_ocr_assignment_audit", mapped[0])

    def test_full_page_mapping_downscales_large_band_and_restores_coordinates(self):
        captured_shapes = []

        class FakePaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                captured_shapes.append(tuple(image.shape[:2]))
                return [[
                    (
                        [[80, 80], [200, 80], [200, 120], [80, 120]],
                        ("HELLO", 0.95),
                    )
                ]]

        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine._model = FakePaddleModel()
        engine.batch_size = 8
        image = np.full((400, 1200, 3), 255, dtype=np.uint8)
        blocks = [SimpleNamespace(xyxy=(120, 120, 420, 260))]

        with patch.dict(os.environ, {"TRADUZAI_PADDLE_FULL_PAGE_MAX_SIDE": "600"}, clear=False):
            mapped = engine._paddle_ocr_full_page_to_blocks(image, blocks, allow_sparse_mapping=True)

        self.assertEqual(captured_shapes[0], (200, 600))
        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertEqual(mapped[0]["text"], "HELLO")
        self.assertEqual(mapped[0]["source_bbox"], [160, 160, 400, 240])
        self.assertEqual(mapped[0]["line_polygons"][0][0], [160, 160])

    def test_crop_fallback_max_limits_empty_mapped_blocks(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34), confidence=0.9),
            SimpleNamespace(xyxy=(70, 8, 118, 34), confidence=0.9),
        ]
        mapped_records = [
            {"text": "", "source_bbox": [], "line_polygons": []},
            {"text": "", "source_bbox": [], "line_polygons": []},
        ]

        with patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=mapped_records,
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            side_effect=["HELLO", "WORLD"],
        ) as recognize:
            records = engine.recognize_blocks_from_page(
                np.full((100, 220, 3), 255, dtype=np.uint8),
                blocks,
                crop_fallback_max=1,
            )

        recognize.assert_called_once()
        self.assertEqual(records[0]["text"], "HELLO")
        self.assertEqual(records[1]["text"], "")
        self.assertEqual(engine._observation_state().stats["crop_fallback_attempts"], 1)
        self.assertEqual(engine._observation_state().stats["crop_fallback_recovered"], 1)

    def test_crop_fallback_shadow_records_recoveries_after_simulated_limit(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34), confidence=0.9),
            SimpleNamespace(xyxy=(70, 8, 118, 34), confidence=0.9),
            SimpleNamespace(xyxy=(132, 8, 180, 34), confidence=0.9),
        ]
        mapped_records = [
            {"text": "", "source_bbox": [], "line_polygons": []},
            {"text": "", "source_bbox": [], "line_polygons": []},
            {"text": "", "source_bbox": [], "line_polygons": []},
        ]

        with patch.dict(
            os.environ,
            {
                "TRADUZAI_OCR_FALLBACK_SHADOW": "1",
                "TRADUZAI_OCR_FALLBACK_SHADOW_MAX": "1",
            },
            clear=False,
        ), patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=mapped_records,
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            side_effect=["", "SECOND", "THIRD"],
        ):
            records = engine.recognize_blocks_from_page(
                np.full((100, 220, 3), 255, dtype=np.uint8),
                blocks,
                crop_fallback_max=3,
            )

        self.assertEqual([record["text"] for record in records], ["", "SECOND", "THIRD"])
        self.assertEqual(engine._observation_state().stats["crop_fallback_attempts"], 3)
        self.assertEqual(engine._observation_state().stats["crop_fallback_recovered"], 2)
        self.assertEqual(engine._observation_state().stats["fallback_shadow_attempt_limit"], 1)
        self.assertEqual(
            engine._observation_state().stats["fallback_shadow_attempts_saved_or_would_skip"],
            2,
        )
        self.assertEqual(engine._observation_state().stats["fallback_shadow_recovered_after_limit"], 2)
        self.assertEqual(
            engine._observation_state().stats["fallback_shadow_full_page_already_resolved_count"],
            0,
        )

    def test_crop_fallback_shadow_stats_are_absent_by_default(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34), confidence=0.9),
            SimpleNamespace(xyxy=(70, 8, 118, 34), confidence=0.9),
        ]
        mapped_records = [
            {"text": "", "source_bbox": [], "line_polygons": []},
            {"text": "", "source_bbox": [], "line_polygons": []},
        ]

        with patch.dict(os.environ, {"TRADUZAI_OCR_FALLBACK_SHADOW": "0"}, clear=False), patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=mapped_records,
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            side_effect=["HELLO", "WORLD"],
        ):
            records = engine.recognize_blocks_from_page(
                np.full((100, 220, 3), 255, dtype=np.uint8),
                blocks,
                crop_fallback_max=2,
            )

        self.assertEqual([record["text"] for record in records], ["HELLO", "WORLD"])
        for key in (
            "fallback_shadow_attempt_limit",
            "fallback_shadow_attempts_saved_or_would_skip",
            "fallback_shadow_recovered_after_limit",
            "fallback_shadow_full_page_already_resolved_count",
        ):
            self.assertNotIn(key, engine._observation_state().stats)

    def test_sparse_crop_fallback_can_be_disabled_separately(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34), confidence=0.9),
            SimpleNamespace(xyxy=(70, 8, 118, 34), confidence=0.9),
        ]
        mapped_records = [
            {"text": "", "source_bbox": [], "line_polygons": []},
            {"text": "", "source_bbox": [], "line_polygons": []},
        ]

        with patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=mapped_records,
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            return_value="SHOULD NOT RUN",
        ) as recognize:
            records = engine.recognize_blocks_from_page(
                np.full((100, 220, 3), 255, dtype=np.uint8),
                blocks,
                crop_fallback_max=3,
                sparse_crop_fallback_max=0,
            )

        recognize.assert_not_called()
        self.assertEqual(records[0]["text"], "")
        self.assertEqual(records[1]["text"], "")
        self.assertEqual(engine._observation_state().stats["crop_fallback_attempts"], 0)
        self.assertEqual(engine._observation_state().stats["crop_fallback_recovered"], 0)
        self.assertEqual(engine._observation_state().stats["crop_fallback_suppressed"], 2)
        self.assertEqual(engine._observation_state().stats["sparse_crop_fallback_max"], 0)

    def test_sparse_crop_fallback_can_be_opted_in_separately(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34), confidence=0.9),
            SimpleNamespace(xyxy=(70, 8, 118, 34), confidence=0.9),
        ]
        mapped_records = [
            {"text": "", "source_bbox": [], "line_polygons": []},
            {"text": "", "source_bbox": [], "line_polygons": []},
        ]

        with patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=mapped_records,
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            side_effect=["HELLO", "WORLD"],
        ) as recognize:
            records = engine.recognize_blocks_from_page(
                np.full((100, 220, 3), 255, dtype=np.uint8),
                blocks,
                crop_fallback_max=3,
                sparse_crop_fallback_max=1,
            )

        recognize.assert_called_once()
        self.assertEqual(records[0]["text"], "HELLO")
        self.assertEqual(records[1]["text"], "")
        self.assertEqual(engine._observation_state().stats["crop_fallback_attempts"], 1)
        self.assertEqual(engine._observation_state().stats["crop_fallback_recovered"], 1)
        self.assertEqual(engine._observation_state().stats["sparse_crop_fallback_max"], 1)

    def test_crop_fallback_max_limits_full_page_mapping_failures(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        blocks = [
            SimpleNamespace(xyxy=(8, 8, 54, 34), confidence=0.9),
            SimpleNamespace(xyxy=(70, 8, 118, 34), confidence=0.9),
            SimpleNamespace(xyxy=(140, 8, 190, 34), confidence=0.9),
        ]

        with patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=None,
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            side_effect=["HELLO", "WORLD"],
        ) as recognize:
            records = engine.recognize_blocks_from_page(
                np.full((100, 220, 3), 255, dtype=np.uint8),
                blocks,
                crop_fallback_max=1,
            )

        recognize.assert_called_once()
        self.assertEqual(records, ["HELLO", "", ""])
        self.assertEqual(engine._observation_state().stats["crop_fallback_max"], 1)
        self.assertEqual(engine._observation_state().stats["crop_fallback_attempts"], 1)
        self.assertEqual(engine._observation_state().stats["crop_fallback_recovered"], 1)

    def test_observation_records_are_reset_for_each_public_request_and_returned_as_a_copy(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine.batch_size = 8

        class FakePaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                return [[
                    (
                        [[10, 10], [70, 10], [70, 28], [10, 28]],
                        ("HELLO", 0.97),
                    )
                ]]

        engine._model = FakePaddleModel()
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(0, 0, 100, 60), confidence=0.95)

        engine.recognize_blocks_from_page(image, [block], allow_sparse_mapping=True)
        first_snapshot = copy.deepcopy(list(engine._observation_state().records))
        self.assertTrue(first_snapshot)

        first_snapshot[0]["text"] = "MUTATED"
        self.assertEqual(engine._observation_state().records[0]["text"], "HELLO")

        engine.recognize_blocks_from_page(image, [])
        self.assertEqual(engine._observation_state().records, [])
        self.assertEqual(engine._observation_state().full_page_lines, [])

    def test_full_page_observations_preserve_unmapped_provider_lines_with_reason(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine.batch_size = 8

        class FakePaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                return [[
                    (
                        [[10, 10], [70, 10], [70, 28], [10, 28]],
                        ("INSIDE", 0.98),
                    ),
                    (
                        [[150, 60], [210, 60], [210, 78], [150, 78]],
                        ("OUTSIDE", 0.91),
                    ),
                ]]

        engine._model = FakePaddleModel()
        image = np.full((100, 240, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(0, 0, 100, 50), confidence=0.95)

        mapped = engine.recognize_blocks_from_page(image, [block], allow_sparse_mapping=True)
        observations = [
            record
            for record in engine._observation_state().records
            if record["provider"] == "paddle_full_page"
        ]

        self.assertEqual(mapped[0]["text"], "INSIDE")
        self.assertEqual([record["text"] for record in observations], ["INSIDE", "OUTSIDE"])
        self.assertTrue(observations[0]["accepted"])
        self.assertFalse(observations[1]["accepted"])
        self.assertEqual(observations[1]["rejection_reason"], "unmapped_to_detector_block")
        self.assertEqual(observations[1]["bbox"], [150, 60, 210, 78])
        self.assertEqual(observations[1]["line_polygons"][0][0], [150, 60])

    def test_full_page_observations_preserve_provider_items_rejected_before_mapping(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine.batch_size = 8

        class FakePaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                return [[
                    None,
                    (
                        [[10, 10], [70, 10], [70, 28], [10, 28]],
                        ("", 0.88),
                    ),
                    (
                        [[20, 20], [40, 20]],
                        ("BROKEN", 0.81),
                    ),
                ]]

        engine._model = FakePaddleModel()
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(0, 0, 100, 60), confidence=0.95)

        mapped = engine.recognize_blocks_from_page(
            image,
            [block],
            allow_sparse_mapping=True,
            crop_fallback_max=0,
        )
        observations = [
            record
            for record in engine._observation_state().records
            if record["provider"] == "paddle_full_page"
        ]

        self.assertEqual(mapped, [""])
        self.assertEqual(
            [record["rejection_reason"] for record in observations],
            ["invalid_provider_item", "empty_text", "invalid_polygon"],
        )
        self.assertEqual(observations[1]["bbox"], [10, 10, 70, 28])
        self.assertEqual(observations[2]["text"], "BROKEN")
        self.assertTrue(all(not record["accepted"] for record in observations))

    def test_full_page_observations_record_provider_failure(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine.batch_size = 8

        class BrokenPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                raise RuntimeError("provider unavailable")

        engine._model = BrokenPaddleModel()
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(0, 0, 100, 60), confidence=0.95)

        engine.recognize_blocks_from_page(image, [block], crop_fallback_max=0)
        observations = list(engine._observation_state().records)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["provider"], "paddle_full_page")
        self.assertEqual(observations[0]["rejection_reason"], "provider_error")
        self.assertIn("provider unavailable", observations[0]["error"])

    def test_full_page_observations_record_provider_no_output(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine.batch_size = 8

        class EmptyPaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                return [[]]

        engine._model = EmptyPaddleModel()
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(0, 0, 100, 60), confidence=0.95)

        engine.recognize_blocks_from_page(image, [block], crop_fallback_max=0)
        observations = list(engine._observation_state().records)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["provider"], "paddle_full_page")
        self.assertEqual(observations[0]["rejection_reason"], "no_output")

    def test_crop_retry_observations_keep_empty_losing_and_selected_attempts(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine.batch_size = 8

        class FakePaddleModel:
            def __init__(self):
                self.calls = 0

            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                self.calls += 1
                if self.calls == 1:
                    return [[]]
                if self.calls == 2:
                    return [[[[0, 0], ("NO", 0.70)]]]
                return [[[[0, 0], ("BETTER TEXT", 0.96)]]]

        engine._model = FakePaddleModel()
        crop = np.full((32, 80, 3), 255, dtype=np.uint8)

        with patch.object(
            engine,
            "_build_paddle_retry_variants",
            return_value=[crop.copy(), crop.copy()],
        ):
            result = engine.recognize_batch([crop])

        observations = [
            record
            for record in engine._observation_state().records
            if record["provider"] == "paddle_crop"
        ]
        self.assertEqual(result, ["BETTER TEXT"])
        self.assertEqual([record["variant"] for record in observations], ["native", "retry_1", "retry_2"])
        self.assertEqual([record["accepted"] for record in observations], [False, False, True])
        self.assertEqual(observations[0]["rejection_reason"], "empty_result")
        self.assertEqual(observations[1]["rejection_reason"], "lower_score")
        self.assertIsNone(observations[2]["rejection_reason"])
        for record in observations:
            self.assertEqual(record["bbox"], [0, 0, 80, 32])
            self.assertIn("attempt_id", record)
            self.assertIn("raw_text", record)
            self.assertIn("confidence", record)

    def test_paddle_batch_observations_keep_their_source_crop_index(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        engine.batch_size = 8

        class FakePaddleModel:
            def __init__(self):
                self.calls = 0

            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                self.calls += 1
                text = "FIRST" if self.calls == 1 else "SECOND"
                return [[[[0, 0], (text, 0.95)]]]

        engine._model = FakePaddleModel()
        crops = [
            np.full((24, 60, 3), 255, dtype=np.uint8),
            np.full((32, 80, 3), 255, dtype=np.uint8),
        ]

        result = engine.recognize_batch(crops)
        accepted = [
            record
            for record in engine._observation_state().records
            if record["provider"] == "paddle_crop" and record["accepted"]
        ]

        self.assertEqual(result, ["FIRST", "SECOND"])
        self.assertEqual([record["crop_index"] for record in accepted], [0, 1])
        self.assertEqual([record["bbox"] for record in accepted], [[0, 0, 60, 24], [0, 0, 80, 32]])

    def test_dedupe_records_rejected_candidate_before_clearing_legacy_text(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        blocks = [
            SimpleNamespace(xyxy=(10, 10, 80, 40), confidence=0.91),
            SimpleNamespace(xyxy=(12, 11, 82, 41), confidence=0.80),
        ]
        mapped_records = [
            {"text": "HELLO THERE", "source_bbox": [10, 10, 80, 40], "line_polygons": []},
            {"text": "HELLO THERE", "source_bbox": [12, 11, 82, 41], "line_polygons": []},
        ]

        with patch.dict(os.environ, {"TRADUZAI_OCR_DEDUP": "1"}, clear=False), patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=mapped_records,
        ):
            result = engine.recognize_blocks_from_page(
                np.full((80, 120, 3), 255, dtype=np.uint8),
                blocks,
                allow_sparse_mapping=True,
            )

        rejected = [
            record
            for record in engine._observation_state().records
            if record["provider"] == "ocr_dedupe"
        ]
        self.assertEqual([record["text"] for record in result], ["HELLO THERE", ""])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["text"], "HELLO THERE")
        self.assertFalse(rejected[0]["accepted"])
        self.assertEqual(rejected[0]["rejection_reason"], "duplicate_ocr_record")
        self.assertEqual(rejected[0]["bbox"], [12, 11, 82, 41])

    def test_rotated_observations_preserve_raw_lines_before_confidence_drop_and_grouping(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"

        class FakePaddleModel:
            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                return [[
                    (
                        [[10, 20], [80, 20], [80, 30], [10, 30]],
                        ("FIRST", 0.96),
                    ),
                    (
                        [[35, 20], [95, 20], [95, 30], [35, 30]],
                        ("SECOND", 0.94),
                    ),
                    (
                        [[10, 50], [80, 50], [80, 60], [10, 60]],
                        ("LOW", 0.40),
                    ),
                ]]

        engine._model = FakePaddleModel()
        image = np.full((100, 200, 3), 255, dtype=np.uint8)

        grouped = engine.recognize_rotated_full_page_lines(image, rotations=(90,), min_confidence=0.80)
        observations = [
            record
            for record in engine._observation_state().records
            if record["provider"] == "paddle_rotated_full_page"
        ]

        self.assertEqual(len(observations), 3)
        self.assertLess(len(grouped), len([record for record in observations if record["accepted"]]))
        low = next(record for record in observations if record["text"] == "LOW")
        self.assertFalse(low["accepted"])
        self.assertEqual(low["rejection_reason"], "below_min_confidence")
        for record in observations:
            self.assertEqual(record["variant"], "rotation_90")
            self.assertTrue(record["line_polygons"])
            for x, y in record["line_polygons"][0]:
                self.assertGreaterEqual(x, 0)
                self.assertLess(x, 200)
                self.assertGreaterEqual(y, 0)
                self.assertLess(y, 100)

    def test_deskew_observations_keep_recovered_lines_before_replacing_partial_mapping(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"

        class FakePaddleModel:
            def __init__(self):
                self.calls = 0

            def ocr(self, image, det=True, rec=True, cls=False):
                del image, det, rec, cls
                self.calls += 1
                if self.calls == 1:
                    return [[
                        (
                            [[20, 100], [120, 40], [135, 60], [35, 120]],
                            ("PARTIAL", 0.93),
                        ),
                        (
                            [[80, 160], [180, 100], [195, 120], [95, 180]],
                            ("READING", 0.91),
                        ),
                    ]]
                return [[
                    ([[10, 10], [90, 10], [90, 28], [10, 28]], ("PARTIAL", 0.95)),
                    ([[10, 36], [70, 36], [70, 54], [10, 54]], ("FULL", 0.94)),
                    ([[10, 62], [100, 62], [100, 80], [10, 80]], ("READING", 0.96)),
                ]]

        engine._model = FakePaddleModel()
        page = np.full((220, 240, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(0, 0, 240, 220), confidence=0.95)

        mapped = engine.recognize_blocks_from_page(page, [block], allow_sparse_mapping=True)
        observations = list(engine._observation_state().records)
        full_page = [record["text"] for record in observations if record["provider"] == "paddle_full_page"]
        deskew = [record["text"] for record in observations if record["provider"] == "paddle_skewed_recovery"]

        self.assertEqual(full_page, ["PARTIAL", "READING"])
        self.assertEqual(deskew, ["PARTIAL", "FULL", "READING"])
        self.assertEqual(mapped[0]["text"], "PARTIAL FULL READING")


if __name__ == "__main__":
    unittest.main()
