"""
Primary OCR pass using PaddleOCR when available.
Fails closed when PaddleOCR is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_paddle_readers: dict[str, object] = {}


class OcrBackendUnavailable(RuntimeError):
    """Raised when PaddleOCR is required but unavailable."""


def _prefer_torch_cuda_dlls() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        pass


def choose_primary_ocr_engine(paddle_ready: bool) -> str:
    if paddle_ready:
        return "paddle"
    raise OcrBackendUnavailable("PaddleOCR indisponivel; EasyOCR fallback removido.")


def is_paddle_available() -> bool:
    try:
        _prefer_torch_cuda_dlls()
        from paddleocr import PaddleOCR  # noqa: F401

        return True
    except Exception:
        return False


def get_paddle_reader(use_gpu: bool = False, lang: str = "en"):
    normalized_lang = (lang or "en").strip() or "en"
    use_gpu = bool(use_gpu)
    reader_key = f"{normalized_lang}|gpu={int(use_gpu)}"
    if reader_key not in _paddle_readers:
        _prefer_torch_cuda_dlls()
        try:
            import paddle.base.libpaddle as libpaddle
            if hasattr(libpaddle, 'AnalysisConfig') and not hasattr(libpaddle.AnalysisConfig, 'set_optimization_level'):
                libpaddle.AnalysisConfig.set_optimization_level = lambda *args, **kwargs: None
        except Exception:
            pass
        from paddleocr import PaddleOCR

        logger.info("Inicializando PaddleOCR lang=%s...", normalized_lang)
        _paddle_readers[reader_key] = PaddleOCR(
            use_angle_cls=False,
            lang=normalized_lang,
            use_gpu=use_gpu,
            enable_mkldnn=not use_gpu,
        )
        logger.info("PaddleOCR pronto lang=%s.", normalized_lang)
    return _paddle_readers[reader_key]


def normalize_paddle_results(raw_results) -> list[dict]:
    page = raw_results[0] if raw_results else []
    runs: list[dict] = []
    for item in page or []:
        if not item or len(item) < 2:
            continue
        bbox_pts = item[0]
        rec = item[1]
        if not rec or len(rec) < 2:
            continue
        text = str(rec[0]).strip()
        confidence = float(rec[1])
        if not text:
            continue
        runs.append(
            {
                "bbox_pts": bbox_pts,
                "text": text,
                "confidence": confidence,
                "source": "primary-paddle",
            }
        )
    return runs


def run_paddle_primary_recognition(image_bgr, use_gpu: bool = False, lang: str = "en") -> list[dict]:
    reader = get_paddle_reader(use_gpu=use_gpu, lang=lang)
    raw_results = reader.ocr(image_bgr, det=True, rec=True, cls=False)
    return normalize_paddle_results(raw_results)
