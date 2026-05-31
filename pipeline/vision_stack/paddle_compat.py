from __future__ import annotations

from typing import Any


class PaddleOCRCompat:
    """Compatibility shim for PaddleOCR 2.x and 3.x call shapes."""

    def __init__(self, model: Any, *, api_version: int) -> None:
        self._model = model
        self.api_version = api_version

    def ocr(self, img: Any, *, det: bool = True, rec: bool = True, cls: bool = False) -> list:
        if self.api_version < 3:
            return self._model.ocr(img, det=det, rec=rec, cls=cls)

        results = self._model.predict(
            img,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=bool(cls),
        )
        if not results:
            return [[]]

        converted_pages = []
        for result in results:
            data = _result_to_dict(result)
            if not rec:
                converted_pages.append(_coerce_polygons(data.get("dt_polys") or data.get("rec_polys") or []))
                continue

            polys = _coerce_polygons(data.get("rec_polys") or data.get("dt_polys") or [])
            texts = list(data.get("rec_texts") or [])
            scores = list(data.get("rec_scores") or [])
            lines = []
            for index, poly in enumerate(polys):
                text = texts[index] if index < len(texts) else ""
                score = scores[index] if index < len(scores) else 0.0
                lines.append([poly, [text, score]])
            converted_pages.append(lines)
        return converted_pages


def create_paddle_ocr(
    paddle_ocr_cls: Any,
    *,
    lang: str,
    use_gpu: bool,
    use_angle_cls: bool = False,
    enable_mkldnn: bool | None = None,
    show_log: bool | None = None,
) -> PaddleOCRCompat:
    device = "gpu:0" if use_gpu else "cpu"
    try:
        model = paddle_ocr_cls(
            lang=lang,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=bool(use_angle_cls),
            enable_mkldnn=bool(enable_mkldnn) if enable_mkldnn is not None else not use_gpu,
        )
        return PaddleOCRCompat(model, api_version=3)
    except (TypeError, ValueError) as exc:
        if "Unknown argument" not in str(exc) and "unexpected keyword" not in str(exc):
            raise

    kwargs: dict[str, Any] = {
        "use_angle_cls": use_angle_cls,
        "lang": lang,
        "use_gpu": use_gpu,
        "enable_mkldnn": bool(enable_mkldnn) if enable_mkldnn is not None else not use_gpu,
    }
    if show_log is not None:
        kwargs["show_log"] = show_log
    return PaddleOCRCompat(paddle_ocr_cls(**kwargs), api_version=2)


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result.get("res", result) if isinstance(result.get("res", result), dict) else result
    json_value = getattr(result, "json", None)
    if isinstance(json_value, dict):
        nested = json_value.get("res", json_value)
        return nested if isinstance(nested, dict) else json_value
    try:
        return dict(result)
    except Exception:
        return {}


def _coerce_polygons(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)
