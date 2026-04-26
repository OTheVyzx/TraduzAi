"""Adapter em-memória do inpainter para o pipeline strip-based."""

import numpy as np


def inpaint_band_image(band_rgb: np.ndarray, ocr_page: dict) -> np.ndarray:
    """Adapter em-memória: aplica inpaint na banda usando os bboxes do ocr_page.

    Dilata cada text bbox em 6 px antes de passar para o inpainter para capturar
    pixels anti-aliased e serifs que ficam fora do bbox OCR exato.

    Retorna ndarray do mesmo shape com texto inglês removido.
    """
    from inpainter.classical import clean_image
    from PIL import Image

    if band_rgb.size == 0 or not ocr_page.get("texts"):
        return band_rgb.copy()

    h, w = band_rgb.shape[:2]
    dilation = 6

    # Construir lista de textos com bbox dilatado
    inflated_texts = []
    for txt in ocr_page["texts"]:
        bb = txt.get("bbox")
        if not bb:
            # Texto sem bbox — ignorar (evita placeholder [0,0,32,32])
            continue
        x1, y1, x2, y2 = bb
        new_txt = dict(txt)
        new_txt["bbox"] = [
            max(0, x1 - dilation),
            max(0, y1 - dilation),
            min(w, x2 + dilation),
            min(h, y2 + dilation),
        ]
        inflated_texts.append(new_txt)

    if not inflated_texts:
        return band_rgb.copy()

    # clean_image espera PIL.Image e retorna PIL.Image
    img = Image.fromarray(band_rgb)
    cleaned_img = clean_image(img, inflated_texts)
    return np.array(cleaned_img)
