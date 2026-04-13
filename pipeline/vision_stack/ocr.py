"""
OCR Engine — Reconhecimento de texto em manga
Suporta manga-ocr (japonês/inglês) e PaddleOCR (multilingual)
Batching para máxima performance na GPU
"""

import logging
import os
from typing import Optional, Union
import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Motor de OCR com suporte a batching.
    
    Backends:
        - "manga-ocr": TrOCR fine-tuned para manga (kha-white/manga-ocr)
          Melhor para japonês, excelente para inglês em manga
        - "paddleocr": PaddleOCR multilingual
          Já está no seu stack — usado como fallback ou alternativa
    """

    def __init__(
        self,
        model: str = "paddleocr",
        device: str = "cuda",
        half: bool = True,
        batch_size: int = 8,
        lang: str = "en",
    ):
        self._requested_model = model
        self.model_name = model
        self.device = self._resolve_device(device)
        self.half = half and self.device.type == "cuda"
        self.batch_size = batch_size
        self.lang = lang
        self._model = None
        self._processor = None
        self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        if self.model_name == "manga-ocr":
            self._load_manga_ocr()
        elif self.model_name == "paddleocr":
            self._load_paddle_ocr()
        else:
            raise ValueError(f"OCR backend '{self.model_name}' não suportado")

    def _load_manga_ocr(self):
        """manga-ocr usa VisionEncoderDecoder (TrOCR) da HuggingFace."""
        try:
            from transformers import AutoFeatureExtractor, VisionEncoderDecoderModel, AutoTokenizer
            
            model_id = "kha-white/manga-ocr-base"
            logger.info(f"Carregando manga-ocr de {model_id}...")
            
            self._processor = AutoFeatureExtractor.from_pretrained(model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_id)
            self._model.to(self.device)
            
            if self.half:
                self._model = self._model.half()
            
            self._model.eval()
            self._backend = "manga-ocr"
            logger.info(f"manga-ocr carregado ({self.device})")

        except ImportError:
            logger.warning("transformers não instalado, usando PaddleOCR como fallback")
            self.model_name = "paddleocr"
            self._load_paddle_ocr()
        except Exception as exc:
            logger.warning("manga-ocr não carregou (%s); usando PaddleOCR como fallback", exc)
            self.model_name = "paddleocr"
            self._load_paddle_ocr()

    def _load_paddle_ocr(self):
        """PaddleOCR — já presente no TraduzAi."""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from paddleocr import PaddleOCR
        
        # Mapeamento do TraduzAi (app) para o PaddleOCR
        # en -> en
        # ja -> japan
        # ko -> korean
        # zh -> ch
        mapped_lang = {
            "en": "en",
            "ja": "japan",
            "ko": "korean",
            "zh": "ch",
        }.get(self.lang, "en")
        
        use_gpu = self.device.type == "cuda"
        self._model = PaddleOCR(
            use_angle_cls=True,
            lang=mapped_lang,
            use_gpu=use_gpu,
            show_log=False,
            enable_mkldnn=not use_gpu,  # MKL-DNN acelera CPU
        )
        self._backend = "paddleocr"
        logger.info(f"PaddleOCR carregado (lang={mapped_lang}, gpu={use_gpu})")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def recognize_batch(self, crops: list[np.ndarray]) -> list[str]:
        """
        Reconhece texto em múltiplas imagens recortadas.
        Processa em batches para máxima eficiência GPU.
        """
        if not crops:
            return []

        results = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i : i + self.batch_size]
            batch_results = self._recognize_batch_impl(batch)
            results.extend(batch_results)
        return results

    def recognize_blocks_from_page(self, page_rgb: np.ndarray, blocks: list) -> list[str]:
        """Reconhece texto para cada bloco detectado, alinhando o resultado ao `blocks`.

        Otimiza o backend PaddleOCR: evita rodar detecção repetidamente por crop.
        Faz 1 pass de OCR na página inteira e associa as linhas reconhecidas aos blocos.
        """
        if not blocks:
            return []

        if not isinstance(page_rgb, np.ndarray) or page_rgb.size == 0:
            return [""] * len(blocks)

        if getattr(self, "_backend", "") != "paddleocr":
            raise ValueError("recognize_blocks_from_page disponível apenas para PaddleOCR")

        page_bgr = page_rgb
        if len(page_rgb.shape) == 3 and page_rgb.shape[2] >= 3:
            try:
                page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                page_bgr = page_rgb

        texts = self._paddle_ocr_full_page_to_blocks(page_bgr, blocks)
        if texts is None:
            return [
                self._recognize_single_paddle_with_retry(self._crop_block_from_page(page_rgb, block))
                for block in blocks
            ]

        # Fallback por crop apenas para casos prováveis, evitando custo alto em falsos positivos.
        max_fallback = 3
        attempted = 0
        for index, text in enumerate(texts):
            if attempted >= max_fallback:
                break
            if text.strip():
                continue
            try:
                block_confidence = float(getattr(blocks[index], "confidence", 1.0) or 0.0)
            except Exception:
                block_confidence = 1.0
            if block_confidence < 0.45:
                continue
            crop = self._crop_block_from_page(page_rgb, blocks[index])
            if not self._crop_might_have_text(crop):
                continue
            attempted += 1
            texts[index] = self._recognize_single_paddle_with_retry(crop)

        return texts

    def recognize_single(self, crop: np.ndarray) -> str:
        """Reconhece texto em uma única imagem."""
        results = self.recognize_batch([crop])
        return results[0] if results else ""

    def _recognize_batch_impl(self, crops: list[np.ndarray]) -> list[str]:
        if self._backend == "manga-ocr":
            return self._manga_ocr_batch(crops)
        else:
            return self._paddle_ocr_batch(crops)

    def _manga_ocr_batch(self, crops: list[np.ndarray]) -> list[str]:
        """Inferência batched com manga-ocr."""
        pil_images = []
        for crop in crops:
            if isinstance(crop, np.ndarray):
                img = Image.fromarray(crop).convert("RGB")
            else:
                img = crop.convert("RGB")
            
            # manga-ocr funciona melhor com imagens quadradas
            img = self._pad_to_square(img)
            pil_images.append(img)

        # Tokeniza batch
        pixel_values = self._processor(
            images=pil_images,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        if self.half:
            pixel_values = pixel_values.half()

        with torch.inference_mode():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=300,
                num_beams=1,          # greedy — mais rápido, boa qualidade
                do_sample=False,
            )

        texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _paddle_ocr_batch(self, crops: list[np.ndarray]) -> list[str]:
        """PaddleOCR processa uma imagem por vez (não tem batch nativo)."""
        texts = []
        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                texts.append("")
                continue
            texts.append(self._recognize_single_paddle_with_retry(crop))
        return texts

    def _recognize_single_paddle(self, crop: np.ndarray, *, cls: bool = False) -> str:
        try:
            result = self._model.ocr(crop, det=True, rec=True, cls=bool(cls))
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line and line[1]]
                return " ".join(lines).strip()
            return ""
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return ""

    @staticmethod
    def _score_ocr_candidate(text: str) -> tuple[int, int, int]:
        cleaned = str(text or "").strip()
        alnum = sum(ch.isalnum() for ch in cleaned)
        alpha = sum(ch.isalpha() for ch in cleaned)
        return (alnum, alpha, len(cleaned))

    def _build_paddle_retry_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = []
        up2 = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(up2)

        # Reuse up2 for grayscale base (avoid redundant resize)
        gray_up2 = cv2.cvtColor(up2, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray_up2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB))

        blur = cv2.GaussianBlur(gray_up2, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(gray_up2, 1.8, blur, -0.8, 0)
        sharp_up15 = cv2.resize(sharp, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        variants.append(cv2.cvtColor(sharp_up15, cv2.COLOR_GRAY2RGB))
        return variants

    @staticmethod
    def _detect_dot_run_fallback(crop: np.ndarray) -> str:
        if crop.size == 0:
            return ""

        height, width = crop.shape[:2]
        if height < 10 or width < 18:
            return ""

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        components: list[dict[str, float]] = []
        max_component_area = max(18, int(height * width * 0.08))
        for index in range(1, component_count):
            x, y, w, h, area = stats[index].tolist()
            if area < 6 or area > max_component_area:
                continue
            if w <= 0 or h <= 0:
                continue
            ratio = max(w / float(h), h / float(w))
            if ratio > 1.8:
                continue
            components.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "area": float(area),
                    "cx": float(centroids[index][0]),
                    "cy": float(centroids[index][1]),
                }
            )

        if not 3 <= len(components) <= 8:
            return ""

        components.sort(key=lambda item: item["cx"])
        ys = [item["cy"] for item in components]
        if max(ys) - min(ys) > max(4.0, height * 0.16):
            return ""

        widths = [item["w"] for item in components]
        heights = [item["h"] for item in components]
        if max(widths) > min(widths) * 1.8 or max(heights) > min(heights) * 1.8:
            return ""

        gaps = [
            components[index + 1]["cx"] - components[index]["cx"]
            for index in range(len(components) - 1)
        ]
        if any(gap <= 0 for gap in gaps):
            return ""
        if max(gaps) > max(16.0, min(gaps) * 2.2):
            return ""

        total_area = sum(item["area"] for item in components)
        if total_area > height * width * 0.22:
            return ""

        return "." * len(components)

    def _recognize_single_paddle_with_retry(self, crop: np.ndarray) -> str:
        text = self._recognize_single_paddle(crop, cls=True)
        if text:
            return text

        best_text = ""
        best_score = self._score_ocr_candidate("")
        for variant in self._build_paddle_retry_variants(crop):
            candidate = self._recognize_single_paddle(variant, cls=True)
            score = self._score_ocr_candidate(candidate)
            if score > best_score:
                best_score = score
                best_text = candidate
            # Early exit: good enough result, skip remaining variants
            if score >= (3, 3, 4):
                break
        if best_text.strip():
            return best_text.strip()
        return self._detect_dot_run_fallback(crop)

    @staticmethod
    def _crop_block_from_page(page_rgb: np.ndarray, block, padding: int = 4) -> np.ndarray:
        height, width = page_rgb.shape[:2]
        try:
            x1 = int(getattr(block, "x1"))
            y1 = int(getattr(block, "y1"))
            x2 = int(getattr(block, "x2"))
            y2 = int(getattr(block, "y2"))
        except Exception:
            xyxy = getattr(block, "xyxy", (0, 0, 0, 0))
            x1, y1, x2, y2 = [int(v) for v in xyxy]

        pad = int(padding)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return page_rgb[y1:y2, x1:x2]

    @staticmethod
    def _crop_might_have_text(crop_rgb: np.ndarray) -> bool:
        if not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
            return False

        height, width = crop_rgb.shape[:2]
        if height < 10 or width < 10:
            return False

        if len(crop_rgb.shape) == 3 and crop_rgb.shape[2] >= 3:
            gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_rgb.astype(np.uint8, copy=False)

        max_dim = max(height, width)
        if max_dim > 256:
            scale = 256.0 / float(max_dim)
            resized = cv2.resize(
                gray,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = gray

        std = float(np.std(resized))
        edges = cv2.Canny(resized, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, resized.size))
        dark_ratio = float(np.mean(resized < 150))

        if std < 9.0 and edge_density < 0.006:
            return False
        if dark_ratio < 0.015 and edge_density < 0.004:
            return False
        return True

    @staticmethod
    def _bbox_intersection_area(a: list[int], b: list[int]) -> int:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return int((ix2 - ix1) * (iy2 - iy1))

    def _paddle_ocr_full_page_to_blocks(self, page_bgr: np.ndarray, blocks: list) -> list[str] | None:
        try:
            result = self._model.ocr(page_bgr, det=True, rec=True, cls=False)
        except Exception as exc:
            logger.warning("PaddleOCR full-page falhou; fallback por crop: %s", exc)
            return None

        raw_lines = result[0] if isinstance(result, list) and result else []
        if not raw_lines:
            return None

        block_bboxes: list[list[int]] = []
        for block in blocks:
            try:
                block_bboxes.append(
                    [
                        int(getattr(block, "x1")),
                        int(getattr(block, "y1")),
                        int(getattr(block, "x2")),
                        int(getattr(block, "y2")),
                    ]
                )
            except Exception:
                xyxy = getattr(block, "xyxy", (0, 0, 0, 0))
                block_bboxes.append([int(v) for v in xyxy])

        assigned: list[list[tuple[list[int], str]]] = [[] for _ in blocks]

        for item in raw_lines:
            if not item or len(item) < 2:
                continue
            box = item[0]
            meta = item[1]
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue

            text = meta[0] if isinstance(meta, (list, tuple)) and len(meta) >= 1 else ""
            if not str(text or "").strip():
                continue

            xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
            if not xs or not ys:
                continue

            line_bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            lx1, ly1, lx2, ly2 = line_bbox
            line_area = max(1, (lx2 - lx1) * (ly2 - ly1))

            best_index = None
            best_score = 0.0
            for idx, block_bbox in enumerate(block_bboxes):
                bx1, by1, bx2, by2 = block_bbox
                inter = self._bbox_intersection_area(line_bbox, block_bbox)
                if inter <= 0:
                    continue
                block_area = max(1, (bx2 - bx1) * (by2 - by1))
                score = inter / float(max(1, min(line_area, block_area)))
                if score > best_score:
                    best_score = score
                    best_index = idx

            if best_index is not None and best_score >= 0.18:
                assigned[best_index].append((line_bbox, str(text).strip()))

        texts: list[str] = []
        non_empty = 0
        for lines in assigned:
            lines.sort(key=lambda entry: (entry[0][1], entry[0][0]))
            joined = " ".join(text for _, text in lines).strip()
            if joined:
                non_empty += 1
            texts.append(joined)

        if non_empty == 0:
            return None

        # Se associou texto em poucos blocos, o mapeamento falhou e devemos preservar qualidade
        # voltando ao caminho antigo por crop.
        if len(blocks) >= 3 and non_empty / max(1, len(blocks)) < 0.5:
            return None

        return texts

    @staticmethod
    def _pad_to_square(img: Image.Image, size: int = 224) -> Image.Image:
        """Adiciona padding para tornar quadrado (melhora manga-ocr)."""
        w, h = img.size
        max_dim = max(w, h, 1)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        
        new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        new_img.paste(img, (pad_w, pad_h))
        return new_img.resize((size, size), Image.LANCZOS)

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
