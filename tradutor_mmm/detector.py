"""
Deteccao de texto em imagens de manga/manhwa usando EasyOCR + docTR fallback.
Classifica regioes em: balao de fala, balao intenso, narracao, SFX coreano, watermark.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import easyocr

from . import config

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Regiao de texto detectada em uma imagem."""
    bbox: List[List[int]]       # 4 cantos do poligono [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text: str                   # Texto reconhecido
    confidence: float           # Confianca do OCR (0-1)
    text_type: str = config.TEXT_TYPE_SPEECH_BUBBLE
    language: str = "en"
    style: Optional[object] = field(default=None, repr=False)

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Retorna (x_min, y_min, x_max, y_max) do retangulo delimitador."""
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    @property
    def center(self) -> Tuple[int, int]:
        x_min, y_min, x_max, y_max = self.rect
        return ((x_min + x_max) // 2, (y_min + y_max) // 2)

    @property
    def width(self) -> int:
        x_min, _, x_max, _ = self.rect
        return x_max - x_min

    @property
    def height(self) -> int:
        _, y_min, _, y_max = self.rect
        return y_max - y_min


class TextDetector:
    """Detector de texto usando EasyOCR com classificacao de tipo."""

    def __init__(self):
        logger.info("Inicializando EasyOCR (pode demorar no primeiro uso)...")
        self._reader = easyocr.Reader(
            config.OCR_LANGUAGES,
            gpu=config.OCR_GPU,
            verbose=False,
        )
        logger.info("EasyOCR inicializado.")
        self._doctr_model = None  # Lazy-loaded

    def detect(self, image_path: str, image_shape: Optional[Tuple[int, int]] = None) -> List[TextRegion]:
        """Detecta todas as regioes de texto em uma imagem.

        Args:
            image_path: Caminho da imagem
            image_shape: (height, width) da imagem, se ja conhecido

        Returns:
            Lista de TextRegion classificadas
        """
        # Carregar imagem original para obter dimensoes
        original_img = cv2.imread(image_path)
        if original_img is None:
            return []

        orig_h, orig_w = original_img.shape[:2]
        if image_shape is None:
            image_shape = (orig_h, orig_w)

        # Preprocessar imagem para melhorar OCR
        preprocessed = self._preprocess_for_ocr(image_path)

        # Calcular escala de preprocessamento
        if isinstance(preprocessed, np.ndarray):
            prep_h = preprocessed.shape[0]
            scale = prep_h / orig_h
        else:
            scale = 1.0

        # OCR normal
        results = self._reader.readtext(
            preprocessed,
            text_threshold=config.OCR_TEXT_THRESHOLD,
            low_text=config.OCR_LOW_TEXT,
            canvas_size=config.OCR_CANVAS_SIZE,
        )

        # OCR com imagem invertida (detecta texto claro em fundo escuro)
        if isinstance(preprocessed, np.ndarray):
            inverted = cv2.bitwise_not(preprocessed)
        else:
            inv_img = cv2.imread(image_path)
            inverted = cv2.bitwise_not(inv_img) if inv_img is not None else None

        if inverted is not None:
            inv_results = self._reader.readtext(
                inverted,
                text_threshold=config.OCR_TEXT_THRESHOLD,
                low_text=config.OCR_LOW_TEXT,
                canvas_size=config.OCR_CANVAS_SIZE,
            )
            # Adicionar resultados da inversao que nao sobreponham os normais
            results = self._merge_ocr_results(results, inv_results)

        # 3o passo: docTR fallback para texto que EasyOCR nao conseguiu ler
        if config.DOCTR_ENABLED:
            try:
                doctr_results = self._doctr_detect(image_path, orig_h, orig_w)
                # Mesclar docTR com EasyOCR (so adicionar regioes novas)
                results = self._merge_ocr_results(results, doctr_results)
            except Exception as e:
                logger.warning(f"docTR fallback falhou: {e}")

        regions = []
        for bbox, text, conf in results:
            if conf < config.OCR_CONFIDENCE_THRESHOLD:
                continue

            text = text.strip()
            if not text:
                continue

            # Filtrar deteccoes muito curtas com baixa confianca (ruido)
            if len(text) <= 2 and conf < 0.6:
                continue
            # Filtrar deteccoes curtas que sao apenas numeros/simbolos
            if len(text) <= 3 and not any(c.isalpha() for c in text):
                continue

            # Corrigir erros comuns de OCR
            text = self._fix_ocr_errors(text)
            if not text:
                continue

            # Converter bbox para lista de ints E escalar de volta para coordenadas originais
            bbox_int = [
                [int(round(p[0] / scale)), int(round(p[1] / scale))]
                for p in bbox
            ]

            # Clampar ao tamanho da imagem original
            for p in bbox_int:
                p[0] = max(0, min(orig_w - 1, p[0]))
                p[1] = max(0, min(orig_h - 1, p[1]))

            region = TextRegion(
                bbox=bbox_int,
                text=text,
                confidence=conf,
            )

            # Classificar lingua
            region.language = self._classify_language(text)

            # Classificar tipo
            if region.language == "ko":
                region.text_type = config.TEXT_TYPE_SFX_KOREAN
            # Tipo final (SPEECH/INTENSE/NARRATION) sera definido pelo StyleAnalyzer
            # Watermarks sao ignoradas e mantidas nas imagens

            regions.append(region)

        # Mesclar regioes proximas do mesmo balao
        regions = self._merge_nearby_regions(regions)

        logger.debug(f"Detectadas {len(regions)} regioes de texto")
        return regions

    def _merge_ocr_results(self, primary: list, secondary: list) -> list:
        """Mescla resultados de OCR, preferindo maior confianca. Detecta duplicatas por
        distancia de centro OU sobreposicao de bounding box (IoU)."""
        if not secondary:
            return primary

        merged = list(primary)

        for s_bbox, s_text, s_conf in secondary:
            s_text = s_text.strip()
            if not s_text or s_conf < config.OCR_CONFIDENCE_THRESHOLD:
                continue

            s_center = (
                sum(p[0] for p in s_bbox) / 4,
                sum(p[1] for p in s_bbox) / 4,
            )
            s_rect = self._bbox_to_rect(s_bbox)

            # Verificar se ja existe uma deteccao similar na mesma area
            best_match_idx = -1
            best_match_score = 0.0  # Maior IoU ou menor distancia

            for idx, (p_bbox, p_text, p_conf) in enumerate(merged):
                p_center = (
                    sum(p[0] for p in p_bbox) / 4,
                    sum(p[1] for p in p_bbox) / 4,
                )
                p_rect = self._bbox_to_rect(p_bbox)

                # Checar distancia de centro
                dist = ((s_center[0] - p_center[0]) ** 2 + (s_center[1] - p_center[1]) ** 2) ** 0.5
                if dist < 50:
                    score = 1.0 - dist / 50
                    if score > best_match_score:
                        best_match_idx = idx
                        best_match_score = score
                    continue

                # Checar sobreposicao de bbox (IoU ou contencao)
                iou = self._bbox_iou(s_rect, p_rect)
                if iou > 0.2:
                    if iou > best_match_score:
                        best_match_idx = idx
                        best_match_score = iou
                    continue

                # Checar se centro de um esta dentro do bbox do outro
                if self._point_in_rect(s_center, p_rect) or self._point_in_rect(p_center, s_rect):
                    score = 0.5
                    if score > best_match_score:
                        best_match_idx = idx
                        best_match_score = score

            if best_match_idx >= 0:
                # Duplicata encontrada: manter a de maior confianca
                _, p_text, p_conf = merged[best_match_idx]
                if s_conf > p_conf:
                    merged[best_match_idx] = (s_bbox, s_text, s_conf)
            else:
                merged.append((s_bbox, s_text, s_conf))

        return merged

    @staticmethod
    def _bbox_to_rect(bbox) -> Tuple[float, float, float, float]:
        """Converte bbox 4-pontos para (xmin, ymin, xmax, ymax)."""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _bbox_iou(r1: Tuple, r2: Tuple) -> float:
        """Calcula IoU (Intersection over Union) entre dois retangulos."""
        x1 = max(r1[0], r2[0])
        y1 = max(r1[1], r2[1])
        x2 = min(r1[2], r2[2])
        y2 = min(r1[3], r2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
        area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _point_in_rect(point: Tuple, rect: Tuple) -> bool:
        """Verifica se um ponto esta dentro de um retangulo."""
        return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

    def _init_doctr(self):
        """Inicializa docTR sob demanda (download de modelos na primeira vez)."""
        if self._doctr_model is not None:
            return
        import torch
        from doctr.models import ocr_predictor

        logger.info("Inicializando docTR (download de modelos na primeira vez)...")
        self._doctr_model = ocr_predictor(
            det_arch=config.DOCTR_DET_ARCH,
            reco_arch=config.DOCTR_RECO_ARCH,
            pretrained=True,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._doctr_model = self._doctr_model.to(device)
        logger.info("docTR inicializado.")

    def _doctr_detect(self, image_path: str, orig_h: int, orig_w: int) -> list:
        """Roda docTR na imagem e retorna resultados no formato EasyOCR: [(bbox, text, conf), ...]"""
        self._init_doctr()
        from doctr.io import DocumentFile

        doc = DocumentFile.from_images(image_path)
        result = self._doctr_model(doc)

        ocr_results = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(w.value for w in line.words)
                    if not line_text.strip():
                        continue
                    line_conf = min(w.confidence for w in line.words)
                    if line_conf < config.DOCTR_CONFIDENCE_THRESHOLD:
                        continue

                    # Coordenadas normalizadas (0-1) -> pixels
                    all_geom = [w.geometry for w in line.words]
                    xmin = min(g[0][0] for g in all_geom) * orig_w
                    ymin = min(g[0][1] for g in all_geom) * orig_h
                    xmax = max(g[1][0] for g in all_geom) * orig_w
                    ymax = max(g[1][1] for g in all_geom) * orig_h

                    bbox = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                    ocr_results.append((bbox, line_text, line_conf))

        return ocr_results

    def _preprocess_for_ocr(self, image_path: str) -> np.ndarray:
        """Preprocessa imagem para melhorar qualidade do OCR.

        Aplica:
        - Upscale 2x se a imagem for pequena
        - Aumento de contraste (CLAHE)
        - Sharpening
        """
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        h, w = img.shape[:2]

        # Upscale se muito pequena (melhora muito a leitura de texto)
        if w < 1000:
            scale = 2.0
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Converter para LAB para aplicar CLAHE no canal L (luminosidade)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # CLAHE - Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Sharpening leve para definir melhor as bordas do texto
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ])
        img = cv2.filter2D(img, -1, kernel)

        return img

    def _fix_ocr_errors(self, text: str) -> str:
        """Corrige erros comuns de OCR com fontes comic estilizadas.

        EasyOCR frequentemente confunde caracteres em fontes bold/comic:
        - Numeros no lugar de letras (5->S, 6->G, 0->O, 1->I/L, 4->A, 7->T)
        - Pipe no lugar de I (|->I)
        - Caracteres coreanos falsos no lugar de combinacoes latinas
        """
        if not text:
            return text

        # Substituir caracteres coreanos falsos comuns por equivalentes latinos
        # (EasyOCR detecta partes de letras estilizadas como hangul)
        korean_to_latin = {
            "\uAEBC": "TH",   # 꺼 -> TH
            "\uB300": "TCH",  # 대 -> TCH (como em MATCH)
            "\uC57C": "",     # 야 -> remove (noise)
            "\uBC84": "B",    # 벼 -> B
            "\uC0C8": "M",    # 새 -> M
            "\uB354": "TU",   # 더 -> TU
            "\uC5B4": "ER",   # 어 -> ER
            "\uAE4A": "",     # 깊 -> remove (noise)
            "\uB9AC": "LL",   # 리 -> LL
            "\uD130": "T",    # 터 -> T
        }
        for kor, lat in korean_to_latin.items():
            if kor in text:
                text = text.replace(kor, lat)
        # Limpar letras duplicadas de substituicoes (TTCHH -> TCH)
        text = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)  # max 2 repeated

        # Remover quaisquer caracteres coreanos restantes se misturados com latino
        korean_pattern = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
        meaningful = re.sub(r"[\s\W]", "", text)
        if meaningful:
            korean_count = len(korean_pattern.findall(text))
            ratio = korean_count / len(meaningful)
            if 0 < ratio < 0.5:
                text = korean_pattern.sub("", text)

        text = re.sub(r"\s{2,}", " ", text).strip()

        if not any(c.isalpha() for c in text):
            return text

        # Substituicoes contextuais: numeros->letras em contexto de palavras
        replacements = [
            # Pipe -> I
            (r"\|", "I"),
            # 15 como palavra -> IS
            (r"\b15\b", "IS"),
            (r"\bI5\b", "IS"),
            # 5 no meio/fim de palavra -> S
            (r"(?<=[A-Za-z])5", "S"),
            (r"5(?=[A-Za-z])", "S"),
            # 6 no inicio/fim de palavra -> G
            (r"\b6(?=[A-Za-z])", "G"),
            (r"(?<=[A-Za-z])6\b", "G"),
            # 0 como letra
            (r"\b0F\b", "OF"),
            (r"(?<=[A-Za-z])0(?=[A-Za-z])", "O"),
            # 1 no meio de palavra -> I (GH1SLAIN -> GHISLAIN)
            (r"(?<=[A-Za-z])1(?=[A-Za-z])", "I"),
            # 4 como letra A
            (r"(?<=[A-Za-z])4(?=[A-Za-z])", "A"),
            (r"(?<=[A-Za-z])4\b", "A"),       # MAN4 -> MANA
            (r"\b4\b(?=\s+[A-Za-z])", "A"),
            # 7 como T
            (r"7(?=[Hh])", "T"),         # 7H -> TH
            (r"\b7-7", "T-T"),           # 7-7 -> T-T (T-THAT'S)
            (r"\b7(?=[A-Za-z]{2,})", "T"),  # 7 before 2+ letters -> T
            # K como R em contexto especifico
            (r"\byouk\b", "YOUR"),
            (r"\bcOkEs\b", "CORES"),
            (r"\bSTKIKE\b", "STRIKE"),
            (r"\bstkike\b", "STRIKE"),
            (r"\bSTKENeTH\b", "STRENGTH"),
            (r"\bMONSTkous\b", "MONSTROUS"),
            (r"\bEEFOKE\b", "BEFORE"),
            (r"\bKzND\b", "KIND"),
            (r"\bTKNIN", "TURNIN"),
            (r"\bFYLL", "FULL"),
            (r"\bDYIN6\b", "DYING"),
            (r"\bTHkOLeH\b", "THROUGH"),
            (r"\bcoLLD\b", "COULD"),
            # 50 como SO em contexto de palavras
            (r"(?<=[A-Za-z]\s)50\b", "SO"),
            (r"\b50\s+(?=[A-Za-z])", "SO "),
            # 47 -> AT em contexto
            (r"47(?=[Ss'])", "AT"),       # 47S -> ATS, 47'S -> AT'S
            # LM no inicio -> IM
            (r"\bLM(?=[A-Za-z])", "IM"),  # LMPOSSIBLE -> IMPOSSIBLE
            # Correcoes de duplas de substituicao coreana
            (r"TTCH", "TCH"),
            (r"TCHH", "TCH"),
        ]

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def _classify_language(self, text: str) -> str:
        """Classifica o idioma do texto baseado na proporcao de caracteres coreanos."""
        meaningful_chars = re.sub(r"[\s\W]", "", text)
        if not meaningful_chars:
            return "en"

        korean_count = len(config.KOREAN_CHAR_PATTERN.findall(text))
        ratio = korean_count / len(meaningful_chars)

        return "ko" if ratio > config.KOREAN_RATIO_THRESHOLD else "en"

    def _is_watermark(self, text: str, region: TextRegion, image_shape: Tuple[int, int]) -> bool:
        """Verifica se o texto e um watermark/logo de site.

        APENAS watermarks reais (URLs, logos de sites) sao filtrados.
        Nao filtra por posicao sozinha para evitar falsos positivos com dialogo.
        """
        # Verificar por padrao de texto (URLs de sites de scan)
        for pattern in config.WATERMARK_PATTERNS:
            if pattern.search(text):
                return True

        return False

    def _merge_nearby_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Mescla regioes de texto proximas que pertencem ao mesmo balao.

        EasyOCR frequentemente separa texto multi-linha em deteccoes individuais.
        Este metodo junta regioes verticalmente proximas com sobreposicao horizontal.
        """
        if len(regions) <= 1:
            return regions

        # Separar regioes por tipo (so mesclar regioes do mesmo tipo)
        translatable = [r for r in regions if r.text_type not in
                       (config.TEXT_TYPE_SFX_KOREAN, config.TEXT_TYPE_WATERMARK)]
        non_translatable = [r for r in regions if r.text_type in
                           (config.TEXT_TYPE_SFX_KOREAN, config.TEXT_TYPE_WATERMARK)]

        merged = self._merge_group(translatable)
        return merged + non_translatable

    def _merge_group(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Mescla um grupo de regioes traduziveis."""
        if len(regions) <= 1:
            return regions

        # Ordenar por Y central
        regions.sort(key=lambda r: r.center[1])

        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            current = r1
            for j in range(i + 1, len(regions)):
                if j in used:
                    continue

                r2 = regions[j]

                # Verificar proximidade vertical
                gap = r2.rect[1] - current.rect[3]  # top de r2 - bottom de current
                max_gap = current.height * config.MERGE_VERTICAL_GAP_RATIO

                if gap > max_gap:
                    break  # Muito longe verticalmente (lista esta ordenada por Y)
                if gap < -current.height:
                    continue  # Sobreposicao excessiva, pular

                # Verificar sobreposicao horizontal
                x_overlap = self._horizontal_overlap(current, r2)
                if x_overlap < config.MERGE_HORIZONTAL_OVERLAP:
                    continue

                # Mesclar
                current = self._merge_two_regions(current, r2)
                used.add(j)

            merged.append(current)

        return merged

    def _horizontal_overlap(self, r1: TextRegion, r2: TextRegion) -> float:
        """Calcula a sobreposicao horizontal relativa entre duas regioes."""
        x1_min, _, x1_max, _ = r1.rect
        x2_min, _, x2_max, _ = r2.rect

        overlap_start = max(x1_min, x2_min)
        overlap_end = min(x1_max, x2_max)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_width = overlap_end - overlap_start
        min_width = min(x1_max - x1_min, x2_max - x2_min)

        return overlap_width / min_width if min_width > 0 else 0.0

    def _merge_two_regions(self, r1: TextRegion, r2: TextRegion) -> TextRegion:
        """Combina duas regioes em uma unica."""
        # Bbox = retangulo que engloba ambas
        x1_min, y1_min, x1_max, y1_max = r1.rect
        x2_min, y2_min, x2_max, y2_max = r2.rect

        new_x_min = min(x1_min, x2_min)
        new_y_min = min(y1_min, y2_min)
        new_x_max = max(x1_max, x2_max)
        new_y_max = max(y1_max, y2_max)

        new_bbox = [
            [new_x_min, new_y_min],
            [new_x_max, new_y_min],
            [new_x_max, new_y_max],
            [new_x_min, new_y_max],
        ]

        return TextRegion(
            bbox=new_bbox,
            text=r1.text + " " + r2.text,
            confidence=min(r1.confidence, r2.confidence),
            text_type=r1.text_type,
            language=r1.language,
        )
