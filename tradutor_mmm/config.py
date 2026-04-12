"""
Configuracoes e constantes do Tradutor Automatico MMM.
"""
import os
import re

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FONT_DIR = os.path.join(BASE_DIR, "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

# Pastas de entrada/saida
INPUT_DIR = os.path.join(PROJECT_ROOT, "nao_traduzidos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "traduzidos")

# ============================================================
# OCR (EasyOCR)
# ============================================================
OCR_LANGUAGES = ["en", "ko"]
OCR_GPU = True
OCR_CONFIDENCE_THRESHOLD = 0.20
OCR_TEXT_THRESHOLD = 0.4
OCR_LOW_TEXT = 0.3
OCR_CANVAS_SIZE = 2560

# ============================================================
# DETECCAO DE LINGUA - SFX Coreano
# ============================================================
KOREAN_CHAR_PATTERN = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
KOREAN_RATIO_THRESHOLD = 0.3  # >30% chars coreanos = SFX

# ============================================================
# WATERMARK
# ============================================================
WATERMARK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"LAGOONSCANS?\.COM",
        r"MEDIOCRESCAN\.COM",
        r"mangabuddy",
        r"mangaflix",
        r"mangaball",
        r"ursaring",
        r"READ\s*ONLY\s*AT",
        r"WARNING\s*!",
        r"BETTER\s*QUALITY",
        r"MORE\s*CHAPTERS",
        r"FAST\s*UPDATES",
        r"MORE\s*CONTENT",
        r"LEIA\s*PRIMEIRO",
        r"DISCORD\.GG",
    ]
]
WATERMARK_Y_THRESHOLD = 0.08  # top/bottom 8% da imagem

# ============================================================
# TIPOS DE TEXTO
# ============================================================
TEXT_TYPE_SPEECH_BUBBLE = "SPEECH_BUBBLE"
TEXT_TYPE_INTENSE_BUBBLE = "INTENSE_BUBBLE"
TEXT_TYPE_NARRATION_OVERLAY = "NARRATION_OVERLAY"
TEXT_TYPE_WATERMARK = "WATERMARK"
TEXT_TYPE_SFX_KOREAN = "SFX_KOREAN"

# ============================================================
# ANALISE DE ESTILO
# ============================================================
BUBBLE_WHITE_THRESHOLD = 190       # RGB > este valor = fundo branco
BUBBLE_DARK_THRESHOLD = 100        # RGB < este valor = fundo escuro
BG_STD_UNIFORM_THRESHOLD = 35      # std < = fundo uniforme (balao)
BG_STD_VARIED_THRESHOLD = 45       # std > = fundo variado (overlay)
BG_SAMPLE_EXPAND_PX = 15           # pixels de expansao para amostragem do fundo
TEXT_SAMPLE_INNER_RATIO = 0.8      # usar 80% interno do bbox para amostra de texto

# ============================================================
# LIMPEZA / INPAINTING
# ============================================================
BUBBLE_FILL_MARGIN = 4             # pixels extras ao redor do texto para fill
INPAINT_MASK_DILATION = 7          # dilatacao da mascara para inpainting
INPAINT_KERNEL_SIZE = 5            # tamanho do kernel de dilatacao

# ============================================================
# FONTE
# ============================================================
FALLBACK_FONT_PATH = "C:/Windows/Fonts/comicbd.ttf"  # Comic Sans MS Bold
MIN_FONT_SIZE = 12
MAX_FONT_SIZE = 60

# Catalogo de fontes comic gratuitas para download
# Cada entrada: (nome, url_download, perfil de caracteristicas)
# Perfil: (peso 0-1, largura 0-1, serifa 0-1, estilo: "comic"|"manga"|"impact")
FONT_CATALOG = {
    "bangers": {
        "name": "Bangers",
        "filename": "bangers.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf",
        "profile": {"weight": 0.8, "width": 0.55, "serif": 0.0, "style": "comic"},
    },
    "permanent_marker": {
        "name": "Permanent Marker",
        "filename": "permanentmarker.ttf",
        "url": "https://github.com/google/fonts/raw/main/apache/permanentmarker/PermanentMarker-Regular.ttf",
        "profile": {"weight": 0.7, "width": 0.5, "serif": 0.0, "style": "comic"},
    },
    "bungee": {
        "name": "Bungee",
        "filename": "badaboom.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/bungee/Bungee-Regular.ttf",
        "profile": {"weight": 0.85, "width": 0.6, "serif": 0.0, "style": "comic"},
    },
}

# ============================================================
# TRADUCAO
# ============================================================
TRANSLATION_SOURCE = "en"
TRANSLATION_TARGET = "pt"
TRANSLATION_BATCH_SEPARATOR = " ||| "
TRANSLATION_MAX_RETRIES = 3
TRANSLATION_RETRY_DELAY = 0.5  # segundos
TRANSLATION_BATCH_MAX_CHARS = 4000

# ============================================================
# OUTPUT
# ============================================================
JPEG_QUALITY = 95

# ============================================================
# OCR FALLBACK (docTR)
# ============================================================
DOCTR_ENABLED = True
DOCTR_DET_ARCH = "db_resnet50"       # Modelo de deteccao
DOCTR_RECO_ARCH = "crnn_vgg16_bn"    # Modelo de reconhecimento
DOCTR_CONFIDENCE_THRESHOLD = 0.30    # Threshold mais alto que EasyOCR

# ============================================================
# MERGE DE REGIOES
# ============================================================
MERGE_VERTICAL_GAP_RATIO = 1.0    # gap < 100% da altura do texto = merge (mais agressivo)
MERGE_HORIZONTAL_OVERLAP = 0.2    # sobreposicao horizontal minima para merge (mais permissivo)
