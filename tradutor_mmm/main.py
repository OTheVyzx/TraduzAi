"""
Tradutor Automatico MMM - CLI principal e orquestrador do pipeline.

Uso:
    python -m tradutor_mmm                          # traduz tudo de nao_traduzidos/
    python -m tradutor_mmm <arquivo.cbz>            # traduz um CBZ especifico
    python -m tradutor_mmm <pasta/>                 # traduz todos CBZ de uma pasta

Estrutura de pastas:
    nao_traduzidos/<obra>/capitulo.cbz   -> entrada
    traduzidos/<obra>/capitulo_PTBR.cbz  -> saida (CBZ)
    traduzidos/<obra>/capitulo/001.jpg   -> saida (JPGs)
"""
import sys
import os
import re
import argparse
import logging
import shutil
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from . import config
from .extractor import extract_cbz, extract_comic_info, pack_cbz, list_cbz_files
from .detector import TextDetector, TextRegion
from .style_analyzer import StyleAnalyzer
from .font_matcher import FontMatcher
from .cleaner import TextCleaner
from .translator import MangaTranslator
from .renderer import TextRenderer

logger = logging.getLogger("tradutor_mmm")


def setup_logging(verbose: bool = False):
    """Configura o sistema de logging."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("tradutor_mmm")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def _format_time(seconds: float) -> str:
    """Formata segundos em formato legivel."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _extract_obra_name(cbz_filename: str) -> str:
    """Extrai o nome da obra do nome do arquivo CBZ.

    Exemplos:
        'Ursaring (mangabuddy)_Chapter 82_787dd0.cbz' -> 'Ursaring'
        'Solo Leveling_Cap 15.cbz' -> 'Solo Leveling'
        'Titulo.cbz' -> 'Titulo'
    """
    name = os.path.splitext(cbz_filename)[0]
    # Tentar extrair antes de '(' ou '_Chapter' ou '_Cap'
    match = re.match(r"^(.+?)(?:\s*\(|_Chapter|_Cap|_ch)", name, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Tentar antes do primeiro '_'
    match = re.match(r"^(.+?)_", name)
    if match:
        return match.group(1).strip()
    return name.strip()


def process_image(
    image_path: str,
    detector: TextDetector,
    analyzer: StyleAnalyzer,
    cleaner: TextCleaner,
    translator: MangaTranslator,
    renderer: TextRenderer,
    output_path: str,
    verbose: bool = False,
) -> dict:
    """Processa uma unica imagem pelo pipeline completo."""
    stats = {
        "regions_found": 0,
        "regions_translated": 0,
        "regions_skipped": 0,
        "had_error": False,
    }

    try:
        pil_image = Image.open(image_path).convert("RGB")
        cv_image = cv2.imread(image_path)

        if cv_image is None:
            logger.warning(f"Impossivel carregar: {image_path}")
            shutil.copy2(image_path, output_path)
            stats["had_error"] = True
            return stats

        image_shape = (cv_image.shape[0], cv_image.shape[1])

        # [1] Detectar texto
        regions = detector.detect(image_path, image_shape)
        stats["regions_found"] = len(regions)

        cleanable = [
            r for r in regions
            if r.text_type != config.TEXT_TYPE_SFX_KOREAN
        ]
        to_translate = [
            r for r in regions
            if r.text_type not in (config.TEXT_TYPE_SFX_KOREAN, config.TEXT_TYPE_WATERMARK)
        ]

        if not cleanable:
            pil_image.save(output_path, "JPEG", quality=config.JPEG_QUALITY)
            return stats

        # [2] Analisar estilo
        for region in cleanable:
            analyzer.analyze(cv_image, region)

        if verbose:
            for r in regions:
                tipo = r.text_type.replace("_", " ").title()
                logger.debug(f"  [{tipo}] '{r.text[:40]}' (conf={r.confidence:.2f})")

        # [3] Traduzir
        texts_to_translate = [r.text for r in to_translate]
        if texts_to_translate:
            translations = translator.translate_batch(texts_to_translate)
        else:
            translations = []

        stats["regions_translated"] = len(translations)
        stats["regions_skipped"] = len(regions) - len(to_translate)

        if verbose and translations:
            for orig, trad in zip(texts_to_translate, translations):
                logger.debug(f"  '{orig[:30]}' -> '{trad[:30]}'")

        # Filtrar garbled
        if translations:
            valid_translate = []
            valid_translations = []
            garbled_regions = set()
            for r, t in zip(to_translate, translations):
                if renderer._is_garbled(t):
                    garbled_regions.add(id(r))
                    logger.debug(f"  Skipping garbled: '{t[:40]}'")
                else:
                    valid_translate.append(r)
                    valid_translations.append(t)
            cleanable = [r for r in cleanable if id(r) not in garbled_regions]
            to_translate = valid_translate
            translations = valid_translations

        # [4] Limpar texto original
        cleaned = cleaner.clean_image(pil_image, cleanable)

        # [5] Renderizar texto traduzido
        if to_translate and translations:
            result = renderer.render_all(cleaned, to_translate, translations)
        else:
            result = cleaned

        result.save(output_path, "JPEG", quality=config.JPEG_QUALITY)

    except Exception as e:
        logger.error(f"Erro processando {os.path.basename(image_path)}: {e}")
        try:
            shutil.copy2(image_path, output_path)
        except Exception:
            pass
        stats["had_error"] = True

    return stats


def process_cbz(
    cbz_path: str,
    obra_output_dir: str,
    verbose: bool = False,
) -> str:
    """Processa um arquivo CBZ completo.

    Args:
        cbz_path: Caminho do arquivo .cbz
        obra_output_dir: Diretorio da obra em traduzidos/ (ex: traduzidos/Ursaring/)

    Returns:
        Caminho do CBZ traduzido
    """
    t_start = time.time()

    cbz_name = os.path.splitext(os.path.basename(cbz_path))[0]
    logger.info(f"Processando: {cbz_name}")

    # Estrutura: traduzidos/<obra>/<capitulo>/001.jpg e traduzidos/<obra>/<capitulo>_PTBR.cbz
    jpg_dir = os.path.join(obra_output_dir, cbz_name)
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(obra_output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mmm_") as temp_dir:
        logger.info("Extraindo CBZ...")
        image_paths = extract_cbz(cbz_path, temp_dir)
        comic_info = extract_comic_info(cbz_path)
        logger.info(f"Extraidas {len(image_paths)} imagens")

        # Inicializar componentes
        logger.info("Inicializando pipeline de traducao...")
        detector = TextDetector()
        analyzer = StyleAnalyzer()
        font_matcher = FontMatcher()
        cleaner = TextCleaner()
        translator = MangaTranslator()

        # Detectar fonte
        logger.info("Detectando fonte mais parecida...")
        text_crops = _collect_text_crops(image_paths[:5], detector, analyzer)
        font_path = font_matcher.get_best_font(text_crops if text_crops else None)
        logger.info(f"Fonte selecionada: {os.path.basename(font_path)}")

        renderer = TextRenderer(font_path)

        # Processar cada imagem
        total = len(image_paths)
        total_stats = {
            "images_processed": 0,
            "images_with_text": 0,
            "total_regions": 0,
            "total_translated": 0,
            "errors": 0,
        }

        for idx, img_path in enumerate(image_paths, 1):
            basename = os.path.basename(img_path)
            ext = os.path.splitext(basename)[1]
            out_name = f"{idx:03d}{ext}"
            out_path = os.path.join(jpg_dir, out_name)

            logger.info(f"[{idx}/{total}] {basename}")

            stats = process_image(
                img_path, detector, analyzer, cleaner,
                translator, renderer, out_path, verbose
            )

            total_stats["images_processed"] += 1
            if stats["regions_found"] > 0:
                total_stats["images_with_text"] += 1
            total_stats["total_regions"] += stats["regions_found"]
            total_stats["total_translated"] += stats["regions_translated"]
            if stats["had_error"]:
                total_stats["errors"] += 1

        # Empacotar CBZ traduzido
        output_cbz = os.path.join(obra_output_dir, f"{cbz_name}_PTBR.cbz")
        logger.info("Empacotando CBZ traduzido...")
        pack_cbz(jpg_dir, output_cbz, comic_info)

    elapsed = time.time() - t_start

    # Resumo
    logger.info("=" * 50)
    logger.info(f"Concluido: {cbz_name}")
    logger.info(f"  Imagens processadas: {total_stats['images_processed']}")
    logger.info(f"  Imagens com texto: {total_stats['images_with_text']}")
    logger.info(f"  Regioes de texto: {total_stats['total_regions']}")
    logger.info(f"  Textos traduzidos: {total_stats['total_translated']}")
    if total_stats["errors"] > 0:
        logger.warning(f"  Erros: {total_stats['errors']}")
    logger.info(f"  JPGs salvos em: {jpg_dir}")
    logger.info(f"  CBZ salvo em: {output_cbz}")
    logger.info(f"  Tempo: {_format_time(elapsed)}")
    logger.info("=" * 50)

    return output_cbz


def _collect_text_crops(
    image_paths: List[str],
    detector: TextDetector,
    analyzer: StyleAnalyzer,
) -> list:
    """Coleta crops de texto das primeiras imagens para analise de fonte."""
    crops = []
    for img_path in image_paths:
        try:
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                continue
            regions = detector.detect(img_path, (cv_image.shape[0], cv_image.shape[1]))
            for r in regions:
                if r.text_type not in (config.TEXT_TYPE_SFX_KOREAN, config.TEXT_TYPE_WATERMARK):
                    x_min, y_min, x_max, y_max = r.rect
                    crop = cv_image[
                        max(0, y_min):min(cv_image.shape[0], y_max),
                        max(0, x_min):min(cv_image.shape[1], x_max),
                    ]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crops.append(crop_rgb)
            if len(crops) >= 5:
                break
        except Exception:
            continue
    return crops


def process_obra(obra_dir: str, verbose: bool = False):
    """Processa todos os CBZs de uma obra em nao_traduzidos/.

    Args:
        obra_dir: Caminho da pasta da obra (ex: nao_traduzidos/Ursaring/)
    """
    obra_name = os.path.basename(obra_dir)
    cbz_files = list_cbz_files(obra_dir)

    if not cbz_files:
        logger.warning(f"Nenhum .cbz encontrado em: {obra_dir}")
        return

    obra_output = os.path.join(config.OUTPUT_DIR, obra_name)
    os.makedirs(obra_output, exist_ok=True)

    logger.info(f"Obra: {obra_name} ({len(cbz_files)} capitulo(s))")

    for cbz_path in cbz_files:
        process_cbz(cbz_path, obra_output, verbose)


def process_all_untranslated(verbose: bool = False):
    """Processa todas as obras em nao_traduzidos/."""
    t_global_start = time.time()

    input_dir = config.INPUT_DIR
    if not os.path.isdir(input_dir):
        logger.error(f"Pasta de entrada nao encontrada: {input_dir}")
        logger.info("Crie a pasta 'nao_traduzidos/' e coloque as obras dentro.")
        return

    # Buscar obras (subpastas) e CBZs soltos
    obras = []
    cbz_soltos = []

    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            obras.append(item_path)
        elif item.lower().endswith(".cbz"):
            cbz_soltos.append(item_path)

    if not obras and not cbz_soltos:
        logger.error("Nenhuma obra ou arquivo .cbz encontrado em nao_traduzidos/")
        logger.info("Estrutura esperada:")
        logger.info("  nao_traduzidos/<nome_da_obra>/capitulo.cbz")
        logger.info("  ou")
        logger.info("  nao_traduzidos/capitulo.cbz")
        return

    total_cbz = 0

    # Processar CBZs soltos (extrair nome da obra do arquivo)
    for cbz_path in cbz_soltos:
        obra_name = _extract_obra_name(os.path.basename(cbz_path))
        obra_output = os.path.join(config.OUTPUT_DIR, obra_name)
        os.makedirs(obra_output, exist_ok=True)
        process_cbz(cbz_path, obra_output, verbose)
        total_cbz += 1

    # Processar obras (subpastas)
    for obra_dir in obras:
        cbz_files = list_cbz_files(obra_dir)
        total_cbz += len(cbz_files)
        process_obra(obra_dir, verbose)

    t_global_elapsed = time.time() - t_global_start

    logger.info("")
    logger.info("=" * 50)
    logger.info("  TRADUCAO COMPLETA")
    logger.info(f"  Total de capitulos: {total_cbz}")
    logger.info(f"  Tempo total: {_format_time(t_global_elapsed)}")
    logger.info(f"  Resultados em: {config.OUTPUT_DIR}")
    logger.info("=" * 50)


def main():
    """Entry point do CLI."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Tradutor Automatico MMM - Traduz manga/manhwa de EN para PT-BR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python -m tradutor_mmm                              # traduz tudo de nao_traduzidos/
  python -m tradutor_mmm capitulo.cbz                 # traduz um CBZ especifico
  python -m tradutor_mmm ./pasta_com_cbz/             # traduz todos CBZ de uma pasta
  python -m tradutor_mmm capitulo.cbz --verbose       # modo detalhado
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Arquivo .cbz ou pasta (se omitido, processa nao_traduzidos/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar detalhes de cada regiao detectada",
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=config.JPEG_QUALITY,
        help=f"Qualidade JPEG de saida (padrao: {config.JPEG_QUALITY})",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    config.JPEG_QUALITY = args.quality

    logger.info("=" * 50)
    logger.info("  Tradutor Automatico MMM")
    logger.info("  Manga/Manhwa EN -> PT-BR")
    logger.info("=" * 50)

    # Sem argumentos: processar tudo de nao_traduzidos/
    if args.input is None:
        process_all_untranslated(args.verbose)
        return

    input_path = os.path.abspath(args.input)

    if os.path.isfile(input_path) and input_path.lower().endswith(".cbz"):
        # CBZ especifico -> extrair nome da obra, salvar em traduzidos/<obra>/
        obra_name = _extract_obra_name(os.path.basename(input_path))
        obra_output = os.path.join(config.OUTPUT_DIR, obra_name)
        os.makedirs(obra_output, exist_ok=True)

        t_start = time.time()
        process_cbz(input_path, obra_output, args.verbose)
        elapsed = time.time() - t_start

        logger.info("")
        logger.info(f"Tempo total: {_format_time(elapsed)}")

    elif os.path.isdir(input_path):
        # Pasta com CBZs -> tratar como obra
        t_start = time.time()
        process_obra(input_path, args.verbose)
        elapsed = time.time() - t_start

        logger.info("")
        logger.info(f"Tempo total: {_format_time(elapsed)}")

    else:
        logger.error(f"Entrada invalida: {input_path}")
        logger.info("Use: python -m tradutor_mmm [arquivo.cbz | pasta/ | (vazio)]")
        sys.exit(1)


if __name__ == "__main__":
    main()
