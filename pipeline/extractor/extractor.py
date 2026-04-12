"""
TraduzAi — Extractor
Responsável por desempacotar .cbz/.zip ou copiar imagens individuais
para uma pasta temporária (_tmp/) dentro do work_dir.

A pasta temporária é criada aqui e deve ser apagada pelo chamador
após o typesetting estar completo.
"""

import zipfile
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def extract(source_path: str | Path, work_dir: str | Path) -> tuple[list[Path], Path]:
    """
    Extrai imagens de source_path para work_dir/_tmp/.

    Suporta:
      - Arquivo .cbz ou .zip
      - Pasta com imagens
      - Imagem única (cópia para _tmp/)

    Retorna:
      (image_files, tmp_dir)
        image_files — lista ordenada de Path das imagens extraídas
        tmp_dir     — Path da pasta temporária criada (_tmp/)

    O chamador é responsável por apagar tmp_dir após o uso
    chamando cleanup(tmp_dir).
    """
    source_path = Path(source_path)
    work_dir = Path(work_dir)

    tmp_dir = work_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix.lower()

    if suffix in IMAGE_EXTS:
        _copy_single_image(source_path, tmp_dir)

    elif suffix in (".cbz", ".zip"):
        _extract_archive(source_path, tmp_dir)

    elif source_path.is_dir():
        _copy_directory(source_path, tmp_dir)

    else:
        raise ValueError(f"Formato não suportado: {source_path}")

    image_files = _sorted_images(tmp_dir)

    if not image_files:
        raise ValueError(f"Nenhuma imagem encontrada em: {source_path}")

    return image_files, tmp_dir


def cleanup(tmp_dir: str | Path) -> None:
    """Remove a pasta temporária criada pela extração."""
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Internos
# ──────────────────────────────────────────────────────────────────────────────

def _copy_single_image(source: Path, dest_dir: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {source}")
    shutil.copy2(source, dest_dir / source.name)


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            basename = Path(info.filename).name
            data = zf.read(info.filename)
            (dest_dir / basename).write_bytes(data)


def _copy_directory(source_dir: Path, dest_dir: Path) -> None:
    for f in sorted(source_dir.rglob("*")):
        if f.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(f, dest_dir / f.name)


def _sorted_images(directory: Path) -> list[Path]:
    files = [f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)
