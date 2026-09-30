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
from pathlib import PurePosixPath
import re
import unicodedata

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

    suffix = source_path.suffix.lower()

    if suffix in (".cbz", ".zip"):
        _raise_if_traduzai_project_archive(source_path)
    elif source_path.is_dir():
        _raise_if_traduzai_project_directory(source_path)

    tmp_dir = work_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

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

    _raise_if_traduzai_project_archive(archive_path)

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [
            (info, _safe_relative_image_path(info.filename))
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_EXTS
        ]
        _require_unique_relative_paths([relative for _, relative in members])
        for info, relative in members:
            destination = dest_dir / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def _copy_directory(source_dir: Path, dest_dir: Path) -> None:
    _raise_if_traduzai_project_directory(source_dir)

    files = sorted(
        (path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: path.relative_to(source_dir).as_posix(),
    )
    relatives = [_safe_relative_image_path(path.relative_to(source_dir).as_posix()) for path in files]
    _require_unique_relative_paths(relatives)
    for source, relative in zip(files, relatives, strict=True):
        destination = dest_dir / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _raise_if_traduzai_project_directory(source_dir: Path) -> None:
    if (source_dir / "project.json").exists():
        raise ValueError(
            "Esta pasta ja e um projeto TraduzAi. Use Abrir projeto para continuar, "
            "nao Nova traducao."
        )


def _raise_if_traduzai_project_archive(archive_path: Path) -> None:
    if not archive_path.exists():
        return
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if not info.is_dir() and Path(info.filename).name.lower() == "project.json":
                raise ValueError(
                    "Este arquivo ja e um projeto/exportacao do TraduzAi. Extraia o ZIP e use "
                    "Abrir projeto para continuar, nao Nova traducao."
                )


def _sorted_images(directory: Path) -> list[Path]:
    files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda path: path.relative_to(directory).as_posix())


def _safe_relative_image_path(value: str) -> PurePosixPath:
    raw = str(value or "")
    if not raw or "\\" in raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ValueError(f"Caminho inseguro no arquivo de origem: {raw!r}")
    relative = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Caminho inseguro no arquivo de origem: {raw!r}")
    if relative.as_posix() != raw:
        raise ValueError(f"Caminho não canônico no arquivo de origem: {raw!r}")
    return relative


def _require_unique_relative_paths(paths: list[PurePosixPath]) -> None:
    seen: set[str] = set()
    for path in paths:
        key = unicodedata.normalize("NFC", path.as_posix()).casefold()
        if key in seen:
            raise ValueError(f"Colisão de caminho no arquivo de origem: {path.as_posix()}")
        seen.add(key)
