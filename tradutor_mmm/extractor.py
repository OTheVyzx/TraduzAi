"""
Extracao e empacotamento de arquivos CBZ.
CBZ = ZIP contendo imagens (JPG/PNG) + opcional ComicInfo.xml
"""
import os
import re
import zipfile
import shutil
from typing import List, Optional, Tuple


def natural_sort_key(path: str) -> List:
    """Chave de ordenacao natural para nomes de arquivo.
    Lida com padroes como '001__002.jpg' e '042.jpg'.
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    parts = re.split(r"(\d+)", basename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def is_image_file(filename: str) -> bool:
    """Verifica se o arquivo e uma imagem suportada."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def extract_cbz(cbz_path: str, output_dir: str) -> List[str]:
    """Extrai um arquivo CBZ para um diretorio.

    Args:
        cbz_path: Caminho do arquivo .cbz
        output_dir: Diretorio de destino

    Returns:
        Lista ordenada de caminhos absolutos das imagens extraidas
    """
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(cbz_path, "r") as zf:
        # Extrair apenas imagens (ignorar ComicInfo.xml e outros)
        image_names = [n for n in zf.namelist() if is_image_file(n)]

        for name in image_names:
            # Extrair para o output_dir com nome flat (sem subdiretorios)
            data = zf.read(name)
            # Usar apenas o basename para evitar subdiretorios do zip
            out_name = os.path.basename(name)
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "wb") as f:
                f.write(data)

    # Listar e ordenar as imagens extraidas
    extracted = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if is_image_file(f)
    ]
    extracted.sort(key=natural_sort_key)
    return extracted


def extract_comic_info(cbz_path: str) -> Optional[str]:
    """Extrai o conteudo do ComicInfo.xml de um CBZ, se existir.

    Returns:
        String XML ou None
    """
    try:
        with zipfile.ZipFile(cbz_path, "r") as zf:
            for name in zf.namelist():
                if name.lower() == "comicinfo.xml":
                    return zf.read(name).decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def pack_cbz(
    image_dir: str,
    output_path: str,
    comic_info_xml: Optional[str] = None,
) -> str:
    """Empacota imagens de um diretorio em um arquivo CBZ.

    Args:
        image_dir: Diretorio contendo as imagens
        output_path: Caminho do arquivo .cbz de saida
        comic_info_xml: Conteudo opcional do ComicInfo.xml

    Returns:
        Caminho do arquivo CBZ criado
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Coletar e ordenar imagens
    images = [
        f for f in os.listdir(image_dir) if is_image_file(f)
    ]
    images.sort(key=natural_sort_key)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        for idx, img_name in enumerate(images, start=1):
            # Renomear para sequencial: 001.jpg, 002.jpg, ...
            ext = os.path.splitext(img_name)[1]
            new_name = f"{idx:03d}{ext}"
            img_path = os.path.join(image_dir, img_name)
            zf.write(img_path, new_name)

        # Adicionar ComicInfo.xml se fornecido
        if comic_info_xml:
            zf.writestr("ComicInfo.xml", comic_info_xml.encode("utf-8"))

    return output_path


def list_cbz_files(folder_path: str) -> List[str]:
    """Lista todos os arquivos .cbz em uma pasta, ordenados naturalmente.

    Args:
        folder_path: Caminho da pasta

    Returns:
        Lista de caminhos completos dos arquivos .cbz
    """
    if not os.path.isdir(folder_path):
        return []

    cbz_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".cbz")
    ]
    cbz_files.sort(key=natural_sort_key)
    return cbz_files
