from __future__ import annotations

import hashlib
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile

from server.config import Settings


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".zip", ".cbz"}
MAGIC = {
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".webp": [b"RIFF"],
    ".zip": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
    ".cbz": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
}


@dataclass(slots=True)
class PreparedUpload:
    path: Path
    filename: str
    suffix: str
    mime_type: str | None
    size: int
    sha256: str
    page_count: int


async def prepare_upload(file: UploadFile, settings: Settings) -> PreparedUpload:
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="tipo de arquivo nao permitido")
    digest = hashlib.sha256()
    total = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    try:
        first = b""
        while chunk := await file.read(1024 * 1024):
            if not first:
                first = chunk[:16]
            total += len(chunk)
            if total > settings.max_file_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail="arquivo muito grande")
            digest.update(chunk)
            tmp.write(chunk)
    finally:
        tmp.close()
    if not any(first.startswith(prefix) for prefix in MAGIC[suffix]):
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="assinatura do arquivo invalida")
    page_count = 1
    if suffix in {".zip", ".cbz"}:
        page_count = _validate_archive(tmp_path, settings)
    return PreparedUpload(
        path=tmp_path,
        filename=filename,
        suffix=suffix,
        mime_type=file.content_type,
        size=total,
        sha256=digest.hexdigest(),
        page_count=page_count,
    )


def prepare_upload_from_drive_link(link: str, settings: Settings) -> PreparedUpload:
    source_url, filename_hint = _google_drive_download_url(link)
    suffix = Path(filename_hint).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".zip"
    digest = hashlib.sha256()
    total = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    first = b""
    try:
        request = urllib.request.Request(source_url, headers={"User-Agent": "TraduzAI/0.2"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("content-type")
            cd_filename = _filename_from_content_disposition(response.headers.get("content-disposition"))
            if cd_filename:
                filename_hint = cd_filename
                suffix = Path(filename_hint).suffix.lower() or suffix
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if not first:
                    first = chunk[:16]
                total += len(chunk)
                if total > settings.max_file_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="arquivo muito grande")
                digest.update(chunk)
                tmp.write(chunk)
    finally:
        tmp.close()
    if suffix not in ALLOWED_SUFFIXES:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="tipo de arquivo do Drive nao permitido")
    if not any(first.startswith(prefix) for prefix in MAGIC[suffix]):
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="link do Drive nao retornou um arquivo valido")
    page_count = _validate_archive(tmp_path, settings) if suffix in {".zip", ".cbz"} else 1
    return PreparedUpload(
        path=tmp_path,
        filename=Path(filename_hint).name or f"google-drive{suffix}",
        suffix=suffix,
        mime_type=content_type,
        size=total,
        sha256=digest.hexdigest(),
        page_count=page_count,
    )


def _validate_archive(path: Path, settings: Settings) -> int:
    expanded = 0
    pages = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > settings.max_files_per_job:
            raise HTTPException(status_code=413, detail="arquivo com itens demais")
        for info in infos:
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise HTTPException(status_code=400, detail="zip inseguro")
            expanded += info.file_size
            if expanded > settings.max_zip_expanded_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail="zip expandido muito grande")
            if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                pages += 1
    return max(pages, 1)


def _google_drive_download_url(link: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(link.strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="link do Google Drive invalido")
    query = urllib.parse.parse_qs(parsed.query)
    file_id = query.get("id", [""])[0]
    parts = [part for part in parsed.path.split("/") if part]
    if "file" in parts and "d" in parts:
        index = parts.index("d")
        if index + 1 < len(parts):
            file_id = parts[index + 1]
    if "drive.google.com" in parsed.netloc and file_id:
        return f"https://drive.google.com/uc?export=download&id={urllib.parse.quote(file_id)}", f"{file_id}.zip"
    return link.strip(), Path(parsed.path).name or "google-drive.zip"


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')
    return None
