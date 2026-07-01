"""Utilities for caching Google Fonts families as local font files.

The typesetter renders local files through FreeType. This module downloads
TTF/OTF files from the public google/fonts repository into ``fonts/google`` so
the rest of the pipeline can keep using local font paths.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GOOGLE_FONTS_API_ROOT = "https://api.github.com/repos/google/fonts/contents"
GOOGLE_FONTS_REF = "main"
GOOGLE_FONT_LICENSE_DIRS = ("ofl", "apache", "ufl")
WEIGHT_NAMES = {
    100: "Thin",
    200: "ExtraLight",
    300: "Light",
    400: "Regular",
    500: "Medium",
    600: "SemiBold",
    700: "Bold",
    800: "ExtraBold",
    900: "Black",
}


@dataclass(frozen=True)
class GoogleFontSpec:
    family: str
    weight: int = 400
    italic: bool = False


def google_family_slug(family: str) -> str:
    """Return the google/fonts repository directory slug for a family name."""
    return re.sub(r"[^a-z0-9]+", "", family.lower())


def _http_json(url: str, timeout: float = 15.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "TraduzAI-font-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, destination: Path, timeout: float = 30.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "TraduzAI-font-cache"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        destination.write_bytes(response.read())


def _score_google_font_file(name: str, spec: GoogleFontSpec) -> tuple[int, str]:
    lower = name.lower()
    if not lower.endswith((".ttf", ".otf")):
        return (-1000, name)

    score = 0
    weight_name = WEIGHT_NAMES.get(spec.weight, str(spec.weight)).lower()
    wants_italic = spec.italic
    has_italic = "italic" in lower

    if wants_italic == has_italic:
        score += 20
    else:
        score -= 20

    if weight_name in lower:
        score += 30
    if spec.weight == 400 and "regular" in lower:
        score += 25
    if spec.weight == 700 and "bold" in lower:
        score += 25
    if "[" in name and "]" in name:
        score += 5
    return (score, name)


def download_google_font_family(
    spec: GoogleFontSpec,
    fonts_dir: Path,
    *,
    cache_subdir: str = "google",
) -> Path:
    """Download one Google Fonts family file and return the local path.

    Raises ``RuntimeError`` if the family cannot be found or no TTF/OTF file is
    available in the public google/fonts repository.
    """
    slug = google_family_slug(spec.family)
    last_error: Exception | None = None

    for license_dir in GOOGLE_FONT_LICENSE_DIRS:
        api_url = (
            f"{GOOGLE_FONTS_API_ROOT}/{license_dir}/{slug}"
            f"?ref={GOOGLE_FONTS_REF}"
        )
        try:
            entries = _http_json(api_url)
        except Exception as exc:
            last_error = exc
            continue

        files = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and str(entry.get("name", "")).lower().endswith((".ttf", ".otf"))
            and entry.get("download_url")
        ]
        if not files:
            continue

        chosen = max(files, key=lambda entry: _score_google_font_file(str(entry["name"]), spec))
        destination = fonts_dir / cache_subdir / str(chosen["name"])
        if not destination.exists():
            _download_file(str(chosen["download_url"]), destination)
        return destination

    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"Google Font family not found: {spec.family}{detail}")


def specs_from_font_map(font_map_path: Path) -> list[GoogleFontSpec]:
    """Read optional Google Fonts entries from fonts/font-map.json."""
    if not font_map_path.exists():
        return []

    try:
        data = json.loads(font_map_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    specs: list[GoogleFontSpec] = []
    for entry in data.get("available", []):
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("google_family") or "").strip()
        if not family:
            continue
        try:
            weight = int(entry.get("google_weight") or 400)
        except (TypeError, ValueError):
            weight = 400
        specs.append(
            GoogleFontSpec(
                family=family,
                weight=weight,
                italic=bool(entry.get("google_italic", False)),
            )
        )
    return specs


def ensure_google_fonts_from_map(fonts_dir: Path, font_map_path: Path) -> list[Path]:
    """Download configured Google Fonts from font-map.json."""
    downloaded: list[Path] = []
    for spec in specs_from_font_map(font_map_path):
        try:
            downloaded.append(download_google_font_family(spec, fonts_dir))
        except Exception:
            continue
    return downloaded
