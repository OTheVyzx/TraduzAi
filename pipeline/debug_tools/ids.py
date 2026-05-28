from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone


def slugify_run_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return label or "run"


def make_run_id(label: str = "") -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = slugify_run_label(label)
    digest = hashlib.sha1(f"{created}:{label}".encode("utf-8", errors="ignore")).hexdigest()[:6]
    return f"{created}_{slug}_{digest}"


def page_id(page_number: int) -> str:
    return f"page_{int(page_number):03d}"


def band_id(page_number: int, band_index: int) -> str:
    return f"{page_id(page_number)}_band_{int(band_index):03d}"


def page_id_from_band_id(value: str | None) -> str | None:
    """Return the source page id encoded in a band id.

    Band ids are the stable bridge across strip/debug artifacts. Keep this
    helper tiny so stages can derive the same page key without importing strip
    internals.
    """

    if not value:
        return None
    match = re.match(r"^(page_\d{3})_band_\d{3}$", str(value))
    return match.group(1) if match else None


def make_trace_id(text_id: str | None, band_id_value: str | None) -> str | None:
    text = str(text_id or "").strip()
    band = str(band_id_value or "").strip()
    if not text or not band:
        return None
    return f"{text}@{band}"


def make_text_instance_id(text_id: str | None, band_id_value: str | None) -> str | None:
    text = str(text_id or "").strip()
    band = str(band_id_value or "").strip()
    if not text or not band:
        return None
    return f"{band}_{text}"


def candidate_id(page_number: int, band_index: int, candidate_index: int) -> str:
    return f"{band_id(page_number, band_index)}_cand_{int(candidate_index):03d}"


def block_id(page_number: int, band_index: int, block_index: int) -> str:
    return f"{band_id(page_number, band_index)}_block_{int(block_index):03d}"
