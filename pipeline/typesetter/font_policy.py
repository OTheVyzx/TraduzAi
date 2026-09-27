"""Closed, file-backed automatic font policy for the private quality replay."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from typesetter.font_identity import resolve_font_identity
from typesetter.font_matcher import (MATCH_MARGIN_THRESHOLD, MATCH_SCORE_THRESHOLD,
                                    MULTILINE_MATCH_SCORE_THRESHOLD, FontShapeMatcher,
                                    load_font_catalog)


BASE = "ComicNeue-Bold.ttf"
EXPECTED_SHA256 = {
    BASE: "c802d1294419c5584ab09cd1b9a674d16e94aa5ddc8b29705ccbf36ccede714e",
    "KOMIKAX_.ttf": "d2c790c5ce96e4453ab7ea2d17f8c71db06cec3d3ab4f7f98db02955e63ab353",
    "LeagueGothic-Regular-VariableFont_wdth.ttf": "b749ae432ebf9715c75d9213301815df40e89754800251aba0e22066f548597a",
    "ReadyforAnythingBB-Regular.ttf": "6a415e1d671f62122a5e126d672a9614d1b7be473e2d636a122b6352bbaa127b",
    "ReadyforAnythingBB-Italic.ttf": "a2a9d04ce79eaed5c203c573d690827f25a6b1f35fa929921904fafcd6cddf99",
    "ReadyforAnythingBB-Bold.ttf": "f3e51a5f2c839ed09521e0556d411588aec51c9f2198652c9bfbad50f745d1d7",
    "ReadyforAnythingBB-BoldItalic.ttf": "2de0e683bc32cc76eadbbaed16dc50947f985e8210bb007bf169aa13fccbad4e",
}

POLICY_PATH = Path(__file__).with_name("font-policy.json")


@lru_cache(maxsize=1)
def _policy() -> dict[str, Any]:
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if (value.get("schema") != "quality_font_policy_v1"
            or value.get("base") != BASE
            or value.get("score_metric") != "shape_score_v2_multiline"
            or value.get("single_line_threshold") != MATCH_SCORE_THRESHOLD
            or value.get("multiline_threshold") != MULTILINE_MATCH_SCORE_THRESHOLD
            or value.get("margin_threshold") != MATCH_MARGIN_THRESHOLD):
        raise ValueError("font policy differs from the active matcher metric")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ClosedFontMap:
    root: Path
    map_path: Path
    files: Mapping[str, Path]
    hashes: Mapping[str, str]
    roles: Mapping[str, tuple[str, ...]]
    map_sha256: str
    catalog_version: str

    @classmethod
    def load(cls, fonts_dir: Path, font_map_path: Path) -> "ClosedFontMap":
        root = Path(fonts_dir).resolve(strict=True)
        map_path = Path(font_map_path).resolve(strict=True)
        if map_path.name.startswith("font-map.closed"):
            closed = json.loads(map_path.read_text(encoding="utf-8"))
            original = root / "font-map.json"
            if closed.get("source_map_sha256") != _sha(original):
                raise ValueError("active font map changed since closed path binding")
        catalog = load_font_catalog(root, map_path)
        files = {item.font_name: item.path for item in catalog}
        if BASE not in files:
            raise ValueError("ComicNeue-Bold.ttf is absent from the active font map")
        hashes = {name: _sha(path) for name, path in files.items()}
        for name, expected in EXPECTED_SHA256.items():
            if name in files and hashes[name] != expected:
                raise ValueError(f"active font asset hash mismatch: {name}")
        return cls(root, map_path, files, hashes,
                   {item.font_name:item.roles for item in catalog},_sha(map_path),
                   FontShapeMatcher(catalog).catalog_version)

    def path(self, name: str) -> Path:
        if name not in self.files:
            raise ValueError(f"font outside active map: {name}")
        path = self.files[name]
        if _sha(path) != self.hashes[name]:
            raise ValueError(f"font asset changed after map load: {name}")
        return path


@lru_cache(maxsize=256)
def _coverage(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont

    font = TTFont(path, lazy=True)
    try:
        return frozenset((font.getBestCmap() or {}).keys())
    finally:
        font.close()


def _covers(path: Path, text: str) -> bool:
    return all(ord(ch) in _coverage(str(path)) for ch in unicodedata.normalize("NFC", text)
               if not ch.isspace())


def decide(
    font_map: ClosedFontMap,
    text: str,
    match_evidence: Mapping[str, Any] | None,
    semantic_role: str | None = None,
) -> dict[str, Any]:
    """Apply the existing shape-score threshold, never OCR confidence.

    The stored evidence must be a result of matching *this* map. Historical
    recipe font names without a scored source comparison are not evidence.
    """
    policy = _policy()
    source = dict(match_evidence or {})
    top = source.get("top_k")
    top = top if isinstance(top, list) and top else []
    best = top[0] if top and isinstance(top[0], dict) else {}
    candidate = str(best.get("font_name") or "")
    raw_score = best.get("score")
    margin = source.get("margin")
    normalization = source.get("normalization") or {}
    line_count = (normalization.get("source_line_count")
                  if isinstance(normalization, Mapping) else None)
    metric = source.get("metric")
    if metric is None and "slant_similarity" in best and isinstance(line_count, int):
        metric = "shape_score_v2_multiline"
    threshold = (policy["multiline_threshold"] if isinstance(line_count, int)
                 and line_count > 1 else policy["single_line_threshold"])
    reason = "font_match_missing"
    chosen = BASE
    try:
        score = float(raw_score)
        separation = float(margin)
        if not math.isfinite(score) or not math.isfinite(separation):
            raise ValueError("nonfinite score")
        if source.get("catalog_version") not in (None, font_map.catalog_version):
            reason = "catalog_version_mismatch"
        elif metric != policy["score_metric"]:
            reason = "score_metric_incompatible"
        elif candidate not in font_map.files:
            reason = "candidate_outside_active_map"
        elif semantic_role is not None and semantic_role not in font_map.roles[candidate]:
            reason = "candidate_role_not_authorized"
        elif score < threshold:
            reason = "shape_score_below_threshold"
        elif separation < policy["margin_threshold"]:
            reason = "insufficient_margin"
        elif not _covers(font_map.path(candidate), text):
            reason = "candidate_missing_glyph_block_fallback"
        else:
            chosen = candidate
            reason = "shape_score_accepted"
    except (TypeError, ValueError):
        if raw_score is not None:
            reason = "score_invalid"
    base_path = font_map.path(BASE)
    if not _covers(base_path, text):
        raise ValueError("ComicNeue-Bold.ttf does not cover required block characters")
    selected_path = font_map.path(chosen)
    identity = resolve_font_identity(selected_path)
    payload = {
        "schema": "font_policy_v1", "selected": chosen, "reason": reason,
        "considered": sorted(font_map.files), "score": raw_score if isinstance(raw_score, (int, float))
        and math.isfinite(float(raw_score)) else None,
        "metric": policy["score_metric"], "threshold": threshold,
        "source_line_count": line_count,
        "margin": margin if isinstance(margin, (int, float)) and math.isfinite(float(margin)) else None,
        "margin_threshold": policy["margin_threshold"],
        "policy_sha256": _sha(POLICY_PATH),
        "map_sha256": font_map.map_sha256, "font_sha256": font_map.hashes[chosen],
        "catalog_version": font_map.catalog_version,
        "identity": identity.to_dict(), "axes": {axis.tag: axis.default_value
                                                 for axis in identity.variation_axes},
        "coverage": "complete_nfc", "text_sha256": hashlib.sha256(
            unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest(),
    }
    payload["decision_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return payload


def validate_decision(font_map: ClosedFontMap, text: str,
                      decision: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(decision)
    digest = payload.pop("decision_sha256", None)
    actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                       sort_keys=True, separators=(",", ":")
                            ).encode("utf-8")).hexdigest()
    if digest != actual or payload.get("schema") != "font_policy_v1":
        raise ValueError("font decision integrity mismatch")
    name = str(payload.get("selected") or "")
    if (payload.get("map_sha256") != font_map.map_sha256
            or payload.get("policy_sha256") != _sha(POLICY_PATH)
            or payload.get("font_sha256") != font_map.hashes.get(name)
            or payload.get("catalog_version") != font_map.catalog_version
            or payload.get("text_sha256") != hashlib.sha256(
                unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()
            or not _covers(font_map.path(name), text)):
        raise ValueError("persisted font decision no longer matches map or text")
    payload["decision_sha256"] = digest
    return payload
