"""Resolved, file-backed font identities and pre-raster font-run plans."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from typesetter.font_matcher import load_font_catalog


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FontAxisDefinition:
    tag: str
    minimum_value: float
    default_value: float
    maximum_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedFontIdentity:
    filename: str
    file_sha256: str
    family: str
    subfamily: str
    postscript_name: str
    weight_class: int
    width_class: int
    variation_axes: tuple[FontAxisDefinition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_sha256": self.file_sha256,
            "family": self.family,
            "subfamily": self.subfamily,
            "postscript_name": self.postscript_name,
            "weight_class": self.weight_class,
            "width_class": self.width_class,
            "variation_axes": [axis.to_dict() for axis in self.variation_axes],
        }


@dataclass(frozen=True)
class FontIdentityCatalogEntry:
    identity: ResolvedFontIdentity
    path: Path
    roles: tuple[str, ...]
    license_status: str
    covered_codepoints: frozenset[int]

    def contract_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "roles": list(self.roles),
            "license_status": self.license_status,
            "covered_codepoints_sha256": _sha256_bytes(
                _canonical_json(sorted(self.covered_codepoints))
            ),
        }


@dataclass(frozen=True)
class FontIdentityCatalog:
    entries: tuple[FontIdentityCatalogEntry, ...]
    catalog_sha256: str

    def resolve(self, alias: str) -> FontIdentityCatalogEntry:
        key = str(alias or "").strip().casefold()
        matches = [
            entry
            for entry in self.entries
            if key
            in {
                entry.identity.filename.casefold(),
                entry.identity.family.casefold(),
                entry.identity.postscript_name.casefold(),
            }
        ]
        if len(matches) != 1:
            raise ValueError(f"font alias must resolve to exactly one file-backed face: {alias!r}")
        return matches[0]

    def entry_by_sha256(self, file_sha256: str) -> FontIdentityCatalogEntry:
        matches = [entry for entry in self.entries if entry.identity.file_sha256 == file_sha256]
        if len(matches) != 1:
            raise ValueError("font identity is not present exactly once in catalog")
        return matches[0]


@dataclass(frozen=True)
class ResolvedFontIntent:
    primary_identity: ResolvedFontIdentity
    primary_path: Path
    requested_weight: int | None
    requested_width: float | None
    intent_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_identity": self.primary_identity.to_dict(),
            "requested_weight": self.requested_weight,
            "requested_width": self.requested_width,
            "intent_sha256": self.intent_sha256,
        }


@dataclass(frozen=True)
class ResolvedFontComparison:
    matches: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FontRunSpanPlan:
    text_start: int
    text_end: int
    codepoints: tuple[int, ...]
    glyph_ids: tuple[int, ...]
    font_identity: ResolvedFontIdentity
    font_path: Path
    fallback_identity: ResolvedFontIdentity | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_start": self.text_start,
            "text_end": self.text_end,
            "codepoints": list(self.codepoints),
            "glyph_ids": list(self.glyph_ids),
            "font_identity": self.font_identity.to_dict(),
            "fallback_identity": (
                self.fallback_identity.to_dict() if self.fallback_identity is not None else None
            ),
        }


@dataclass(frozen=True)
class FontRunPlan:
    text: str
    primary_identity: ResolvedFontIdentity
    primary_path: Path
    spans: tuple[FontRunSpanPlan, ...]
    configured_axes: tuple[tuple[str, float], ...]
    instance_path: Path
    instance_file_sha256: str
    resolution_kind: str
    reason: str
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "primary_identity": self.primary_identity.to_dict(),
            "spans": [span.to_dict() for span in self.spans],
            "configured_axes": [list(item) for item in self.configured_axes],
            "instance_file_sha256": self.instance_file_sha256,
            "resolution_kind": self.resolution_kind,
            "reason": self.reason,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True)
class FontRunObservation:
    text: str
    primary_identity: ResolvedFontIdentity
    spans: tuple[FontRunSpanPlan, ...]
    configured_axes: tuple[tuple[str, float], ...]
    instance_file_sha256: str
    observation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "primary_identity": self.primary_identity.to_dict(),
            "spans": [span.to_dict() for span in self.spans],
            "configured_axes": [list(item) for item in self.configured_axes],
            "instance_file_sha256": self.instance_file_sha256,
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True)
class FontRunComparison:
    matches: bool
    reasons: tuple[str, ...]


def _font_name(font: Any, name_id: int, fallback: str = "") -> str:
    table = font.get("name")
    if table is None:
        return fallback
    value = table.getDebugName(name_id)
    return str(value or fallback).strip()


def _read_font_contract(path: Path) -> tuple[ResolvedFontIdentity, frozenset[int]]:
    from fontTools.ttLib import TTFont

    resolved = Path(path).resolve(strict=True)
    font = TTFont(resolved, lazy=True)
    try:
        os2 = font.get("OS/2")
        axes = tuple(
            FontAxisDefinition(
                tag=str(axis.axisTag),
                minimum_value=float(axis.minValue),
                default_value=float(axis.defaultValue),
                maximum_value=float(axis.maxValue),
            )
            for axis in sorted(
                getattr(font.get("fvar"), "axes", ()) or (), key=lambda item: str(item.axisTag)
            )
        )
        identity = ResolvedFontIdentity(
            filename=resolved.name,
            file_sha256=_sha256_file(resolved),
            family=_font_name(font, 1, resolved.stem),
            subfamily=_font_name(font, 2, "Regular"),
            postscript_name=_font_name(font, 6, resolved.stem.replace(" ", "-")),
            weight_class=int(getattr(os2, "usWeightClass", 400)),
            width_class=int(getattr(os2, "usWidthClass", 5)),
            variation_axes=axes,
        )
        coverage = frozenset(int(value) for value in (font.getBestCmap() or {}))
        return identity, coverage
    finally:
        font.close()


def resolve_font_identity(path: Path) -> ResolvedFontIdentity:
    return _read_font_contract(path)[0]


def observe_resolved_font(path: Path) -> ResolvedFontIdentity:
    return resolve_font_identity(path)


def load_font_identity_catalog(fonts_dir: Path, font_map_path: Path) -> FontIdentityCatalog:
    legacy_catalog = load_font_catalog(fonts_dir, font_map_path)
    entries = []
    for legacy in legacy_catalog:
        identity, coverage = _read_font_contract(legacy.path)
        entries.append(
            FontIdentityCatalogEntry(
                identity=identity,
                path=legacy.path.resolve(),
                roles=legacy.roles,
                license_status=legacy.license_status,
                covered_codepoints=coverage,
            )
        )
    ordered = tuple(sorted(entries, key=lambda item: item.identity.filename.casefold()))
    catalog_sha256 = _sha256_bytes(_canonical_json([entry.contract_dict() for entry in ordered]))
    return FontIdentityCatalog(entries=ordered, catalog_sha256=catalog_sha256)


def canonicalize_font_intent(
    value: str | Mapping[str, Any], catalog: FontIdentityCatalog
) -> ResolvedFontIntent:
    payload = {"font_name": value} if isinstance(value, str) else dict(value)
    entry = catalog.resolve(str(payload.get("font_name") or ""))
    raw_weight = payload.get("font_weight")
    raw_width = payload.get("font_width")
    requested_weight = int(raw_weight) if raw_weight is not None else None
    requested_width = float(raw_width) if raw_width is not None else None
    contract = {
        "primary_identity": entry.identity.to_dict(),
        "requested_weight": requested_weight,
        "requested_width": requested_width,
    }
    return ResolvedFontIntent(
        primary_identity=entry.identity,
        primary_path=entry.path,
        requested_weight=requested_weight,
        requested_width=requested_width,
        intent_sha256=_sha256_bytes(_canonical_json(contract)),
    )


def compare_resolved_font(
    expected: ResolvedFontIntent | ResolvedFontIdentity,
    observed: ResolvedFontIdentity,
) -> ResolvedFontComparison:
    intent = expected if isinstance(expected, ResolvedFontIntent) else None
    identity = intent.primary_identity if intent is not None else expected
    reasons: list[str] = []
    if identity.file_sha256 != observed.file_sha256:
        reasons.append("file_sha256")
    if intent is not None and intent.requested_weight is not None:
        if intent.requested_weight != observed.weight_class:
            reasons.append("weight_class")
    if intent is not None and intent.requested_width is not None and 1 <= intent.requested_width <= 9:
        if round(intent.requested_width) != observed.width_class:
            reasons.append("width_class")
    return ResolvedFontComparison(matches=not reasons, reasons=tuple(reasons))


def _glyph_id(path: Path, codepoint: int) -> int:
    from fontTools.ttLib import TTFont

    font = TTFont(path, lazy=True)
    try:
        glyph_name = (font.getBestCmap() or {}).get(codepoint)
        return int(font.getGlyphID(glyph_name)) if glyph_name is not None else 0
    finally:
        font.close()


def _validate_axes(
    identity: ResolvedFontIdentity, requested_axes: Mapping[str, float] | None
) -> tuple[tuple[str, float], ...]:
    requested = dict(requested_axes or {})
    definitions = {axis.tag: axis for axis in identity.variation_axes}
    if set(requested) - set(definitions):
        raise ValueError(f"font does not define axes: {sorted(set(requested) - set(definitions))}")
    configured: list[tuple[str, float]] = []
    for tag, raw_value in sorted(requested.items()):
        value = float(raw_value)
        definition = definitions[tag]
        if not definition.minimum_value <= value <= definition.maximum_value:
            raise ValueError(f"font axis {tag} is outside its declared range")
        configured.append((tag, value))
    return tuple(configured)


def _materialize_variable_instance(
    source_path: Path,
    source_identity: ResolvedFontIdentity,
    configured_axes: tuple[tuple[str, float], ...],
    cache_root: Path,
) -> tuple[Path, str]:
    if not configured_axes:
        return source_path, source_identity.file_sha256
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = _sha256_bytes(
        _canonical_json(
            {"source_file_sha256": source_identity.file_sha256, "axes": configured_axes}
        )
    )
    target = root / f"{source_path.stem}-{key[:20]}.ttf"
    if target.is_file():
        return target, _sha256_file(target)
    font = TTFont(source_path)
    try:
        instance = instantiateVariableFont(font, dict(configured_axes), inplace=False)
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}-", suffix=".tmp", dir=root, delete=False
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            instance.save(temporary)
            generated_sha256 = _sha256_file(temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        font.close()
    if _sha256_file(target) != generated_sha256:
        raise ValueError("variable font cache write failed SHA-256 revalidation")
    return target, generated_sha256


def resolve_font_run_plan(
    *,
    intent: ResolvedFontIntent,
    text: str,
    catalog: FontIdentityCatalog,
    requested_axes: Mapping[str, float] | None = None,
    cache_root: Path | None = None,
) -> FontRunPlan:
    if not text:
        raise ValueError("font run text must not be empty")
    primary_entry = catalog.entry_by_sha256(intent.primary_identity.file_sha256)
    configured_axes = _validate_axes(intent.primary_identity, requested_axes)
    if configured_axes and cache_root is None:
        raise ValueError("variable font instances require an explicit cache root")
    instance_path, instance_sha256 = _materialize_variable_instance(
        primary_entry.path,
        primary_entry.identity,
        configured_axes,
        Path(cache_root) if cache_root is not None else primary_entry.path.parent,
    )
    selected: list[FontIdentityCatalogEntry] = []
    for character in text:
        codepoint = ord(character)
        if codepoint in primary_entry.covered_codepoints:
            selected.append(primary_entry)
            continue
        fallbacks = [entry for entry in catalog.entries if codepoint in entry.covered_codepoints]
        if not fallbacks:
            raise ValueError(f"font catalog cannot render codepoint U+{codepoint:04X}")
        fallbacks.sort(
            key=lambda entry: (
                abs(entry.identity.weight_class - primary_entry.identity.weight_class),
                abs(entry.identity.width_class - primary_entry.identity.width_class),
                entry.identity.filename.casefold(),
            )
        )
        selected.append(fallbacks[0])
    spans: list[FontRunSpanPlan] = []
    start = 0
    for index in range(1, len(text) + 1):
        if index < len(text) and selected[index] == selected[start]:
            continue
        entry = selected[start]
        font_path = instance_path if entry == primary_entry else entry.path
        codepoints = tuple(ord(character) for character in text[start:index])
        glyph_ids = tuple(_glyph_id(font_path, codepoint) for codepoint in codepoints)
        spans.append(
            FontRunSpanPlan(
                text_start=start,
                text_end=index,
                codepoints=codepoints,
                glyph_ids=glyph_ids,
                font_identity=entry.identity,
                font_path=font_path,
                fallback_identity=entry.identity if entry != primary_entry else None,
            )
        )
        start = index
    resolution_kind = "derived" if any(span.fallback_identity for span in spans) else "exact"
    reason = "glyph_coverage_fallback" if resolution_kind == "derived" else ""
    contract = {
        "text": text,
        "primary_identity": intent.primary_identity.to_dict(),
        "spans": [span.to_dict() for span in spans],
        "configured_axes": configured_axes,
        "instance_file_sha256": instance_sha256,
        "resolution_kind": resolution_kind,
        "reason": reason,
    }
    return FontRunPlan(
        text=text,
        primary_identity=intent.primary_identity,
        primary_path=primary_entry.path,
        spans=tuple(spans),
        configured_axes=configured_axes,
        instance_path=instance_path,
        instance_file_sha256=instance_sha256,
        resolution_kind=resolution_kind,
        reason=reason,
        plan_sha256=_sha256_bytes(_canonical_json(contract)),
    )


def observe_font_run(plan: FontRunPlan) -> FontRunObservation:
    observed_primary = observe_resolved_font(plan.primary_path)
    observed_spans = []
    for span in plan.spans:
        observed_identity = observe_resolved_font(span.font_path)
        source_identity = span.font_identity
        if span.font_path == plan.instance_path and plan.configured_axes:
            # The static instance is observed by its own SHA below while the
            # source face remains the identity named by the plan.
            observed_identity = source_identity
        observed_spans.append(
            replace(
                span,
                font_identity=observed_identity,
                glyph_ids=tuple(_glyph_id(span.font_path, codepoint) for codepoint in span.codepoints),
            )
        )
    contract = {
        "text": plan.text,
        "primary_identity": observed_primary.to_dict(),
        "spans": [span.to_dict() for span in observed_spans],
        "configured_axes": plan.configured_axes,
        "instance_file_sha256": _sha256_file(plan.instance_path),
    }
    return FontRunObservation(
        text=plan.text,
        primary_identity=observed_primary,
        spans=tuple(observed_spans),
        configured_axes=plan.configured_axes,
        instance_file_sha256=contract["instance_file_sha256"],
        observation_sha256=_sha256_bytes(_canonical_json(contract)),
    )


def replace_font_run_observation_primary(
    observation: FontRunObservation, identity: ResolvedFontIdentity
) -> FontRunObservation:
    contract = observation.to_dict()
    contract["primary_identity"] = identity.to_dict()
    contract.pop("observation_sha256", None)
    return replace(
        observation,
        primary_identity=identity,
        observation_sha256=_sha256_bytes(_canonical_json(contract)),
    )


def compare_font_run(plan: FontRunPlan, observation: FontRunObservation) -> FontRunComparison:
    reasons: list[str] = []
    if plan.text != observation.text:
        reasons.append("text")
    if plan.primary_identity.file_sha256 != observation.primary_identity.file_sha256:
        reasons.append("primary_file_sha256")
    if plan.configured_axes != observation.configured_axes:
        reasons.append("configured_axes")
    if plan.instance_file_sha256 != observation.instance_file_sha256:
        reasons.append("instance_file_sha256")
    if len(plan.spans) != len(observation.spans):
        reasons.append("span_count")
    else:
        for index, (expected, observed) in enumerate(zip(plan.spans, observation.spans)):
            if expected.text_start != observed.text_start or expected.text_end != observed.text_end:
                reasons.append(f"span_{index}_range")
            if expected.codepoints != observed.codepoints:
                reasons.append(f"span_{index}_codepoints")
            if expected.glyph_ids != observed.glyph_ids:
                reasons.append(f"span_{index}_glyph_ids")
            if expected.font_identity.file_sha256 != observed.font_identity.file_sha256:
                reasons.append(f"span_{index}_font_identity")
    return FontRunComparison(matches=not reasons, reasons=tuple(reasons))


def glyph_advance(plan: FontRunPlan, character: str) -> int:
    if len(character) != 1:
        raise ValueError("glyph advance accepts exactly one character")
    from fontTools.ttLib import TTFont

    path = plan.instance_path
    font = TTFont(path, lazy=True)
    try:
        glyph_name = (font.getBestCmap() or {}).get(ord(character))
        if glyph_name is None:
            return 0
        advance, _ = font["hmtx"].metrics[glyph_name]
        return int(advance)
    finally:
        font.close()
