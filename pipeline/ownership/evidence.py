"""Pure comparison and audit ledger for immutable source OCR evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence


_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _items(value: Any, key: str) -> list[Any]:
    result = _field(value, key, ()) or ()
    return list(result) if isinstance(result, Sequence) and not isinstance(result, str) else []


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = _field(value, "bbox_page") or _field(value, "bbox")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_contains(container: Any, contained: Any, *, tolerance: float = 2.0) -> bool:
    outer = _bbox(container)
    inner = _bbox(contained)
    if outer is None or inner is None:
        return False
    return bool(
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] + tolerance >= inner[2]
        and outer[3] + tolerance >= inner[3]
    )


def normalize_evidence_tokens(text: str) -> tuple[str, ...]:
    """Normalize OCR evidence while preserving letter/digit token boundaries."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = _LETTER_DIGIT_BOUNDARY.sub(" ", normalized)
    return tuple(_TOKEN.findall(normalized))


def _contains_contiguous(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(haystack) <= len(needle):
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def safely_dominates(
    *,
    complete: Any,
    truncated: Any,
    same_region: bool,
    corroboration_count: int,
) -> bool:
    """Return whether fuller coherent evidence may safely dominate a truncation."""

    complete_tokens = normalize_evidence_tokens(str(_field(complete, "text", "")))
    truncated_tokens = normalize_evidence_tokens(str(_field(truncated, "text", "")))
    if not same_region or not _contains_contiguous(complete_tokens, truncated_tokens):
        return False

    complete_components = {str(value) for value in _items(complete, "component_ids")}
    truncated_components = {str(value) for value in _items(truncated, "component_ids")}
    if truncated_components and not truncated_components.issubset(complete_components):
        return False

    geometry_sufficient = bool(
        complete_components > truncated_components
        and _bbox_contains(complete, truncated)
    )
    return int(corroboration_count) >= 2 or geometry_sufficient


@dataclass(frozen=True)
class SourceEvidenceRecord:
    observation_id: str
    owner_id: str | None
    component_ids: tuple[str, ...]
    normalized_tokens: tuple[str, ...]
    component_coverage: float
    observation_precision: float
    material: bool
    disposition: str
    reason: str | None = None
    dominant_observation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_source_evidence_ledger(graph: Any) -> tuple[SourceEvidenceRecord, ...]:
    """Derive a complete write-only audit ledger from an owner graph snapshot."""

    observations = _items(graph, "observations")
    owners = _items(graph, "owners")
    owner_by_observation: dict[str, Any] = {}
    for owner in owners:
        for observation_id in _items(owner, "observation_ids"):
            owner_by_observation[str(observation_id)] = owner

    observation_by_id = {
        str(_field(observation, "observation_id", "")): observation
        for observation in observations
    }
    rows: list[SourceEvidenceRecord] = []
    for observation_id in sorted(observation_by_id):
        observation = observation_by_id[observation_id]
        component_ids = tuple(
            sorted({str(value) for value in _items(observation, "component_ids")})
        )
        tokens = normalize_evidence_tokens(str(_field(observation, "text", "")))
        owner = owner_by_observation.get(observation_id)
        owner_id = str(_field(owner, "owner_id", "")).strip() or None
        owner_components = {str(value) for value in _items(owner, "component_ids")}
        observation_components = set(component_ids)
        intersection = owner_components & observation_components
        component_coverage = (
            len(intersection) / len(owner_components) if owner_components else 0.0
        )
        observation_precision = (
            len(intersection) / len(observation_components)
            if observation_components
            else 0.0
        )

        rejection_reason = str(_field(observation, "rejection_reason", "") or "").strip()
        selected_ids = {str(value) for value in _items(owner, "selected_observation_ids")}
        dominant_id: str | None = None
        reason: str | None = None
        if owner is None:
            disposition = "unowned"
            reason = "no_owner_assignment"
        elif observation_id in selected_ids:
            disposition = "selected"
            reason = "selected_for_owner"
        elif rejection_reason.startswith("policy:"):
            disposition = "preserve"
            reason = rejection_reason
        elif rejection_reason:
            disposition = "rejected"
            reason = rejection_reason
        else:
            disposition = "unselected"
            reason = "not_selected"
            for candidate_id in sorted(selected_ids):
                candidate = observation_by_id.get(candidate_id)
                if candidate is None:
                    continue
                same_region = bool(
                    observation_components
                    and observation_components.issubset(owner_components)
                    and {str(value) for value in _items(candidate, "component_ids")}.issubset(
                        owner_components
                    )
                )
                if safely_dominates(
                    complete=candidate,
                    truncated=observation,
                    same_region=same_region,
                    corroboration_count=len(selected_ids),
                ):
                    disposition = "dominated"
                    reason = "safely_dominated_by_complete_evidence"
                    dominant_id = candidate_id
                    break

        rows.append(
            SourceEvidenceRecord(
                observation_id=observation_id,
                owner_id=owner_id,
                component_ids=component_ids,
                normalized_tokens=tokens,
                component_coverage=round(component_coverage, 6),
                observation_precision=round(observation_precision, 6),
                material=bool(tokens),
                disposition=disposition,
                reason=reason,
                dominant_observation_id=dominant_id,
            )
        )
    return tuple(rows)


class OwnerObservationCollisionError(ValueError):
    """Raised when one observation ID is reused for incompatible evidence."""


def merge_observation_strict(
    existing: Any,
    incoming: Any,
) -> Any:
    """Merge provenance only when identity, payload and physical input agree."""

    from .model import TextObservation

    if isinstance(existing, TextObservation) and isinstance(incoming, TextObservation):
        return _merge_typed_observation_strict(existing, incoming)
    if isinstance(existing, Mapping) and isinstance(incoming, Mapping):
        return _merge_mapping_observation_strict(existing, incoming)
    raise TypeError("observation merge requires two typed observations or two mappings")


def _merge_typed_observation_strict(existing: Any, incoming: Any) -> Any:
    from dataclasses import replace

    if existing.observation_id != incoming.observation_id:
        raise OwnerObservationCollisionError("observation identities differ")
    immutable_fields = (
        "run_id",
        "origin_execution_id",
        "page_id",
        "page_source_sha256",
        "root_input_pixel_sha256",
        "input_pixel_sha256",
        "invocation_id",
        "attempt_id",
        "provider_family",
        "payload_sha256",
        "text",
        "bbox_page",
    )
    conflicts = tuple(
        field_name
        for field_name in immutable_fields
        if getattr(existing, field_name) != getattr(incoming, field_name)
    )
    if conflicts:
        raise OwnerObservationCollisionError(
            "observation collision changed " + ", ".join(conflicts)
        )
    return replace(
        existing,
        component_ids=tuple(sorted(set(existing.component_ids) | set(incoming.component_ids))),
        polygons_page=tuple(
            sorted(set(existing.polygons_page) | set(incoming.polygons_page))
        ),
        tile_provenance=tuple(
            sorted(set(existing.tile_provenance) | set(incoming.tile_provenance))
        ),
        projection_ids=tuple(
            sorted(set(existing.projection_ids) | set(incoming.projection_ids))
        ),
    )


def _merge_mapping_observation_strict(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    import copy
    import json

    existing_id = str(existing.get("observation_id") or "")
    incoming_id = str(incoming.get("observation_id") or "")
    if existing_id != incoming_id:
        raise OwnerObservationCollisionError("observation identities differ")

    immutable_fields = (
        "run_id",
        "origin_execution_id",
        "page_id",
        "page_source_sha256",
        "root_input_pixel_sha256",
        "input_pixel_sha256",
        "invocation_id",
        "attempt_id",
        "provider_family",
        "payload_sha256",
        "text",
        "bbox_page",
    )
    conflicts = tuple(
        field_name
        for field_name in immutable_fields
        if existing.get(field_name) not in (None, "", [], ())
        and incoming.get(field_name) not in (None, "", [], ())
        and existing.get(field_name) != incoming.get(field_name)
    )
    if conflicts:
        raise OwnerObservationCollisionError(
            f"observation_id collision for {existing_id!r}: conflicting "
            + ", ".join(conflicts)
        )

    merged = copy.deepcopy(dict(existing))
    for field_name in (
        "component_ids",
        "polygons_page",
        "tile_provenance",
        "projection_ids",
    ):
        values: list[Any] = []
        seen: set[str] = set()
        for value in [
            *list(existing.get(field_name) or ()),
            *list(incoming.get(field_name) or ()),
        ]:
            token = json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            if token in seen:
                continue
            seen.add(token)
            values.append(copy.deepcopy(value))
        merged[field_name] = values

    for field_name, incoming_value in incoming.items():
        if field_name in {
            "component_ids",
            "polygons_page",
            "tile_provenance",
            "projection_ids",
            "legacy_selected",
        }:
            continue
        if merged.get(field_name) in (None, "", [], ()) and incoming_value not in (
            None,
            "",
            [],
            (),
        ):
            merged[field_name] = copy.deepcopy(incoming_value)
    merged["legacy_selected"] = bool(existing.get("legacy_selected")) or bool(
        incoming.get("legacy_selected")
    )
    return merged
