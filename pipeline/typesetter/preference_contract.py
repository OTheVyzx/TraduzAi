"""Renderer-owned contract for safe A/B layout preference comparisons.

The renderer publishes immutable candidate identities and resolves a displayed
position back to that identity.  Studio owns collection/persistence of the
answer; no answer produced here is treated as a calibrated human profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping

from ownership.hash_contract import canonical_json_sha256


SCHEMA_VERSION = "traduzai.renderer-preference.v1"
RESOLUTION_SCHEMA = "traduzai.renderer-preference-resolution.v1"
PREFERENCE_PROFILE = "uncalibrated"
PreferenceChoice = Literal["A", "B", "equivalent", "neither", "unsure"]
_CHOICES = {"A", "B", "equivalent", "neither", "unsure"}


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))


def _require_sha256(value: Any, field: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return candidate


def _require_relative_path(value: Any, field: str) -> str:
    candidate = str(value or "")
    if not candidate or "\\" in candidate or candidate.startswith("/") or ":" in candidate:
        raise ValueError(f"{field} must be a portable relative path")
    if any(part in {"", ".", ".."} for part in candidate.split("/")):
        raise ValueError(f"{field} must not escape its root")
    return candidate


def _validate_artifact_ref(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    copied = _canonical_copy(value)
    if set(copied) != {"relative_path", "sha256"}:
        raise ValueError(f"{field} must contain relative_path and sha256")
    _require_relative_path(copied["relative_path"], f"{field}.relative_path")
    _require_sha256(copied["sha256"], f"{field}.sha256")
    return copied


@dataclass(frozen=True)
class PreferenceCandidate:
    schema: str
    owner_id: str
    target_text: str
    target_sha256: str
    source_sha256: str
    style_sha256: str
    layout_plan_sha256: str
    recipe_sha256: str
    output_sha256: str
    preview_ref: dict[str, Any]
    context_ref: dict[str, Any]
    metrics: dict[str, Any]
    hard_safety_passed: bool
    preference_profile: str
    candidate_id: str

    @classmethod
    def build(
        cls,
        *,
        owner_id: str,
        target_text: str,
        source_sha256: str,
        style_sha256: str,
        layout_plan_sha256: str,
        recipe_sha256: str,
        output_sha256: str,
        preview_ref: Mapping[str, Any],
        context_ref: Mapping[str, Any],
        metrics: Mapping[str, Any],
        hard_safety_passed: bool,
        schema: str = SCHEMA_VERSION,
        target_sha256: str | None = None,
        preference_profile: str = PREFERENCE_PROFILE,
    ) -> "PreferenceCandidate":
        if schema != SCHEMA_VERSION:
            raise ValueError("candidate schema version is unsupported")
        if not owner_id or not target_text:
            raise ValueError("candidate owner and target text are required")
        if preference_profile != PREFERENCE_PROFILE:
            raise ValueError("renderer preference profile must remain uncalibrated")
        if type(hard_safety_passed) is not bool:
            raise ValueError("candidate hard safety result must be boolean")
        hashes = {
            "source_sha256": _require_sha256(source_sha256, "source_sha256"),
            "style_sha256": _require_sha256(style_sha256, "style_sha256"),
            "layout_plan_sha256": _require_sha256(layout_plan_sha256, "layout_plan_sha256"),
            "recipe_sha256": _require_sha256(recipe_sha256, "recipe_sha256"),
            "output_sha256": _require_sha256(output_sha256, "output_sha256"),
        }
        expected_target_sha256 = sha256(target_text.encode("utf-8")).hexdigest()
        if target_sha256 is not None and target_sha256 != expected_target_sha256:
            raise ValueError("target_sha256 does not match the exact UTF-8 target text")
        exact_metrics = _canonical_copy(metrics)
        from typesetter.raster_safety import validate_raster_safety_evidence

        safety = validate_raster_safety_evidence(exact_metrics.get("raster_safety"))
        if hard_safety_passed and safety["status"] != "pass":
            raise ValueError("candidate cannot pass hard safety with raster collisions")
        if not hard_safety_passed and safety["status"] == "pass":
            raise ValueError("candidate hard safety flag disagrees with raster safety evidence")
        body = {
            "schema": schema,
            "owner_id": owner_id,
            "target_text": target_text,
            "target_sha256": expected_target_sha256,
            **hashes,
            "preview_ref": _validate_artifact_ref(preview_ref, "preview_ref"),
            "context_ref": _validate_artifact_ref(context_ref, "context_ref"),
            "metrics": exact_metrics,
            "hard_safety_passed": hard_safety_passed,
            "preference_profile": preference_profile,
        }
        return cls(
            **body,
            candidate_id=f"renderer-candidate:{canonical_json_sha256(body)[:32]}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreferenceCandidate":
        payload = _canonical_copy(value)
        expected_fields = {
            "schema", "owner_id", "target_text", "target_sha256", "source_sha256",
            "style_sha256", "layout_plan_sha256", "recipe_sha256", "output_sha256",
            "preview_ref", "context_ref", "metrics", "hard_safety_passed",
            "preference_profile", "candidate_id",
        }
        if set(payload) != expected_fields:
            raise ValueError("candidate schema is incomplete")
        candidate_id = payload.pop("candidate_id")
        candidate = cls.build(**payload)
        if candidate.candidate_id != candidate_id:
            raise ValueError("candidate hash mismatch")
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return _canonical_copy(asdict(self))


def _validate_comparison_pair(
    candidates: tuple[PreferenceCandidate, PreferenceCandidate],
) -> None:
    if candidates[0].candidate_id == candidates[1].candidate_id:
        raise ValueError("comparison candidates must be distinct")
    if not all(candidate.hard_safety_passed for candidate in candidates):
        raise ValueError("comparison candidates must pass hard safety")
    if candidates[0].owner_id != candidates[1].owner_id:
        raise ValueError("comparison candidates must belong to the same owner")
    if candidates[0].target_sha256 != candidates[1].target_sha256:
        raise ValueError("comparison candidates must use the same target text")
    if candidates[0].style_sha256 != candidates[1].style_sha256:
        raise ValueError("comparison candidates must use the same source style")
    if candidates[0].source_sha256 != candidates[1].source_sha256:
        raise ValueError("comparison candidates must use the same source image")


@dataclass(frozen=True)
class PreferenceComparison:
    schema: str
    candidates: tuple[PreferenceCandidate, PreferenceCandidate]
    positions: dict[str, str]
    randomization_sha256: str
    preference_profile: str
    comparison_sha256: str

    @classmethod
    def build(
        cls,
        first: PreferenceCandidate,
        second: PreferenceCandidate,
        *,
        randomization_nonce: str,
    ) -> "PreferenceComparison":
        candidates = tuple(sorted((first, second), key=lambda item: item.candidate_id))
        _validate_comparison_pair(candidates)
        if not randomization_nonce:
            raise ValueError("comparison randomization nonce is required")
        randomization_body = {
            "nonce": randomization_nonce,
            "candidate_ids": [item.candidate_id for item in candidates],
        }
        randomization_sha256 = canonical_json_sha256(randomization_body)
        displayed = candidates if int(randomization_sha256[:2], 16) % 2 == 0 else tuple(reversed(candidates))
        positions = {"A": displayed[0].candidate_id, "B": displayed[1].candidate_id}
        body = {
            "schema": SCHEMA_VERSION,
            "candidates": [item.to_dict() for item in candidates],
            "positions": positions,
            "randomization_sha256": randomization_sha256,
            "preference_profile": PREFERENCE_PROFILE,
        }
        return cls(
            schema=SCHEMA_VERSION,
            candidates=candidates,
            positions=positions,
            randomization_sha256=randomization_sha256,
            preference_profile=PREFERENCE_PROFILE,
            comparison_sha256=canonical_json_sha256(body),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreferenceComparison":
        payload = _canonical_copy(value)
        expected_fields = {
            "schema", "candidates", "positions", "randomization_sha256",
            "preference_profile", "comparison_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("comparison schema is incomplete")
        if payload["schema"] != SCHEMA_VERSION or payload["preference_profile"] != PREFERENCE_PROFILE:
            raise ValueError("comparison schema or preference profile is unsupported")
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) != 2:
            raise ValueError("comparison requires exactly two candidates")
        candidates = tuple(PreferenceCandidate.from_dict(item) for item in raw_candidates)
        _validate_comparison_pair(candidates)
        positions = payload["positions"]
        if not isinstance(positions, dict) or set(positions) != {"A", "B"}:
            raise ValueError("comparison positions must be A and B")
        if set(positions.values()) != {item.candidate_id for item in candidates}:
            raise ValueError("comparison positions do not match candidates")
        _require_sha256(payload["randomization_sha256"], "randomization_sha256")
        comparison_sha256 = payload.pop("comparison_sha256")
        if canonical_json_sha256(payload) != comparison_sha256:
            raise ValueError("comparison hash mismatch")
        return cls(
            schema=SCHEMA_VERSION,
            candidates=(candidates[0], candidates[1]),
            positions=dict(positions),
            randomization_sha256=payload["randomization_sha256"],
            preference_profile=PREFERENCE_PROFILE,
            comparison_sha256=comparison_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidates": [item.to_dict() for item in self.candidates],
            "positions": dict(self.positions),
            "randomization_sha256": self.randomization_sha256,
            "preference_profile": self.preference_profile,
            "comparison_sha256": self.comparison_sha256,
        }


def resolve_preference_choice(
    comparison: PreferenceComparison | Mapping[str, Any],
    choice: PreferenceChoice | str,
) -> dict[str, Any]:
    """Resolve a UI position to a stable candidate without persisting it."""

    validated = (
        comparison
        if isinstance(comparison, PreferenceComparison)
        else PreferenceComparison.from_dict(comparison)
    )
    if choice not in _CHOICES:
        raise ValueError("preference choice must be A, B, equivalent, neither, or unsure")
    selected_candidate_id = validated.positions[str(choice)] if choice in {"A", "B"} else None
    by_id = {item.candidate_id: item for item in validated.candidates}
    displayed = [by_id[validated.positions[position]] for position in ("A", "B")]
    body = {
        "schema": RESOLUTION_SCHEMA,
        "comparison_sha256": validated.comparison_sha256,
        "choice": choice,
        "selected_candidate_id": selected_candidate_id,
        "displayed_candidate_ids": dict(validated.positions),
        "candidate_recipe_sha256s": [item.recipe_sha256 for item in displayed],
        "candidate_output_sha256s": [item.output_sha256 for item in displayed],
        "target_sha256": displayed[0].target_sha256,
        "randomization_sha256": validated.randomization_sha256,
        "preference_profile": PREFERENCE_PROFILE,
    }
    return body | {"response_sha256": canonical_json_sha256(body)}


def rank_preference_candidates(
    candidates: tuple[PreferenceCandidate, ...] | list[PreferenceCandidate],
    *,
    ranker: Any | None = None,
) -> dict[str, Any]:
    """Rank already-safe candidates with a capability-gated optional adapter.

    The deterministic fallback deliberately uses no synthetic preference score.
    A future ranker may reorder only the complete set of candidate identities;
    it cannot add an unsafe candidate or silently drop one.
    """

    items = tuple(candidates)
    if not items:
        raise ValueError("candidate ranking requires at least one candidate")
    if not all(isinstance(item, PreferenceCandidate) for item in items):
        raise ValueError("candidate ranking received an unsupported value")
    if not all(item.hard_safety_passed for item in items):
        raise ValueError("ranker may only receive candidates that passed hard safety")
    candidate_ids = [item.candidate_id for item in items]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ranking requires distinct candidate identities")

    ordered_ids = sorted(candidate_ids)
    policy = "deterministic_candidate_id_v1"
    ranker_version: str | None = None
    fallback_reason = "ranker_not_configured"
    if ranker is not None:
        capabilities = set(getattr(ranker, "capabilities", ()))
        version = str(getattr(ranker, "version", ""))
        method = getattr(ranker, "rank_candidate_ids", None)
        if "renderer_safe_candidate_ranking_v1" not in capabilities:
            fallback_reason = "ranker_capability_missing"
        elif not version or not callable(method):
            fallback_reason = "ranker_interface_invalid"
        else:
            try:
                proposed = [str(value) for value in method(items)]
            except Exception:
                fallback_reason = "ranker_failed"
            else:
                if len(proposed) != len(candidate_ids) or set(proposed) != set(candidate_ids):
                    fallback_reason = "ranker_changed_candidate_set"
                else:
                    ordered_ids = proposed
                    policy = "capability_ranker_v1"
                    ranker_version = version
                    fallback_reason = ""

    body = {
        "schema": SCHEMA_VERSION,
        "candidate_ids": ordered_ids,
        "policy": policy,
        "ranker_version": ranker_version,
        "fallback_reason": fallback_reason,
        "preference_profile": PREFERENCE_PROFILE,
    }
    return body | {"ranking_sha256": canonical_json_sha256(body)}
