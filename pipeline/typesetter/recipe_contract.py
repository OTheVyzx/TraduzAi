"""Canonical renderer recipe for exact local replay.

This version is additive: legacy recipes keep their original loader and hash.
Only records declaring ``traduzai.renderer-recipe.v1`` are accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from ownership.hash_contract import canonical_json_sha256


RECIPE_SCHEMA = "traduzai.renderer-recipe.v1"
LINE_PLAN_SCHEMA = "traduzai.exact-line-plan.v1"


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))


def _sha(value: Any, field: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return candidate


def _relative_path(value: Any, field: str) -> str:
    candidate = str(value or "")
    if not candidate or "\\" in candidate or candidate.startswith("/") or ":" in candidate:
        raise ValueError(f"{field} must be a portable relative path")
    if any(part in {"", ".", ".."} for part in candidate.split("/")):
        raise ValueError(f"{field} must not escape its root")
    return candidate


def _bbox(value: Sequence[Any]) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("recipe bbox must contain four coordinates")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("recipe bbox coordinates must be integers") from exc
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("recipe bbox must have positive area")
    return result


def _hash_mapping(value: Mapping[str, Any], field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, digest in sorted(value.items()):
        name = str(key or "")
        if not name:
            raise ValueError(f"{field} contains an empty dependency name")
        result[name] = _sha(digest, f"{field}.{name}")
    return result


@dataclass(frozen=True)
class ExactLinePlan:
    schema: str
    lines: tuple[str, ...]
    separators: tuple[str, ...]
    target_utf8_sha256: str
    plan_sha256: str

    @classmethod
    def build(
        cls,
        *,
        target_text: str,
        lines: Sequence[str],
        separators: Sequence[str],
    ) -> "ExactLinePlan":
        exact_lines = tuple(lines)
        exact_separators = tuple(separators)
        if not target_text or not exact_lines or any(not isinstance(line, str) or not line for line in exact_lines):
            raise ValueError("exact line plan requires non-empty text lines")
        if len(exact_separators) != len(exact_lines) - 1:
            raise ValueError("exact line plan requires one separator between adjacent lines")
        if any(not isinstance(item, str) or not item or not item.isspace() for item in exact_separators):
            raise ValueError("exact line plan separators must preserve whitespace")
        reconstructed = "".join(
            line + (exact_separators[index] if index < len(exact_separators) else "")
            for index, line in enumerate(exact_lines)
        )
        if reconstructed != target_text:
            raise ValueError("exact line plan must reconstruct target text byte for byte")
        target_utf8_sha256 = sha256(target_text.encode("utf-8")).hexdigest()
        body = {
            "schema": LINE_PLAN_SCHEMA,
            "lines": list(exact_lines),
            "separators": list(exact_separators),
            "target_utf8_sha256": target_utf8_sha256,
        }
        return cls(
            schema=LINE_PLAN_SCHEMA,
            lines=exact_lines,
            separators=exact_separators,
            target_utf8_sha256=target_utf8_sha256,
            plan_sha256=canonical_json_sha256(body),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactLinePlan":
        payload = _copy(value)
        required = {"schema", "lines", "separators", "target_utf8_sha256", "plan_sha256"}
        if set(payload) != required or payload["schema"] != LINE_PLAN_SCHEMA:
            raise ValueError("exact line plan schema is unsupported or incomplete")
        target_text = "".join(
            line + (payload["separators"][index] if index < len(payload["separators"]) else "")
            for index, line in enumerate(payload["lines"])
        )
        rebuilt = cls.build(
            target_text=target_text,
            lines=payload["lines"],
            separators=payload["separators"],
        )
        if rebuilt.target_utf8_sha256 != payload["target_utf8_sha256"]:
            raise ValueError("exact line plan target hash mismatch")
        if rebuilt.plan_sha256 != payload["plan_sha256"]:
            raise ValueError("exact line plan hash mismatch")
        return rebuilt

    def reconstruct(self) -> str:
        return "".join(
            line + (self.separators[index] if index < len(self.separators) else "")
            for index, line in enumerate(self.lines)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lines": list(self.lines),
            "separators": list(self.separators),
            "target_utf8_sha256": self.target_utf8_sha256,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True)
class RendererRecipe:
    schema: str
    owner_id: str
    source_sha256: str
    output_sha256: str
    target_text: str
    target_utf8_sha256: str
    font: dict[str, Any]
    rasterizer: dict[str, Any]
    line_plan: ExactLinePlan
    bbox: tuple[int, int, int, int]
    font_size_px: int
    line_advance_px: int
    effects: dict[str, Any]
    anchors: dict[str, Any]
    geometry: dict[str, Any]
    policy_versions: dict[str, Any]
    dependency_hashes: dict[str, str]
    recipe_sha256: str

    @classmethod
    def build(
        cls,
        *,
        owner_id: str,
        source_sha256: str,
        output_sha256: str,
        target_text: str,
        font: Mapping[str, Any],
        rasterizer: Mapping[str, Any],
        line_plan: ExactLinePlan | Mapping[str, Any],
        bbox: Sequence[Any],
        font_size_px: int,
        line_advance_px: int,
        effects: Mapping[str, Any],
        anchors: Mapping[str, Any],
        geometry: Mapping[str, Any],
        policy_versions: Mapping[str, Any],
        dependency_hashes: Mapping[str, Any],
        schema: str = RECIPE_SCHEMA,
        target_utf8_sha256: str | None = None,
    ) -> "RendererRecipe":
        if schema != RECIPE_SCHEMA:
            raise ValueError("renderer recipe schema is unsupported")
        if not owner_id or not target_text:
            raise ValueError("renderer recipe owner and target text are required")
        exact_plan = line_plan if isinstance(line_plan, ExactLinePlan) else ExactLinePlan.from_dict(line_plan)
        if exact_plan.reconstruct() != target_text:
            raise ValueError("renderer recipe line plan changes target text")
        exact_target_sha256 = sha256(target_text.encode("utf-8")).hexdigest()
        if target_utf8_sha256 is not None and target_utf8_sha256 != exact_target_sha256:
            raise ValueError("renderer recipe target hash mismatch")

        exact_font = _copy(font)
        if set(exact_font) != {"family", "relative_path", "sha256"} or not exact_font["family"]:
            raise ValueError("renderer recipe font identity is incomplete")
        _relative_path(exact_font["relative_path"], "font.relative_path")
        _sha(exact_font["sha256"], "font.sha256")
        exact_rasterizer = _copy(rasterizer)
        if set(exact_rasterizer) != {"runtime_id", "runtime_sha256", "config_sha256"} or not exact_rasterizer["runtime_id"]:
            raise ValueError("renderer recipe rasterizer identity is incomplete")
        _sha(exact_rasterizer["runtime_sha256"], "rasterizer.runtime_sha256")
        _sha(exact_rasterizer["config_sha256"], "rasterizer.config_sha256")
        exact_bbox = _bbox(bbox)
        if int(font_size_px) <= 0 or int(line_advance_px) <= 0:
            raise ValueError("renderer recipe font size and line advance must be positive")
        exact_policies = _copy(policy_versions)
        if not exact_policies or any(not str(key) or not str(value) for key, value in exact_policies.items()):
            raise ValueError("renderer recipe policy versions are required")
        exact_dependencies = _hash_mapping(dependency_hashes, "dependency_hashes")
        body = {
            "schema": RECIPE_SCHEMA,
            "owner_id": owner_id,
            "source_sha256": _sha(source_sha256, "source_sha256"),
            "output_sha256": _sha(output_sha256, "output_sha256"),
            "target_text": target_text,
            "target_utf8_sha256": exact_target_sha256,
            "font": exact_font,
            "rasterizer": exact_rasterizer,
            "line_plan": exact_plan.to_dict(),
            "bbox": list(exact_bbox),
            "font_size_px": int(font_size_px),
            "line_advance_px": int(line_advance_px),
            "effects": _copy(effects),
            "anchors": _copy(anchors),
            "geometry": _copy(geometry),
            "policy_versions": exact_policies,
            "dependency_hashes": exact_dependencies,
        }
        return cls(
            schema=RECIPE_SCHEMA,
            owner_id=owner_id,
            source_sha256=body["source_sha256"],
            output_sha256=body["output_sha256"],
            target_text=target_text,
            target_utf8_sha256=exact_target_sha256,
            font=exact_font,
            rasterizer=exact_rasterizer,
            line_plan=exact_plan,
            bbox=exact_bbox,
            font_size_px=int(font_size_px),
            line_advance_px=int(line_advance_px),
            effects=body["effects"],
            anchors=body["anchors"],
            geometry=body["geometry"],
            policy_versions=exact_policies,
            dependency_hashes=exact_dependencies,
            recipe_sha256=canonical_json_sha256(body),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RendererRecipe":
        payload = _copy(value)
        required = {
            "schema", "owner_id", "source_sha256", "output_sha256", "target_text",
            "target_utf8_sha256", "font", "rasterizer", "line_plan", "bbox",
            "font_size_px", "line_advance_px", "effects", "anchors", "geometry",
            "policy_versions", "dependency_hashes", "recipe_sha256",
        }
        if set(payload) != required:
            raise ValueError("renderer recipe schema is incomplete")
        expected = payload.pop("recipe_sha256")
        rebuilt = cls.build(**payload)
        if rebuilt.recipe_sha256 != expected:
            raise ValueError("renderer recipe hash mismatch")
        return rebuilt

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "owner_id": self.owner_id,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
            "target_text": self.target_text,
            "target_utf8_sha256": self.target_utf8_sha256,
            "font": _copy(self.font),
            "rasterizer": _copy(self.rasterizer),
            "line_plan": self.line_plan.to_dict(),
            "bbox": list(self.bbox),
            "font_size_px": self.font_size_px,
            "line_advance_px": self.line_advance_px,
            "effects": _copy(self.effects),
            "anchors": _copy(self.anchors),
            "geometry": _copy(self.geometry),
            "policy_versions": _copy(self.policy_versions),
            "dependency_hashes": dict(self.dependency_hashes),
            "recipe_sha256": self.recipe_sha256,
        }

    def verify_output(self, canonical_output_bytes: bytes) -> None:
        if sha256(canonical_output_bytes).hexdigest() != self.output_sha256:
            raise ValueError("rendered output SHA-256 does not match recipe")

    def to_receipt(self, relative_path: str) -> dict[str, Any]:
        """Adapt this full recipe to the published Integration Recipe shape."""

        return {
            "recipe_sha256": self.recipe_sha256,
            "output_sha256": self.output_sha256,
            "relative_path": _relative_path(relative_path, "relative_path"),
            "runtime_id": self.rasterizer["runtime_id"],
            "runtime_sha256": self.rasterizer["runtime_sha256"],
            "source_sha256": self.source_sha256,
            "dependency_hashes": dict(self.dependency_hashes),
        }

    def assert_replay_compatible(
        self,
        *,
        runtime_id: str,
        runtime_sha256: str,
        font_sha256: str,
        dependency_hashes: Mapping[str, Any],
    ) -> None:
        if runtime_id != self.rasterizer["runtime_id"] or _sha(runtime_sha256, "runtime_sha256") != self.rasterizer["runtime_sha256"]:
            raise ValueError("renderer runtime is incompatible with recipe")
        if _sha(font_sha256, "font_sha256") != self.font["sha256"]:
            raise ValueError("renderer font is incompatible with recipe")
        if _hash_mapping(dependency_hashes, "dependency_hashes") != self.dependency_hashes:
            raise ValueError("renderer dependencies are incompatible with recipe")
