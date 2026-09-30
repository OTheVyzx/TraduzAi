"""Owner- and region-aware final language residual classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping
import unicodedata

import numpy as np

from ownership.hash_contract import canonical_json_sha256, sha256_text
from ownership.model import LanguageResidualIssue
from ownership.translation import TranslationBinding


@dataclass(frozen=True)
class ResidualRegion:
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    page_output_pixel_sha256: str
    bbox_page: tuple[int, int, int, int]
    owner_id: str | None = None
    component_id: str | None = None
    container_id: str | None = None
    invocation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.run_id, self.execution_id, self.page_id)):
            raise ValueError("language residual region identity is incomplete")
        if len(self.page_source_sha256) != 64 or len(self.page_output_pixel_sha256) != 64:
            raise ValueError("language residual region hash identity is invalid")
        if (
            len(self.bbox_page) != 4
            or self.bbox_page[0] < 0
            or self.bbox_page[1] < 0
            or self.bbox_page[0] >= self.bbox_page[2]
            or self.bbox_page[1] >= self.bbox_page[3]
        ):
            raise ValueError("language residual region bbox is invalid")
        object.__setattr__(
            self,
            "invocation_ids",
            tuple(sorted({str(value) for value in self.invocation_ids if str(value)})),
        )

    @property
    def region_sha256(self) -> str:
        return canonical_json_sha256(
            {
                "page_id": self.page_id,
                "page_output_pixel_sha256": self.page_output_pixel_sha256,
                "bbox_page": list(self.bbox_page),
                "owner_id": self.owner_id,
                "component_id": self.component_id,
                "container_id": self.container_id,
            }
        )


def _tokens(value: Any) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if token
    )


def _subtract_shared(source: Iterable[str], target: Iterable[str]) -> tuple[str, ...]:
    remaining = Counter(target)
    result: list[str] = []
    for token in source:
        if remaining[token] > 0:
            remaining[token] -= 1
        elif not token.isdigit():
            result.append(token)
    return tuple(result)


def _visible(expected: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    if not expected or not observed:
        return False
    observed_set = set(observed)
    matched = tuple(token for token in expected if token in observed_set)
    if len(expected) == 1:
        return len(expected[0]) >= 4 and bool(matched)
    bigrams = set(zip(expected, expected[1:])) & set(zip(observed, observed[1:]))
    return bool(bigrams) or (len(matched) >= 2 and len(matched) / len(expected) >= 0.5)


def build_language_residual_issue(
    *,
    kind: str,
    observed_text: str,
    region: ResidualRegion,
    binding: TranslationBinding | None = None,
    source_only_tokens: Iterable[str] = (),
    repair_required: bool = True,
) -> LanguageResidualIssue:
    if binding is not None:
        TranslationBinding.from_dict(binding.to_dict())
        if (
            binding.run_id != region.run_id
            or binding.page_id != region.page_id
            or binding.page_source_sha256 != region.page_source_sha256
            or binding.owner_id != region.owner_id
        ):
            raise ValueError("language residual binding belongs to another region")
    source_binding_sha256 = (
        binding.translation_binding_sha256 if binding is not None else None
    )
    issue_id = canonical_json_sha256(
        {
            "run_id": region.run_id,
            "execution_id": region.execution_id,
            "page_id": region.page_id,
            "page_output_pixel_sha256": region.page_output_pixel_sha256,
            "owner_or_region": region.owner_id or region.region_sha256,
            "kind": str(kind),
            "source_binding_sha256": source_binding_sha256,
        }
    )
    return LanguageResidualIssue.build(
        issue_id=issue_id,
        run_id=region.run_id,
        execution_id=region.execution_id,
        page_id=region.page_id,
        page_source_sha256=region.page_source_sha256,
        page_output_pixel_sha256=region.page_output_pixel_sha256,
        owner_id=region.owner_id,
        component_id=region.component_id,
        container_id=region.container_id,
        invocation_ids=region.invocation_ids,
        kind=str(kind),
        observed_text=str(observed_text),
        observed_text_sha256=sha256_text(str(observed_text)),
        source_binding_sha256=source_binding_sha256,
        region_sha256=region.region_sha256,
        source_only_tokens=tuple(sorted(set(str(value) for value in source_only_tokens))),
        repair_required=bool(repair_required),
    )


def classify_language_residual(
    *,
    observed: str,
    binding: TranslationBinding,
    region: ResidualRegion,
) -> tuple[LanguageResidualIssue, ...]:
    source_only = _subtract_shared(_tokens(binding.source_text), _tokens(binding.target_text))
    target_only = _subtract_shared(_tokens(binding.target_text), _tokens(binding.source_text))
    observed_tokens = _tokens(observed)
    source_visible = _visible(source_only, observed_tokens)
    if not source_visible:
        return ()
    target_visible = _visible(target_only, observed_tokens)
    kind = "mixed_language_overlay" if target_visible else "source_language_visible"
    return (
        build_language_residual_issue(
            kind=kind,
            observed_text=observed,
            region=region,
            binding=binding,
            source_only_tokens=tuple(
                token for token in source_only if token in set(observed_tokens)
            ),
        ),
    )


def _mapping_bbox(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("unowned OCR line has no canonical bbox")
    bbox = tuple(int(item) for item in value)
    if bbox[0] < 0 or bbox[1] < 0 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("unowned OCR bbox is invalid")
    return bbox


def classify_unowned_text(
    line: Mapping[str, Any],
    container: Mapping[str, Any] | None,
) -> tuple[LanguageResidualIssue, ...]:
    observed = str(line.get("text") or "").strip()
    if not re.search(r"[^\W\d_]", observed, flags=re.UNICODE):
        return ()
    if not bool(line.get("glyph_support", True)):
        return ()
    if container is not None and not bool(container.get("translatable", True)):
        return ()
    invocation_id = str(line.get("invocation_id") or "")
    region = ResidualRegion(
        run_id=str(line.get("run_id") or ""),
        execution_id=str(line.get("execution_id") or ""),
        page_id=str(line.get("page_id") or ""),
        page_source_sha256=str(line.get("page_source_sha256") or ""),
        page_output_pixel_sha256=str(line.get("page_output_pixel_sha256") or ""),
        bbox_page=_mapping_bbox(line.get("bbox")),
        owner_id=None,
        component_id=(str(line["component_id"]) if line.get("component_id") else None),
        container_id=(
            str(container["container_id"])
            if container is not None and container.get("container_id") else None
        ),
        invocation_ids=((invocation_id,) if invocation_id else ()),
    )
    return (
        build_language_residual_issue(
            kind="independently_detected_text_without_owner",
            observed_text=observed,
            region=region,
            source_only_tokens=_tokens(observed),
        ),
    )


def deduplicate_language_issues(
    issues: Iterable[LanguageResidualIssue],
) -> tuple[LanguageResidualIssue, ...]:
    grouped: dict[str, list[LanguageResidualIssue]] = {}
    for issue in issues:
        LanguageResidualIssue.from_dict(issue.to_dict())
        grouped.setdefault(issue.issue_id, []).append(issue)
    result: list[LanguageResidualIssue] = []
    for issue_id in sorted(grouped):
        group = grouped[issue_id]
        primary = group[0]
        payload = primary.to_dict()
        payload.pop("issue_sha256")
        payload["invocation_ids"] = tuple(
            sorted({value for item in group for value in item.invocation_ids})
        )
        payload["source_only_tokens"] = tuple(
            sorted({value for item in group for value in item.source_only_tokens})
        )
        result.append(LanguageResidualIssue.build(**payload))
    return tuple(result)


def verify_replacement_pixels(
    *,
    source_support_mask: Any,
    cleanup_mask: Any,
    target_glyph_mask: Any,
    binding: TranslationBinding,
    region: ResidualRegion,
) -> LanguageResidualIssue | None:
    masks = [np.asarray(value) for value in (source_support_mask, cleanup_mask, target_glyph_mask)]
    if any(mask.ndim == 3 for mask in masks):
        masks = [mask[:, :, 0] if mask.ndim == 3 else mask for mask in masks]
    if len({mask.shape for mask in masks}) != 1 or masks[0].ndim != 2:
        raise ValueError("replacement pixel masks have incompatible geometry")
    source, cleanup, target = (mask > 0 for mask in masks)
    if np.any(source & ~cleanup):
        return build_language_residual_issue(
            kind="cleanup_incomplete",
            observed_text="",
            region=region,
            binding=binding,
            source_only_tokens=_tokens(binding.source_text),
        )
    if not np.any(target):
        return build_language_residual_issue(
            kind="target_glyphs_missing",
            observed_text="",
            region=region,
            binding=binding,
        )
    return None


__all__ = [
    "ResidualRegion",
    "build_language_residual_issue",
    "classify_language_residual",
    "classify_unowned_text",
    "deduplicate_language_issues",
    "verify_replacement_pixels",
]
