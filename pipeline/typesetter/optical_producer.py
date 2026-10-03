"""Collect existing OCR scale evidence for opt-in fixed lettering.

The producer never creates a fixed contract or upgrades a manual interior mask.
It only fills missing calibration in contracts already authorized by the caller.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from .optical_estimator import estimate_optical_profile


def _scope(page: Mapping[str, Any]) -> tuple[str, str]:
    work = str(page.get('work_id') or page.get('obra_id') or '')
    chapter = str(page.get('chapter_id') or page.get('capitulo_id') or '')
    if not work or not chapter:
        return ('unscoped_page', str(page.get('page_id') or page.get('numero') or id(page)))
    return (work, chapter)


def _sample(record: Mapping[str, Any], width: int) -> dict[str, Any]:
    role = str(record.get('visual_role') or record.get('semantic_role') or '').lower()
    if role in {'dialogue', 'speech', 'speech_balloon', 'balloon_dialogue'}:
        role = 'ordinary_dialogue'
    if not role and str(record.get('content_class') or '').lower() in {'dialogue', 'speech'}:
        role = 'ordinary_dialogue'
    return {
        'owner_id': str(record.get('owner_id') or ''),
        'visual_role': role,
        'source_x_heights_px': list(record.get('source_x_heights_px') or []),
        'source_scale_evidence_ids': list(record.get('source_scale_evidence_ids') or []),
        'source_scale_evidence_confidence': record.get('source_scale_evidence_confidence'),
        'page_width_px': int(record.get('page_width_px') or width or 0),
    }


def attach_optical_calibrations(
    pages: Sequence[dict[str, Any]], *, widths: Sequence[int],
    find_font: Callable[[str], str | None],
) -> dict[str, Any]:
    """Reuse one source-backed profile per scope/font/reference width.

    The only mutation is in an existing fixed_optical_contract. A weak
    sample leaves that contract unresolved for the caller's review path.
    """
    if len(pages) != len(widths):
        raise ValueError('page_width_count_mismatch')
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for page, width in zip(pages, widths, strict=True):
        for record in page.get('texts') or []:
            if isinstance(record, Mapping):
                pools[_scope(page)].append(_sample(record, width))
    cache: dict[tuple[tuple[str, str], str, int], tuple[dict[str, Any] | None, str]] = {}
    status_counts: dict[str, int] = defaultdict(int)
    for page, width in zip(pages, widths, strict=True):
        scope = _scope(page)
        for record in page.get('texts') or []:
            if not isinstance(record, dict):
                continue
            contract = record.get('fixed_optical_contract')
            if not isinstance(contract, dict) or contract.get('optical_calibration') is not None:
                continue
            name = str(contract.get('font_name') or '')
            key = (scope, name, int(width))
            if key not in cache:
                path = find_font(name) if name else None
                cache[key] = (estimate_optical_profile(
                    pools[scope], font_path=path, font_name=name,
                    page_width_px=int(width),
                ) if path else (None, 'calibration_font_unavailable'))
            profile, status = cache[key]
            contract['optical_calibration_producer_status'] = status
            status_counts[status] += 1
            if profile is not None:
                contract['optical_calibration'] = dict(profile)
                contract['optical_calibration_provenance'] = {
                    'scope_work_id': scope[0], 'scope_chapter_id': scope[1],
                    'source_evidence_sha256': profile['source_evidence_sha256'],
                    'sample_count': len(pools[scope]),
                    'evidence_count': sum(len(item['source_x_heights_px']) for item in pools[scope]),
                    'producer': 'chapter_ocr_source_scale_v1',
                }
    return {'profiles_evaluated': len(cache), 'statuses': dict(status_counts)}
