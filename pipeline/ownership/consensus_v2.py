"""Consensus authority that counts physical OCR invocations, not aliases."""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher
import re
import unicodedata

from .model import TextObservation


def independent_origins(group: Sequence[TextObservation]) -> frozenset[str]:
    return frozenset(
        str(observation.invocation_id)
        for observation in group
        if str(observation.invocation_id or "").strip()
    )


def _payload_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def _compact_text(text: str) -> str:
    return "".join(character for character in _payload_key(text) if character.isalnum())


def _same_spatial_reading(left: TextObservation, right: TextObservation) -> bool:
    """Collapse overlapping readings of one physical fragment, never distant repeats."""
    if left.invocation_id != right.invocation_id:
        return False
    a, b = left.bbox_page, right.bbox_page
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min(max(1, (a[2]-a[0])*(a[3]-a[1])), max(1, (b[2]-b[0])*(b[3]-b[1])))
    if overlap / smaller < 0.80:
        return False
    left_words = _payload_key(left.text).split()
    right_words = _payload_key(right.text).split()
    short, long = sorted((left_words, right_words), key=len)
    return bool(short) and any(long[i:i+len(short)] == short for i in range(len(long)-len(short)+1))


def _same_physical_line(left: TextObservation, right: TextObservation) -> bool:
    """Compare page-space line supports, independently of OCR call and component IDs."""
    a, b = left.bbox_page, right.bbox_page
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 0 or max(ah, bh) > 1.8 * min(ah, bh):
        return False  # A block is not interchangeable with one of its lines.
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    if iy < 0.65 * min(ah, bh) or ix < 0.65 * min(aw, bw):
        return False
    if abs((a[1] + a[3]) - (b[1] + b[3])) > 0.8 * max(ah, bh):
        return False
    return True


def _reading_penalty(item: TextObservation, alternatives: Sequence[TextObservation]) -> tuple[int, int, int]:
    """Prefer supported legible readings without rewriting any OCR character."""
    text = str(item.text or "")
    anomaly = len(re.findall(r"[A-Za-z]![A-Za-z]", text)) * 3
    anomaly += len(re.findall(r"(?i)['’](?:ll|ve|re|d|m|s|t)[a-z]", text)) * 3
    compact = re.sub(r"[^A-Za-z0-9]", "", text).casefold()
    spacing = 0
    for other in alternatives:
        if other is item:
            continue
        other_text = str(other.text or "")
        other_compact = re.sub(r"[^A-Za-z0-9]", "", other_text).casefold()
        if compact == other_compact:
            spacing = max(spacing, max(0, len(other_text.split()) - len(text.split())))
        # Treat a typical OCR digit confusion as weaker only when an aligned
        # independent reading supplies an alphabetic alternative. Never edit a
        # code or infer its characters from a single observation.
        if re.search(r"[A-Za-z][015][A-Za-z]", text) and not re.search(r"\d", other_text):
            if SequenceMatcher(None, compact, other_compact).ratio() >= 0.75:
                anomaly += 2
                break
    case_flips = sum(
        sum(a.islower() != b.islower() for a, b in zip(word, word[1:])) >= 2
        for word in re.findall(r"[A-Za-z]{3,}", text)
    )
    return anomaly, spacing, case_flips


def _choose_line_reading(group: Sequence[TextObservation]) -> TextObservation:
    return min(group, key=lambda item: (
        _reading_penalty(item, group),
        -float(item.confidence),
        -len(str(item.text or "")),
        str(item.observation_id),
    ))


def select_consensus_observation(
    observations: Sequence[TextObservation],
) -> TextObservation:
    eligible = tuple(
        observation
        for observation in observations
        if observation.rejection_reason is None and str(observation.text or "").strip()
    )
    if not eligible:
        raise ValueError("consensus requires at least one eligible text observation")

    groups: dict[str, list[TextObservation]] = {}
    for observation in eligible:
        groups.setdefault(_payload_key(observation.text), []).append(observation)

    def group_rank(item: tuple[str, list[TextObservation]]) -> tuple[int, float, int, str]:
        key, group = item
        return (
            len(independent_origins(group)),
            max(float(observation.confidence) for observation in group),
            len(key),
            key,
        )

    _, winner_group = max(groups.items(), key=group_rank)
    return max(
        winner_group,
        key=lambda observation: (
            float(observation.confidence),
            len(str(observation.text or "")),
            str(observation.observation_id),
        ),
    )


def select_ordered_consensus_body(
    observation_groups: Sequence[Sequence[TextObservation]],
) -> tuple[tuple[TextObservation, ...], str]:
    selected, payload, _ = select_ordered_consensus_body_with_decisions(observation_groups)
    return selected, payload


def select_ordered_consensus_body_with_decisions(
    observation_groups: Sequence[Sequence[TextObservation]],
) -> tuple[tuple[TextObservation, ...], str, dict[str, dict[str, str | None]]]:
    """Select one invocation reading per component in canonical page order.

    A provider invocation may legitimately return multiple spatial lines for
    one component.  Those lines are fragments of one body, not competing OCR
    votes, so they must remain together.  Multiple invocation origins still
    use the strict consensus selector.
    """

    selected_candidates: list[TextObservation] = []
    decisions: dict[str, dict[str, str | None]] = {}
    for group in observation_groups:
        materialized = tuple(group)
        invocation_ids = {
            observation.invocation_id for observation in materialized
        }
        if len(invocation_ids) == 1:
            selected_candidates.extend(materialized)
        else:
            winner = select_consensus_observation(materialized)
            selected_candidates.append(winner)
            for item in materialized:
                decisions[item.observation_id] = {
                    "decision": "selected" if item is winner else "alternate_same_component",
                    "selected_observation_id": winner.observation_id,
                }
    selected_by_id = {
        observation.observation_id: observation
        for observation in selected_candidates
    }
    line_groups: list[list[TextObservation]] = []
    for candidate in sorted(selected_by_id.values(), key=lambda item: (
        item.bbox_page[1], item.bbox_page[0], item.observation_id
    )):
        # Complete-link membership prevents a chain of neighbouring lines from
        # turning into one group through a broad intermediate observation.
        matching = [group for group in line_groups if all(
            _same_physical_line(candidate, member) for member in group
        )]
        if len(matching) == 1:
            matching[0].append(candidate)
        else:
            line_groups.append([candidate])
    suppressed_blocks: dict[int, str] = {}
    representatives = [_choose_line_reading(group) for group in line_groups]
    for index, block in enumerate(representatives):
        bx1, by1, bx2, by2 = block.bbox_page
        block_height = by2 - by1
        contained = [
            other_index for other_index, line in enumerate(representatives)
            if other_index != index
            and block_height >= 1.8 * (line.bbox_page[3] - line.bbox_page[1])
            and bx1 <= (line.bbox_page[0] + line.bbox_page[2]) / 2 < bx2
            and by1 <= (line.bbox_page[1] + line.bbox_page[3]) / 2 < by2
            and min(bx2, line.bbox_page[2]) - max(bx1, line.bbox_page[0])
            >= 0.55 * (line.bbox_page[2] - line.bbox_page[0])
        ]
        contained.sort(key=lambda item: (representatives[item].bbox_page[1], representatives[item].bbox_page[0]))
        if len(contained) < 2:
            continue
        if any(
            representatives[right].bbox_page[1] < representatives[left].bbox_page[3]
            - 0.2 * min(
                representatives[left].bbox_page[3] - representatives[left].bbox_page[1],
                representatives[right].bbox_page[3] - representatives[right].bbox_page[1],
            )
            for left, right in zip(contained, contained[1:])
        ):
            continue
        line_text = " ".join(representatives[item].text for item in contained)
        similarity = SequenceMatcher(None, _compact_text(block.text), _compact_text(line_text)).ratio()
        suppressed_blocks[index] = (
            "alternate_block_covered_by_lines" if similarity >= 0.75
            else "ambiguous_block_line_overlap"
        )
    deduplicated: list[TextObservation] = []
    for group_index, group in enumerate(line_groups):
        if group_index in suppressed_blocks:
            for item in group:
                decisions[item.observation_id] = {
                    "decision": suppressed_blocks[group_index],
                    "selected_observation_id": None,
                }
            continue
        winner = _choose_line_reading(group)
        deduplicated.append(winner)
        for item in group:
            conflicting = (
                item is not winner
                and _reading_penalty(item, group) == _reading_penalty(winner, group)
                and SequenceMatcher(None, _compact_text(item.text), _compact_text(winner.text)).ratio() < 0.65
            )
            decisions[item.observation_id] = {
                "decision": "selected" if item is winner else (
                    "ambiguous_same_physical_line" if conflicting else "alternate_same_physical_line"
                ),
                "selected_observation_id": winner.observation_id,
            }
    selected = tuple(
        sorted(
            deduplicated,
            key=lambda item: (
                item.bbox_page[1],
                item.bbox_page[0],
                item.bbox_page[3],
                item.bbox_page[2],
                item.observation_id,
            ),
        )
    )
    payload = " ".join(
        " ".join(str(item.text or "").split()) for item in selected
    ).strip()
    if not payload:
        raise ValueError("consensus body requires non-empty selected observations")
    return selected, payload, decisions
