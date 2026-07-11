"""Score and render reports for isolated Style Atlas v2 runs without Pillow."""

from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path
from typing import Any

import cv2


ATTRIBUTE_NAMES = (
    "font_name",
    "font_weight",
    "font_width",
    "font_size_px",
    "alignment",
    "fill",
    "stroke",
    "shadow",
    "gradient",
    "rotation_deg",
    "container",
)


def _observation_index(observations: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(item["case_id"]), str(item["variant"])): item for item in observations}


def _value(attributes: dict[str, Any], attribute: str) -> tuple[bool, Any, list[Any]]:
    item = attributes.get(attribute)
    if not isinstance(item, dict) or "value" not in item or item["value"] == "unknown":
        return False, None, []
    top_k = item.get("top_k")
    return True, item["value"], list(top_k) if isinstance(top_k, list) else []


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.upper() == actual.upper()
    return expected == actual


def score_benchmark(manifest: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute attribute metrics and abstention gates from detector observations."""
    indexed = _observation_index(observations)
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hard_negative = {"evaluated": 0, "abstained": 0}
    round_trip = {"evaluated": 0, "passed": 0}

    for case in manifest.get("cases", []):
        case_id = str(case["id"])
        records = [indexed.get((case_id, variant)) for variant in ("a", "b")]
        if case.get("level") == "hard-negative":
            for record in records:
                if record is None:
                    continue
                hard_negative["evaluated"] += 1
                attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
                if not any(_value(attributes, name)[0] for name in ATTRIBUTE_NAMES):
                    hard_negative["abstained"] += 1
            continue

        if all(record is not None for record in records):
            round_trip["evaluated"] += 1
            if records[0].get("source_text") == case.get("text_a") and records[1].get("source_text") == case.get("text_b"):
                round_trip["passed"] += 1

        for record in records:
            if record is None:
                continue
            attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
            for attribute in ATTRIBUTE_NAMES:
                if attribute not in case:
                    continue
                stats = totals[attribute]
                stats["evaluated"] += 1
                known, actual, top_k = _value(attributes, attribute)
                if not known:
                    stats["unknown"] += 1
                    continue
                stats["known"] += 1
                if _equal(case[attribute], actual):
                    stats["correct"] += 1
                if any(_equal(case[attribute], candidate) for candidate in top_k):
                    stats["top_k_hits"] += 1

    attribute_report = {}
    for attribute, stats in totals.items():
        evaluated = stats["evaluated"]
        known = stats["known"]
        attribute_report[attribute] = {
            "coverage": round(known / evaluated, 4) if evaluated else 0.0,
            "evaluated": evaluated,
            "known": known,
            "precision": round(stats["correct"] / known, 4) if known else 0.0,
            "top_k_hits": stats["top_k_hits"],
            "unknown": stats["unknown"],
        }

    hard_negative_rate = (
        round(hard_negative["abstained"] / hard_negative["evaluated"], 4)
        if hard_negative["evaluated"]
        else 0.0
    )
    round_trip_rate = (
        round(round_trip["passed"] / round_trip["evaluated"], 4)
        if round_trip["evaluated"]
        else 0.0
    )
    return {
        "attributes": attribute_report,
        "gates": {
            "hard_negative_abstention": bool(hard_negative["evaluated"])
            and hard_negative_rate == 1.0,
        },
        "hard_negative": {**hard_negative, "rate": hard_negative_rate},
        "round_trip": {**round_trip, "rate": round_trip_rate},
    }


def write_run_reports(run_dir: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Write score artifacts beneath an already-isolated benchmark run."""
    run_dir = Path(run_dir)
    records_path = run_dir / "style_benchmark_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {"schema_version": 2, "score": score_benchmark(manifest, records)}
    (run_dir / "style_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Style Benchmark v2</title>"
        "<h1>Style Benchmark v2</h1><pre>"
        + html.escape(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        + "</pre><p><a href=\"contact_sheets/contact_sheet.jpg\">Contact sheet</a></p>",
        encoding="utf-8",
    )
    _write_contact_sheet(run_dir, manifest)
    return summary


def _write_contact_sheet(run_dir: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for case in manifest.get("cases", []):
        images = [cv2.imread(str(run_dir / case[key]), cv2.IMREAD_COLOR) for key in ("image_a", "image_b")]
        if any(image is None for image in images):
            raise FileNotFoundError(f"missing benchmark image for {case.get('id')}")
        target_height = 140
        resized = [
            cv2.resize(image, (round(image.shape[1] * target_height / image.shape[0]), target_height))
            for image in images
        ]
        rows.append(cv2.hconcat(resized))
    if not rows:
        raise ValueError("cannot create a contact sheet without benchmark cases")
    width = max(row.shape[1] for row in rows)
    normalized = [
        cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(32, 32, 32))
        for row in rows
    ]
    contact_dir = run_dir / "contact_sheets"
    contact_dir.mkdir(exist_ok=False)
    if not cv2.imwrite(str(contact_dir / "contact_sheet.jpg"), cv2.vconcat(normalized)):
        raise RuntimeError("failed to write style benchmark contact sheet")
