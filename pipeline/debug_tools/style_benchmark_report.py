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


def evaluate_validation_thresholds(
    score: dict[str, Any], thresholds: dict[str, Any] | None
) -> dict[str, Any]:
    """Evaluate versioned benchmark thresholds without allowing missing data to pass."""

    policy = thresholds if isinstance(thresholds, dict) else {}
    findings: list[dict[str, Any]] = []
    attributes = score.get("attributes") if isinstance(score.get("attributes"), dict) else {}
    required_attributes = policy.get("attributes") if isinstance(policy.get("attributes"), dict) else {}
    for attribute, raw_rules in sorted(required_attributes.items()):
        rules = raw_rules if isinstance(raw_rules, dict) else {}
        metrics = attributes.get(attribute)
        if not isinstance(metrics, dict):
            findings.append({"code": "required_metric_missing", "metric": f"attributes.{attribute}"})
            continue
        evaluated = int(metrics.get("evaluated") or 0)
        minimum_evaluated = int(rules.get("minimum_evaluated") or 1)
        if evaluated < minimum_evaluated:
            findings.append(
                {
                    "code": "zero_metric_denominator" if evaluated == 0 else "metric_denominator_below_minimum",
                    "metric": f"attributes.{attribute}",
                    "actual": evaluated,
                    "minimum": minimum_evaluated,
                }
            )
        for rule_name, metric_name in (("precision_min", "precision"), ("coverage_min", "coverage")):
            if rule_name not in rules:
                continue
            if metric_name not in metrics:
                findings.append({"code": "required_metric_missing", "metric": f"attributes.{attribute}.{metric_name}"})
                continue
            actual = float(metrics[metric_name])
            minimum = float(rules[rule_name])
            if actual < minimum:
                findings.append(
                    {
                        "code": "metric_threshold_breach",
                        "metric": f"attributes.{attribute}.{metric_name}",
                        "actual": actual,
                        "minimum": minimum,
                    }
                )
    for policy_name, report_name in (
        ("round_trip_min", "round_trip"),
        ("hard_negative_abstention_min", "hard_negative"),
    ):
        if policy_name not in policy:
            continue
        metrics = score.get(report_name)
        if not isinstance(metrics, dict):
            findings.append({"code": "required_metric_missing", "metric": report_name})
            continue
        evaluated = int(metrics.get("evaluated") or 0)
        if evaluated == 0:
            findings.append({"code": "zero_metric_denominator", "metric": report_name})
        if float(metrics.get("rate") or 0.0) < float(policy[policy_name]):
            findings.append(
                {
                    "code": "metric_threshold_breach",
                    "metric": f"{report_name}.rate",
                    "actual": float(metrics.get("rate") or 0.0),
                    "minimum": float(policy[policy_name]),
                }
            )
    if "font_top1_min" in policy:
        font = score.get("font_top1") if isinstance(score.get("font_top1"), dict) else None
        if font is not None:
            evaluated = int(font.get("evaluated") or 0)
            if not evaluated:
                findings.append({"code": "zero_metric_denominator", "metric": "font_top1"})
            actual = float(font.get("rate") or 0.0)
            if actual < float(policy["font_top1_min"]):
                findings.append({"code": "metric_threshold_breach", "metric": "font_top1", "actual": actual, "minimum": float(policy["font_top1_min"])})
        else:
            font = attributes.get("font_name") if isinstance(attributes.get("font_name"), dict) else None
        if font is None:
            findings.append({"code": "required_metric_missing", "metric": "font_top1"})
        elif "rate" not in font and float(font.get("precision") or 0.0) < float(policy["font_top1_min"]):
            findings.append({"code": "metric_threshold_breach", "metric": "font_top1", "actual": float(font.get("precision") or 0.0), "minimum": float(policy["font_top1_min"])})
    if "font_top3_min" in policy:
        font = score.get("font_top3") if isinstance(score.get("font_top3"), dict) else None
        if font is None:
            font = attributes.get("font_name") if isinstance(attributes.get("font_name"), dict) else None
        evaluated = int((font or {}).get("evaluated") or 0)
        if not evaluated:
            findings.append({"code": "zero_metric_denominator", "metric": "font_top3"})
        else:
            actual = float(font.get("rate")) if "rate" in font else float(font.get("top_k_hits") or 0) / evaluated
            if actual < float(policy["font_top3_min"]):
                findings.append({"code": "metric_threshold_breach", "metric": "font_top3", "actual": actual, "minimum": float(policy["font_top3_min"])})
    flat_metric_contracts = {
        "fill_delta_e_2000_median_max": ("fill_delta_e_2000", "median", "maximum"),
        "fill_delta_e_2000_p95_max": ("fill_delta_e_2000", "p95", "maximum"),
    }
    for policy_name, (section_name, metric_name, direction) in flat_metric_contracts.items():
        if policy_name not in policy:
            continue
        section = score.get(section_name)
        if not isinstance(section, dict) or metric_name not in section:
            findings.append({"code": "required_metric_missing", "metric": f"{section_name}.{metric_name}"})
            continue
        if section_name == "fill_delta_e_2000" and int(section.get("count") or 0) == 0:
            findings.append({"code": "zero_metric_denominator", "metric": section_name})
        actual = float(section[metric_name])
        threshold = float(policy[policy_name])
        breached = actual < threshold if direction == "minimum" else actual > threshold
        if breached:
            findings.append({"code": "metric_threshold_breach", "metric": f"{section_name}.{metric_name}", "actual": actual, direction: threshold})
    return {"status": "BLOCK" if findings else "PASS", "findings": findings}


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
    font_top1 = {"evaluated": 0, "hits": 0}
    font_top3 = {"evaluated": 0, "hits": 0}
    fill_deltas: list[float] = []

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
            top_k = record.get("font_top_k") if isinstance(record.get("font_top_k"), list) else []
            if top_k:
                font_top1["evaluated"] += 1
                font_top3["evaluated"] += 1
                if _equal(case.get("font_name"), top_k[0]):
                    font_top1["hits"] += 1
                if any(_equal(case.get("font_name"), candidate) for candidate in top_k[:3]):
                    font_top3["hits"] += 1
            delta_e = record.get("fill_delta_e_2000")
            if isinstance(delta_e, (int, float)):
                fill_deltas.append(float(delta_e))
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
    def rate(section: dict[str, int]) -> float:
        return round(section["hits"] / section["evaluated"], 4) if section["evaluated"] else 0.0

    import numpy as np
    return {
        "attributes": attribute_report,
        "font_top1": {**font_top1, "rate": rate(font_top1)},
        "font_top3": {**font_top3, "rate": rate(font_top3)},
        "fill_delta_e_2000": {
            "count": len(fill_deltas),
            "median": round(float(np.median(fill_deltas)), 6) if fill_deltas else None,
            "p95": round(float(np.percentile(fill_deltas, 95)), 6) if fill_deltas else None,
            "evidence": "owner_glyph_core_mask",
        },
        "gates": {
            "hard_negative_abstention": bool(hard_negative["evaluated"])
            and hard_negative_rate == 1.0,
        },
        "hard_negative": {**hard_negative, "rate": hard_negative_rate},
        "round_trip": {**round_trip, "rate": round_trip_rate},
    }


def write_run_reports(
    run_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    legacy_records: list[dict[str, Any]] | None = None,
    validation_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write score artifacts beneath an already-isolated benchmark run."""
    run_dir = Path(run_dir)
    records_path = run_dir / "style_benchmark_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    score = score_benchmark(manifest, records)
    validation = evaluate_validation_thresholds(score, validation_thresholds)
    validation["inputs"] = ["v2_mask_backed"]
    summary = {
        "schema_version": 3,
        "run_id": run_dir.name,
        "level": manifest.get("level"),
        "seed": manifest.get("seed"),
        "authoritative_score": "v2_mask_backed",
        "provenance": {
            "manifest": "benchmark_manifest.json",
            "ground_truth": "source_composition_masks",
            "measurement": "owner_mask_v2",
        },
        "score": score,
        "legacy_diagnostic_score": score_benchmark(manifest, legacy_records or []),
        "validation": validation,
    }
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
