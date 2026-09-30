"""Executable cache-invalidation probe bound to an exact AnalysisRecord."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ownership.hash_contract import canonical_json_sha256
from vision_runtime.analysis_cache import AnalysisCacheIdentity


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provider_request(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_sha256": record["source_sha256"],
        "authenticated_neighbor_sha256s": record.get("authenticated_neighbor_sha256s", []),
        "region": record["region"],
        "coordinate_space": record["coordinate_space"],
        "transform_sha256": record["transform_sha256"],
        "source_language": record["source_language"],
        "analysis_config_sha256": record["analysis_config_sha256"],
        "providers": {
            "detection": {
                "family": "vision-runtime",
                "name": "detector",
                "model": "configured-detector",
                "version": str(record.get("capability_version") or "vision-runtime-v1"),
            },
            "ocr": {
                "family": record["provider_family"],
                "name": record["provider_name"],
                "model": record["provider_model"],
                "version": record["provider_version"],
            },
            "structure": {
                "family": "vision-runtime",
                "name": "structure",
                "model": "contract-v1",
                "version": str(record.get("capability_version") or "vision-runtime-v1"),
            },
            "masks": {
                "family": "vision-runtime",
                "name": "mask-channels",
                "model": "contract-v1",
                "version": str(record.get("capability_version") or "vision-runtime-v1"),
            },
        },
        "target_text": "TEXTO",
        "font_size": 30,
        "line_spacing": 1.1,
    }


def run_invalidation_probe(analysis_record_path: str | Path) -> dict[str, Any]:
    """Run counting providers behind capability-scoped cache keys."""

    record_path = Path(analysis_record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    embedded_sha256 = str(record.get("analysis_record_sha256") or "")
    semantic = dict(record)
    semantic.pop("analysis_record_sha256", None)
    if canonical_json_sha256(semantic) != embedded_sha256:
        raise ValueError("AnalysisRecord semantic SHA-256 mismatch")

    counters = {"detection": 0, "ocr": 0}
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def resolve(identity: AnalysisCacheIdentity) -> None:
        for capability in ("detection", "ocr"):
            key = (capability, identity.capability_sha256(capability))
            if key not in cache:
                counters[capability] += 1
                cache[key] = {"provider_call": counters[capability]}

    baseline_request = _provider_request(record)
    baseline = AnalysisCacheIdentity.build(baseline_request)
    resolve(baseline)
    after_baseline = dict(counters)

    typography_request = deepcopy(baseline_request)
    typography_request.update({
        "target_text": "OUTRO TEXTO",
        "font_size": 18,
        "line_spacing": 1.4,
    })
    typography = AnalysisCacheIdentity.build(typography_request)
    resolve(typography)
    after_typography = dict(counters)

    provider_request = deepcopy(baseline_request)
    provider_request["providers"]["ocr"]["version"] = (
        f"{provider_request['providers']['ocr']['version']}+probe"
    )
    provider_changed = AnalysisCacheIdentity.build(provider_request)
    resolve(provider_changed)
    after_provider_change = dict(counters)

    selected_observation_id = str(record.get("selected_observation_id") or "")
    payload = {
        "schema": "traduzai.integration.invalidation-probe.v1",
        "analysis_record_ref": {
            "path": record_path.as_posix(),
            "file_sha256": _file_sha256(record_path),
            "analysis_record_sha256": embedded_sha256,
        },
        "baseline": {
            "analysis_identity_sha256": baseline.sha256,
            "visual_revision_sha256": baseline.revision_sha256(selected_observation_id),
            "provider_calls_total": after_baseline,
        },
        "typography_only": {
            "analysis_identity_sha256": typography.sha256,
            "visual_revision_sha256": typography.revision_sha256(selected_observation_id),
            "visual_revision_unchanged": (
                typography.revision_sha256(selected_observation_id)
                == baseline.revision_sha256(selected_observation_id)
            ),
            "provider_calls_delta": {
                key: after_typography[key] - after_baseline[key] for key in counters
            },
        },
        "ocr_provider_change": {
            "analysis_identity_sha256": provider_changed.sha256,
            "detection_identity_unchanged": (
                provider_changed.capability_sha256("detection")
                == baseline.capability_sha256("detection")
            ),
            "ocr_identity_changed": (
                provider_changed.capability_sha256("ocr")
                != baseline.capability_sha256("ocr")
            ),
            "provider_calls_delta": {
                key: after_provider_change[key] - after_typography[key] for key in counters
            },
        },
        "provider_call_scope": "instrumented capability resolvers; no external inference invoked",
        "published_pixels": False,
        "human_approval_claimed": False,
    }
    return {**payload, "trace_sha256": canonical_json_sha256(payload)}


def write_trace(path: str | Path, trace: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(trace), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_record")
    parser.add_argument("output")
    args = parser.parse_args()
    trace = run_invalidation_probe(args.analysis_record)
    write_trace(args.output, trace)
    print(json.dumps(trace, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
