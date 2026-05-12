"""Runtime profile decisions for TraduzAi pipeline execution."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VALID_PROFILES = {"balanced", "performance", "eco"}
PROFILE_ALIASES = {
    "default": "balanced",
    "normal": "balanced",
    "padrao": "balanced",
    "padrão": "balanced",
    "fast": "performance",
    "perf": "performance",
    "speed": "performance",
    "economy": "eco",
    "economia": "eco",
}


@dataclass(frozen=True)
class RuntimeProfileDecision:
    requested_profile: str
    profile: str
    ready_for_default: bool
    visual_stack_warmup: bool
    strip_inpainter_prewarm: bool
    semantic_review: bool
    smart_skip: str
    macro_ocr: str
    cpu_thread_limit: int | None
    env_defaults: dict[str, str]
    blocked_features: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_runtime_profile(config: dict[str, Any] | None) -> RuntimeProfileDecision:
    config = config or {}
    requested = _requested_profile(config)
    profile = _normalize_profile(requested)

    common_env = {
        "TRADUZAI_SEMANTIC_REVIEW": "0",
    }

    if profile == "performance":
        return RuntimeProfileDecision(
            requested_profile=requested,
            profile="performance",
            ready_for_default=False,
            visual_stack_warmup=True,
            strip_inpainter_prewarm=True,
            semantic_review=False,
            smart_skip="off",
            macro_ocr="off",
            cpu_thread_limit=None,
            env_defaults=common_env,
            blocked_features=["TRADUZAI_SMART_SKIP", "TRADUZAI_MACRO_OCR"],
            notes=[
                "Performance profile is declared but real accelerators remain disabled until gates pass.",
                "Use shadow diagnostics separately when investigating Smart Skip or Macro OCR.",
            ],
        )

    if profile == "eco":
        env_defaults = {
            **common_env,
            "TRADUZAI_STRIP_INPAINTER_PREWARM": "0",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
        return RuntimeProfileDecision(
            requested_profile=requested,
            profile="eco",
            ready_for_default=True,
            visual_stack_warmup=False,
            strip_inpainter_prewarm=False,
            semantic_review=False,
            smart_skip="off",
            macro_ocr="off",
            cpu_thread_limit=2,
            env_defaults=env_defaults,
            blocked_features=[],
            notes=[
                "Eco profile avoids optional prewarm and keeps Google-only translation.",
                "It may be slower on cold starts, but should reduce background warmup pressure.",
            ],
        )

    return RuntimeProfileDecision(
        requested_profile=requested,
        profile="balanced",
        ready_for_default=True,
        visual_stack_warmup=True,
        strip_inpainter_prewarm=True,
        semantic_review=False,
        smart_skip="off",
        macro_ocr="off",
        cpu_thread_limit=None,
        env_defaults=common_env,
        blocked_features=[],
        notes=["Balanced profile preserves current default behavior."],
    )


def apply_runtime_profile_environment(decision: RuntimeProfileDecision) -> dict[str, str]:
    applied: dict[str, str] = {}
    for key, value in decision.env_defaults.items():
        if key in os.environ:
            applied[key] = "preserved"
            continue
        os.environ[key] = value
        applied[key] = "applied"
    return applied


def evaluate_runtime_profile_gate(out_dir: str | Path | None = None) -> dict[str, Any]:
    profile_decisions = {
        name: resolve_runtime_profile({"runtime_profile": name}).to_dict()
        for name in ("balanced", "performance", "eco")
    }
    performance_blocked = bool(profile_decisions["performance"]["blocked_features"])
    eco_ready = bool(profile_decisions["eco"]["ready_for_default"])
    status = "PASS" if performance_blocked and eco_ready else "FAIL"
    reasons = []
    if performance_blocked:
        reasons.append("performance profile documents blocked accelerators")
    else:
        reasons.append("performance profile did not document blocked accelerators")
    if eco_ready:
        reasons.append("eco profile has a runnable low-resource contract")
    else:
        reasons.append("eco profile is not runnable")

    result = {
        "gate": {
            "name": "runtime_profile_contract",
            "status": status,
            "reasons": reasons,
            "profiles": profile_decisions,
        }
    }
    return _write_result(result, out_dir)


def build_chapter_route_shadow(
    page_results: list[dict[str, Any]],
    *,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Summarize route risk without changing execution automatically."""

    sample = _select_shadow_sample(page_results, sample_size=max(1, int(sample_size)))
    if not sample:
        return {
            "mode": "shadow",
            "status": "NO_SAMPLE",
            "sample_size": 0,
            "text_count_ratio": None,
            "vlm_failure_phrase_rate": 0.0,
            "suspicious_bbox_rate": 0.0,
            "recommendation": "keep_current_route",
            "route_history": [],
        }

    text_counts = [len(page.get("texts") or []) for page in sample]
    block_counts = [len(page.get("_vision_blocks") or []) for page in sample]
    total_texts = sum(text_counts)
    total_blocks = sum(block_counts)
    text_count_ratio = total_texts / float(max(1, total_blocks))
    vlm_failure_count = 0
    suspicious_bbox_count = 0
    route_history = []

    for page in sample:
        page_number = page.get("numero") or page.get("_source_page_number") or page.get("image") or len(route_history) + 1
        page_flags = []
        for text in page.get("texts") or []:
            flags = {str(flag) for flag in text.get("qa_flags") or []}
            if "vlm_failure_phrase" in flags:
                vlm_failure_count += 1
                page_flags.append("vlm_failure_phrase")
            if _is_suspicious_bbox(text.get("bbox")):
                suspicious_bbox_count += 1
                page_flags.append("suspicious_bbox")
        route_history.append(
            {
                "page": page_number,
                "text_count": len(page.get("texts") or []),
                "vision_block_count": len(page.get("_vision_blocks") or []),
                "flags": sorted(set(page_flags)),
                "route": "shadow_only",
            }
        )

    denominator = float(max(1, total_texts))
    vlm_failure_rate = vlm_failure_count / denominator
    suspicious_bbox_rate = suspicious_bbox_count / denominator
    recommendation = "keep_current_route"
    if text_count_ratio < 0.55 or suspicious_bbox_rate > 0.25:
        recommendation = "enable_bbox_expanded_reocr"
    if text_count_ratio < 0.35 and suspicious_bbox_rate > 0.35:
        recommendation = "consider_page_detect_after_reocr"

    return {
        "mode": "shadow",
        "status": "PASS",
        "sample_size": len(sample),
        "text_count_ratio": round(text_count_ratio, 4),
        "vlm_failure_phrase_rate": round(vlm_failure_rate, 4),
        "suspicious_bbox_rate": round(suspicious_bbox_rate, 4),
        "recommendation": recommendation,
        "route_history": route_history,
    }


def _requested_profile(config: dict[str, Any]) -> str:
    for key in ("runtime_profile", "execution_profile", "performance_profile"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    preset = config.get("preset")
    if isinstance(preset, dict):
        for key in ("runtime_profile", "execution_profile", "profile", "id"):
            value = preset.get(key)
            if isinstance(value, str) and _normalize_profile(value) in VALID_PROFILES:
                return value.strip()
    elif isinstance(preset, str) and preset.strip():
        return preset.strip()

    return "balanced"


def _select_shadow_sample(page_results: list[dict[str, Any]], *, sample_size: int) -> list[dict[str, Any]]:
    pages = [page for page in page_results if isinstance(page, dict)]
    if len(pages) <= sample_size:
        return pages
    if sample_size == 1:
        return [pages[len(pages) // 2]]
    selected = []
    for slot in range(sample_size):
        index = round(slot * (len(pages) - 1) / float(sample_size - 1))
        selected.append(pages[int(index)])
    return selected


def _is_suspicious_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return True
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except Exception:
        return True
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < 8 or height < 8:
        return True
    aspect = width / max(1.0, height)
    return aspect > 14.0 or aspect < 0.05


def _normalize_profile(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    normalized = PROFILE_ALIASES.get(normalized, normalized)
    return normalized if normalized in VALID_PROFILES else "balanced"


def _write_result(result: dict[str, Any], out_dir: str | Path | None) -> dict[str, Any]:
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = evaluate_runtime_profile_gate(args.out)
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
