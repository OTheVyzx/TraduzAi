from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from ocr.text_router import route_text


HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
CJK_RE = re.compile(r"[\u3000-\u303F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")


def probe_sfx_candidate_script(candidate: dict[str, Any], recognized_text: str = "") -> dict[str, Any]:
    """Promote a visual SFX candidate only when script evidence is strong."""

    result = deepcopy(candidate)
    text = str(recognized_text or result.get("text") or result.get("original") or "").strip()
    sfx = result.get("sfx") if isinstance(result.get("sfx"), dict) else {}
    flags = list(dict.fromkeys([*(result.get("qa_flags") or []), *(sfx.get("qa_flags") or [])]))

    result["content_class"] = "sfx"
    result["tipo"] = "sfx"
    result["text"] = text
    result["original"] = text

    if text and HANGUL_RE.search(text):
        routed = route_text(text, tipo="sfx")
        result.update(
            {
                "script": routed.get("script", "hangul"),
                "route_action": routed.get("route_action", "translate_sfx_inpaint_render"),
                "translate_policy": routed.get("translate_policy", "adapt_sfx"),
                "render_policy": routed.get("render_policy", "sfx_style"),
                "route_reason": routed.get("route_reason", "hangul_sfx_candidate"),
            }
        )
        flags = [flag for flag in flags if flag != "sfx_script_unknown"]
        result["sfx"] = {
            **sfx,
            "source_text": text,
            "qa_flags": [flag for flag in sfx.get("qa_flags") or [] if flag != "sfx_script_unknown"],
        }
        result["qa_flags"] = flags
        return result

    flags = list(dict.fromkeys([*flags, "sfx_visual_candidate", "sfx_script_unknown"]))
    result.update(
        {
            "script": "unknown" if not text else "cjk_unknown" if CJK_RE.search(text) else "unknown",
            "route_action": "review_required",
            "translate_policy": "review",
            "render_policy": "review_required",
            "route_reason": "sfx_script_unknown",
            "qa_flags": flags,
        }
    )
    result["sfx"] = {
        **sfx,
        "source_text": text,
        "adapted_text": sfx.get("adapted_text") or "",
        "inpaint_allowed": False,
        "qa_flags": list(dict.fromkeys([*(sfx.get("qa_flags") or []), "sfx_script_unknown"])),
    }
    return result
