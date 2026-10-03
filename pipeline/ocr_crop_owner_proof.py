"""Compare the real detect/OCR owner manifest without translation or rendering."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_stack import runtime
from ownership.model import SourceTextComponent, TextObservation
from ownership.reconcile import SemanticRegion, build_page_owner_graph
from ownership.translation import owners_to_translation_page
from strip.run import _associate_page_observations

ROOT = Path(r"N:\TraduzAI")
SOURCE = ROOT / "artifacts/e2e/optimized_e2e_20260907/full_chapter_observation_57_complete_02/source_input/023.webp"
runtime._configure_model_roots(str(ROOT / "models"))
engine = runtime._get_ocr_engine("max", lang="en")
provider = engine._model
original = provider.ocr
counter = {"calls": 0}


def counted(*args, **kwargs):
    counter["calls"] += 1
    return original(*args, **kwargs)


provider.ocr = counted
rows = []
for mode in ("baseline", "crop_first"):
    os.environ["TRADUZAI_OCR_CROP_FIRST"] = "1" if mode == "crop_first" else "0"
    before = counter["calls"]
    started = time.perf_counter()
    page = runtime.run_detect_ocr(str(SOURCE), models_dir=str(ROOT / "models"), profile="max", idioma_origem="en", engine_preset_id="default")
    components = []
    regions = []
    for index, item in enumerate(page.get("_vision_blocks", []), 1):
        bbox = tuple(int(value) for value in item["bbox"])
        x1, y1, x2, y2 = bbox
        component_id = f"selected_component_{index}"
        components.append(SourceTextComponent(
            component_id=component_id, page_id="page_023", bbox_page=bbox,
            polygon_page=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            detector_sources=("selected_vision_block",),
        ))
        regions.append(SemanticRegion(f"selected_region_{index}", (component_id,), "body"))
    observations = [TextObservation(
        observation_id=item["observation_id"], page_id="page_023",
        component_ids=tuple(item.get("component_ids") or ()),
        text=item["text"], confidence=float(item.get("confidence") or 0),
        provider=item["provider"], bbox_page=tuple(item["bbox_page"]),
        rejection_reason=item.get("rejection_reason"),
        legacy_selected=bool(item.get("legacy_selected")),
        run_id=str(item.get("run_id") or ""),
        origin_execution_id=str(item.get("origin_execution_id") or ""),
        invocation_id=str(item.get("invocation_id") or ""),
        attempt_id=str(item.get("attempt_id") or ""),
        provider_family=str(item.get("provider_family") or ""),
        page_source_sha256=str(item.get("page_source_sha256") or ""),
        root_input_pixel_sha256=str(item.get("root_input_pixel_sha256") or ""),
        input_pixel_sha256=str(item.get("input_pixel_sha256") or ""),
        payload_sha256=str(item.get("payload_sha256") or ""),
    ) for item in page.get("owner_observations", [])]
    associated = _associate_page_observations(observations, components)
    graph = build_page_owner_graph(
        page_id="page_023", components=components, observations=associated,
        semantic_regions=regions,
    )
    try:
        translation_input = owners_to_translation_page(graph)
        translation_error = None
    except Exception as exc:
        translation_input = None
        translation_error = f"{type(exc).__name__}: {exc}"
    row = {
        "mode": mode,
        "seconds": round(time.perf_counter() - started, 3),
        "provider_calls": counter["calls"] - before,
        "texts": [{"text": item.get("text"), "bbox": item.get("bbox"), "route_action": item.get("route_action")}
                  for item in page.get("texts", [])],
        "owner_observations": [{"text": item.get("text"), "bbox_page": item.get("bbox_page"),
                                "provider": item.get("provider"), "legacy_selected": item.get("legacy_selected")}
                               for item in page.get("owner_observations", [])],
        "vision_blocks": [{"bbox": item.get("bbox"), "text": item.get("text")}
                          for item in page.get("_vision_blocks", [])],
        "derived_graph": {
            "owners": [{"owner_id": owner.owner_id, "source_payload": owner.source_payload,
                        "state": owner.state, "disposition": owner.disposition,
                        "route_action": owner.route_action} for owner in graph.owners],
            "validation": [item.code for item in graph.validate()],
            "translation_input": translation_input,
            "translation_error": translation_error,
        },
    }
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

Path(__file__).with_name("ocr_crop_owner_proof_results.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
)
