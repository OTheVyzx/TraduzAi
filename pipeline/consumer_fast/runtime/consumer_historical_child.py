"""Trusted child entrypoint for one historical recipe; never returns stored raster."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from consumer_fast_recipe import load_recipe, rerender_recipe


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(sys.argv[1]).resolve(strict=True)
    relative_recipe, digest, output = sys.argv[2], sys.argv[3], Path(sys.argv[4])
    project = json.loads((root / "project.json").read_text(encoding="utf8"))
    found = []
    for page in project["pages"]:
        references = json.loads(
            (root / page["rerender_metadata"]).read_text(encoding="utf8")
        )["recipes"]
        for reference in references:
            if reference["path"] == relative_recipe and reference["sha256"] == digest:
                found.append((page, reference))
    if len(found) != 1:
        raise ValueError("historical recipe reference ambiguous")
    page, reference = found[0]
    recipe = load_recipe(root, reference)
    clean = np.asarray(Image.open(root / page["clean_base"]).convert("RGB"), dtype=np.uint8)
    rendered = rerender_recipe(clean, recipe)
    layers = rendered["text_layers"]
    if rendered.get("exception") is not None or len(layers) != 1:
        raise ValueError("historical recipe did not rasterize one layer")
    layer = layers[0]

    import consumer_fast_render
    import integrated_render
    import typesetter.font_policy as font_policy
    import typesetter.renderer as renderer
    import typesetter.stable_baseline as baseline

    effective = renderer._quality_closed_font_map()
    blocks = rendered.get("render_blocks") or []
    selected = (blocks[0].get("font_policy_v1") or {}).get("selected") if blocks else None
    face = effective.path(selected) if selected else None
    np.save(output / "rgba.npy", layer.rgba, allow_pickle=False)
    result = {
        "recipe_sha256": digest,
        "owner_id": layer.owner_id,
        "bbox": list(layer.bbox),
        "clean_pixels_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
        "rgba_sha256": hashlib.sha256(layer.rgba.tobytes()).hexdigest(),
        "interpreter": sys.executable,
        "closed_fonts": os.environ.get("TRADUZAI_QUALITY_CLOSED_FONTS"),
        "font_map_path": str(effective.map_path),
        "font_map_sha256": effective.map_sha256,
        "font_policy_sha256": sha(font_policy.POLICY_PATH),
        "selected_font": selected,
        "face_path": str(face) if face else None,
        "face_sha256": sha(face) if face else None,
        "modules": {
            name: {"path": str(module.__file__), "sha256": sha(Path(module.__file__))}
            for name, module in (
                ("renderer", renderer),
                ("stable_baseline", baseline),
                ("font_policy", font_policy),
                ("integrated_render", integrated_render),
                ("consumer_fast_render", consumer_fast_render),
            )
        },
        "render_blocks": blocks,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, default=str), encoding="utf8"
    )


if __name__ == "__main__":
    main()
