from pathlib import Path

from main import build_project_json


def test_build_project_json_persists_page_engine_metadata():
    project = build_project_json(
        {
            "obra": "Teste",
            "capitulo": 1,
            "idioma_origem": "ko",
            "idioma_destino": "pt-BR",
            "engine_preset_id": "manhwa_manhua",
            "engine_preset": {
                "id": "manhwa_manhua",
                "content_family": "manhwa_manhua",
                "mask_strategy": "roi_segmentation_assisted",
            },
        },
        {},
        [
            {
                "texts": [],
                "_vision_blocks": [],
                "engine_preset_id": "manhwa_manhua",
                "_engine_preset": {
                    "engine_preset_id": "manhwa_manhua",
                    "content_family": "manhwa_manhua",
                    "mask_strategy": "roi_segmentation_assisted",
                    "engine_steps": [
                        "comic-text-bubble-detector",
                        "comic-text-detector-seg",
                        "speech-bubble-segmentation",
                        "paddle-ocr-vl-1.5",
                        "aot-inpainting",
                    ],
                },
            }
        ],
        [{"texts": []}],
        [Path("001.jpg")],
        1,
        1.2,
    )

    assert project["engine_preset_id"] == "manhwa_manhua"
    assert project["paginas"][0]["vision_engine"] == {
        "engine_preset_id": "manhwa_manhua",
        "content_family": "manhwa_manhua",
        "mask_strategy": "roi_segmentation_assisted",
        "engine_steps": [
            "comic-text-bubble-detector",
            "comic-text-detector-seg",
            "speech-bubble-segmentation",
            "paddle-ocr-vl-1.5",
            "aot-inpainting",
        ],
    }
