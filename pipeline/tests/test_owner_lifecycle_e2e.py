from __future__ import annotations

from strip.page_pipeline import run_page_owner_pipeline
from test_page_owner_pipeline import _request, _services


def test_no_band_dialogue_recipe_completes_basic_page_first_lifecycle():
    request = _request(bands=())

    result = run_page_owner_pipeline(request, _services())

    assert result.coverage.entries
    assert result.owner_graph.read().owners[0].source_payload
    assert result.translations[0].target_locale == "pt-BR"
    assert result.status == "candidate_ready"
    assert result.final_page is None
