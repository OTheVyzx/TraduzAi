from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time

import pytest


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _request() -> dict[str, object]:
    return {
        "source_sha256": SHA_A,
        "authenticated_neighbor_sha256s": [SHA_B],
        "region": {"bbox": [83, 422, 407, 650], "coordinate_space": "logical_page"},
        "coordinate_space": "logical_page",
        "transform_sha256": SHA_B,
        "source_language": "en",
        "analysis_config_sha256": SHA_C,
        "providers": {
            "detection": {"family": "vision-stack", "name": "detector", "model": "craft", "version": "1"},
            "ocr": {"family": "vision-paddleocr", "name": "PaddleOCR", "model": "v4", "version": "2.9.1"},
            "structure": {"family": "vision-runtime", "name": "relations", "model": "rules", "version": "1"},
            "masks": {"family": "vision-runtime", "name": "mask-channels", "model": "rules", "version": "1"},
        },
        "target_text": "TEXTO",
        "font_size": 30,
        "line_spacing": 1.1,
    }


def test_typography_and_target_text_do_not_change_analysis_identity() -> None:
    from vision_runtime.analysis_cache import AnalysisCacheIdentity

    first = AnalysisCacheIdentity.build(_request())
    changed = _request()
    changed.update({"target_text": "OUTRO TEXTO", "font_size": 18, "line_spacing": 1.4})
    second = AnalysisCacheIdentity.build(changed)

    assert first.sha256 == second.sha256
    assert first.capability_sha256("ocr") == second.capability_sha256("ocr")


@pytest.mark.parametrize("field,value", [
    ("source_sha256", SHA_C),
    ("region", {"bbox": [84, 422, 407, 650], "coordinate_space": "logical_page"}),
])
def test_source_or_region_change_invalidates_analysis_identity(field: str, value: object) -> None:
    from vision_runtime.analysis_cache import AnalysisCacheIdentity

    first = AnalysisCacheIdentity.build(_request())
    changed = _request()
    changed[field] = value

    assert first.sha256 != AnalysisCacheIdentity.build(changed).sha256


def test_ocr_provider_change_invalidates_ocr_but_not_detection() -> None:
    from vision_runtime.analysis_cache import AnalysisCacheIdentity

    first = AnalysisCacheIdentity.build(_request())
    changed = _request()
    providers = deepcopy(changed["providers"])
    providers["ocr"]["version"] = "2.9.2"
    changed["providers"] = providers
    second = AnalysisCacheIdentity.build(changed)

    assert first.capability_sha256("ocr") != second.capability_sha256("ocr")
    assert first.capability_sha256("detection") == second.capability_sha256("detection")


def test_selection_revision_changes_without_invalidating_detection_or_ocr() -> None:
    from vision_runtime.analysis_cache import AnalysisCacheIdentity

    identity = AnalysisCacheIdentity.build(_request())

    assert identity.revision_sha256("ocr-001") != identity.revision_sha256("ocr-002")
    assert identity.capability_sha256("detection") == identity.capability_sha256("detection")
    assert identity.capability_sha256("ocr") == identity.capability_sha256("ocr")


def test_complete_result_is_reused_without_second_provider_call(tmp_path) -> None:
    from vision_runtime.analysis_cache import AnalysisArtifactCache, AnalysisCacheIdentity

    cache = AnalysisArtifactCache(tmp_path)
    identity = AnalysisCacheIdentity.build(_request())
    calls = 0

    def produce() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "complete", "analysis_record_sha256": SHA_A}

    first = cache.get_or_compute(identity, produce)
    second = cache.get_or_compute(identity, produce)

    assert first.payload == second.payload
    assert calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True


def test_cancelled_result_is_never_published_as_complete(tmp_path) -> None:
    from vision_runtime.analysis_cache import AnalysisArtifactCache, AnalysisCacheIdentity

    cache = AnalysisArtifactCache(tmp_path)
    identity = AnalysisCacheIdentity.build(_request())

    with pytest.raises(ValueError, match="complete"):
        cache.get_or_compute(identity, lambda: {"status": "cancelled"})

    assert cache.lookup(identity) is None


def test_two_consumers_share_one_inflight_provider_call(tmp_path) -> None:
    from vision_runtime.analysis_cache import AnalysisArtifactCache, AnalysisCacheIdentity

    cache = AnalysisArtifactCache(tmp_path)
    identity = AnalysisCacheIdentity.build(_request())
    calls = 0
    lock = threading.Lock()

    def produce() -> dict[str, object]:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return {"status": "complete", "analysis_record_sha256": SHA_A}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: cache.get_or_compute(identity, produce), range(2)))

    assert calls == 1
    assert sorted(result.cache_hit for result in results) == [False, True]


def test_deterministic_failure_is_reused_for_same_capability_identity(tmp_path) -> None:
    from vision_runtime.analysis_cache import (
        AnalysisCacheIdentity, DeterministicFailureCache,
    )

    identity = AnalysisCacheIdentity.build(_request())
    cache = DeterministicFailureCache(tmp_path / "failures")
    recorded = cache.record(
        identity,
        capability="ocr",
        reason_code="illegible_or_unrecognized",
        evidence_sha256=SHA_A,
    )

    reused = cache.lookup(identity, capability="ocr")
    assert reused == recorded
    assert reused["cache_hit"] is True
    assert reused["status"] == "deterministic_failure"


def test_transient_provider_failure_cannot_be_cached_as_deterministic(tmp_path) -> None:
    from vision_runtime.analysis_cache import (
        AnalysisCacheIdentity, DeterministicFailureCache,
    )

    identity = AnalysisCacheIdentity.build(_request())
    cache = DeterministicFailureCache(tmp_path / "failures")

    with pytest.raises(ValueError, match="not deterministic"):
        cache.record(
            identity,
            capability="ocr",
            reason_code="provider_unavailable",
            evidence_sha256=SHA_A,
        )


def test_failure_cache_invalidates_only_when_capability_identity_changes(tmp_path) -> None:
    from vision_runtime.analysis_cache import (
        AnalysisCacheIdentity, DeterministicFailureCache,
    )

    first = AnalysisCacheIdentity.build(_request())
    cache = DeterministicFailureCache(tmp_path / "failures")
    cache.record(
        first,
        capability="ocr",
        reason_code="no_observations",
        evidence_sha256=SHA_A,
    )
    typography = _request()
    typography.update({"target_text": "OUTRO", "font_size": 12})
    assert cache.lookup(
        AnalysisCacheIdentity.build(typography), capability="ocr"
    ) is not None

    provider_changed = _request()
    providers = deepcopy(provider_changed["providers"])
    providers["ocr"]["version"] = "new"
    provider_changed["providers"] = providers
    assert cache.lookup(
        AnalysisCacheIdentity.build(provider_changed), capability="ocr"
    ) is None
