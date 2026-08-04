from pathlib import Path
import re


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (
    PIPELINE_ROOT / "vision_stack",
    PIPELINE_ROOT / "strip",
    PIPELINE_ROOT / "ownership",
)
FORBIDDEN = re.compile(
    r"_last_full_page_line_records|_last_recognize_blocks_stats|"
    r"_last_batch_cache_stats|_snapshot_ocr_engine_observations|"
    r"get_last_observation_records"
)


def test_enforced_ocr_path_has_no_mutable_engine_side_channel_names():
    hits = []
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN.search(line):
                    hits.append(f"{path.relative_to(PIPELINE_ROOT)}:{line_number}")
    assert hits == []
