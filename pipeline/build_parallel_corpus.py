from __future__ import annotations

import json
from pathlib import Path

from corpus.parallel_dataset import write_corpus_artifacts


def main():
    pipeline_dir = Path(__file__).resolve().parent
    root_dir = pipeline_dir.parent
    pt_dir = root_dir / "exemplos" / "exemploptbr"
    en_dir = root_dir / "exemplos" / "exemploen"
    output_dir = pipeline_dir / "models" / "corpus" / "the-regressed-mercenary-has-a-plan"

    result = write_corpus_artifacts(
        pt_directory=pt_dir,
        en_directory=en_dir,
        output_directory=output_dir,
        work_slug="the-regressed-mercenary-has-a-plan",
        max_ocr_page_pairs=24,
    )
    print(json.dumps(result["work_profile"], ensure_ascii=False, indent=2))
    print(json.dumps(result["visual_benchmark_profile"], ensure_ascii=False, indent=2))
    print(json.dumps(result["page_alignment_profile"]["chapters"][:3], ensure_ascii=False, indent=2))
    print(json.dumps({
        "sampled_page_pairs": result["translation_memory_candidates"]["sampled_page_pairs"],
        "candidate_count": result["translation_memory_candidates"]["candidate_count"],
        "glossary_candidate_count": result["translation_memory_candidates"]["glossary_candidate_count"],
    }, ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {result['output_directory']}")


if __name__ == "__main__":
    main()
