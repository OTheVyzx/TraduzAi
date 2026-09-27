"""Focal launcher for the same Consumer Fast recovery executor used by dispatch."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

from PIL import Image

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime"


def runtime_roots() -> tuple[Path, ...]:
    """Return the only code roots authorized for the isolated runtime."""

    return (RUNTIME_ROOT, PIPELINE_ROOT)


sys.path[:0] = [str(path) for path in runtime_roots()]
from consumer_fast_resume import load as load_frontier
from consumer_operational_recovery import prepare_member, read, sha
from consumer_operational_publish import publish


def run(*, source_project: Path, control_project: Path, frontier: Path,
        ocr_cache: Path, members: list[str], output: Path,
        font_path: Path,
        publish_to: Path | None = None,
        translation_review: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source = read(source_project / "project.json")
    restored = load_frontier(frontier, structural=True)
    graphs = {row["graph"].page_id: row["graph"] for row in restored["pages"]}
    output.mkdir(parents=True)
    audits = []
    candidates = []
    for member in members:
        matches = [row for row in source["pages"] if row.get("source_member") == member]
        if len(matches) != 1:
            raise ValueError("selected source member absent or ambiguous")
        page = matches[0]
        graph = graphs[page["page_id"]]
        observations = {row.observation_id: dict(observation_id=row.observation_id,
            text=row.text, bbox_page=list(row.bbox_page),
            component_ids=list(row.component_ids), confidence=row.confidence)
            for row in graph.observations}
        outcomes = prepare_member(source_root=source_project,
            control_root=control_project, member=member,
            graph_observations=observations, cache_root=ocr_cache,
            font_path=font_path,
            translation_review=translation_review)
        for ordinal, outcome in enumerate(outcomes):
            if outcome["status"] == "rendered_candidate":
                candidates.append(outcome)
            row = {key: value for key, value in outcome.items() if key not in
                   {"clean", "final", "layer", "recipe", "page", "subunits"}}
            row["source_member"] = member
            if outcome["status"] == "rendered_candidate":
                prefix = f"{Path(member).stem}_{ordinal:02d}"
                checkpoint = output / f"{prefix}_candidate.pickle"
                checkpoint.write_bytes(pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL))
                row["candidate_checkpoint"] = checkpoint.name
                row["candidate_checkpoint_sha256"] = sha(checkpoint.read_bytes())
                Image.fromarray(outcome["clean"]).save(output / f"{prefix}_clean.png")
                Image.fromarray(outcome["final"]).save(output / f"{prefix}_final.png")
                if "subunits" in outcome:
                    row["subunits"] = []
                    for unit in outcome["subunits"]:
                        Image.fromarray(unit["layer"].rgba).save(
                            output / f"{prefix}_subblock_{unit['order']}_layer.png")
                        row["subunits"].append({key: value for key, value in unit.items()
                            if key not in {"layer", "recipe"}})
                else:
                    Image.fromarray(outcome["layer"].rgba).save(output / f"{prefix}_layer.png")
                row["clean_pixels_sha256"] = sha(outcome["clean"].tobytes())
                row["final_pixels_sha256"] = sha(outcome["final"].tobytes())
                if "layer" in outcome:
                    row["layer_alpha_sha256"] = sha(outcome["layer"].rgba.tobytes())
                    row["recipe_summary"] = {key: value for key, value in outcome["recipe"].items()
                        if key not in {"visual_comfort_policy", "layout_search"}}
            audits.append(row)
    report = dict(schema="consumer_operational_closure_run_v1",
        source_project_sha256=sha((source_project/"project.json").read_bytes()),
        control_project_sha256=sha((control_project/"project.json").read_bytes()),
        source_frontier=str(frontier), ocr_cache=str(ocr_cache),
        members=members, outcomes=audits, publication_status="candidate_only")
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False,
        indent=2, default=str)+"\n", encoding="utf-8")
    if publish_to is not None and candidates:
        result = publish(control_root=control_project, destination=publish_to,
                         outcomes=candidates)
        report["publication_status"] = "published"
        report["publication_audit"] = result
        (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False,
            indent=2, default=str)+"\n", encoding="utf-8")
    elif publish_to is not None and audits and all(
            row["status"] == "already_published" for row in audits):
        report["publication_status"] = "no_op_already_published"
        (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False,
            indent=2, default=str)+"\n", encoding="utf-8")
    elif publish_to is not None and not audits:
        report["publication_status"] = "no_op_no_proposals"
        (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False,
            indent=2, default=str)+"\n", encoding="utf-8")
    print(json.dumps(dict(status="prepared", outcomes=[row["status"] for row in audits]),
                     ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--control-project", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--publish-to", type=Path)
    parser.add_argument("--translation-review", type=Path)
    args = parser.parse_args()
    os.environ["TRADUZAI_QUALITY_CLOSED_FONTS"] = "1"
    run(source_project=args.source_project, control_project=args.control_project,
        frontier=args.frontier, ocr_cache=args.ocr_cache,
        members=args.member, output=args.output, font_path=args.font_path,
        publish_to=args.publish_to,
        translation_review=args.translation_review)


if __name__ == "__main__":
    main()
