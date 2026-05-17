from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_page_server import FastPageSession, serve_jsonl  # noqa: E402


def _write_fake_translated_page(config_path: str, calls: list[dict]) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    calls.append(config)
    work_dir = Path(config["work_dir"])
    translated_dir = work_dir / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)
    (translated_dir / "001.png").write_bytes(b"fake image")
    (work_dir / "project.json").write_text(
        json.dumps({"paginas": [{"numero": 1}]}, ensure_ascii=True),
        encoding="utf-8",
    )


def test_fast_page_session_warms_once_and_reports_page_artifact(tmp_path: Path) -> None:
    runner_calls: list[dict] = []
    warmup_calls: list[dict] = []

    session = FastPageSession(
        pipeline_runner=lambda config_path: _write_fake_translated_page(config_path, runner_calls),
        warmup_runner=lambda **kwargs: warmup_calls.append(kwargs),
    )

    warmup_events = session.handle(
        {"type": "warmup", "models_dir": str(tmp_path / "models"), "profile": "max", "idioma_origem": "ko"}
    )
    page_events = session.handle(
        {
            "type": "process_page",
            "source_path": str(tmp_path / "001.png"),
            "work_dir": str(tmp_path / "jobs" / "one"),
            "models_dir": str(tmp_path / "models"),
            "obra": "Teste",
            "capitulo": 7,
            "idioma_origem": "ko",
            "idioma_destino": "pt-BR",
            "mode": "manual",
        }
    )

    assert warmup_calls == [{"models_dir": str(tmp_path / "models"), "profile": "max", "lang": "ko"}]
    assert warmup_events == [{"type": "ready", "session_id": session.session_id, "warm": True}]
    assert [event["type"] for event in page_events] == ["page_completed", "complete"]
    assert page_events[0]["artifact_path"] == "translated/001.png"
    assert page_events[1]["page_count"] == 1
    assert runner_calls[0]["mode"] == "manual"
    assert runner_calls[0]["source_path"] == str(tmp_path / "001.png")


def test_fast_page_session_warms_inpaint_once(tmp_path: Path) -> None:
    warmup_calls: list[dict] = []
    session = FastPageSession(
        pipeline_runner=lambda _config_path: None,
        inpaint_warmup_runner=lambda **kwargs: warmup_calls.append(kwargs),
        session_id="fixed-session",
    )

    first = session.handle({"type": "warmup_inpaint", "models_dir": str(tmp_path / "models"), "profile": "quality"})
    second = session.handle({"type": "warmup_inpaint", "models_dir": str(tmp_path / "models"), "profile": "quality"})

    assert warmup_calls == [{"models_dir": str(tmp_path / "models"), "profile": "quality"}]
    assert first == [{"type": "ready", "session_id": "fixed-session", "warm": True, "target": "inpaint"}]
    assert second == [
        {"type": "ready", "session_id": "fixed-session", "warm": True, "target": "inpaint", "reused": True}
    ]


def test_fast_page_session_routes_editor_reinpaint_and_captures_stdout(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("TRADUZAI_INPAINT_ROI_TIGHTEN", raising=False)
    calls: list[tuple[Path, int, dict | None]] = []
    roi_flags: list[str | None] = []

    def reinpaint_runner(project_path: Path, page_index: int, region: dict | None) -> None:
        calls.append((project_path, page_index, region))
        roi_flags.append(os.environ.get("TRADUZAI_INPAINT_ROI_TIGHTEN"))
        print(json.dumps({"type": "progress", "step": "inpaint", "message": "regional"}))
        print(json.dumps({"type": "complete", "output_path": str(tmp_path / "images" / "001.png")}))

    session = FastPageSession(
        pipeline_runner=lambda _config_path: None,
        reinpaint_runner=reinpaint_runner,
        session_id="fixed-session",
    )

    events = session.handle(
        {
            "type": "editor_reinpaint",
            "project_path": str(tmp_path / "project.json"),
            "page_index": 2,
            "region": {"bbox": [1, 2, 30, 40], "mask_path": str(tmp_path / "mask.png")},
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert calls == [
        (tmp_path / "project.json", 2, {"bbox": [1, 2, 30, 40], "mask_path": str(tmp_path / "mask.png")})
    ]
    assert roi_flags == ["1"]
    assert os.environ.get("TRADUZAI_INPAINT_ROI_TIGHTEN") is None
    assert [event["type"] for event in events] == ["progress", "complete"]
    assert events[0]["message"] == "regional"


def test_fast_page_session_routes_editor_detect_and_ocr_actions(tmp_path: Path, capsys) -> None:
    calls: list[tuple[str, Path, int, dict | None, dict | None]] = []

    def detect_runner(project_path: Path, page_index: int, region: dict | None, options: dict | None) -> None:
        calls.append(("detect", project_path, page_index, region, options))
        print(json.dumps({"type": "progress", "step": "ocr", "message": "detectando"}))
        print(json.dumps({"type": "complete", "output_path": str(tmp_path / "translated" / "001.png")}))

    def ocr_runner(project_path: Path, page_index: int, region: dict | None, options: dict | None) -> None:
        calls.append(("ocr", project_path, page_index, region, options))
        print(json.dumps({"type": "progress", "step": "ocr", "message": "lendo"}))
        print(json.dumps({"type": "complete", "output_path": str(tmp_path / "translated" / "002.png")}))

    session = FastPageSession(
        pipeline_runner=lambda _config_path: None,
        detect_runner=detect_runner,
        ocr_runner=ocr_runner,
        session_id="fixed-session",
    )

    detect_events = session.handle(
        {
            "type": "editor_detect_page",
            "project_path": str(tmp_path / "project.json"),
            "page_index": 1,
            "idioma_origem": "ja",
            "engine_preset_id": "manga",
        }
    )
    ocr_events = session.handle(
        {
            "type": "editor_ocr_page",
            "project_path": str(tmp_path / "project.json"),
            "page_index": 2,
            "region": {"bbox": [10, 20, 30, 40]},
            "idioma_origem": "ja",
            "engine_preset_id": "manga",
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert calls == [
        (
            "detect",
            tmp_path / "project.json",
            1,
            None,
            {"idioma_origem": "ja", "idioma_destino": None, "engine_preset_id": "manga"},
        ),
        (
            "ocr",
            tmp_path / "project.json",
            2,
            {"bbox": [10, 20, 30, 40]},
            {"idioma_origem": "ja", "idioma_destino": None, "engine_preset_id": "manga"},
        ),
    ]
    assert [event["type"] for event in detect_events] == ["progress", "complete"]
    assert detect_events[0]["message"] == "detectando"
    assert detect_events[1]["output_path"].endswith("001.png")
    assert [event["type"] for event in ocr_events] == ["progress", "complete"]
    assert ocr_events[0]["message"] == "lendo"
    assert ocr_events[1]["output_path"].endswith("002.png")


def test_serve_jsonl_keeps_one_session_for_multiple_page_requests(tmp_path: Path) -> None:
    runner_calls: list[dict] = []
    input_stream = StringIO(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "process_page",
                        "source_path": str(tmp_path / "001.png"),
                        "work_dir": str(tmp_path / "jobs" / "one"),
                        "models_dir": str(tmp_path / "models"),
                    }
                ),
                json.dumps(
                    {
                        "type": "process_page",
                        "source_path": str(tmp_path / "002.png"),
                        "work_dir": str(tmp_path / "jobs" / "two"),
                        "models_dir": str(tmp_path / "models"),
                    }
                ),
                json.dumps({"type": "shutdown"}),
            ]
        )
        + "\n"
    )
    output_stream = StringIO()
    session = FastPageSession(
        pipeline_runner=lambda config_path: _write_fake_translated_page(config_path, runner_calls),
        session_id="fixed-session",
    )

    serve_jsonl(input_stream, output_stream, session=session)

    events = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    page_complete_events = [event for event in events if event["type"] == "complete"]
    assert len(runner_calls) == 2
    assert len(page_complete_events) == 2
    assert {event["session_id"] for event in events} == {"fixed-session"}
    assert events[-1]["type"] == "bye"


def test_fast_page_session_captures_inner_pipeline_stdout(tmp_path: Path, capsys) -> None:
    def noisy_runner(config_path: str) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        work_dir = Path(config["work_dir"])
        translated_dir = work_dir / "translated"
        translated_dir.mkdir(parents=True, exist_ok=True)
        (translated_dir / "001.png").write_bytes(b"fake image")
        print(json.dumps({"type": "progress", "step": "extract", "message": "inner"}))
        print(json.dumps({"type": "complete", "output_path": str(work_dir)}))

    session = FastPageSession(pipeline_runner=noisy_runner, session_id="fixed-session")

    events = session.handle(
        {
            "type": "process_page",
            "source_path": str(tmp_path / "001.png"),
            "work_dir": str(tmp_path / "jobs" / "one"),
            "models_dir": str(tmp_path / "models"),
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert [event["type"] for event in events] == ["progress", "page_completed", "complete"]
    assert events[0]["message"] == "inner"


def test_fast_page_session_reports_source_page_when_pipeline_renumbers_single_input(tmp_path: Path) -> None:
    def renumbering_runner(config_path: str) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        translated_dir = Path(config["work_dir"]) / "translated"
        translated_dir.mkdir(parents=True, exist_ok=True)
        (translated_dir / "001.jpg").write_bytes(b"fake image")

    session = FastPageSession(pipeline_runner=renumbering_runner, session_id="fixed-session")

    events = session.handle(
        {
            "type": "process_page",
            "source_path": str(tmp_path / "002.jpg"),
            "work_dir": str(tmp_path / "jobs" / "page-002"),
            "models_dir": str(tmp_path / "models"),
        }
    )

    page_event = next(event for event in events if event["type"] == "page_completed")
    assert page_event["current_page"] == 2
    assert page_event["source_page_number"] == 2
    assert page_event["artifact_filename"] == "001.jpg"


def test_fast_page_session_does_not_use_archive_number_for_each_output_page(tmp_path: Path) -> None:
    def chapter_runner(config_path: str) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        translated_dir = Path(config["work_dir"]) / "translated"
        translated_dir.mkdir(parents=True, exist_ok=True)
        (translated_dir / "001.jpg").write_bytes(b"page 1")
        (translated_dir / "002.jpg").write_bytes(b"page 2")

    session = FastPageSession(pipeline_runner=chapter_runner, session_id="fixed-session")

    events = session.handle(
        {
            "type": "process_page",
            "source_path": str(tmp_path / "113.cbz"),
            "work_dir": str(tmp_path / "jobs" / "chapter"),
            "models_dir": str(tmp_path / "models"),
        }
    )

    page_events = [event for event in events if event["type"] == "page_completed"]
    assert [event["current_page"] for event in page_events] == [1, 2]
    assert [event["source_page_number"] for event in page_events] == [1, 2]
