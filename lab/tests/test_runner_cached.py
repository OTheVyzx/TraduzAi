from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image
from lab.reference_ingestor import ChapterPair
from lab.runner import build_pipeline_config, enforce_gpu_policy, filter_chapter_pairs, run_pipeline_for_pair


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 1200), color=color).save(path)


def _write_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        temp_path = path.parent / "page-001.jpg"
        _write_image(temp_path, (220, 220, 220))
        archive.write(temp_path, arcname="001.jpg")
        temp_path.unlink()


class RunnerCachedFlowTests(unittest.TestCase):
    def test_run_pipeline_for_pair_ignores_non_json_stdout_lines(self) -> None:
        class _FakeProcess:
            def __init__(self, stdout_lines: list[str], return_code: int = 0) -> None:
                self.stdout = stdout_lines
                self._return_code = return_code

            def wait(self) -> int:
                return self._return_code

        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pair = ChapterPair(
                chapter_number=1,
                source_path=str(temp_root / "source.cbz"),
                reference_path=str(temp_root / "reference.cbz"),
                source_pages=10,
                reference_pages=10,
                reference_group="ArinVale",
            )
            run_dir = temp_root / "run"
            pause_file = temp_root / "lab.pause"
            noisy_stdout = [
                "Novo stack visual falhou no inpainting, fallback para legacy\n",
                json.dumps(
                    {
                        "type": "progress",
                        "step": "ocr",
                        "step_progress": 12.5,
                        "overall_progress": 12.5,
                        "current_page": 1,
                        "total_pages": 8,
                        "message": "OCR pagina 1/8",
                        "eta_seconds": 5.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
            ]

            with patch("lab.runner.subprocess.Popen", return_value=_FakeProcess(noisy_stdout)):
                with patch("lab.runner.emit"):
                    with patch("lab.runner.emit_agent"):
                        output_path = run_pipeline_for_pair(
                            root=repo_root,
                            pair=pair,
                            run_dir=run_dir,
                            pause_file=pause_file,
                            processed_before=0,
                            total_pairs=1,
                            work_slug="the-regressed-mercenary-has-a-plan",
                        )

            expected_output_dir = run_dir / "chapters" / "chapter-0001" / "output"
            self.assertEqual(output_path, expected_output_dir)
            pipeline_log = (run_dir / "chapters" / "chapter-0001" / "pipeline.log").read_text(encoding="utf-8")
            self.assertIn("[stdout-non-json] Novo stack visual falhou no inpainting, fallback para legacy", pipeline_log)

    def test_filter_chapter_pairs_supports_all_first_n_and_range(self) -> None:
        pairs = [
            ChapterPair(i, f"source-{i}.cbz", f"ref-{i}.cbz", 10, 10, "ArinVale")
            for i in (1, 2, 3, 4, 5)
        ]

        self.assertEqual(
            [pair.chapter_number for pair in filter_chapter_pairs(pairs, [])],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [pair.chapter_number for pair in filter_chapter_pairs(pairs, [1, 2, 3])],
            [1, 2, 3],
        )
        self.assertEqual(
            [pair.chapter_number for pair in filter_chapter_pairs(pairs, [5, 2, 99])],
            [2, 5],
        )

    def test_enforce_gpu_policy_rejects_missing_gpu_capabilities_when_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "OCR GPU indisponivel"):
            enforce_gpu_policy(
                "require_gpu",
                {
                    "ocr_worker_ready": False,
                    "torch_cuda_ready": True,
                    "onnx_gpu_ready": True,
                    "ollama_ready": True,
                },
            )

    def test_enforce_gpu_policy_accepts_partial_gpu_stack_when_only_preferred(self) -> None:
        enforce_gpu_policy(
            "prefer_gpu",
            {
                "ocr_worker_ready": False,
                "torch_cuda_ready": False,
                "onnx_gpu_ready": False,
                "ollama_ready": False,
            },
        )

    def test_build_pipeline_config_preserves_vision_worker_path(self) -> None:
        pair = ChapterPair(
            chapter_number=7,
            source_path="D:/TraduzAi/exemplos/exemploen/Chapter 7.cbz",
            reference_path="D:/TraduzAi/exemplos/exemploptbr/Capitulo 7.cbz",
            source_pages=72,
            reference_pages=80,
            reference_group="ArinVale",
        )

        config = build_pipeline_config(
            pair=pair,
            output_dir=Path("D:/traduzai_data/lab/runs/test/chapter-0007/output"),
            models_dir=Path("D:/traduzai_data/models"),
            pause_file=Path("D:/traduzai_data/lab/runs/test/lab.pause"),
            work_slug="the-regressed-mercenary-has-a-plan",
            vision_worker_path="D:/TraduzAi/vision-worker/target/debug/mangatl-vision.exe",
        )

        self.assertEqual(
            config["vision_worker_path"],
            "D:/TraduzAi/vision-worker/target/debug/mangatl-vision.exe",
        )

    def test_runner_reuses_cached_artifact_and_emits_benchmark(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            reference_dir = temp_root / "reference"
            run_dir = temp_root / "run"
            output_dir = run_dir / "chapters" / "chapter-0010" / "output"

            source_archive = source_dir / "Chapter 10_aaaaaa.cbz"
            reference_archive = reference_dir / "ArinVale_Capítulo 10_bbbbbb.cbz"
            _write_archive(source_archive)
            _write_archive(reference_archive)

            _write_image(output_dir / "translated" / "001.jpg", (215, 215, 215))
            (output_dir / "project.json").write_text(
                json.dumps(
                    {
                        "paginas": [
                            {
                                "numero": 1,
                                "arquivo_traduzido": "translated/001.jpg",
                                "textos": [
                                    {
                                        "original": "hello knight",
                                        "traduzido": "ola cavaleiro",
                                        "confianca_ocr": 0.95,
                                        "bbox": [0, 0, 320, 140],
                                        "estilo": {"tamanho": 18},
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "chapters" / "chapter-0010" / "chapter_artifact.json").write_text(
                json.dumps(
                    {
                        "chapter_number": 10,
                        "source_path": str(source_archive),
                        "reference_path": str(reference_archive),
                        "reference_group": "ArinVale",
                        "output_dir": str(output_dir),
                        "project_json": str(output_dir / "project.json"),
                        "benchmark": {
                            "score_before": 40.0,
                            "score_after": 72.0,
                            "green": True,
                            "summary": "Cache existente.",
                            "metrics": {
                                "textual_similarity": 75.0,
                                "term_consistency": 90.0,
                                "layout_occupancy": 70.0,
                                "readability": 74.0,
                                "visual_cleanup": 78.0,
                                "manual_edits_saved": 95.0,
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config_path = run_dir / "lab_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "test-run-1234",
                        "source_dir": str(source_dir),
                        "reference_dir": str(reference_dir),
                        "pause_file": str(run_dir / "lab.pause"),
                        "git_available": False,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "-m", "lab.runner", str(config_path)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            messages = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            self.assertTrue(any(message.get("type") == "benchmark_result" for message in messages))
            self.assertTrue(any(message.get("type") == "review_requested" for message in messages))
            self.assertTrue(
                any(
                    message.get("type") == "lab_state"
                    and message.get("status") == "completed"
                    for message in messages
                )
            )
            persisted_snapshot = temp_root / "snapshot.json"
            self.assertTrue(persisted_snapshot.exists())
            snapshot_payload = json.loads(persisted_snapshot.read_text(encoding="utf-8"))
            self.assertEqual(snapshot_payload["status"], "completed")
            self.assertEqual(snapshot_payload["processed_pairs"], 1)

    def test_runner_script_entrypoint_supports_direct_execution(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            reference_dir = temp_root / "reference"
            run_dir = temp_root / "run"
            output_dir = run_dir / "chapters" / "chapter-0010" / "output"

            source_archive = source_dir / "Chapter 10_aaaaaa.cbz"
            reference_archive = reference_dir / "ArinVale_Capitulo 10_bbbbbb.cbz"
            _write_archive(source_archive)
            _write_archive(reference_archive)

            _write_image(output_dir / "translated" / "001.jpg", (215, 215, 215))
            (output_dir / "project.json").write_text(
                json.dumps(
                    {
                        "paginas": [
                            {
                                "numero": 1,
                                "arquivo_traduzido": "translated/001.jpg",
                                "textos": [
                                    {
                                        "original": "hello knight",
                                        "traduzido": "ola cavaleiro",
                                        "confianca_ocr": 0.95,
                                        "bbox": [0, 0, 320, 140],
                                        "estilo": {"tamanho": 18},
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "chapters" / "chapter-0010" / "chapter_artifact.json").write_text(
                json.dumps(
                    {
                        "chapter_number": 10,
                        "source_path": str(source_archive),
                        "reference_path": str(reference_archive),
                        "reference_group": "ArinVale",
                        "output_dir": str(output_dir),
                        "project_json": str(output_dir / "project.json"),
                        "benchmark": {
                            "score_before": 40.0,
                            "score_after": 72.0,
                            "green": True,
                            "summary": "Cache existente.",
                            "metrics": {
                                "textual_similarity": 75.0,
                                "term_consistency": 90.0,
                                "layout_occupancy": 70.0,
                                "readability": 74.0,
                                "visual_cleanup": 78.0,
                                "manual_edits_saved": 95.0,
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config_path = run_dir / "lab_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "test-run-direct-script",
                        "source_dir": str(source_dir),
                        "reference_dir": str(reference_dir),
                        "pause_file": str(run_dir / "lab.pause"),
                        "git_available": False,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(repo_root / "lab" / "runner.py"), str(config_path)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            messages = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            self.assertTrue(any(message.get("type") == "benchmark_result" for message in messages))
            self.assertTrue(
                any(
                    message.get("type") == "lab_state"
                    and message.get("status") == "completed"
                    for message in messages
                )
            )
            persisted_snapshot = temp_root / "snapshot.json"
            self.assertTrue(persisted_snapshot.exists())
            snapshot_payload = json.loads(persisted_snapshot.read_text(encoding="utf-8"))
            self.assertEqual(snapshot_payload["status"], "completed")


if __name__ == "__main__":
    unittest.main()
