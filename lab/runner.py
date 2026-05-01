from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from lab.agents import ClaudeReviewerAgent
from lab.benchmarking import aggregate_benchmark_results, benchmark_chapter_output, load_corpus_profiles
from lab.critics import run_all_critics_over_chapters
from lab.planner import build_proposals, proposal_to_lab_payload
from lab.reference_ingestor import ChapterPair, pair_chapters

SNAPSHOT_STATE: dict[str, object] | None = None
SNAPSHOT_GLOBAL_PATH: Path | None = None
SNAPSHOT_RUN_PATH: Path | None = None


def now_ms() -> int:
    return int(time.time() * 1000)


def lab_data_root_for_run(run_dir: Path) -> Path:
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent
    return run_dir.parent


def recompute_pending_proposals(snapshot: dict[str, object]) -> None:
    proposals = snapshot.get("proposals", [])
    if not isinstance(proposals, list):
        snapshot["pending_proposals"] = 0
        return

    pending_statuses = {"needs_approval", "reviewing", "benchmark_passed"}
    snapshot["pending_proposals"] = sum(
        1
        for proposal in proposals
        if isinstance(proposal, dict) and str(proposal.get("proposal_status", "")) in pending_statuses
    )


def upsert_snapshot_item(items: list[dict], key_name: str, payload: dict) -> None:
    key_value = payload.get(key_name)
    for index, existing in enumerate(items):
        if existing.get(key_name) == key_value:
            items[index] = payload
            return
    items.append(payload)


def current_history_entry(snapshot: dict[str, object]) -> dict[str, object] | None:
    history = snapshot.get("history", [])
    run_id = str(snapshot.get("run_id", "")).strip()
    if not isinstance(history, list) or not run_id:
        return None

    for entry in reversed(history):
        if isinstance(entry, dict) and entry.get("run_id") == run_id:
            return entry
    return None


def persist_snapshot_state() -> None:
    global SNAPSHOT_STATE, SNAPSHOT_GLOBAL_PATH, SNAPSHOT_RUN_PATH

    if SNAPSHOT_STATE is None or SNAPSHOT_GLOBAL_PATH is None or SNAPSHOT_RUN_PATH is None:
        return

    SNAPSHOT_STATE["updated_at_ms"] = now_ms()
    payload = json.dumps(SNAPSHOT_STATE, ensure_ascii=False, indent=2)
    SNAPSHOT_GLOBAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_GLOBAL_PATH.write_text(payload, encoding="utf-8")
    SNAPSHOT_RUN_PATH.write_text(payload, encoding="utf-8")


def apply_snapshot_event(message_type: str, payload: dict[str, object]) -> None:
    global SNAPSHOT_STATE

    if SNAPSHOT_STATE is None:
        return

    snapshot = SNAPSHOT_STATE

    if message_type == "lab_state":
        for field in (
            "status",
            "run_id",
            "current_stage",
            "message",
            "acceleration_summary",
            "total_pairs",
            "processed_pairs",
            "eta_seconds",
            "pending_proposals",
            "active_batch_id",
            "git_available",
            "source_dir",
            "reference_dir",
            "chapter_pairs",
            "available_chapter_pairs",
            "scope_label",
            "gpu_policy",
        ):
            if field in payload:
                snapshot[field] = payload[field]

        history_entry = current_history_entry(snapshot)
        if history_entry is not None:
            history_entry["status"] = snapshot.get("status", history_entry.get("status", ""))
            history_entry["summary"] = str(snapshot.get("message", history_entry.get("summary", "")))
            history_entry["processed_pairs"] = int(snapshot.get("processed_pairs", 0) or 0)
            history_entry["total_pairs"] = int(snapshot.get("total_pairs", 0) or 0)
            if str(snapshot.get("status", "")) in {"completed", "error", "stopped"}:
                history_entry["finished_at_ms"] = now_ms()

    elif message_type == "agent_status":
        agent = payload.get("agent", {})
        if isinstance(agent, dict):
            agent["updated_at_ms"] = now_ms()
            agents = snapshot.setdefault("agents", [])
            if isinstance(agents, list):
                upsert_snapshot_item(agents, "agent_id", agent)

    elif message_type == "review_requested":
        proposal = payload.get("proposal", {})
        if isinstance(proposal, dict):
            proposal.setdefault("created_at_ms", now_ms())
            proposals = snapshot.setdefault("proposals", [])
            if isinstance(proposals, list):
                upsert_snapshot_item(proposals, "proposal_id", proposal)
            recompute_pending_proposals(snapshot)

    elif message_type == "review_result":
        review = payload.get("review", {})
        if isinstance(review, dict):
            review["reviewed_at_ms"] = now_ms()
            reviews = snapshot.setdefault("reviews", [])
            if isinstance(reviews, list):
                upsert_snapshot_item(reviews, "reviewer_id", review)

            proposals = snapshot.get("proposals", [])
            if isinstance(proposals, list):
                for proposal in proposals:
                    if not isinstance(proposal, dict):
                        continue
                    if proposal.get("proposal_id") != review.get("proposal_id"):
                        continue
                    findings = proposal.setdefault("review_findings", [])
                    if isinstance(findings, list):
                        findings.extend(review.get("findings", []))
                    if review.get("reviewer_id") == "integration_architect":
                        proposal["integration_verdict"] = review.get("verdict", "")
                    break

    elif message_type == "benchmark_result":
        benchmark = payload.get("benchmark", {})
        if isinstance(benchmark, dict):
            benchmark["generated_at_ms"] = now_ms()
            benchmarks = snapshot.setdefault("benchmarks", [])
            if isinstance(benchmarks, list):
                upsert_snapshot_item(benchmarks, "proposal_id", benchmark)

            proposals = snapshot.get("proposals", [])
            if isinstance(proposals, list):
                for proposal in proposals:
                    if not isinstance(proposal, dict):
                        continue
                    if proposal.get("proposal_id") != benchmark.get("proposal_id"):
                        continue
                    proposal["benchmark_batch_id"] = benchmark.get("batch_id", "")
                    proposal["proposal_status"] = "benchmark_passed" if benchmark.get("green") else "benchmark_failed"
                    proposal["pr_status"] = benchmark.get("pr_status", "")
                    break
            recompute_pending_proposals(snapshot)

    elif message_type == "proposal_promoted":
        promotion = payload.get("promotion", {})
        if isinstance(promotion, dict):
            proposals = snapshot.get("proposals", [])
            if isinstance(proposals, list):
                for proposal in proposals:
                    if not isinstance(proposal, dict):
                        continue
                    if proposal.get("proposal_id") != promotion.get("proposal_id"):
                        continue
                    proposal["proposal_status"] = promotion.get("proposal_status", proposal.get("proposal_status", ""))
                    proposal["pr_status"] = promotion.get("pr_status", proposal.get("pr_status", ""))
                    break
            snapshot["message"] = promotion.get("summary", snapshot.get("message", ""))
            recompute_pending_proposals(snapshot)

    persist_snapshot_state()


def initialize_snapshot_state(
    *,
    run_id: str,
    source_dir: Path,
    reference_dir: Path,
    chapter_pairs: list[ChapterPair],
    available_chapter_pairs: list[ChapterPair],
    scope_label: str,
    gpu_policy: str,
    git_available: bool,
    run_dir: Path,
) -> None:
    global SNAPSHOT_STATE, SNAPSHOT_GLOBAL_PATH, SNAPSHOT_RUN_PATH

    data_root = lab_data_root_for_run(run_dir)
    SNAPSHOT_GLOBAL_PATH = data_root / "snapshot.json"
    SNAPSHOT_RUN_PATH = run_dir / "runtime_snapshot.json"
    started_at = now_ms()
    SNAPSHOT_STATE = {
        "status": "starting",
        "run_id": run_id,
        "current_stage": "boot",
        "message": "Inicializando Improvement Lab",
        "acceleration_summary": "",
        "total_pairs": len(chapter_pairs),
        "processed_pairs": 0,
        "eta_seconds": 0.0,
        "pending_proposals": 0,
        "active_batch_id": f"batch-{run_id[:8]}",
        "git_available": git_available,
        "pr_ready": False,
        "source_dir": str(source_dir),
        "reference_dir": str(reference_dir),
        "chapter_pairs": [pair.to_dict() for pair in chapter_pairs],
        "available_chapter_pairs": [pair.to_dict() for pair in available_chapter_pairs],
        "scope_label": scope_label,
        "gpu_policy": normalize_gpu_policy(gpu_policy),
        "agents": [],
        "proposals": [],
        "reviews": [],
        "benchmarks": [],
        "history": [
            {
                "run_id": run_id,
                "status": "starting",
                "summary": f"Rodada iniciada pelo operador ({scope_label})",
                "total_pairs": len(chapter_pairs),
                "processed_pairs": 0,
                "started_at_ms": started_at,
                "finished_at_ms": 0,
            }
        ],
        "updated_at_ms": started_at,
    }
    persist_snapshot_state()


_stdout_pipe_broken = False


def emit(message_type: str, **payload: object) -> None:
    global _stdout_pipe_broken
    apply_snapshot_event(message_type, payload)
    if _stdout_pipe_broken:
        return
    try:
        print(json.dumps({"type": message_type, **payload}, ensure_ascii=False), flush=True)
    except OSError:
        _stdout_pipe_broken = True


def wait_if_paused(pause_file: Path) -> None:
    while pause_file.exists():
        time.sleep(0.25)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def discover_work_slug(root: Path) -> str:
    corpus_root = root / "pipeline" / "models" / "corpus"
    if corpus_root.exists():
        directories = sorted(path.name for path in corpus_root.iterdir() if path.is_dir())
        if len(directories) == 1:
            return directories[0]
        if "the-regressed-mercenary-has-a-plan" in directories:
            return "the-regressed-mercenary-has-a-plan"
    return "the-regressed-mercenary-has-a-plan"


def preferred_models_dir(root: Path) -> Path:
    app_models = Path("D:/traduzai_data/models")
    if app_models.exists():
        return app_models
    return root / "pipeline" / "models"


def normalize_gpu_policy(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return "require_gpu" if normalized == "require_gpu" else "prefer_gpu"


def filter_chapter_pairs(chapter_pairs: list[ChapterPair], selected_chapters: list[int] | None) -> list[ChapterPair]:
    selected = sorted({int(chapter) for chapter in (selected_chapters or []) if int(chapter) > 0})
    if not selected:
        return list(chapter_pairs)

    selected_set = set(selected)
    return [pair for pair in chapter_pairs if pair.chapter_number in selected_set]


def describe_chapter_scope(chapter_pairs: list[ChapterPair], selected_pairs: list[ChapterPair]) -> str:
    if not selected_pairs:
        return "Nenhum capitulo selecionado"
    if len(selected_pairs) == len(chapter_pairs):
        return "Todos os capitulos"
    if len(selected_pairs) == 1:
        return f"Capitulo {selected_pairs[0].chapter_number}"
    numbers = [pair.chapter_number for pair in selected_pairs]
    contiguous = numbers == list(range(numbers[0], numbers[-1] + 1))
    if contiguous:
        return f"Capitulos {numbers[0]}-{numbers[-1]}"
    return f"{len(selected_pairs)} capitulos selecionados"


def collect_gpu_runtime_status(vision_worker_path: str) -> dict[str, object]:
    status: dict[str, object] = {
        "ocr_worker_ready": False,
        "torch_cuda_ready": False,
        "onnx_gpu_ready": False,
        "ollama_ready": False,
    }

    worker = str(vision_worker_path or "").strip()
    if worker and Path(worker).exists():
        status["ocr_worker_ready"] = True

    try:
        import torch

        if torch.cuda.is_available():
            status["torch_cuda_ready"] = True
            status["torch_gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import onnxruntime as ort
        from pipeline.inpainter.lama_onnx import prepare_windows_onnxruntime_gpu_runtime

        prepare_windows_onnxruntime_gpu_runtime()
        providers = [str(provider) for provider in ort.get_available_providers()]
        status["onnx_available_providers"] = providers
        if "TensorrtExecutionProvider" in providers or "CUDAExecutionProvider" in providers:
            status["onnx_gpu_ready"] = True
    except Exception:
        pass

    try:
        from pipeline.translator.translate import _check_ollama, _pick_ollama_model

        ollama = _check_ollama("http://localhost:11434")
        if ollama.get("running") and ollama.get("models"):
            status["ollama_ready"] = True
            status["ollama_model"] = _pick_ollama_model(list(ollama["models"]), "traduzai-translator")
    except Exception:
        pass

    return status


def enforce_gpu_policy(gpu_policy: str, runtime_status: dict[str, object]) -> None:
    if normalize_gpu_policy(gpu_policy) != "require_gpu":
        return

    required_checks = [
        ("ocr_worker_ready", "OCR GPU indisponivel: o vision-worker nao foi localizado."),
        ("torch_cuda_ready", "PyTorch CUDA indisponivel: o detector visual ainda cairia para CPU."),
        ("onnx_gpu_ready", "ONNX GPU indisponivel: o inpainting ainda cairia para CPU."),
        ("ollama_ready", "Ollama local indisponivel: a traducao GPU local nao esta pronta."),
    ]
    missing = [message for key, message in required_checks if not bool(runtime_status.get(key))]
    if missing:
        raise RuntimeError(" ".join(missing))


def required_reviewers_for(domains: list[str]) -> list[str]:
    reviewers = {"integration_architect"}
    for domain in domains:
        normalized = domain.lower()
        if normalized.startswith("pipeline/") or "pipeline/**" in normalized:
            reviewers.add("python_senior_reviewer")
        if normalized.startswith("src-tauri/") or "src-tauri/**" in normalized:
            reviewers.add("rust_senior_reviewer")
        if normalized.startswith("src/") or "src/**" in normalized:
            reviewers.add("react_ts_senior_reviewer")
        if "ipc" in normalized or "event" in normalized or "artifact" in normalized:
            reviewers.add("tauri_boundary_reviewer")
    return sorted(reviewers)


def emit_agent(
    *,
    agent_id: str,
    label: str,
    layer: str,
    status: str,
    current_task: str,
    last_action: str,
    confidence: float,
    touched_domains: list[str] | None = None,
    proposal_id: str = "",
) -> None:
    emit(
        "agent_status",
        agent={
            "agent_id": agent_id,
            "label": label,
            "layer": layer,
            "status": status,
            "current_task": current_task,
            "last_action": last_action,
            "confidence": confidence,
            "touched_domains": touched_domains or [],
            "proposal_id": proposal_id,
        },
    )


def runtime_agent_id_for_step(step: str) -> tuple[str, str]:
    mapping = {
        "extract": ("runtime_orchestrator", "Runtime Orchestrator"),
        "context": ("runtime_orchestrator", "Runtime Orchestrator"),
        "ocr": ("ocr_critic", "OCR Critic"),
        "translate": ("translation_critic", "Translation Critic"),
        "inpaint": ("inpaint_critic", "Inpaint Critic"),
        "typeset": ("typeset_critic", "Typeset Critic"),
    }
    return mapping.get(step, ("runtime_orchestrator", "Runtime Orchestrator"))


def reviewer_label(reviewer_id: str) -> str:
    labels = {
        "python_senior_reviewer": "Python Senior Reviewer",
        "rust_senior_reviewer": "Rust Senior Reviewer",
        "react_ts_senior_reviewer": "React/TS Senior Reviewer",
        "tauri_boundary_reviewer": "Tauri Boundary Reviewer",
        "integration_architect": "Integration Architect",
    }
    return labels.get(reviewer_id, reviewer_id.replace("_", " ").title())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_acceleration_mode(vision_worker_path: str, runtime_status: dict[str, object], gpu_policy: str) -> str:
    notes: list[str] = []
    worker = str(vision_worker_path or "").strip()
    if bool(runtime_status.get("ocr_worker_ready")) and worker:
        notes.append(f"OCR via {Path(worker).name}")
    else:
        notes.append("OCR na stack Python")

    if bool(runtime_status.get("torch_cuda_ready")):
        gpu_name = str(runtime_status.get("torch_gpu_name", "")).strip()
        notes.append(f"PyTorch CUDA{f' ({gpu_name})' if gpu_name else ''}")

    providers = list(runtime_status.get("onnx_available_providers", []) or [])
    if "TensorrtExecutionProvider" in providers:
        notes.append("inpainting ONNX com TensorRT")
    elif "CUDAExecutionProvider" in providers:
        notes.append("inpainting ONNX com CUDA")

    if bool(runtime_status.get("ollama_ready")):
        model = str(runtime_status.get("ollama_model", "")).strip()
        notes.append(f"traducao local preferida via {model or 'Ollama'}")

    notes.append("GPU estrita" if normalize_gpu_policy(gpu_policy) == "require_gpu" else "GPU preferencial")

    notes.append("typesetting final segue em CPU/Pillow")
    return " | ".join(notes)


def build_pipeline_config(
    *,
    pair: ChapterPair,
    output_dir: Path,
    models_dir: Path,
    pause_file: Path,
    work_slug: str,
    vision_worker_path: str = "",
    gpu_policy: str = "prefer_gpu",
) -> dict:
    return {
        "job_id": f"lab-chapter-{pair.chapter_number:04d}",
        "source_path": pair.source_path,
        "work_dir": str(output_dir),
        "obra": work_slug,
        "capitulo": pair.chapter_number,
        "idioma_origem": "en",
        "idioma_destino": "pt-BR",
        "qualidade": "alta",
        "glossario": {},
        "contexto": {
            "sinopse": "",
            "genero": [],
            "personagens": [],
            "aliases": [],
            "termos": [],
            "relacoes": [],
            "faccoes": [],
            "resumo_por_arco": [],
            "memoria_lexical": {},
            "fontes_usadas": [],
        },
        "models_dir": str(models_dir),
        "ollama_host": "http://localhost:11434",
        "ollama_model": "traduzai-translator",
        "vision_worker_path": str(vision_worker_path or "").strip(),
        "gpu_policy": normalize_gpu_policy(gpu_policy),
        "pause_file": str(pause_file),
    }


def chapter_artifact_path(run_dir: Path, chapter_number: int) -> Path:
    return run_dir / "chapters" / f"chapter-{chapter_number:04d}"


def build_artifact_record(
    *,
    pair: ChapterPair,
    chapter_dir: Path,
    benchmark_payload: dict,
) -> dict:
    return {
        "chapter_number": pair.chapter_number,
        "source_path": pair.source_path,
        "reference_path": pair.reference_path,
        "reference_group": pair.reference_group,
        "output_dir": str(chapter_dir / "output"),
        "project_json": str(chapter_dir / "output" / "project.json"),
        "benchmark": benchmark_payload,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def load_existing_artifact(chapter_dir: Path) -> dict | None:
    artifact_path = chapter_dir / "chapter_artifact.json"
    if not artifact_path.exists():
        return None
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def decode_pipeline_stdout_line(raw_line: str, log_file) -> dict[str, object] | None:
    line = raw_line.strip()
    if not line:
        return None

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        log_file.write(f"[stdout-non-json] {line}\n")
        log_file.flush()
        return None

    if not isinstance(payload, dict):
        log_file.write(f"[stdout-non-object-json] {line}\n")
        log_file.flush()
        return None

    return payload


def run_pipeline_for_pair(
    *,
    root: Path,
    pair: ChapterPair,
    run_dir: Path,
    pause_file: Path,
    processed_before: int,
    total_pairs: int,
    work_slug: str,
    vision_worker_path: str = "",
    gpu_policy: str = "prefer_gpu",
) -> Path:
    chapter_dir = chapter_artifact_path(run_dir, pair.chapter_number)
    output_dir = chapter_dir / "output"
    config_path = chapter_dir / "pipeline_config.json"
    log_path = chapter_dir / "pipeline.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = build_pipeline_config(
        pair=pair,
        output_dir=output_dir,
        models_dir=preferred_models_dir(root),
        pause_file=pause_file,
        work_slug=work_slug,
        vision_worker_path=vision_worker_path,
        gpu_policy=gpu_policy,
    )
    write_json(config_path, pipeline_config)

    command = [
        sys.executable,
        str(root / "pipeline" / "main.py"),
        str(config_path),
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=log_file,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        completed_output_path: Path | None = None

        assert process.stdout is not None
        for raw_line in process.stdout:
            message = decode_pipeline_stdout_line(raw_line, log_file)
            if message is None:
                continue

            message_type = message.get("type", "")

            if message_type == "progress":
                step = str(message.get("step", "pipeline"))
                step_progress = float(message.get("step_progress", 0.0))
                page = int(message.get("current_page", 0))
                total_pages = int(message.get("total_pages", 0))
                pipeline_message = str(message.get("message", "Executando pipeline"))
                chapter_fraction = processed_before + max(0.0, min(1.0, message.get("overall_progress", 0.0) / 100.0))
                remaining_pairs = max(0.0, total_pairs - chapter_fraction)
                chapter_eta = float(message.get("eta_seconds", 0.0))
                overall_eta = chapter_eta + remaining_pairs * 18.0

                emit(
                    "lab_state",
                    status="running",
                    current_stage=f"pipeline:{step}",
                    message=f"Capitulo {pair.chapter_number}: {pipeline_message}",
                    processed_pairs=processed_before,
                    total_pairs=total_pairs,
                    eta_seconds=overall_eta,
                )

                agent_id, label = runtime_agent_id_for_step(step)
                emit_agent(
                    agent_id=agent_id,
                    label=label,
                    layer="runtime",
                    status="running",
                    current_task=f"Capitulo {pair.chapter_number} - {step}",
                    last_action=f"{pipeline_message} ({step_progress:.0f}%)",
                    confidence=max(0.25, min(0.98, step_progress / 100.0)),
                    touched_domains=["pipeline/**"],
                )

                if total_pages > 0:
                    emit_agent(
                        agent_id="runtime_orchestrator",
                        label="Runtime Orchestrator",
                        layer="runtime",
                        status="running",
                        current_task=f"Capitulo {pair.chapter_number} - pagina {page}/{total_pages}",
                        last_action=pipeline_message,
                        confidence=max(0.35, min(0.99, float(message.get("overall_progress", 0.0)) / 100.0)),
                        touched_domains=["pipeline/**"],
                    )

            elif message_type == "error":
                raise RuntimeError(str(message.get("message", "Pipeline retornou erro sem detalhe.")))
            elif message_type == "complete":
                completed_output_path = Path(str(message.get("output_path", output_dir)))

        return_code = process.wait()
        if return_code != 0:
            error_log = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Pipeline falhou no capitulo {pair.chapter_number} com codigo {return_code}.\n{error_log[-2000:]}"
            )

    return completed_output_path or output_dir


def touched_domains_for_benchmark(aggregate_benchmark: dict, chapter_failures: int) -> list[str]:
    metrics = aggregate_benchmark.get("metrics", {})
    domains = {"pipeline/**"}

    if chapter_failures > 0:
        domains.add("src-tauri/**")
        domains.add("ipc_artifact_contract")

    if float(metrics.get("visual_cleanup", 0.0)) < 65.0:
        domains.add("src/**")
    if float(metrics.get("layout_occupancy", 0.0)) < 65.0:
        domains.add("src/**")

    return sorted(domains)


def build_proposal(run_id: str, git_available: bool, aggregate_benchmark: dict, chapter_failures: int) -> dict:
    touched_domains = touched_domains_for_benchmark(aggregate_benchmark, chapter_failures)
    metrics = aggregate_benchmark.get("metrics", {})
    weakest_metric = min(metrics.items(), key=lambda item: item[1]) if metrics else ("textual_similarity", 0.0)
    benchmark_green = bool(aggregate_benchmark.get("green", False))

    return {
        "proposal_id": f"proposal-{run_id[:8]}",
        "batch_id": f"batch-{run_id[:8]}",
        "title": "Aprimoramento guiado pelo benchmark real do Lab",
        "summary": (
            f"Rodada consolidada com foco em {weakest_metric[0]} "
            f"({float(weakest_metric[1]):.1f}) e "
            f"{chapter_failures} capitulo(s) com falha."
        ),
        "author": "code_author",
        "risk": "medio" if benchmark_green else "alto",
        "touched_domains": touched_domains,
        "required_reviewers": required_reviewers_for(touched_domains),
        "review_findings": [],
        "integration_verdict": "",
        "benchmark_batch_id": f"batch-{run_id[:8]}",
        "proposal_status": "reviewing",
        "pr_status": "awaiting_review",
        "git_available": git_available,
    }


def reviewer_result_for(proposal: dict, aggregate_benchmark: dict, reviewer_id: str) -> tuple[str, dict]:
    green = bool(aggregate_benchmark.get("green", False))
    metrics = aggregate_benchmark.get("metrics", {})
    weakest_metric = min(metrics.items(), key=lambda item: item[1]) if metrics else ("textual_similarity", 0.0)

    findings = [
        {
            "title": f"{reviewer_id} revisou o lote {proposal['batch_id']}",
            "body": aggregate_benchmark.get("summary", "Benchmark consolidado sem resumo."),
            "severity": "info" if green else "warning",
            "file_path": "pipeline/**" if reviewer_id != "integration_architect" else "",
        }
    ]

    if reviewer_id == "integration_architect":
        verdict = "approve" if green else "request_changes"
        findings.append(
            {
                "title": "Gate de integracao",
                "body": (
                    "A rodada esta pronta para promocao manual."
                    if green
                    else f"Antes de promover, precisamos subir {weakest_metric[0]}."
                ),
                "severity": "info" if green else "warning",
                "file_path": "",
            }
        )
        return verdict, {"findings": findings}

    if reviewer_id == "tauri_boundary_reviewer":
        verdict = "approve" if green else "needs_benchmark_focus"
    else:
        verdict = "approve" if green else "request_changes"
    return verdict, {"findings": findings}


def main() -> int:
    if len(sys.argv) < 2:
        emit("lab_state", status="error", message="Nenhum arquivo de configuracao do Lab foi fornecido.")
        return 1

    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pause_file = Path(config["pause_file"])
    run_id = str(config["run_id"])
    git_available = bool(config.get("git_available", False))
    vision_worker_path = str(config.get("vision_worker_path", "") or "").strip()
    gpu_policy = normalize_gpu_policy(str(config.get("gpu_policy", "prefer_gpu")))
    selected_chapters = [int(value) for value in config.get("selected_chapters", []) or [] if int(value) > 0]
    run_dir = config_path.parent
    root = project_root()
    work_slug = discover_work_slug(root)

    source_dir = Path(config["source_dir"])
    reference_dir = Path(config["reference_dir"])
    available_chapter_pairs = pair_chapters(source_dir, reference_dir)
    chapter_pairs = filter_chapter_pairs(available_chapter_pairs, selected_chapters)
    scope_label = str(config.get("scope_label") or describe_chapter_scope(available_chapter_pairs, chapter_pairs))
    runtime_status = collect_gpu_runtime_status(vision_worker_path)
    acceleration_summary = summarize_acceleration_mode(vision_worker_path, runtime_status, gpu_policy)
    total_pairs = len(chapter_pairs)
    initialize_snapshot_state(
        run_id=run_id,
        source_dir=source_dir,
        reference_dir=reference_dir,
        chapter_pairs=chapter_pairs,
        available_chapter_pairs=available_chapter_pairs,
        scope_label=scope_label,
        gpu_policy=gpu_policy,
        git_available=git_available,
        run_dir=run_dir,
    )
    if SNAPSHOT_STATE is not None:
        SNAPSHOT_STATE["acceleration_summary"] = acceleration_summary
        SNAPSHOT_STATE["available_chapter_pairs"] = [pair.to_dict() for pair in available_chapter_pairs]
        SNAPSHOT_STATE["scope_label"] = scope_label
        SNAPSHOT_STATE["gpu_policy"] = gpu_policy
        persist_snapshot_state()

    if not chapter_pairs:
        emit(
            "lab_state",
            status="error",
            run_id=run_id,
            current_stage="scope_error",
            message="A selecao atual nao corresponde a nenhum capitulo pareado do corpus.",
            acceleration_summary=acceleration_summary,
            total_pairs=0,
            processed_pairs=0,
            eta_seconds=0,
            pending_proposals=0,
            active_batch_id=f"batch-{run_id[:8]}",
            git_available=git_available,
            source_dir=str(source_dir),
            reference_dir=str(reference_dir),
            chapter_pairs=[],
            available_chapter_pairs=[pair.to_dict() for pair in available_chapter_pairs],
            scope_label=scope_label,
            gpu_policy=gpu_policy,
        )
        return 1

    try:
        enforce_gpu_policy(gpu_policy, runtime_status)
    except RuntimeError as exc:
        emit(
            "lab_state",
            status="error",
            run_id=run_id,
            current_stage="gpu_guard",
            message=str(exc),
            acceleration_summary=acceleration_summary,
            total_pairs=total_pairs,
            processed_pairs=0,
            eta_seconds=0,
            pending_proposals=0,
            active_batch_id=f"batch-{run_id[:8]}",
            git_available=git_available,
            source_dir=str(source_dir),
            reference_dir=str(reference_dir),
            chapter_pairs=[pair.to_dict() for pair in chapter_pairs],
            available_chapter_pairs=[pair.to_dict() for pair in available_chapter_pairs],
            scope_label=scope_label,
            gpu_policy=gpu_policy,
        )
        return 1
    textual_profile, visual_profile = load_corpus_profiles(root, work_slug)
    artifacts: list[dict] = []
    chapter_results = []
    chapter_failures = 0

    write_json(
        run_dir / "chapter_pairs.json",
        {
            "chapter_pairs": [pair.to_dict() for pair in chapter_pairs],
            "available_chapter_pairs": [pair.to_dict() for pair in available_chapter_pairs],
            "scope_label": scope_label,
            "gpu_policy": gpu_policy,
        },
    )

    emit(
        "lab_state",
        status="running",
        run_id=run_id,
        current_stage="discover",
        message=(
            f"Escopo: {scope_label}. "
            "Mapeando corpus de referencia e preparando o pipeline real do Lab. "
            + acceleration_summary
        ),
        acceleration_summary=acceleration_summary,
        total_pairs=total_pairs,
        processed_pairs=0,
        eta_seconds=max(0, total_pairs * 120),
        pending_proposals=0,
        active_batch_id=f"batch-{run_id[:8]}",
        git_available=git_available,
        source_dir=str(source_dir),
        reference_dir=str(reference_dir),
        chapter_pairs=[pair.to_dict() for pair in chapter_pairs],
        available_chapter_pairs=[pair.to_dict() for pair in available_chapter_pairs],
        scope_label=scope_label,
        gpu_policy=gpu_policy,
    )

    emit_agent(
        agent_id="improvement_planner",
        label="Improvement Planner",
        layer="lab",
        status="running",
        current_task="Planejando rodada real do corpus",
        last_action=f"{total_pairs} capitulos no escopo atual ({scope_label})",
        confidence=0.93,
    )
    emit_agent(
        agent_id="runtime_orchestrator",
        label="Runtime Orchestrator",
        layer="runtime",
        status="idle",
        current_task="Aguardando capitulo",
        last_action="Pipeline ainda nao iniciou nenhum capitulo",
        confidence=0.6,
        touched_domains=["pipeline/**"],
    )

    for processed_before, pair in enumerate(chapter_pairs):
        wait_if_paused(pause_file)
        chapter_dir = chapter_artifact_path(run_dir, pair.chapter_number)
        output_dir = chapter_dir / "output"
        emit(
            "lab_state",
            status="running",
            current_stage="chapter_setup",
            message=f"Preparando capitulo {pair.chapter_number} para execucao real do pipeline.",
            processed_pairs=processed_before,
            total_pairs=total_pairs,
            eta_seconds=max(0, (total_pairs - processed_before) * 120),
        )

        emit_agent(
            agent_id="runtime_orchestrator",
            label="Runtime Orchestrator",
            layer="runtime",
            status="running",
            current_task=f"Capitulo {pair.chapter_number}",
            last_action="Gerando config e preparando diretorio de artefatos",
            confidence=max(0.4, processed_before / max(1, total_pairs)),
            touched_domains=["pipeline/**"],
        )

        try:
            existing_artifact = load_existing_artifact(chapter_dir)
            if existing_artifact and Path(existing_artifact.get("project_json", "")).exists():
                benchmark_payload = existing_artifact.get("benchmark", {})
                artifacts.append(existing_artifact)
                chapter_results.append(benchmark_payload)
                emit(
                    "lab_state",
                    status="running",
                    current_stage="chapter_cached",
                    message=f"Capitulo {pair.chapter_number} reaproveitado do cache persistido.",
                    processed_pairs=processed_before + 1,
                    total_pairs=total_pairs,
                    eta_seconds=max(0, (total_pairs - (processed_before + 1)) * 90),
                )
                continue

            finished_output_dir = run_pipeline_for_pair(
                root=root,
                pair=pair,
                run_dir=run_dir,
                pause_file=pause_file,
                processed_before=processed_before,
                total_pairs=total_pairs,
                work_slug=work_slug,
                vision_worker_path=vision_worker_path,
                gpu_policy=gpu_policy,
            )

            benchmark = benchmark_chapter_output(
                output_dir=finished_output_dir,
                source_archive=Path(pair.source_path),
                reference_archive=Path(pair.reference_path),
                textual_profile=textual_profile,
                visual_profile=visual_profile,
            )
            benchmark_payload = benchmark.to_dict()
            chapter_results.append(benchmark_payload)

            artifact_record = build_artifact_record(
                pair=pair,
                chapter_dir=chapter_dir,
                benchmark_payload=benchmark_payload,
            )
            artifacts.append(artifact_record)
            write_json(chapter_dir / "chapter_artifact.json", artifact_record)
            write_json(run_dir / "artifacts.json", {"chapters": artifacts})

            emit(
                "lab_state",
                status="running",
                current_stage="chapter_done",
                message=f"Capitulo {pair.chapter_number} concluido com score {benchmark.score_after:.1f}.",
                processed_pairs=processed_before + 1,
                total_pairs=total_pairs,
                eta_seconds=max(0, (total_pairs - (processed_before + 1)) * 110),
            )
        except Exception as exc:
            chapter_failures += 1
            failure_payload = {
                "chapter_number": pair.chapter_number,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "source_path": pair.source_path,
                "reference_path": pair.reference_path,
            }
            write_json(chapter_dir / "chapter_error.json", failure_payload)
            artifacts.append(
                {
                    "chapter_number": pair.chapter_number,
                    "source_path": pair.source_path,
                    "reference_path": pair.reference_path,
                    "error": str(exc),
                }
            )
            write_json(run_dir / "artifacts.json", {"chapters": artifacts})
            emit(
                "lab_state",
                status="running",
                current_stage="chapter_error",
                message=f"Capitulo {pair.chapter_number} falhou: {exc}",
                processed_pairs=processed_before,
                total_pairs=total_pairs,
                eta_seconds=max(0, (total_pairs - processed_before) * 120),
            )
            emit_agent(
                agent_id="runtime_orchestrator",
                label="Runtime Orchestrator",
                layer="runtime",
                status="error",
                current_task=f"Capitulo {pair.chapter_number}",
                last_action=str(exc),
                confidence=0.2,
                touched_domains=["pipeline/**"],
            )

    if not chapter_results:
        emit(
            "lab_state",
            status="error",
            run_id=run_id,
            current_stage="error",
            message="Nenhum capitulo foi concluido. O Lab nao conseguiu gerar benchmark real.",
            total_pairs=total_pairs,
            processed_pairs=0,
            eta_seconds=0,
            pending_proposals=0,
            active_batch_id=f"batch-{run_id[:8]}",
            git_available=git_available,
            source_dir=str(source_dir),
            reference_dir=str(reference_dir),
            chapter_pairs=[pair.to_dict() for pair in chapter_pairs],
            available_chapter_pairs=[pair.to_dict() for pair in available_chapter_pairs],
            scope_label=scope_label,
            gpu_policy=gpu_policy,
        )
        return 1

    aggregate = aggregate_benchmark_results(
        [
            benchmark_chapter_output(
                output_dir=Path(artifact["output_dir"]),
                source_archive=Path(artifact["source_path"]),
                reference_archive=Path(artifact["reference_path"]),
                textual_profile=textual_profile,
                visual_profile=visual_profile,
            )
            for artifact in artifacts
            if artifact.get("output_dir")
        ]
    )
    aggregate_payload = aggregate.to_dict()
    write_json(run_dir / "benchmark_summary.json", aggregate_payload)

    # === Critics rule-based (Fase 2) =====================================
    # Roda os 4 critics locais (OCR / Translation / Typeset / Inpaint) sobre
    # os artefatos completados, agrega os findings e gera propostas
    # priorizadas via Planner. Cada proposal mira um problema especifico
    # detectado e carrega `local_patch_hint` p/ o Coder downstream.
    successful_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("project_json")
        and Path(artifact.get("project_json", "")).exists()
        and not artifact.get("error")
    ]
    try:
        findings = run_all_critics_over_chapters(successful_artifacts)
    except Exception as exc:  # pragma: no cover - nunca deve derrubar o Lab
        findings = []
        write_json(
            run_dir / "critics_error.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
    write_json(
        run_dir / "critics_findings.json",
        {"findings": [f.to_dict() for f in findings], "count": len(findings)},
    )

    planner_proposals = build_proposals(findings, aggregate_payload, run_id)

    # Converte para payload compativel com LabProposal + enriquece reviewers.
    proposals: list[dict] = []
    for proposal_obj in planner_proposals:
        payload = proposal_to_lab_payload(proposal_obj, run_id, git_available)
        payload["required_reviewers"] = required_reviewers_for(payload["touched_domains"])
        proposals.append(payload)

    # Fallback: mesmo sem findings o Lab ainda emite uma proposta consolidada
    # para manter o contrato do frontend (1 rodada = >=1 proposta).
    if not proposals:
        proposals.append(build_proposal(run_id, git_available, aggregate_payload, chapter_failures))

    write_json(run_dir / "proposals.json", {"proposals": proposals})
    # Mantem proposal.json (singular) apontando para a proposta top-priority
    # p/ retrocompatibilidade com consumidores antigos.
    write_json(run_dir / "proposal.json", proposals[0])

    reviewer_agent = ClaudeReviewerAgent()

    for proposal in proposals:
        wait_if_paused(pause_file)
        emit("review_requested", proposal=proposal)

        for reviewer_id in proposal["required_reviewers"]:
            wait_if_paused(pause_file)
            label = reviewer_label(reviewer_id)
            verdict, payload = reviewer_agent.review(
                proposal, aggregate_payload, reviewer_id, repo_root=root
            )
            emit_agent(
                agent_id=reviewer_id,
                label=label,
                layer="review",
                status="running",
                current_task=f"Revisando {proposal.get('title', proposal['proposal_id'])}",
                last_action=f"Analisando proposta {proposal['proposal_id']}",
                confidence=0.82,
                touched_domains=proposal["touched_domains"],
                proposal_id=proposal["proposal_id"],
            )
            time.sleep(0.04)
            emit(
                "review_result",
                review={
                    "proposal_id": proposal["proposal_id"],
                    "reviewer_id": reviewer_id,
                    "reviewer_label": label,
                    "verdict": verdict,
                    "touched_domains": proposal["touched_domains"],
                    "findings": payload["findings"],
                },
            )
            emit_agent(
                agent_id=reviewer_id,
                label=label,
                layer="review",
                status="idle",
                current_task="Aguardando proxima proposta",
                last_action=f"Veredito emitido: {verdict}",
                confidence=0.82,
                touched_domains=proposal["touched_domains"],
                proposal_id=proposal["proposal_id"],
            )

    top_proposal = proposals[0]

    emit_agent(
        agent_id="eval_judge",
        label="Eval Judge",
        layer="lab",
        status="running",
        current_task="Consolidando benchmark real do corpus completo",
        last_action=(
            f"{len(chapter_results)} capitulos concluidos, "
            f"{chapter_failures} com falha e {len(proposals)} proposta(s) geradas"
        ),
        confidence=0.94,
        proposal_id=top_proposal["proposal_id"],
    )
    time.sleep(0.08)
    emit(
        "benchmark_result",
        benchmark={
            "proposal_id": top_proposal["proposal_id"],
            "batch_id": top_proposal["batch_id"],
            "score_before": aggregate.score_before,
            "score_after": aggregate.score_after,
            "green": aggregate.green,
            "summary": aggregate.summary,
            "metrics": aggregate.metrics.to_dict(),
            "git_available": git_available,
            "pr_status": "awaiting_manual_approval" if git_available else "blocked_no_git",
        },
    )
    emit_agent(
        agent_id="eval_judge",
        label="Eval Judge",
        layer="lab",
        status="idle",
        current_task="Aguardando proxima execucao",
        last_action="Benchmark real consolidado e persistido",
        confidence=0.94,
        proposal_id=top_proposal["proposal_id"],
    )

    write_json(
        run_dir / "run_summary.json",
        {
            "run_id": run_id,
            "processed_pairs": len(chapter_results),
            "total_pairs": total_pairs,
            "chapter_failures": chapter_failures,
            "benchmark": aggregate_payload,
            "proposal": top_proposal,
            "proposals": proposals,
            "findings_count": len(findings),
        },
    )

    emit(
        "lab_state",
        status="completed",
        run_id=run_id,
        current_stage="awaiting_decision",
        message=(
            f"Rodada finalizada com {len(chapter_results)} capitulos benchmarkados, "
            f"{chapter_failures} falha(s) e {len(proposals)} proposta(s). "
            f"Escopo: {scope_label}."
        ),
        total_pairs=total_pairs,
        processed_pairs=len(chapter_results),
        eta_seconds=0,
        pending_proposals=len(proposals),
        active_batch_id=top_proposal["batch_id"],
        git_available=git_available,
        source_dir=str(source_dir),
        reference_dir=str(reference_dir),
        chapter_pairs=[pair.to_dict() for pair in chapter_pairs],
        available_chapter_pairs=[pair.to_dict() for pair in available_chapter_pairs],
        scope_label=scope_label,
        gpu_policy=gpu_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
