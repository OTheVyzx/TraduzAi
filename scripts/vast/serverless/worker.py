from __future__ import annotations

import os
import time

try:
    from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig
except ImportError as exc:  # pragma: no cover - executed only inside the Vast serverless image.
    raise SystemExit(
        "Instale o pacote vastai no template serverless: "
        "python -m pip install -r scripts/vast/requirements-serverless.txt"
    ) from exc


MODEL_SERVER_URL = f"http://127.0.0.1:{os.environ.get('TRADUZAI_SERVERLESS_PORT', '18000')}"


def workload_cost(request) -> int:
    payload = getattr(request, "json", None) or {}
    if isinstance(payload, dict):
        pages = payload.get("page_count") or payload.get("pages") or 1
    else:
        pages = 1
    try:
        return max(1, int(pages)) * int(os.environ.get("TRADUZAI_SERVERLESS_COST_PER_PAGE", "100"))
    except (TypeError, ValueError):
        return int(os.environ.get("TRADUZAI_SERVERLESS_COST_PER_PAGE", "100"))


def main() -> None:
    wait_for_model_server()
    model_log_file = os.environ.get("TRADUZAI_SERVERLESS_MODEL_LOG", "/tmp/traduzai-serverless-model.log")
    worker = Worker(
        WorkerConfig(
            model_server_url=MODEL_SERVER_URL,
            model_server_port=int(os.environ.get("TRADUZAI_SERVERLESS_PORT", "18000")),
            model_log_file=model_log_file,
            handlers=[
                HandlerConfig(
                    route="/run",
                    methods=["POST"],
                    max_queue_time=int(os.environ.get("TRADUZAI_SERVERLESS_MAX_QUEUE_TIME", "7200")),
                    workload_calculator=workload_cost,
                    benchmark_config=BenchmarkConfig(num_requests=1),
                )
            ],
            log_action_config=LogActionConfig(on_load=["TRADUZAI_SERVERLESS_READY"]),
        ),
    )
    worker.run()


def wait_for_model_server() -> None:
    import urllib.request

    deadline = time.monotonic() + float(os.environ.get("TRADUZAI_SERVERLESS_STARTUP_TIMEOUT", "900"))
    health_url = f"{MODEL_SERVER_URL}/health"
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"model server nao ficou pronto: {last_error}")


if __name__ == "__main__":
    main()
